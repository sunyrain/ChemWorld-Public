"""Finite phase-selected material on original energy and transaction boundaries."""

import json
from dataclasses import replace

import pytest
from tests.reference.humid_air_material_fixtures import packed, record, world
from tests.test_humid_air_volume import thermal_inputs

from chemworld.physchem.humid_air_thermal import HumidAirIsochoricThermodynamics
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot


def execute(runtime, operation, **kwargs):
    print(f"humid material {operation}: {kwargs}", flush=True)
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", {
        k: result.info.get(k)
        for k in (
            "outcome_diagnostics",
            "rollback_reason",
            "error_message",
            "constitution_checks",
            "action_preconditions",
        )
    }
    return result


def test_optional_outlet_json_and_rejected_inputs():
    thermal, state = thermal_inputs()
    assert "material_outlet_phase" not in thermal.to_dict()
    with pytest.raises(ValueError, match="declared"):
        thermal.material_outlet(state, 0.25)
    with pytest.raises(ValueError, match="outlet"):
        replace(thermal, material_outlet_phase="bulk")
    for phase in ("gas", "condensate"):
        model = replace(thermal, material_outlet_phase=phase)
        assert (
            HumidAirIsochoricThermodynamics.from_dict(json.loads(json.dumps(model.to_dict())))
            == model
        )
        for fraction in (True, 0.0, -1.0, float("nan"), float("inf"), 1.01):
            with pytest.raises(ValueError):
                model.material_outlet(state, fraction)
    with pytest.raises(ValueError, match="retain"):
        replace(thermal, material_outlet_phase="gas").material_outlet(state, 1.0)


@pytest.mark.parametrize("phase", ("liquid", "ice"))
@pytest.mark.parametrize("mode", ("mixing", "dewatering"))
def test_real_phase_packet_history_and_exact_snapshot(phase, mode, tmp_path, record_property):
    payload, models, dew = world(phase, "gas" if mode == "mixing" else "condensate")
    runtime, resumed = ChemWorldRuntime(composition=payload), None
    history = {"phase": phase, "mode": mode, "stages": [], "completed": False}
    thermo = models["reactor"].thermodynamics
    try:
        runtime.reset(seed=25)
        initial = record(runtime._state, models)
        history["initial"] = initial

        def check(label):
            now = record(runtime._state, models)
            history["stages"].append({"label": label, **now})
            for key in runtime._state.species_amounts:
                assert sum(v["amounts_mol"][key] for v in now["vessels"].values()) == pytest.approx(
                    sum(v["amounts_mol"][key] for v in initial["vessels"].values()),
                    abs=1e-11,
                    rel=0,
                )
            balance = (
                sum(v["energy_J"] for v in now["vessels"].values())
                - sum(v["energy_J"] for v in initial["vessels"].values())
                - now["jacket_heat_J"]
            )
            history["stages"][-1]["energy_balance_error_J"] = balance
            assert abs(balance) <= 1e-4
            assert all(
                v["volume_L"] == initial["vessels"][name]["volume_L"]
                for name, v in now["vessels"].items()
            )
            return now

        if mode == "dewatering":
            current = packed(runtime._state)
            duration = (
                thermo.internal_energy_j(current)
                - thermo.internal_energy_j(replace(current, temperature_K=dew - 10))
            ) / 2
            execute(runtime, "heat", target_temperature_K=dew - 15, duration_s=duration / 2)
            execute(runtime, "wait", duration_s=duration / 2)
            cold = check("cooled")
            assert cold["vessels"]["reactor"]["temperature_K"] == pytest.approx(dew - 10, abs=1e-5)
            assert cold["vessels"]["reactor"]["equilibrium"]["condensed_water_mass_kg"] > 0
        before = record(runtime._state, models)
        path = tmp_path / f"{phase}-{mode}.json"
        runtime.snapshot().save(path)
        resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
        fraction = 0.2 if mode == "mixing" else 0.9
        action = {"connection": "reactor-to-receiver", "transfer_fraction": fraction, "mixing": 1}
        result = execute(runtime, "route_material", **action)
        execute(resumed, "route_material", **action)
        receipt = runtime._state.metadata["last_material_packet"]
        history["packet"] = receipt
        after = check("packet")
        assert result.info["actual_duration_s"] == 0
        assert after["time_s"] == before["time_s"]
        assert after["cost"] - before["cost"] == pytest.approx(0.005, abs=1e-12)
        assert after["jacket_heat_J"] == before["jacket_heat_J"]
        for name, sign in (("reactor", -1), ("receiver", 1)):
            error = (
                after["vessels"][name]["energy_J"]
                - before["vessels"][name]["energy_J"]
                - sign * receipt["enthalpy_J"]
            )
            assert abs(error) <= 1e-6
        assert runtime._state.to_dict() == resumed._state.to_dict()
        post_packet_path = tmp_path / f"{phase}-{mode}-after-packet.json"
        runtime.snapshot().save(post_packet_path)
        resumed.close()
        resumed = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(post_packet_path))
        for run in (runtime, resumed):
            execute(run, "pause_control")
            execute(run, "wait", duration_s=1.0)
        assert runtime._state.to_dict() == resumed._state.to_dict()
        check("isolated_wait")
        history["replay_equal"] = True
        if mode == "dewatering":
            current = packed(runtime._state)
            duration = (
                thermo.internal_energy_j(replace(current, temperature_K=dew + 10))
                - thermo.internal_energy_j(current)
            ) / 2
            execute(runtime, "heat", target_temperature_K=dew + 15, duration_s=duration / 2)
            execute(runtime, "wait", duration_s=duration / 2)
        final = check("final")
        if mode == "dewatering":
            source, original = final["vessels"]["reactor"], initial["vessels"]["reactor"]
            assert source["temperature_K"] == pytest.approx(dew + 10, abs=1e-5)
            assert source["amounts_mol"]["water"] < original["amounts_mol"]["water"]
            assert (
                source["equilibrium"]["gas"]["humidity_ratio_kg_water_kg_dry_air"]
                < (original["equilibrium"]["gas"]["humidity_ratio_kg_water_kg_dry_air"])
            )
            assert all(receipt["amounts_mol"][key] == 0 for key in thermo.inventory_keys[:3])
        else:
            assert receipt["material_phase"] == "gas"
            assert (
                final["vessels"]["receiver"]["amounts_mol"]["water"]
                > (initial["vessels"]["receiver"]["amounts_mol"]["water"])
            )
        history["completed"] = True
    finally:
        record_property("humid_air_material_history", json.dumps(history, allow_nan=False))
        runtime.close()
        if resumed is not None:
            resumed.close()


@pytest.mark.parametrize("outlet,fraction", [(None, 0.2), ("condensate", 0.2), ("gas", 1.0)])
def test_undeclared_absent_phase_and_empty_gas_are_atomic_rejections(outlet, fraction):
    payload, _, _ = world("liquid", outlet)
    runtime = ChemWorldRuntime(composition=payload)
    try:
        runtime.reset(seed=25)
        before = runtime._state
        result = runtime.execute(
            {
                "operation": "route_material",
                "connection": "reactor-to-receiver",
                "transfer_fraction": fraction,
                "mixing": 1,
            }
        )
        assert result.info["transaction_status"] != "committed"
        after = runtime._state
        assert after.species_amounts == before.species_amounts
        assert after.temperature_K == before.temperature_K
        assert after.inactive_vessels == before.inactive_vessels
        assert after.ledger.time_s == before.ledger.time_s
        assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
    finally:
        runtime.close()
