"""Actual finite water/solute inventories, coupled H, limits and exact replay."""

import json
from dataclasses import replace

import pytest
from tests.reference.incompressible_fixtures import amounts, transport_properties

from chemworld.physchem.transport_devices import FiniteTransportSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.runtime.transport_devices import unpack_transport
from chemworld.world.transport_world import transport_world_spec


def execute(runtime, operation, **kwargs):
    result = runtime.execute({"operation": operation, **kwargs})
    assert result.info["transaction_status"] == "committed", {
        "error": result.info.get("error_message"),
        "outcome": result.info.get("outcome_diagnostics"),
        "checks": [c for c in result.info.get("constitution_checks", []) if not c["passed"]],
    }
    return result


def world(name, area, *, feed_volume_L=0.006, receiver_capacity_L=0.1):
    prop = transport_properties(name)
    spec = FiniteTransportSpec(prop, receiver_capacity_L=receiver_capacity_L)
    spec = replace(spec, path=replace(spec.path, exchanger_area_m2=area))
    source_x, receiver_x = (0.25, 0.05) if name == "MCA" else (0.4, 0.1)

    def fractions(x):
        n = amounts(prop.incompressible_model, x)
        return {k: v / sum(n.values()) for k, v in n.items()}

    return transport_world_spec(
        spec,
        feed_temperature_K=310.0,
        receiver_temperature_K=295.0,
        feed_volume_L=feed_volume_L,
        receiver_volume_L=0.003,
        feed_mole_fractions=fractions(source_x),
        receiver_mole_fractions=fractions(receiver_x),
    )


def inventory_record(state):
    _, spec, inventory = unpack_transport(state)
    model = spec.properties.incompressible_model
    tanks = (state.inactive_vessels[spec.source_vessel_id], state)
    return {
        "amounts_mol": inventory.amounts_mol,
        "enthalpies_J": inventory.enthalpies_J,
        "time_s": state.ledger.time_s,
        "temperatures_K": [s.temperature_K for s in tanks],
        "volumes_L": [s.volume_L for s in tanks],
        "mass_fractions": [model.inventory_coordinates(n)[3] for n in inventory.amounts_mol],
        "component_masses_kg_mol": {
            c.identifier: c.molecular_weight_g_mol / 1000 for c in model.components
        },
        "solute_id": model.solute_id,
        "pressure_Pa": spec.properties.pressure_Pa,
        "support_capacities_J_K": spec.capacities_j_k,
        "support_reference_K": spec.support_reference_temperature_K,
    }


@pytest.mark.parametrize("name", ("MEG", "MPG", "MCA"))
@pytest.mark.parametrize("area", (0.0, 0.001))
def test_real_solution_transfer_mixing_and_snapshot(name, area, tmp_path, record_property):
    runtime = ChemWorldRuntime(composition=world(name, area))
    restored = None
    history = {"fluid": name, "area_m2": area, "stages": [], "replay_equal": False}
    try:
        runtime.reset(seed=25)
        start = inventory_record(runtime._state)
        history["initial"] = start
        execute(runtime, "set_flow_rate", flow_rate_mL_min=5.0)
        duty = 0.0

        def advance(operation, duration):
            nonlocal duty
            result = execute(runtime, operation, duration_s=duration)
            diagnostics = result.info["outcome_diagnostics"]["transport"]
            duty += diagnostics["hydraulic_work_J"] + diagnostics["utility_heat_J"]
            current = inventory_record(runtime._state)
            current["diagnostics"] = diagnostics
            current["cumulative_duty_J"] = duty
            history["stages"].append(current)
            for key in start["amounts_mol"][0]:
                assert sum(n[key] for n in current["amounts_mol"]) == pytest.approx(
                    sum(n[key] for n in start["amounts_mol"]), abs=1e-11, rel=0
                )
            energy_error = sum(current["enthalpies_J"]) - sum(start["enthalpies_J"]) - duty
            current["enthalpy_error_J"] = energy_error
            assert abs(energy_error) <= 1e-5
            return current

        advance("run_flow", 20.0)
        path = tmp_path / (name + str(area) + ".json")
        runtime.snapshot().save(path)
        restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(path))
        advance("wait", 20.0)
        execute(restored, "wait", duration_s=20.0)
        assert restored._state.to_dict() == runtime._state.to_dict()
        before_stop = inventory_record(runtime._state)
        execute(runtime, "set_flow_rate", flow_rate_mL_min=0.0)
        execute(restored, "set_flow_rate", flow_rate_mL_min=0.0)
        final = advance("wait", 10.0)
        execute(restored, "wait", duration_s=10.0)
        assert restored._state.to_dict() == runtime._state.to_dict()
        assert final["time_s"] == pytest.approx(50.0, abs=1e-10, rel=0)
        assert final["amounts_mol"] == before_stop["amounts_mol"]
        assert final["mass_fractions"][1] > start["mass_fractions"][1]
        assert final["mass_fractions"][1] < start["mass_fractions"][0]
        assert final["volumes_L"][0] < start["volumes_L"][0]
        assert final["volumes_L"][1] > start["volumes_L"][1]
        history["replay_equal"] = True
        history["completed"] = True
    finally:
        record_property("incompressible_history", json.dumps(history, allow_nan=False))
        runtime.close()
        if restored is not None:
            restored.close()


@pytest.mark.parametrize("name", ("MEG", "MPG", "MCA"))
@pytest.mark.parametrize("event", ("feed_pickup_exhausted", "receiver_full"))
def test_finite_solution_events_stop_at_physical_inventory(name, event, record_property):
    options = (
        {"feed_volume_L": 0.0002}
        if event == "feed_pickup_exhausted"
        else {"receiver_capacity_L": 0.0031}
    )
    runtime = ChemWorldRuntime(composition=world(name, 0.0, **options))
    history = {"fluid": name, "event": event}
    try:
        runtime.reset(seed=25)
        start = inventory_record(runtime._state)
        execute(runtime, "set_flow_rate", flow_rate_mL_min=20.0)
        result = execute(runtime, "run_flow", duration_s=120.0)
        final = inventory_record(runtime._state)
        diagnostics = result.info["outcome_diagnostics"]
        history.update(initial=start, final=final, diagnostics=diagnostics)
        assert diagnostics["stopping_event"] == event
        assert 0 < result.info["actual_duration_s"] < 120
        index, target = (0, 1e-6) if event == "feed_pickup_exhausted" else (1, 0.0031)
        assert final["volumes_L"][index] == pytest.approx(target, rel=0, abs=1e-10)
        for key in start["amounts_mol"][0]:
            assert sum(n[key] for n in final["amounts_mol"]) == pytest.approx(
                sum(n[key] for n in start["amounts_mol"]), abs=1e-11, rel=0
            )
        duty = diagnostics["transport"]["hydraulic_work_J"]
        history["enthalpy_error_J"] = sum(final["enthalpies_J"]) - sum(start["enthalpies_J"]) - duty
        assert abs(history["enthalpy_error_J"]) <= 1e-5
        history["completed"] = True
    finally:
        record_property("incompressible_event", json.dumps(history, allow_nan=False))
        runtime.close()
