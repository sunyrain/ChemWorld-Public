"""Real finite resin cycles, thermodynamics, original owners and exact replay."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from math import exp, log
from pathlib import Path

import pytest

from chemworld.physchem.cation_exchange import (
    CationExchangeSpec,
    advance_cation_exchange,
    cation_exchange_inventory,
    cation_exchange_profile,
)
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.cation_exchange import CationExchangeAdapter, unpack_cation_exchange
from chemworld.world.cation_exchange_world import cation_exchange_world_spec
from reference.cation_exchange49_fixtures import (
    DATA,
    fixed_cases,
    initial_amounts,
    inventory_case,
    spec_case,
)
from reference.cation_exchange49_reference import independent


def setup(spec=None, *, shared=False, fraction=0.0, contact="ordinary", amounts=None):
    spec = spec or spec_case()
    rt = ChemWorldRuntime(
        composition=cation_exchange_world_spec(
            spec,
            initial_amounts(spec, contact=contact) if amounts is None else amounts,
            initial_calcium_equivalent_fraction=fraction,
            shared_clock=shared,
        )
    )
    rt.reset(seed=49)
    return rt


def commit(rt, operation, **payload):
    result = rt.execute({"operation": operation, **payload})
    assert result.info["transaction_status"] == "committed", result.info
    return result


def inventory(rt):
    current = rt._state
    states = {current.vessel_id: current.replace(inactive_vessels={}), **current.inactive_vessels}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    return unpack_cation_exchange(anchor)[2]


@pytest.mark.parametrize("index", range(19))
def test_finite_reference_and_equilibrium(index):
    name, spec, options, flow, duration, path, _ = fixed_cases()[index]
    initial = inventory_case(spec, **options)
    expected = independent(spec, initial, flow, duration, path)
    result, diagnostic = advance_cation_exchange(spec, initial, flow, duration, flow_path=path)
    for n, row in zip(result.final_state.amounts_mol, expected["amounts"], strict=True):
        assert [n[k] for k in DATA] == pytest.approx(row, abs=3e-8, rel=3e-7)
        assert n["Na"] + 2 * n["Ca"] - n["Cl"] == pytest.approx(0, abs=1e-11)
    final = result.final_state
    assert [
        final.resin_slot.values[k] for k in ("bound_sodium_mol", "bound_calcium_mol")
    ] == pytest.approx(expected["loading"], abs=3e-8, rel=3e-7)
    assert final.enthalpies_J == pytest.approx(expected["enthalpies"], abs=3e-6, rel=3e-7)
    assert result.actual_duration_s == pytest.approx(expected["dt"], abs=2e-7)
    assert result.stopping_event == expected["stop"]
    assert diagnostic["bath_heat_into_system_J"] == pytest.approx(
        expected["bath_heat_into_system_J"], abs=3e-6
    )
    assert result.energy_balance_residual_J == pytest.approx(0, abs=3e-6)
    if name.startswith("batch"):
        assert final.resin_slot.values["bound_calcium_mol"] == pytest.approx(
            expected["equilibrium_Ca_mol"], abs=3e-8
        )


def test_detailed_balance_activity_standard_and_entropy_reference():
    spec = spec_case()
    inv = inventory_case(spec, fraction=0.5)
    point = cation_exchange_profile(spec, inv, 0.0)
    original = spec.database.chemical_potentials(
        inv.amounts_mol[1], temperature_K=spec.temperature_K, pressure_Pa=spec.pressure_Pa
    )
    assert point["sodium_activity"] == pytest.approx(exp(original.log_activities["Na"]), rel=1e-12)
    assert point["calcium_activity"] == pytest.approx(exp(original.log_activities["Ca"]), rel=1e-12)
    q = (
        point["calcium_equivalent_fraction"]
        * point["sodium_activity"] ** 2
        / (point["sodium_equivalent_fraction"] ** 2 * point["calcium_activity"])
    )
    assert point["exchange_gibbs_J_mol"] == pytest.approx(
        8.31446261815324 * spec.temperature_K * log(q / spec.selectivity(spec.temperature_K))
    )
    assert point["dissipation_W"] >= 0
    assert (
        spec.exchange_enthalpy_J_mol - spec.reference_temperature_K * spec.exchange_entropy_J_mol_K
        == pytest.approx(
            -8.31446261815324 * spec.reference_temperature_K * log(spec.reference_selectivity)
        )
    )
    dt = 1e-3
    derivative = (
        log(spec.selectivity(spec.temperature_K + dt))
        - log(spec.selectivity(spec.temperature_K - dt))
    ) / (2 * dt)
    assert 8.31446261815324 * spec.temperature_K**2 * derivative == pytest.approx(
        spec.exchange_enthalpy_J_mol, rel=1e-8
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0])
def test_pure_resin_endpoints_are_finite_and_react_in_allowed_direction(fraction):
    spec = spec_case()
    inv = inventory_case(spec, fraction=fraction, contact="salt" if fraction else "ordinary")
    point = cation_exchange_profile(spec, inv, 0.0)
    assert point["exchange_gibbs_J_mol"] is None
    assert point["exchange_rate_mol_s"] < 0 if fraction else point["exchange_rate_mol_s"] > 0
    result, _ = advance_cation_exchange(spec, inv, 0.0, 2.0)
    slot = result.final_state.resin_slot
    assert 0 < slot.values["bound_calcium_mol"] < spec.charge_equivalents_mol / 2
    assert slot.values["bound_sodium_mol"] + 2 * slot.values["bound_calcium_mol"] == pytest.approx(
        spec.charge_equivalents_mol, abs=1e-11
    )


def test_blocked_pump_continues_exchange_without_transport():
    spec = replace(spec_case(), pump=PumpSpec(0.0, 0.0, 0.7))
    initial = inventory_case(spec)
    result, d = advance_cation_exchange(spec, initial, 10 / 6e7, 5.0)
    assert result.final_state.resin_slot.values["bound_calcium_mol"] > 0
    assert d["throughput_m3"] == d["pump_work_J"] == 0
    for i in (0, 2, 3, 4, 5):
        assert result.final_state.amounts_mol[i] == initial.amounts_mol[i]


@pytest.mark.parametrize("shared", [False, True])
def test_actual_four_stage_cycle_preserves_resin_and_routes(shared):
    rt = setup(shared=shared)
    operation = "wait" if shared else "run_flow"
    original = inventory(rt)
    for path in ("process", "regenerate", "rinse", "process"):
        commit(rt, "select_vessel", vessel="reactor")
        before = inventory(rt)
        commit(rt, "set_flow_rate", flow_rate_mL_min=10.0, flow_path=path)
        if shared:
            commit(rt, "select_vessel", vessel="exchange_contact")
        commit(rt, operation, duration_s=30.0)
        after = inventory(rt)
        source, target = {"process": (0, 2), "regenerate": (3, 5), "rinse": (4, 5)}[path]
        for i in set(range(6)) - {source, target, 1}:
            assert after.amounts_mol[i] == before.amounts_mol[i]
        assert after.amounts_mol[source]["water"] < before.amounts_mol[source]["water"]
        assert after.amounts_mol[target]["water"] > before.amounts_mol[target]["water"]
        assert after.resin_slot.initial_values == original.resin_slot.initial_values
        if path == "regenerate":
            assert (
                after.resin_slot.values["bound_calcium_mol"]
                < before.resin_slot.values["bound_calcium_mol"]
            )
            assert after.amounts_mol[5]["Ca"] > 0
    assert rt._state.ledger.time_s == 120.0
    assert rt._state.ledger.flow_pump_work_J > 0


@pytest.mark.parametrize("shared", [False, True])
def test_cycle_prefix_new_process_exact_snapshot(tmp_path, shared):
    rt = setup(shared=shared)
    operation = "wait" if shared else "run_flow"
    for path in ("process", "regenerate", "rinse", "process"):
        commit(rt, "select_vessel", vessel="reactor")
        commit(rt, "set_flow_rate", flow_rate_mL_min=10.0, flow_path=path)
        if shared:
            commit(rt, "select_vessel", vessel="exchange_contact")
        commit(rt, operation, duration_s=5.0 if rt._state.ledger.time_s == 90 else 30.0)
    source, output = tmp_path / "prefix.json", tmp_path / "continued.json"
    source.write_text(json.dumps(rt.snapshot().to_dict()), encoding="utf-8")
    commit(rt, operation, duration_s=25.0)
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/cation_exchange49_resume.py",
            str(source),
            str(output),
            operation,
            "25",
        ],
        capture_output=True,
        text=True,
        timeout=50,
        env={**os.environ, "PYTHONPATH": str(Path("tests").resolve())},
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(output.read_text()) == json.loads(json.dumps(rt.snapshot().to_dict()))


@pytest.mark.parametrize(
    "mutation",
    [
        "owner",
        "location",
        "fixed_charge",
        "unbalanced_resin",
        "liquid_charge",
        "H",
        "phase",
        "model",
        "preload",
    ],
)
def test_restore_rejects_mutated_original_stocks(mutation):
    rt = setup(shared=True, fraction=0.5)
    commit(rt, "select_vessel", vessel="exchange_contact")
    payload = json.loads(json.dumps(rt.snapshot().to_dict()))
    contact = payload["world_state"]
    anchor = contact["inactive_vessels"]["reactor"]
    slot = contact["state_slots"]["exchange_resin:resin"]
    if mutation == "owner":
        slot["spec"]["owner_id"] = "foreign"
    elif mutation == "location":
        slot["location_id"] = "product"
    elif mutation == "fixed_charge":
        slot["values"]["fixed_charge_eq_mol"] *= 2
    elif mutation == "unbalanced_resin":
        slot["values"]["bound_sodium_mol"] *= 0.5
    elif mutation == "liquid_charge":
        contact["species_amounts"]["Na"] *= 2
    elif mutation == "H":
        contact["state_slots"]["exchange_contact:material:enthalpy"]["values"]["enthalpy_J"] += 1
    elif mutation == "phase":
        contact["phases"]["liquid"]["vessel_id"] = "product"
    elif mutation == "model":
        anchor["equipment"]["cation_exchange"]["settings"]["physical_model"][
            "reference_selectivity"
        ] *= 2
    else:
        slot["initial_values"]["bound_sodium_mol"] += 0.0001
        slot["initial_values"]["bound_calcium_mol"] -= 0.00005
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


def test_slot_charge_and_liquid_enthalpy_are_not_silently_projected():
    spec = spec_case()
    inv = inventory_case(spec)
    with pytest.raises(ValueError, match="original aqueous liquid H"):
        advance_cation_exchange(
            spec,
            replace(inv, enthalpies_J=(inv.enthalpies_J[0] + 1, *inv.enthalpies_J[1:])),
            0.0,
            1.0,
        )
    with pytest.raises(ValueError, match="counterions"):
        spec.validate_slot(
            replace(inv.resin_slot, values={**inv.resin_slot.values, "bound_sodium_mol": 0.0009})
        )


@pytest.mark.parametrize(
    "mutation",
    ["nonideal", "ion_volume", "elevation", "partial_holdup", "negative_flow", "backwash"],
)
def test_rejects_scope_outside_declared_model(mutation):
    spec = spec_case()
    with pytest.raises(ValueError):
        if mutation == "nonideal":
            replace(
                spec,
                properties=replace(
                    spec.properties,
                    database=replace(
                        spec.database,
                        activity_model=replace(spec.database.activity_model, family="davies"),
                    ),
                ),
            )
        elif mutation == "ion_volume":
            species = list(spec.database.species)
            species[1] = replace(
                species[1],
                standard_state=replace(species[1].standard_state, molar_volume_m3_mol=1e-6),
            )
            replace(
                spec,
                properties=replace(
                    spec.properties, database=replace(spec.database, species=tuple(species))
                ),
            )
        elif mutation == "elevation":
            replace(spec, pipe=replace(spec.pipe, elevation_change_m=1.0))
        elif mutation == "partial_holdup":
            rows = list(initial_amounts(spec))
            rows[1] = {k: v / 2 for k, v in rows[1].items()}
            cation_exchange_inventory(spec, rows)
        else:
            advance_cation_exchange(
                spec,
                inventory_case(spec),
                -1.0 if mutation == "negative_flow" else 0.0,
                1.0,
                flow_path="backwash" if mutation == "backwash" else "process",
            )


def test_spec_roundtrip_and_external_mapping_isolation():
    spec = spec_case()
    payload = json.loads(json.dumps(spec.to_dict()))
    other = CationExchangeSpec.from_dict(payload)
    assert other.to_dict() == spec.to_dict()
    payload["properties"]["database"]["species"][0]["composition"]["H"] = 200
    assert other.database.solvent_species.composition["H"] == 2


def test_shared_lifecycle_and_solver_rollback(monkeypatch):
    import chemworld.runtime.cation_exchange as module

    rt = setup(shared=True)
    before = rt._state.to_dict()
    with pytest.raises(ValueError):
        rt.reset(options={"scope": "material", "vessel_ids": ["exchange_contact"]})
    assert rt._state.to_dict() == before

    def fail(*args, **kwargs):
        raise RuntimeError("injected resin failure")

    monkeypatch.setattr(module, "advance_cation_exchange", fail)
    result = rt.execute({"operation": "wait", "duration_s": 5.0})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert rt._state.species_amounts == before["species_amounts"]
    assert rt._state.ledger.time_s == 0
    assert inventory(rt).resin_slot.values["bound_calcium_mol"] == 0


def test_thermal_control_rejected():
    rt = setup()
    result = CationExchangeAdapter(rt.compiled_composition.spec).advance(
        rt._state, {"operation": "wait", "duration_s": 1.0}, heat=True
    )
    assert result.state is None and result.outcome == "rejected"


def test_actual_bath_capacity_crossing_and_single_owner():
    base = spec_case()
    spec = replace(base, bath=replace(base.bath, maximum_cooling_power_W=0.0015))
    rows = list(initial_amounts(spec))
    rows[1] = {**rows[1], "Na": 0.0, "Ca": 0.0, "Cl": 0.0}
    initial = cation_exchange_inventory(spec, rows)
    result, diagnostic = advance_cation_exchange(spec, initial, 10 / 6e7, 20.0)
    assert result.stopping_event == "bath_cooling_limit"
    assert 0 < result.actual_duration_s < 20
    rt = setup(spec, shared=True, amounts=rows)
    commit(rt, "set_flow_rate", flow_rate_mL_min=10.0)
    commit(rt, "select_vessel", vessel="exchange_contact")
    outcome = commit(rt, "wait", duration_s=20.0)
    assert outcome.info["operation_outcome"] == "stopped"
    assert rt._state.ledger.time_s == pytest.approx(result.actual_duration_s, abs=2e-7)
    assert rt._state.ledger.heat_loss_J == pytest.approx(
        diagnostic["bath_cooling_J"] + diagnostic["motor_loss_J"], abs=3e-6
    )
    for name in ("product", "regenerant", "rinse", "waste"):
        assert rt._state.inactive_vessels[name].ledger.energy_jacket_J == 0


def test_actual_cation_stoichiometry_can_hit_total_ion_concentration_limit():
    spec = replace(spec_case(), maximum_ion_concentration_mol_L=0.007)
    rows = [{**n, "Na": 0.0, "Ca": 0.0, "Cl": 0.0} for n in initial_amounts(spec)]
    rows[1].update({"Ca": 1e-5, "Cl": 2e-5})
    rt = setup(spec, shared=True, amounts=rows)
    commit(rt, "select_vessel", vessel="exchange_contact")
    outcome = commit(rt, "wait", duration_s=20.0)
    final = inventory(rt)
    assert outcome.info["operation_outcome"] == "stopped"
    assert 0 < rt._state.ledger.time_s < 20
    assert sum(final.amounts_mol[1][k] for k in ("Na", "Ca", "Cl")) / 0.005 == pytest.approx(
        0.007, abs=1e-9
    )
    assert final.amounts_mol[1]["Na"] == pytest.approx(
        2 * (1e-5 - final.amounts_mol[1]["Ca"]), abs=1e-12
    )
    assert rt._state.ledger.flow_pump_work_J == 0
