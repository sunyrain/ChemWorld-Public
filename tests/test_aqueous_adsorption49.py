"""Actual finite adsorption/rinse, original media ownership, events and replay."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from chemworld.physchem.aqueous_adsorption import (
    AqueousAdsorptionSpec,
    advance_aqueous_adsorption,
    aqueous_adsorption_inventory,
)
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.aqueous_adsorption import AqueousAdsorptionAdapter, unpack_aqueous_adsorption
from chemworld.world.aqueous_adsorption_world import aqueous_adsorption_world_spec
from reference.aqueous_adsorption49_fixtures import (
    DATA,
    fixed_cases,
    initial_amounts,
    inventory_case,
    spec_case,
)
from reference.aqueous_adsorption49_reference import independent


def setup(spec=None, *, shared=False, preloaded=False, amounts=None):
    spec = spec or spec_case()
    runtime = ChemWorldRuntime(
        composition=aqueous_adsorption_world_spec(
            spec,
            initial_amounts(spec, preloaded=preloaded) if amounts is None else amounts,
            initial_loading_mol_per_kg={"A": 0.6, "B": 0.2} if preloaded else None,
            shared_clock=shared,
        )
    )
    runtime.reset(seed=49)
    return runtime


def commit(runtime, operation, **payload):
    result = runtime.execute({"operation": operation, **payload})
    assert result.info["transaction_status"] == "committed", result.info.get(
        "outcome_diagnostics"
    ) or result.info.get("error_message")
    return result


def inventory(runtime):
    current = runtime._state
    states = {current.vessel_id: current.replace(inactive_vessels={}), **current.inactive_vessels}
    anchor = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    return unpack_aqueous_adsorption(anchor)[2]


@pytest.mark.parametrize("index", range(16))
def test_physical_final_state_matches_independent_reference(index):
    name, spec, options, flow, duration, path, _ = fixed_cases()[index]
    initial = inventory_case(spec, **options)
    reference = independent(spec, initial, flow, duration, path)
    result, diagnostic = advance_aqueous_adsorption(spec, initial, flow, duration, flow_path=path)
    final = result.final_state
    for row, expected in zip(final.amounts_mol, reference["amounts"], strict=True):
        assert [row[k] for k in DATA] == pytest.approx(expected, abs=3e-8, rel=3e-7)
    assert [final.sorbent_slot.values[f"loading:{k}"] for k in ("A", "B")] == pytest.approx(
        reference["loading"], abs=3e-8, rel=3e-7
    )
    assert final.enthalpies_J == pytest.approx(reference["enthalpies"], abs=3e-6, rel=3e-7)
    assert result.actual_duration_s == pytest.approx(reference["dt"], abs=2e-7)
    assert result.stopping_event == reference["stop"]
    assert diagnostic["bath_heat_into_system_J"] == pytest.approx(
        reference["bath_heat_into_system_J"], abs=3e-6
    )
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=3e-6)
    assert spec.volume_m3(final.amounts_mol[1]) * 1000 == pytest.approx(0.005)
    if name.startswith("batch-single"):
        assert final.sorbent_slot.values["loading:A"] == pytest.approx(
            reference["equilibrium_A_mol"], abs=3e-8
        )


def test_blocked_pump_keeps_batch_contact_and_no_transfer_work():
    spec = replace(spec_case(), pump=PumpSpec(0.0, 0.0, 0.7))
    initial = inventory_case(spec)
    result, diagnostic = advance_aqueous_adsorption(spec, initial, 10 / 6e7, 10.0)
    assert result.actual_duration_s == 10.0 and result.stopping_event is None
    assert (
        result.final_state.sorbent_slot.values["loading:A"]
        > initial.sorbent_slot.values["loading:A"]
    )
    assert diagnostic["pump_work_J"] == diagnostic["throughput_m3"] == 0.0
    for i in (0, 2, 3, 4):
        assert result.final_state.amounts_mol[i] == initial.amounts_mol[i]


def test_full_competitive_lot_releases_sites_without_capacity_reset():
    spec = spec_case()
    initial = aqueous_adsorption_inventory(
        spec, initial_amounts(spec), initial_loading_mol_per_kg={"A": 0.0, "B": 1.0}
    )
    result, _ = advance_aqueous_adsorption(spec, initial, 0.0, 20.0)
    slot = result.final_state.sorbent_slot
    assert slot.values["loading:A"] > 0 and slot.values["loading:B"] < 0.001
    assert slot.values["loading:A"] + slot.values["loading:B"] <= slot.values["sites_mol"] * (
        1 + 1e-12
    )
    assert slot.initial_values == initial.sorbent_slot.initial_values


def test_no_hidden_liquid_enthalpy_projection():
    spec = spec_case()
    initial = inventory_case(spec)
    with pytest.raises(ValueError, match="original aqueous liquid H"):
        advance_aqueous_adsorption(
            spec,
            replace(
                initial, enthalpies_J=(initial.enthalpies_J[0] + 1.0, *initial.enthalpies_J[1:])
            ),
            0.0,
            10.0,
        )


@pytest.mark.parametrize(
    "mutation", ["charged", "solute_volume", "nonideal", "holdup", "elevation", "source"]
)
def test_rejects_undeclared_physical_scope(mutation):
    spec = spec_case()
    with pytest.raises(ValueError):
        if mutation == "charged":
            species = list(spec.database.species)
            species[1] = replace(species[1], charge=1)
            replace(spec, database=replace(spec.database, species=tuple(species)))
        elif mutation == "solute_volume":
            species = list(spec.database.species)
            species[1] = replace(
                species[1],
                standard_state=replace(species[1].standard_state, molar_volume_m3_mol=1e-6),
            )
            replace(spec, database=replace(spec.database, species=tuple(species)))
        elif mutation == "nonideal":
            replace(
                spec,
                database=replace(
                    spec.database,
                    activity_model=replace(spec.database.activity_model, family="davies"),
                ),
            )
        elif mutation == "holdup":
            rows = list(initial_amounts(spec))
            rows[1] = {k: v / 2 for k, v in rows[1].items()}
            aqueous_adsorption_inventory(spec, rows)
        elif mutation == "elevation":
            replace(spec, pipe=replace(spec.pipe, elevation_change_m=1.0))
        else:
            replace(spec, viscosity=replace(spec.viscosity, source_id="undeclared"))


def test_declaration_roundtrip_and_external_data_copies():
    spec = spec_case()
    payload = json.loads(json.dumps(spec.to_dict()))
    other = AqueousAdsorptionSpec.from_dict(payload)
    assert other.to_dict() == spec.to_dict()
    payload["sorbent"]["adsorption_energy_J_mol"]["A"] = -1
    assert other.sorbent.adsorption_energy_J_mol["A"] == -2000


@pytest.mark.parametrize("shared", [False, True])
def test_actual_process_then_rinse_changes_original_media_and_correct_collectors(shared):
    runtime = setup(shared=shared)
    operation = "wait" if shared else "run_flow"
    initial = inventory(runtime)
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="process")
    if shared:
        commit(runtime, "select_vessel", vessel="adsorber")
    commit(runtime, operation, duration_s=30.0)
    after_process = inventory(runtime)
    assert after_process.amounts_mol[3:] == initial.amounts_mol[3:]
    commit(runtime, "select_vessel", vessel="reactor")
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="rinse")
    if shared:
        commit(runtime, "select_vessel", vessel="adsorber")
    commit(runtime, operation, duration_s=30.0)
    final = inventory(runtime)
    for i in (0, 2):
        assert final.amounts_mol[i] == after_process.amounts_mol[i]
    assert final.amounts_mol[4]["A"] > 0
    # A finite contact liquid retains solute after switching; a short rinse
    # need not make total bound loading decrease monotonically.
    expected = independent(spec_case(), after_process, 10 / 6e7, 30.0, "rinse")
    for k, n in zip(("A", "B"), expected["loading"], strict=True):
        assert final.sorbent_slot.values[f"loading:{k}"] == pytest.approx(n, abs=3e-8)
    assert final.sorbent_slot.initial_values == initial.sorbent_slot.initial_values
    assert runtime._state.ledger.time_s == 60.0
    assert runtime._state.ledger.flow_pump_work_J > 0


def test_actual_bath_event_and_single_anchor_heat_accounting():
    original = spec_case()
    spec = replace(original, bath=replace(original.bath, maximum_cooling_power_W=0.01))
    rows = initial_amounts(spec, preloaded=True)
    initial = aqueous_adsorption_inventory(spec, rows)
    result, diagnostic = advance_aqueous_adsorption(spec, initial, 10 / 6e7, 20.0)
    assert result.stopping_event == "bath_cooling_limit" and 0 < result.actual_duration_s < 20
    runtime = setup(spec, shared=True, amounts=rows)
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0)
    commit(runtime, "select_vessel", vessel="adsorber")
    outcome = commit(runtime, "wait", duration_s=20.0)
    assert outcome.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(result.actual_duration_s, abs=2e-7)
    assert runtime._state.ledger.heat_loss_J == pytest.approx(
        diagnostic["bath_cooling_J"] + diagnostic["motor_loss_J"], abs=3e-6
    )
    assert all(
        runtime._state.inactive_vessels[n].ledger.energy_jacket_J == 0
        for n in ("product", "rinse", "waste")
    )


def test_actual_preloaded_rinse_releases_bound_stock_to_finite_waste():
    runtime = setup(shared=True, preloaded=True)
    initial = inventory(runtime)
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="rinse")
    commit(runtime, "select_vessel", vessel="adsorber")
    commit(runtime, "wait", duration_s=30.0)
    final = inventory(runtime)
    for k in ("A", "B"):
        assert (
            final.sorbent_slot.values[f"loading:{k}"] < initial.sorbent_slot.values[f"loading:{k}"]
        )
        assert final.amounts_mol[4][k] > 0
    assert final.sorbent_slot.initial_values == initial.sorbent_slot.initial_values
    assert final.amounts_mol[0] == initial.amounts_mol[0]
    assert final.amounts_mol[2] == initial.amounts_mol[2]


def test_actual_desorption_stops_at_contact_concentration_limit():
    spec = replace(spec_case(), maximum_solute_concentration_mol_L=0.005)
    rows = tuple({**n, "A": 0.0, "B": 0.0} for n in initial_amounts(spec))
    runtime = setup(spec, shared=True, preloaded=True, amounts=rows)
    commit(runtime, "select_vessel", vessel="adsorber")
    outcome = commit(runtime, "wait", duration_s=20.0)
    final = inventory(runtime)
    assert outcome.info["operation_outcome"] == "stopped"
    assert 0 < runtime._state.ledger.time_s < 20
    concentration = sum(final.amounts_mol[1][k] for k in ("A", "B")) / 0.005
    assert concentration == pytest.approx(0.005, abs=1e-9)
    assert runtime._state.ledger.flow_pump_work_J == 0


@pytest.mark.parametrize(
    "mutation", ["media_owner", "media_location", "media_static", "H", "phase", "model", "preload"]
)
def test_restore_rejects_foreign_original_media_and_liquid_owners(mutation):
    runtime = setup(shared=True, preloaded=True)
    commit(runtime, "select_vessel", vessel="adsorber")
    payload = json.loads(json.dumps(runtime.snapshot().to_dict()))
    contact = payload["world_state"]
    anchor = contact["inactive_vessels"]["reactor"]
    slot = contact["state_slots"]["adsorption_media:sorbent"]
    if mutation == "media_owner":
        slot["spec"]["owner_id"] = "foreign"
    elif mutation == "media_location":
        slot["location_id"] = "product"
    elif mutation == "media_static":
        slot["values"]["mass_kg"] *= 2
    elif mutation == "H":
        contact["state_slots"]["adsorber:material:enthalpy"]["values"]["enthalpy_J"] += 1
    elif mutation == "phase":
        contact["phases"]["liquid"]["vessel_id"] = "product"
    elif mutation == "model":
        anchor["equipment"]["aqueous_adsorption"]["settings"]["physical_model"]["pump"][
            "maximum_shaft_power_W"
        ] *= 2
    else:
        slot["initial_values"]["loading:A"] *= 0.5
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(payload)


@pytest.mark.parametrize("shared", [False, True])
def test_actual_rinse_prefix_and_new_process_full_snapshot(tmp_path, shared):
    runtime = setup(shared=shared)
    operation = "wait" if shared else "run_flow"
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="process")
    commit(runtime, operation, duration_s=30.0)
    commit(runtime, "set_flow_rate", flow_rate_mL_min=10.0, flow_path="rinse")
    if shared:
        commit(runtime, "select_vessel", vessel="adsorber")
    commit(runtime, operation, duration_s=5.0)
    source, output = tmp_path / "prefix.json", tmp_path / "continued.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    commit(runtime, operation, duration_s=25.0)
    child = subprocess.run(
        [
            sys.executable,
            "tests/reference/aqueous_adsorption49_resume.py",
            str(source),
            str(output),
            operation,
            "25",
        ],
        env={**os.environ, "PYTHONPATH": str(Path("tests").resolve())},
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert child.returncode == 0, child.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        json.dumps(runtime.snapshot().to_dict())
    )


def test_actual_solver_failure_rolls_back_media_and_five_liquids(monkeypatch):
    import chemworld.runtime.aqueous_adsorption as module

    runtime = setup(shared=True)
    before = runtime._state

    def fail(*args, **kwargs):
        raise RuntimeError("injected adsorption failure")

    monkeypatch.setattr(module, "advance_aqueous_adsorption", fail)
    result = runtime.execute({"operation": "wait", "duration_s": 10.0})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert runtime._state.species_amounts == before.species_amounts
    assert runtime._state.inactive_vessels == before.inactive_vessels
    assert runtime._state.ledger.time_s == before.ledger.time_s


def test_shared_lifecycle_cannot_replace_contact_or_media():
    runtime = setup(shared=True, preloaded=True)
    before = runtime._state.to_dict()
    with pytest.raises(ValueError):
        runtime.reset(options={"scope": "material", "vessel_ids": ["adsorber"]})
    assert runtime._state.to_dict() == before


def test_thermal_control_is_rejected_without_candidate():
    runtime = setup()
    result = AqueousAdsorptionAdapter(runtime.compiled_composition.spec).advance(
        runtime._state, {"operation": "wait", "duration_s": 5.0}, heat=True
    )
    assert result.state is None and result.outcome == "rejected"
