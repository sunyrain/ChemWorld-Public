"""Signed finite osmosis, original owners/resources, actual wait and replay."""

import json
import subprocess
from dataclasses import replace
from math import exp
from pathlib import Path

import pytest

from chemworld.physchem.osmotic_membrane import (
    OsmoticMembraneSpec,
    advance_osmotic_membrane,
    osmotic_membrane_inventory,
    osmotic_membrane_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.composition import WorldCompositionSpec
from chemworld.world.osmotic_membrane_world import osmotic_membrane_world_spec
from reference.osmotic_membrane48_fixtures import inventories, spec
from reference.osmotic_membrane48_reference import endpoint, event_time, point, trajectory


def setup(model=None, *, shared=True, select=False):
    declaration = osmotic_membrane_world_spec(model or spec(), *inventories(), shared_clock=shared)
    if select:
        payload = declaration.to_dict()
        payload["task"]["operations"].append("select_vessel")
        declaration = WorldCompositionSpec.from_dict(payload)
    runtime = ChemWorldRuntime(composition=declaration)
    runtime.reset(seed=48)
    return runtime


def physical(runtime):
    return {
        name: {
            "n": tank.species_amounts,
            "H": tank.state_slots.to_dict(),
            "T": tank.temperature_K,
            "P": tank.pressure_Pa,
            "V": tank.volume_L,
            "phases": tank.phases.to_dict(),
        }
        for name, tank in {
            runtime._state.vessel_id: runtime._state,
            **runtime._state.inactive_vessels,
        }.items()
    }


@pytest.mark.parametrize(
    "family,mode,left,right",
    [
        ("ideal", "FO", 10, 60),
        ("ideal", "PRO", 10, 60),
        ("ideal", "reverse", 10, 60),
        ("ideal", "FO", 30, 30),
        ("ideal", "FO", 0, 0),
        ("Davies", "FO", 10, 60),
    ],
)
def test_signed_chemical_potential_drives_one_law(family, mode, left, right):
    model, amounts = spec(family, mode), inventories(left, right)
    actual, reference = osmotic_membrane_point(model, amounts), point(model, amounts)
    assert actual["water_chemical_potential_difference_J_mol"] == pytest.approx(
        reference["delta_mu"], abs=2e-9
    )
    assert actual["water_transfer_mol_s"] == pytest.approx(reference["flow"], abs=2e-12)
    assert actual["enthalpy_rates_W"] == pytest.approx(reference["enthalpy_rates_W"], abs=2e-9)
    assert actual["bath_heat_into_liquids_W"] == pytest.approx(reference["bath_W"], abs=2e-9)
    assert actual["entropy_production_W_K"] >= 0
    assert actual["volumes_L"] == pytest.approx(reference["volumes_L"], abs=2e-10)
    if mode == "reverse":
        assert actual["water_transfer_mol_s"] < 0
    elif left == right:
        assert actual["water_transfer_mol_s"] == 0
    else:
        assert actual["water_transfer_mol_s"] > 0


@pytest.mark.parametrize(
    "family,mode", [("ideal", "FO"), ("ideal", "PRO"), ("ideal", "reverse"), ("Davies", "FO")]
)
def test_full_original_n_h_trajectories_and_pressure_boundary_work(family, mode):
    model, amounts = spec(family, mode), inventories()
    initial = osmotic_membrane_inventory(model, *amounts)
    result, diagnostic = advance_osmotic_membrane(
        model, initial, duration_s=20, evaluation_times_s=(0, 5, 10, 20)
    )
    reference = trajectory(model, amounts)
    assert result.stopping_event is None
    for actual, expected in zip(result.trajectory, reference, strict=True):
        assert [n["H2O"] for n in actual.amounts_mol] == pytest.approx(
            [n["H2O"] for n in expected["amounts"]], abs=2e-8
        )
        assert actual.enthalpies_J == pytest.approx(expected["enthalpies_J"], abs=3e-6)
        assert all(
            n[k] == amounts[i][k] for i, n in enumerate(actual.amounts_mol) for k in ("Na+", "Cl-")
        )
    assert diagnostic["bath_heat_into_liquids_J"] == pytest.approx(
        reference[-1]["heat_J"], abs=5e-6
    )
    assert diagnostic["boundary_work_J"] == pytest.approx(reference[-1]["work_J"], abs=5e-6)
    assert abs(diagnostic["internal_energy_closure_J"]) < 5e-6
    assert diagnostic["entropy_production_J_K"] == pytest.approx(
        reference[-1]["entropy_J_K"], abs=2e-8
    )
    assert abs(result.component_residuals_mol["H2O"]) < 2e-10
    assert result.component_residuals_mol["Na+"] == result.component_residuals_mol["Cl-"] == 0


@pytest.mark.parametrize("shared", [False, True])
def test_actual_wait_commits_original_clock_and_single_heat_work(shared):
    runtime = setup(spec(mode="PRO"), shared=shared)
    result = runtime.execute({"operation": "wait", "duration_s": 20})
    assert result.info["transaction_status"] == "committed", result.info["outcome_diagnostics"]
    ledger = runtime._state.ledger
    reference = trajectory(spec(mode="PRO"), inventories())[-1]
    assert ledger.energy_jacket_J - ledger.heat_loss_J == pytest.approx(
        reference["heat_J"], abs=5e-6
    )
    assert ledger.boundary_work_J == pytest.approx(reference["work_J"], abs=5e-6)
    assert ledger.flow_electrical_work_J == ledger.flow_shaft_work_J == ledger.flow_pump_work_J == 0
    assert ledger.time_s == runtime._state.inactive_vessels["draw"].ledger.time_s == 20


def test_water_reserve_is_real_earliest_shared_stop():
    model = spec(mode="PRO", minimum_water_mol=0.045)
    runtime = setup(model)
    result = runtime.execute({"operation": "wait", "duration_s": 120})
    assert result.info["transaction_status"] == "committed", result.info["outcome_diagnostics"]
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(event_time(model, inventories()), abs=2e-4)
    assert runtime._state.species_amounts["H2O"] == pytest.approx(0.045, abs=2e-10)
    assert runtime._state.inactive_vessels["draw"].ledger.time_s == pytest.approx(
        runtime._state.ledger.time_s,
        rel=0,
        abs=2e-7,
    )


def test_full_json_new_process_continuation_to_capacity(tmp_path):
    model = spec()
    capacity = endpoint(model, inventories(), 0.005)["volumes_L"][1]
    model = replace(model, capacities_L=(0.003, capacity))
    runtime = setup(model)
    prefix = runtime.execute({"operation": "wait", "duration_s": 5})
    assert prefix.info["transaction_status"] == "committed"
    assert runtime._state.ledger.time_s == 5
    source, output = tmp_path / "snapshot.json", tmp_path / "continued.json"
    source.write_text(json.dumps(runtime.snapshot().to_dict()), encoding="utf-8")
    result = runtime.execute({"operation": "wait", "duration_s": 60})
    assert result.info["transaction_status"] == "committed", result.info["outcome_diagnostics"]
    assert result.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(event_time(model, inventories()), abs=2e-4)
    process = subprocess.run(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            str(Path(__file__).parent / "reference" / "osmotic_membrane48_resume.py"),
            str(source),
            str(output),
            "60",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot"] == json.loads(
        json.dumps(runtime.snapshot().to_dict())
    )


def test_finite_bath_power_crossing_stops_without_capping_flux():
    model = spec(mode="PRO")
    model = replace(
        model,
        water_permeability=replace(
            model.water_permeability,
            value=model.water_permeability.value * exp(0.3 * 35),
            per_mol_m3=-0.3,
        ),
    )
    q0 = point(model, inventories())["bath_W"]
    model = replace(model, bath=replace(model.bath, maximum_power_W=q0 * 1.02))
    result, diagnostic = advance_osmotic_membrane(
        model, osmotic_membrane_inventory(model, *inventories()), duration_s=20
    )
    assert result.stopping_event == "bath_heating_power"
    assert 0 < result.actual_duration_s < 20
    assert diagnostic["final"]["bath_heat_into_liquids_W"] == pytest.approx(
        model.bath.maximum_power_W, abs=2e-11
    )
    assert diagnostic["final"]["water_transfer_mol_s"] == pytest.approx(
        point(model, result.final_state.amounts_mol)["flow"], abs=2e-12
    )
    runtime = setup(model)
    outcome = runtime.execute({"operation": "wait", "duration_s": 20})
    assert outcome.info["transaction_status"] == "committed", outcome.info["outcome_diagnostics"]
    assert outcome.info["operation_outcome"] == "stopped"
    assert runtime._state.ledger.time_s == pytest.approx(result.actual_duration_s, abs=2e-7)
    assert runtime._state.ledger.energy_jacket_J == pytest.approx(
        diagnostic["bath_heating_J"], abs=5e-6
    )


def test_insufficient_bath_rejection_is_atomic():
    model = spec(mode="PRO")
    model = replace(model, bath=replace(model.bath, maximum_power_W=1e-10))
    runtime = setup(model)
    before = physical(runtime)
    result = runtime.execute({"operation": "wait", "duration_s": 20})
    assert result.info["operation_outcome"] == "numerical_failure"
    assert "bath power rating" in str(result.info["outcome_diagnostics"])
    assert physical(runtime) == before
    ledger = runtime._state.ledger
    assert (
        ledger.time_s == ledger.boundary_work_J == ledger.energy_jacket_J == ledger.heat_loss_J == 0
    )


@pytest.mark.parametrize("kind", ["salt", "slot_location", "phase_owner", "model"])
def test_snapshot_rejects_foreign_original_owners_even_when_anchor_inactive(kind):
    runtime = setup(select=True)
    selected = runtime.execute({"operation": "select_vessel", "vessel": "draw"})
    assert selected.info["transaction_status"] == "committed"
    assert runtime._state.vessel_id == "draw"
    snapshot = json.loads(json.dumps(runtime.snapshot().to_dict()))
    anchor = snapshot["world_state"]["inactive_vessels"]["reactor"]
    if kind == "salt":
        snapshot["world_state"]["phases"]["liquid"]["species_amounts_mol"]["Na+"] *= 2
        snapshot["world_state"]["species_amounts"]["Na+"] *= 2
    elif kind == "slot_location":
        anchor["state_slots"]["reactor:material:enthalpy"]["location_id"] = "draw"
    elif kind == "phase_owner":
        anchor["phases"]["liquid"]["vessel_id"] = "draw"
    else:
        anchor["equipment"]["membrane"]["settings"]["physical_model"]["membrane_area_m2"] *= 2
    with pytest.raises(ValueError):
        ChemWorldRuntime.from_snapshot(snapshot)


def test_source_domain_codec_and_unsupported_controls():
    model = spec()
    assert (
        OsmoticMembraneSpec.from_dict(json.loads(json.dumps(model.to_dict()))).to_dict()
        == model.to_dict()
    )
    with pytest.raises(ValueError, match="pressure-difference"):
        replace(spec(mode="PRO"), maximum_pressure_difference_Pa=1)
    with pytest.raises(ValueError, match=r"domain|outside|range"):
        replace(model, bath=replace(model.bath, setpoint_K=350, maximum_setpoint_K=360))
    runtime = setup()
    before = physical(runtime)
    rejected = runtime.execute({"operation": "set_flow_rate", "flow_rate_mL_min": 1})
    assert rejected.info["transaction_status"] != "committed"
    assert physical(runtime) == before


def test_zero_flux_at_reserve_has_exact_initial_stop_and_no_fabricated_stock():
    model = spec(minimum_water_mol=0.05)
    initial = osmotic_membrane_inventory(model, *inventories(30, 30))
    result, diagnostic = advance_osmotic_membrane(model, initial, duration_s=20)
    assert result.stopping_event == "water_reserve_0"
    assert result.final_state == initial and result.actual_duration_s == 0
    assert diagnostic["bath_heat_into_liquids_J"] == diagnostic["boundary_work_J"] == 0
