"""Passive RED circuit, finite energy ports and original-world ownership."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.examples.reverse_electrodialysis import example_report
from chemworld.physchem.electrodialysis import electrodialysis_point
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.reverse_electrodialysis import (
    ReverseElectrodialysisSpec,
    advance_reverse_electrodialysis,
    reverse_electrodialysis_inventory,
    reverse_electrodialysis_point,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.reverse_electrodialysis import (
    configured_red_load_program,
    reverse_electrodialysis_record,
    unpack_reverse_electrodialysis,
)
from chemworld.world.reverse_electrodialysis_world import reverse_electrodialysis_world_spec
from reference.run_reverse_electrodialysis48_canary import fixtures


@pytest.mark.parametrize("resistance", (None, 0, 30, 100, 1000))
def test_load_circuit_endpoint_and_original_pore_equations(resistance):
    spec, ns = fixtures(4)
    row = reverse_electrodialysis_point(spec, ns, resistance)
    assert row["status"] == "converged"
    assert abs(row["terminal_load_power_residual_W"]) < 1e-10
    assert abs(row["free_energy_power_residual_W"]) < 1e-10
    assert row["dissipation_W"] >= 0
    if resistance is None:
        assert row["current_A"] == pytest.approx(0, abs=1e-15)
    elif resistance == 0:
        assert row["voltage_V"] == pytest.approx(0, abs=1e-10)
    else:
        assert row["load_power_W"] > 0
        assert row["current_A"] * row["voltage_V"] < 0
    original = electrodialysis_point(spec.cell_pair, ns, row["current_A"], "A")
    assert original["voltage_V"] == pytest.approx(row["voltage_V"], abs=1e-10)
    for name in ("aem", "cem"):
        assert list(original[name]["ion_fluxes_mol_m2_s"].values()) == pytest.approx(
            list(row[name]["ion_fluxes_mol_m2_s"].values()), abs=1e-12
        )


def test_gradient_reversal_changes_terminal_sign_and_equal_liquids_generate_nothing():
    spec, ns = fixtures(3)
    forward = reverse_electrodialysis_point(spec, ns, 100)
    backward = reverse_electrodialysis_point(spec, (ns[1], ns[0]), 100)
    assert forward["voltage_V"] * backward["voltage_V"] < 0
    assert forward["current_A"] * backward["current_A"] < 0
    assert backward["load_power_W"] > 0
    equal = reverse_electrodialysis_point(spec, (ns[0], ns[0]), 100)
    assert equal["load_power_W"] < 1e-20
    assert max(map(abs, equal["diluate_rates_mol_s"].values())) < 1e-15


def test_finite_load_energy_and_bath_use_nonnegative_internal_integrals_once():
    spec, ns = fixtures(4)
    initial = reverse_electrodialysis_inventory(spec, *ns)
    solved, receipt = advance_reverse_electrodialysis(
        spec, initial, ScalarControlProgram(((0, 100), (1, 200)), "Ohm"), 3
    )
    assert solved.actual_duration_s == 3
    assert receipt["load_energy_delivered_J"] > 0
    assert receipt["electrical_work_into_liquid_J"] == -receipt["load_energy_delivered_J"]
    assert receipt["bath_heat_into_liquid_J"] == pytest.approx(
        receipt["load_energy_delivered_J"], abs=1e-9
    )
    assert solved.final_state.heater_energy_J == receipt["bath_heat_into_liquid_J"]
    assert solved.final_state.condenser_energy_J == receipt["load_energy_delivered_J"]
    assert receipt["free_energy_decrease_minus_load_J"] >= 0
    assert abs(solved.enthalpy_residual_J) < 1e-8
    assert max(map(abs, solved.component_residuals_mol.values())) < 1e-10
    for n in solved.final_state.amounts_mol:
        assert abs(sum(spec.properties.charges[k] * n[k] for k in spec.properties.ion_ids)) < 1e-12


@pytest.mark.parametrize("kind", ("current", "voltage", "load", "bath"))
def test_physical_compliance_returns_zero_duration_stop_without_clipping(kind):
    spec, ns = fixtures(3)
    expected = {
        "current": "current_compliance",
        "voltage": "voltage_compliance",
        "load": "load_power_limit",
        "bath": "bath_heating_limit",
    }[kind]
    if kind in {"current", "voltage"}:
        pair = replace(
            spec.cell_pair, **{"maximum_" + kind + ("_A" if kind == "current" else "_V"): 1e-20}
        )
        spec = replace(spec, cell_pair=pair)
    else:
        spec = replace(
            spec, **{"maximum_load_power_W" if kind == "load" else "maximum_bath_heating_W": 1e-20}
        )
    initial = reverse_electrodialysis_inventory(spec, *ns)
    solved, receipt = advance_reverse_electrodialysis(
        spec, initial, ScalarControlProgram(((0, 100),), "Ohm"), 2
    )
    assert solved.stopping_event == expected
    assert solved.actual_duration_s == 0
    assert receipt["load_energy_delivered_J"] == 0
    assert receipt["final"]["load_power_W"] > 0
    assert solved.final_state.amounts_mol == initial.amounts_mol


@pytest.mark.parametrize("resistance", (-1, True, np.inf, 1e7))
def test_invalid_passive_resistance_is_rejected(resistance):
    spec, ns = fixtures(3)
    with pytest.raises(ValueError, match="resistance"):
        reverse_electrodialysis_point(spec, ns, resistance)


def test_model_json_and_unsupported_active_controls_are_explicit():
    spec, ns = fixtures(3)
    assert (
        ReverseElectrodialysisSpec.from_dict(json.loads(json.dumps(spec.to_dict()))).to_dict()
        == spec.to_dict()
    )
    initial = reverse_electrodialysis_inventory(spec, *ns)
    for program in (ScalarControlProgram(((0, 0.1),), "A"), ScalarControlProgram(((0, 0.1),), "V")):
        with pytest.raises(ValueError):
            advance_reverse_electrodialysis(spec, initial, program, 1)


@pytest.mark.parametrize("shared_clock", (True, False))
def test_real_world_meter_clock_signed_ledgers_and_json_resume(shared_clock):
    spec, ns = fixtures(3)
    result = example_report(spec, ns, shared_clock=shared_clock)
    assert result["complete"]
    assert result["exact_json_resume"]
    state = result["final_state"]
    assert state["ledger"]["time_s"] == 4
    assert state["ledger"]["flow_electrical_work_J"] < 0
    assert state["ledger"]["heat_loss_J"] == pytest.approx(
        state["ledger"]["flow_electrical_work_J"], abs=1e-8
    )


def test_load_switch_ramp_and_restore_metadata_have_one_owner():
    spec, ns = fixtures(3)
    runtime = ChemWorldRuntime(composition=reverse_electrodialysis_world_spec(spec, *ns))
    try:
        runtime.reset(seed=48)
        state = runtime._state
        record = reverse_electrodialysis_record(state)
        with pytest.raises(ValueError, match="open circuit"):
            configured_red_load_program(
                state, record, {"resistance_Ohm": 100, "ramp_duration_s": 1}
            )
        for action in (
            {"control_mode": "open_circuit", "current_A": 0},
            {"control_mode": "open_circuit", "resistance_Ohm": 100},
        ):
            with pytest.raises(ValueError):
                configured_red_load_program(state, record, action)
        current, model, inventory, program = unpack_reverse_electrodialysis(state)
        assert current.equipment_id == record.equipment_id
        assert model.to_dict() == spec.to_dict()
        assert inventory.amounts_mol == ns
        assert program.unit == "A"
    finally:
        runtime.close()
