"""Original inventory ownership, finite momentum and persistent sorbent operation."""

import importlib.util
import json
import sys
from dataclasses import replace
from math import sqrt
from pathlib import Path

import pytest

from chemworld.foundation.state import WorldState
from chemworld.foundation.state_ledgers import PhaseLedger
from chemworld.physchem.fixed_bed import (
    FixedBedSpec,
    fixed_bed_face_velocity,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime.fixed_bed import advance_fixed_bed

_loader = importlib.util.spec_from_file_location(
    "bed_cases", Path(__file__).parent / "reference/fixed_bed_cases.py"
)
_fixtures = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(_fixtures)


def test_ergun_inverse_uses_pressure_direction_and_public_equation():
    spec = _fixtures.specification(_fixtures.cases()[0], 2)
    for rho, difference in ((1.2, 100.0), (1.5, -300.0), (1.2, 0.0)):
        velocity, diagnostic = fixed_bed_face_velocity(
            spec, distance_m=0.05, density_kg_m3=rho, pressure_difference_Pa=difference
        )
        e, d, mu = 0.4, 1e-4, 1.8e-5
        expected = 0.05 * (
            150 * mu * (1 - e) ** 2 / (e**3 * d * d) * velocity
            + 1.75 * rho * (1 - e) / (e**3 * d) * velocity * abs(velocity)
        )
        assert expected == pytest.approx(difference, abs=1e-7)
        assert diagnostic


def test_world_consumes_actual_cell_and_source_amounts_with_persistent_carrier():
    spec, state = _fixtures.native_world(_fixtures.cases()[2], 2)
    original = state.to_dict()
    result = advance_fixed_bed(
        state,
        spec,
        0.2,
        inlet_phase_id="source",
        outlet_phase_id="collector",
        evaluation_times_s=(0.0, 0.1, 0.2),
    )
    assert result.success, result.to_dict()
    assert (
        result.state.phases.phases["source"].species_amounts_mol["CO2"]
        < (state.phases.phases["source"].species_amounts_mol["CO2"])
    )
    assert result.state.state_slots.species_amounts()["CO2"] > 0
    assert result.state.ledger.time_s == pytest.approx(0.2)
    assert result.state.ledger.energy_jacket_J < 0
    for before, after in zip(state.state_slots.slots, result.state.state_slots.slots, strict=True):
        assert before.spec == after.spec
        assert before.values["sites_mol"] == after.values["sites_mol"]
        assert before.values["mass_kg"] == after.values["mass_kg"]
    assert result.state.phases.phases["purge"] == state.phases.phases["purge"]
    assert state.to_dict() == original
    assert abs(result.diagnostics["energy_balance_residual_J"]) < 1e-7


def test_snapshot_roundtrip_continues_loading_and_time():
    spec, state = _fixtures.native_world(_fixtures.cases()[2], 2)
    first = advance_fixed_bed(
        state, spec, 0.1, inlet_phase_id="source", outlet_phase_id="collector"
    )
    assert first.success, first.to_dict()
    snapshot = WorldState.from_dict(json.loads(json.dumps(first.state.to_dict())))
    second = advance_fixed_bed(
        snapshot, spec, 0.1, inlet_phase_id="source", outlet_phase_id="collector"
    )
    whole = advance_fixed_bed(
        state, spec, 0.2, inlet_phase_id="source", outlet_phase_id="collector"
    )
    assert second.success and whole.success
    assert second.state.ledger.time_s == pytest.approx(0.2)
    assert second.state.state_slots.species_amounts() == pytest.approx(
        whole.state.state_slots.species_amounts(), rel=1e-6, abs=1e-11
    )
    assert second.state.ledger.energy_jacket_J == pytest.approx(
        whole.state.ledger.energy_jacket_J, rel=1e-6, abs=1e-7
    )


def test_bad_ownership_or_cell_volume_returns_no_candidate():
    spec, state = _fixtures.native_world(_fixtures.cases()[2], 2)
    for update in ({"vessel_id": "another"}, {"volume_L": 0.004}, {"phase_type": "liquid"}):
        records = dict(state.phases.phases)
        records["cell_0"] = replace(records["cell_0"], **update)
        invalid = state.replace(phases=PhaseLedger(records))
        result = advance_fixed_bed(
            invalid, spec, 1.0, inlet_phase_id="source", outlet_phase_id="collector"
        )
        assert not result.success and result.state is None and result.actual_duration_s == 0


def test_spec_codec_and_duplicate_boundaries():
    spec, state = _fixtures.native_world(_fixtures.cases()[2], 2)
    clone = FixedBedSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert clone.to_dict() == spec.to_dict()
    failed = advance_fixed_bed(
        state, spec, 1.0, inlet_phase_id="cell_0", outlet_phase_id="collector"
    )
    assert not failed.success and failed.state is None


def test_closed_cell_reaches_finite_langmuir_quadratic():
    spec, state = _fixtures.native_world(_fixtures.cases()[1], 1)
    cell = state.phases.phases["cell_0"]
    total_gas = sum(cell.species_amounts_mol.values())
    records = dict(state.phases.phases)
    records["cell_0"] = replace(
        cell, species_amounts_mol={"CO2": 0.2 * total_gas, "H2O": 0.0, "N2": 0.8 * total_gas}
    )
    phases = PhaseLedger(records)
    state = state.replace(phases=phases, species_amounts=phases.total_amounts_mol())
    result = advance_fixed_bed(state, spec, 10.0, inlet_phase_id=None, outlet_phase_id=None)
    assert result.success, result.to_dict()
    amount, volume, b, sites = 0.2 * total_gas, 0.004, 100.0, 0.001
    linear = volume + sites * b - amount * b
    concentration = 2 * amount / (linear + sqrt(linear * linear + 4 * volume * b * amount))
    expected = sites * b * concentration / (1 + b * concentration)
    assert result.state.state_slots.species_amounts()["CO2"] == pytest.approx(expected, abs=1e-10)
    assert result.state.ledger.energy_jacket_J == pytest.approx(-20000 * expected, abs=1e-7)


def test_stage_switch_does_not_install_fresh_capacity_or_loading():
    spec, state = _fixtures.native_world(_fixtures.cases()[3], 2)
    switched = advance_fixed_bed(
        state, spec, 0.0, inlet_phase_id="purge", outlet_phase_id="collector"
    )
    assert switched.success, switched.to_dict()
    assert switched.state.state_slots == state.state_slots
    assert switched.state.species_amounts == state.species_amounts
    assert switched.state.ledger.time_s == state.ledger.time_s


def test_momentum_failure_does_not_commit_partial_state_or_time():
    spec, state = _fixtures.native_world(_fixtures.cases()[2], 2)
    limited = replace(spec, momentum_root=RootPolicy(tolerance=1e-30, maximum_iterations=1))
    result = advance_fixed_bed(
        state, limited, 1.0, inlet_phase_id="source", outlet_phase_id="collector"
    )
    assert not result.success and result.state is None and result.actual_duration_s == 0
    assert "momentum root failed" in result.diagnostics["error"]


def test_initial_stage_speed_violation_preserves_original_state(monkeypatch):
    case = {**_fixtures.cases()[4], "stages": (("purge", 1.0),)}
    spec, state = _fixtures.native_world(case, 8)
    original = state.to_dict()
    result = advance_fixed_bed(
        state, spec, 1.0, inlet_phase_id="purge", outlet_phase_id="collector"
    )
    assert not result.success and result.state is None and result.actual_duration_s == 0
    assert result.diagnostics["momentum"]["maximum_observed_speed_m_s"] > 0.5
    assert state.to_dict() == original

    # solve_ivp's event crossing alone misses an already violated initial domain.
    monkeypatch.setitem(sys.modules, "fixed_bed_cases", _fixtures)
    loader = importlib.util.spec_from_file_location(
        "bed_reference", Path(__file__).parent / "reference/fixed_bed_reference.py"
    )
    reference_module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(reference_module)
    reference = reference_module.reference_case(case, 8)
    assert not reference["success"]
    assert reference["end"] == 0
    assert reference["error"]["reason"] == "stage_initial_domain_violation"
    assert reference["error"]["event"] == "bed:maximum_face_velocity"
    assert reference["evaluate"](0)["gas"] == pytest.approx(reference["initial_gas"])


@pytest.mark.parametrize("difference", [-6111.5, 6111.5])
def test_inlet_resistance_is_bidirectional_and_independent_of_grid(difference):
    spec = _fixtures.specification(_fixtures.cases()[0], 2)
    resistance = 1e6
    for cells in (4, 8, 16):
        length, rho = .05 / cells, 1.2
        velocity, _ = fixed_bed_face_velocity(
            spec, distance_m=length, density_kg_m3=rho,
            pressure_difference_Pa=difference, series_resistance_Pa_s_m=resistance,
        )
        ergun = length * (150 * 1.8e-5 * .6**2 / (.4**3 * 1e-8) * velocity
                         + 1.75 * rho * .6 / (.4**3 * 1e-4) * velocity * abs(velocity))
        assert ergun + resistance * velocity == pytest.approx(difference, abs=1e-6)
        assert abs(velocity) < abs(difference) / resistance


def test_valve_declaration_codec_and_actual_fine_grid_purge():
    case = {**_fixtures.cases()[4], "inlet_resistance_Pa_s_m": 1e6}
    spec, state = _fixtures.native_world(case, 16)
    assert FixedBedSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec
    old = spec.to_dict()
    old.pop("inlet_resistance_Pa_s_m")
    assert FixedBedSpec.from_dict(old).inlet_resistance_Pa_s_m == 0
    for invalid in (-1, float("inf"), True):
        with pytest.raises(ValueError):
            replace(spec, inlet_resistance_Pa_s_m=invalid)
    original = state.to_dict()
    result = advance_fixed_bed(
        state, spec, .1, inlet_phase_id="purge", outlet_phase_id="collector")
    assert result.success, result.to_dict()
    assert result.state.ledger.time_s == pytest.approx(.1)
    assert result.diagnostics["momentum"]["maximum_observed_speed_m_s"] < .5
    assert result.state.phases.phases["purge"].species_amounts_mol["N2"] < (
        state.phases.phases["purge"].species_amounts_mol["N2"])
    assert state.to_dict() == original
