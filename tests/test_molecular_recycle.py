"""One real tear root, same-reference units and original finite material/energy."""

import json
from dataclasses import replace

import pytest

from chemworld.foundation import WorldState
from chemworld.physchem.molecular_processes import MolecularHeatDuty, MolecularPressureStage
from chemworld.physchem.molecular_recycle import MolecularRecyclePath, MolecularRecyclePolicy
from chemworld.runtime.molecular_process_path import advance_molecular_process_receiver
from reference.molecular_recycle_cases import expected, path, receiver


@pytest.mark.parametrize("cycle", [False, True])
@pytest.mark.parametrize("recycle", [0.2, 0.5])
def test_tear_solves_original_devices_and_net_boundary(cycle, recycle):
    model = path(recycle=recycle, pressure_cycle=cycle)
    result = model.evaluate()
    target = expected((0.25, 0.75), recycle, cycle)
    assert result.success, result.to_dict()
    assert result.outlet.state.temperature_K == pytest.approx(target["outlet_T"], rel=2e-7)
    assert result.outlet.molar_rate_mol_s == pytest.approx(0.05, abs=1e-10)
    assert result.diagnostics["tear_molar_rate_mol_s"] == pytest.approx(target["tear_flow"])
    assert result.shaft_work_W == pytest.approx(target["work"], abs=2e-7)
    assert abs(result.diagnostics["energy_residual_W"]) < 1e-5
    assert result.diagnostics["trial_evaluations"] > 1
    json.dumps(result.to_dict(), allow_nan=False)


@pytest.mark.parametrize("recycle", [0.0, 0.3])
def test_zero_recycle_and_exactly_absent_component_do_not_create_stock(recycle):
    model = path(x=(1.0, 0.0), recycle=recycle)
    result = model.evaluate()
    assert result.success, result.to_dict()
    assert result.outlet.state.mole_fractions == pytest.approx((1.0, 0.0), abs=1e-14)
    assert result.outlet.molar_rate_mol_s == pytest.approx(0.05)
    if not recycle:
        assert result.diagnostics["unknown_count"] == 0


def test_declared_pressure_closure_and_source_are_required():
    model = path()
    with pytest.raises(ValueError, match="pressure loop"):
        replace(model, devices=(MolecularPressureStage(model.domain, "compressor", 4e5),))
    with pytest.raises(ValueError, match="feed"):
        replace(model, recycle_fraction=1.0)
    with pytest.raises(ValueError, match="feed"):
        replace(model, initial_recycle_rates_mol_s=(-0.01, 0.02))


def test_failed_guesses_devices_and_budgets_retain_diagnostics():
    model = path()
    outside = replace(model, initial_temperature_K=200.0).evaluate()
    assert not outside.success and outside.failure_reason == "initial_outside_bounds"
    bad = replace(model, devices=(MolecularHeatDuty(model.domain, 1e9),)).evaluate()
    assert not bad.success and bad.diagnostics["trials"][0]["success"] is False
    budget = replace(model, policy=MolecularRecyclePolicy(maximum_trial_evaluations=1)).evaluate()
    assert not budget.success
    assert budget.diagnostics["trial_evaluations"] == 1


def test_original_receiver_owns_product_once_and_json_continues():
    model = path(real=True, recycle=0.3)
    state = receiver(model)
    before = state.to_dict(include_hidden=True)
    full = advance_molecular_process_receiver(
        state, model, 0.2, phase_id="bulk", maximum_amount_mol=0.3
    )
    assert full.success, full.diagnostics
    first = advance_molecular_process_receiver(
        state, model, 0.1, phase_id="bulk", maximum_amount_mol=0.3
    )
    assert first.success, first.diagnostics
    recovered = WorldState.from_dict(
        json.loads(json.dumps(first.state.to_dict(include_hidden=True)))
    )
    decoded = MolecularRecyclePath.from_dict(json.loads(json.dumps(model.to_dict())))
    second = advance_molecular_process_receiver(
        recovered, decoded, 0.1, phase_id="bulk", maximum_amount_mol=0.3
    )
    assert second.success, second.diagnostics
    assert decoded == model
    assert state.to_dict(include_hidden=True) == before
    assert full.state.ledger.flow_heat_J == pytest.approx(1.0, abs=2e-5)
    assert full.state.ledger.flow_shaft_work_J == 0
    assert full.state.ledger.energy_jacket_J == 0
    for key, original in state.species_amounts.items():
        x = model.inlet.state.mole_fractions[model.domain.model.component_ids.index(key)]
        assert full.state.species_amounts[key] == pytest.approx(original + 0.01 * x, abs=2e-8)
    assert second.state.species_amounts == pytest.approx(full.state.species_amounts, abs=2e-8)
    assert second.state.temperature_K == pytest.approx(full.state.temperature_K, rel=2e-7)
    assert abs(full.diagnostics["receiver_energy_residual_J"]) < 2e-5
    rejected = advance_molecular_process_receiver(
        state,
        replace(model, devices=(MolecularHeatDuty(model.domain, 1e9),)),
        0.2,
        phase_id="bulk",
        maximum_amount_mol=0.3,
    )
    assert not rejected.success and rejected.state is None
    assert state.to_dict(include_hidden=True) == before
