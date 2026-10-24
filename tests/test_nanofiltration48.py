"""Multi-ion pore selectivity on the original finite pump/phase/clock path."""

import json
from dataclasses import replace

import pytest

from chemworld.physchem.nanofiltration import (
    advance_nanofiltration,
    nanofiltration_inventory,
    nanofiltration_point,
    nanofiltration_profile,
)
from chemworld.physchem.residual_system import RootPolicy
from chemworld.physchem.transport import PumpSpec
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.nanofiltration import NanofiltrationRuntime, unpack_nanofiltration
from chemworld.world.nanofiltration_world import nanofiltration_world_spec
from reference.nanofiltration48_fixtures import amounts, concentrations, spec


@pytest.mark.parametrize("count,charge", [(3, -20), (3, 20), (4, -20), (4, 20)])
def test_ion_flux_and_every_pore_node_are_electroneutral(count, charge):
    model = spec(count, charge)
    profile = nanofiltration_profile(model, amounts(model), 10e-6 / 60)
    pore = profile["pore"]
    assert pore["status"] == "converged"
    assert pore["solver"]["scaled_residual_norm"] <= 1e-10
    z = model.properties.charges
    for row in pore["pore_concentrations_mol_m3"]:
        assert sum(z[k] * row[k] for k in z) + charge == pytest.approx(0, abs=2e-7)
    for row in (profile["inlet_mol_s"], profile["permeate_mol_s"], profile["retentate_mol_s"]):
        assert sum(z[k] * row[k] for k in z) == pytest.approx(0, abs=1e-14)
    for k in model.properties.component_ids:
        assert profile["inlet_mol_s"][k] == pytest.approx(
            profile["permeate_mol_s"][k] + profile["retentate_mol_s"][k]
        )
    if charge > 0:
        assert pore["rejection"]["Na+"] < 0


def test_finite_nh_conserves_original_species_and_pump_energy():
    model = spec()
    initial = nanofiltration_inventory(model, amounts(model))
    result, receipt = advance_nanofiltration(model, initial, 10e-6 / 60, 5)
    for key in model.properties.component_ids:
        assert sum(n[key] for n in result.final_state.amounts_mol) == pytest.approx(
            initial.amounts_mol[0][key], abs=1e-12
        )
    change = sum(result.final_state.enthalpies_J) - sum(initial.enthalpies_J)
    assert change == pytest.approx(
        receipt["shaft_work_J"] - receipt["motor_loss_J"] + receipt["bath_heat_into_liquid_J"],
        abs=1e-9,
    )
    assert receipt["hydraulic_work_J"] == pytest.approx(
        (model.inlet_pressure_Pa - model.reservoir_pressure_Pa) * 10e-6 / 60 * 5
    )


def test_actual_bounded_pump_and_finite_capacity():
    model = replace(spec(), pump=PumpSpec(1.5e6, 0.01, 0.75), retentate_capacity_L=1e-4)
    initial = nanofiltration_inventory(model, amounts(model))
    result, receipt = advance_nanofiltration(model, initial, 10e-6 / 60, 20)
    assert receipt["profile"]["flow_limited"]
    assert result.stopping_event == "retentate_full"
    assert 0 < result.actual_duration_s < 20
    assert model.properties.volume(
        result.final_state.amounts_mol[2], 310, 1e5
    ) * 1000 == pytest.approx(1e-4, abs=1e-10)


def test_infeasible_or_iteration_limited_pore_is_not_repaired():
    model = spec()
    c = concentrations(model)
    failed = nanofiltration_point(model, c, policy=RootPolicy(maximum_iterations=1))
    assert failed["status"] == "not_converged"
    assert failed["solver"]["failure_reason"] == "iteration_limit"
    with pytest.raises(RuntimeError, match="iteration_limit"):
        nanofiltration_profile(
            model, amounts(model), 10e-6 / 60, policy=RootPolicy(maximum_iterations=1)
        )
    with pytest.raises(ValueError, match="electroneutral"):
        nanofiltration_point(model, {**c, "Na+": 11})
    with pytest.raises(ValueError, match="solvent"):
        model.properties.validate_amounts({**amounts(model), "H2O": 0})
    with pytest.raises(ValueError, match="radius"):
        replace(model, pore_radius_m=1e-10)


@pytest.mark.parametrize("preloaded", [False, True])
def test_real_flow_wait_preloaded_collectors_and_exact_snapshot(preloaded):
    model = spec(4)
    kwargs = (
        {"permeate": amounts(model, 0.05, 0.5), "retentate": amounts(model, 0.05, 1.5)}
        if preloaded
        else {}
    )
    composition = nanofiltration_world_spec(model, amounts(model), **kwargs)
    runtime = ChemWorldRuntime(composition=composition)
    restored = None
    try:
        runtime.reset(seed=48)
        initial = unpack_nanofiltration(runtime._state)[2]
        for action in (
            {"operation": "set_flow_rate", "flow_rate_mL_min": 10},
            {"operation": "run_flow", "duration_s": 2},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            result = session.execute({"operation": "wait", "duration_s": 3})
            assert result.info["transaction_status"] == "committed", result.info
            _, _, final = unpack_nanofiltration(session._state)
            assert session._state.ledger.time_s == 5
            assert session._state.ledger.flow_pump_work_J > 0
            for key in model.properties.component_ids:
                assert sum(n[key] for n in final.amounts_mol) == pytest.approx(
                    sum(n[key] for n in initial.amounts_mol), abs=2e-10
                )
            for row in final.amounts_mol:
                assert sum(
                    model.properties.charges[k] * row[k] for k in model.properties.ion_ids
                ) == pytest.approx(0, abs=2e-12)
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_isothermal_world_rejects_external_heat_without_advancing():
    model = spec()
    composition = nanofiltration_world_spec(model, amounts(model))
    states = composition.physics.initial_states
    state = states["reactor"].replace(
        inactive_vessels={k: v for k, v in states.items() if k != "reactor"}
    )
    result = NanofiltrationRuntime(composition).advance(state, {"duration_s": 1}, heat=True)
    assert result.outcome == "rejected"
    assert result.actual_duration_s == 0


def test_legacy_flow_meter_does_not_invent_a_binary_salt_observation():
    model = spec()
    runtime = ChemWorldRuntime(composition=nanofiltration_world_spec(model, amounts(model)))
    try:
        runtime.reset(seed=48)
        before = runtime._state.to_dict()
        result = runtime.execute({"operation": "measure", "instrument": "flow_meter"})
        assert result.info["transaction_status"] != "committed"
        assert runtime._state.species_amounts == before["species_amounts"]
        assert runtime._state.ledger.time_s == before["ledger"]["time_s"]
        assert runtime._state.process.last_observation_record is None
    finally:
        runtime.close()
