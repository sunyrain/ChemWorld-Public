"""Periodic dual-membrane transport and original electrical world behavior."""

import json
from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.electrodialysis import (
    advance_electrodialysis,
    electrodialysis_inventory,
    electrodialysis_point,
)
from chemworld.physchem.ion_exchange_pore import PoreResidualBlock
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import RootPolicy
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.electrodialysis import unpack_electrodialysis
from chemworld.world.electrodialysis_world import electrodialysis_world_spec
from reference.electrodialysis48_fixtures import inventories, spec


@pytest.mark.parametrize("current_density", [0.0, 300.0, -300.0])
def test_exponential_flux_analytic_jacobian(current_density):
    model = spec(4)
    ns = inventories(model)
    block = PoreResidualBlock(
        replace(model.aem, segments=4),
        model.properties,
        model.properties.concentrations(ns[1]),
        model.properties.concentrations(ns[0]),
        model.temperature_K,
        model.reservoir_pressure_Pa,
    )
    y = block.initial(current_density)
    analytical, current_derivative = block.jacobian(y, current_density)
    step = 1e-5
    for column in range(block.width):
        delta = np.zeros(block.width)
        delta[column] = step
        numerical = (
            block.residual(y + delta, current_density)
            - block.residual(y - delta, current_density)
        ) / (2 * step)
        np.testing.assert_allclose(analytical[:, column], numerical, rtol=2e-6, atol=2e-8)
    numerical = (
        block.residual(y, current_density + step) - block.residual(y, current_density - step)
    ) / (2 * step)
    np.testing.assert_allclose(current_derivative, numerical, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("count,direction", [(3, -1), (3, 1), (4, -1), (4, 1)])
def test_two_opposite_membranes_preserve_each_tank_charge(count, direction):
    model = spec(count)
    ns = inventories(model, direction)
    row = electrodialysis_point(model, ns, 0.001, "A")
    assert row["status"] == "converged"
    assert row["dissipation_W"] > 0
    assert row["aem"]["current_density_A_m2"] == row["cem"]["current_density_A_m2"]
    assert row["aem"]["water_flux_m_s"] < row["cem"]["water_flux_m_s"]
    for field in ("diluate_rates_mol_s", "concentrate_rates_mol_s"):
        assert sum(
            model.properties.charges[k] * row[field][k] for k in model.properties.ion_ids
        ) == pytest.approx(0, abs=1e-22)
    reverse = electrodialysis_point(model, ns, row["voltage_V"], "V")
    assert reverse["current_A"] == pytest.approx(0.001, abs=1e-11)


def test_finite_material_work_and_bath_use_one_shared_nh_advance():
    model = spec(4)
    initial = electrodialysis_inventory(model, *inventories(model))
    result, receipt = advance_electrodialysis(
        model, initial, ScalarControlProgram(((0, 0.001),), "A"), 5
    )
    for key in model.properties.component_ids:
        assert sum(n[key] for n in result.final_state.amounts_mol) == pytest.approx(
            sum(n[key] for n in initial.amounts_mol), abs=1e-12
        )
    assert sum(result.final_state.enthalpies_J) == pytest.approx(
        sum(initial.enthalpies_J), abs=1e-8
    )
    assert receipt["electrical_work_J"] == pytest.approx(
        -receipt["bath_heat_into_liquid_J"], abs=1e-10
    )
    assert result.solver_diagnostic["pore_solves"] > 0


def test_actual_compliance_stops_without_clipping_the_source():
    model = replace(spec(), maximum_voltage_V=0.15)
    initial = electrodialysis_inventory(model, *inventories(model))
    result, receipt = advance_electrodialysis(
        model, initial, ScalarControlProgram(((0, 0.003),), "A"), 2
    )
    assert result.stopping_event == "voltage_compliance"
    assert result.actual_duration_s == 0
    assert receipt["electrical_work_J"] == 0
    assert receipt["final"]["current_A"] == pytest.approx(0.003)


def test_bad_charge_orientation_and_nonconverged_root_are_explicit():
    model = spec()
    ns = inventories(model)
    with pytest.raises(ValueError, match="orientation"):
        replace(model, aem=replace(model.aem, fixed_charge_mol_m3=-100))
    with pytest.raises(ValueError, match="electroneutral"):
        electrodialysis_point(model, ({**ns[0], "Na+": 0.001}, ns[1]), 0.001, "A")
    failed = electrodialysis_point(model, ns, 0.003, "A", policy=RootPolicy(maximum_iterations=1))
    assert failed["status"] == "not_converged"
    assert failed["solver"]["failure_reason"] == "iteration_limit"


def test_real_cc_cv_ramp_and_exact_snapshot_account_power_once():
    model = spec(4)
    runtime = ChemWorldRuntime(composition=electrodialysis_world_spec(model, *inventories(model)))
    restored = None
    try:
        runtime.reset(seed=48)
        for action in (
            {"operation": "configure_electrical_control", "current_A": 0.001},
            {"operation": "wait", "duration_s": 1},
            {"operation": "configure_electrical_control", "potential_V": 0.2},
            {"operation": "configure_electrical_control", "potential_V": 0.3, "ramp_duration_s": 1},
        ):
            result = runtime.execute(action)
            assert result.info["transaction_status"] == "committed", result.info
        restored = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(runtime.snapshot().to_dict()))
        )
        for session in (runtime, restored):
            result = session.execute({"operation": "wait", "duration_s": 2})
            assert result.info["transaction_status"] == "committed", result.info
            _, _, inventory = unpack_electrodialysis(session._state)
            assert session._state.ledger.time_s == 3
            assert session._state.ledger.energy_jacket_J == 0
            assert session._state.ledger.flow_electrical_work_J == pytest.approx(
                session._state.ledger.heat_loss_J, abs=1e-9
            )
            for n in inventory.amounts_mol:
                assert sum(
                    model.properties.charges[k] * n[k] for k in model.properties.ion_ids
                ) == pytest.approx(0, abs=1e-12)
            for name in ("reactor", "concentrate"):
                tank = (
                    session._state if name == "reactor" else session._state.inactive_vessels[name]
                )
                assert tank.thermal.vessels[name].heat_loss_J == pytest.approx(
                    session._state.ledger.heat_loss_J / 2, abs=1e-9
                )
        assert runtime._state.to_dict() == restored._state.to_dict()
    finally:
        runtime.close()
        if restored:
            restored.close()


def test_empty_concentrate_and_old_instruments_are_not_supported():
    model = spec()
    ns = inventories(model)
    with pytest.raises(ValueError, match="solvent"):
        electrodialysis_world_spec(model, ns[0], dict.fromkeys(model.properties.component_ids, 0))
    runtime = ChemWorldRuntime(composition=electrodialysis_world_spec(model, *ns))
    try:
        runtime.reset(seed=48)
        for instrument in ("flow_meter", "electrical_meter"):
            before = dict(runtime._state.species_amounts)
            result = runtime.execute({"operation": "measure", "instrument": instrument})
            assert result.info["transaction_status"] != "committed"
            assert runtime._state.species_amounts == before
            assert runtime._state.ledger.time_s == 0
            assert runtime._state.process.last_observation_record is None
    finally:
        runtime.close()
