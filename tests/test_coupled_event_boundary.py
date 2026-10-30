"""Rounded depletion roots must preserve the entire conservative trajectory."""

from dataclasses import replace

import numpy as np
import pytest

from chemworld.physchem.coupled_events import feasible_terminal_point
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledCoordinate,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.solver_backend import ODESolverPolicy


def test_roundoff_uses_one_full_trajectory_point_without_clipping_material_or_work():
    root = float(np.nextafter(0.7, np.inf))

    def trajectory(t):
        return np.array([0.7 - t, t, 3 * t])

    time, state = feasible_terminal_point(
        root, trajectory(root), trajectory, lambda _t, y: y[0] >= 0)
    assert time < root
    assert np.array_equal(state, trajectory(time))
    assert state[0] + state[1] == 0.7
    assert state[2] == 3 * time


def test_large_domain_error_is_not_repaired_by_moving_the_event_or_inventory():
    with pytest.raises(RuntimeError, match="roundoff interval"):
        feasible_terminal_point(1., np.array([-0.1]), lambda t: np.array([.9 - t]),
                                lambda _t, y: y[0] >= 0)


def test_already_feasible_event_keeps_its_original_vector_and_time():
    vector = np.array([0., 0.7])
    time, result = feasible_terminal_point(.7, vector, None, lambda _t, _y: True)
    assert time == .7 and result is vector


@pytest.mark.parametrize("grid", (None, (0., .4, 1., 2.)))
def test_actual_coupled_depletion_preserves_progress_heat_and_requested_grid(grid):
    system = CoupledSystem(
        ConstantCvThermodynamics({}, 10.),
        (CoupledProcess("consume", lambda _s: ProcessContribution(
            coordinate_rates={"supply": -0.3, "used": 0.3})),),
        heat_flux=lambda _s: HeatFlux(heater_W=3.),
        coordinates=(CoupledCoordinate("supply", "kg", 1., minimum=0.),
                     CoupledCoordinate("used", "kg", 1., minimum=0.)),
        conservation_laws={"mass": {"supply": 1., "used": 1.}},
        events=(CoupledEvent("empty", lambda s: s.coordinates["supply"], feasible_side=True),),
        policy=ODESolverPolicy("finite-event", method="BDF"),
    )
    initial = CoupledState({}, 300., 1., coordinates={"supply": .2, "used": 0.})
    result = system.advance(initial, 2., evaluation_times_s=grid)
    assert result.stopping_event == "empty"
    assert result.actual_duration_s == pytest.approx(2 / 3, abs=1e-12)
    assert result.final_state.coordinates["supply"] >= 0
    assert sum(result.final_state.coordinates.values()) == pytest.approx(.2, abs=1e-15)
    assert result.final_state.boundary_heat_J == pytest.approx(3 * result.actual_duration_s)
    assert all(s.time_s <= result.final_state.time_s for s in result.trajectory)
    assert result.solver_diagnostic["solver_calls"] == 1
    with pytest.raises(ValueError, match="ODE interpolant"):
        replace(system, numerical_form="residual")


@pytest.mark.parametrize("changes", ({"terminal": False}, {"direction": 0}, {"feasible_side": 1}))
def test_feasible_side_requires_an_explicit_directed_terminal_event(changes):
    with pytest.raises(ValueError, match="directed terminal"):
        CoupledEvent("bad", lambda _s: 1., **{"feasible_side": True, **changes})


@pytest.mark.parametrize("negative,accepted", [(-1e-15, True), (-1e-10, False)])
def test_terminal_event_uses_original_snapshot_tolerance_for_unrelated_zero_stock(
    monkeypatch, negative, accepted,
):
    import chemworld.physchem.coupled_system as coupled

    original_solve = coupled.solve_ode
    inventory_scale = 1e-3

    def solve_with_zero_stock_roundoff(*args, **kwargs):
        report = original_solve(*args, **kwargs)
        interpolant = report.raw_result.sol

        def perturbed(t):
            values = np.asarray(interpolant(t)).copy()
            values[0] = negative/inventory_scale
            return values

        report.raw_result.sol = perturbed
        for vector in report.raw_result.y_events[0]:
            vector[0] = negative/inventory_scale
        return report

    monkeypatch.setattr(coupled, "solve_ode", solve_with_zero_stock_roundoff)
    system = CoupledSystem(
        ConstantCvThermodynamics({"tracer": SpeciesInternalEnergy(0., 0.)}, 10.), (),
        heat_flux=lambda _s: HeatFlux(heater_W=3.),
        inventory_scales_mol={"tracer": inventory_scale},
        conservation_laws={"tracer": {"tracer": 1.}},
        events=(CoupledEvent("temperature-limit", lambda s: 301.-s.temperature_K,
                             feasible_side=True),),
        policy=ODESolverPolicy("zero-stock-event", method="BDF"),
    )
    initial = CoupledState({"tracer": 0.}, 300., 1.)
    if not accepted:
        with pytest.raises(RuntimeError, match="roundoff interval"):
            system.advance(initial, 5.)
        return
    result = system.advance(initial, 5.)
    assert result.stopping_event == "temperature-limit"
    assert result.actual_duration_s == pytest.approx(10/3, rel=0, abs=1e-10)
    assert result.final_state.amounts_mol["tracer"] == 0.
    assert result.roundoff_corrections_mol["tracer"] == pytest.approx(-negative, rel=1e-14, abs=0)
    assert abs(result.energy_balance_residual_J) <= 1e-10
    assert initial.amounts_mol["tracer"] == 0.
