"""Common residual consumers retain physical balances and temporary algebraics."""

from collections.abc import Mapping
from dataclasses import fields, replace

import numpy as np
import pytest

from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledAlgebraic,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    InternalTransfer,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.finite_volume import ConservativeDiffusion, DiffusionBoundary, DiffusionMesh
from chemworld.physchem.residual_system import ResidualEquation, ResidualVariable
from chemworld.physchem.solver_backend import ODESolverPolicy
from test_conservative_diffusion import coupled
from test_coupled_system import initial, shared_system


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_residual_assembly_preserves_common_reaction_heat_and_material_laws(method):
    ode = shared_system()
    residual = replace(
        ode, numerical_form="residual", policy=ODESolverPolicy("residual-test", method=method)
    )
    before = initial()
    expected = ode.advance(before, 3, evaluation_times_s=[0, 3])
    result = residual.advance(before, 3, evaluation_times_s=[0, 3])
    assert result.final_state.amounts_mol == pytest.approx(
        expected.final_state.amounts_mol, rel=2e-6, abs=1e-11
    )
    assert result.final_state.temperature_K == pytest.approx(
        expected.final_state.temperature_K, abs=1e-6
    )
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-6)
    assert result.actual_duration_s == 3
    assert result.final_state.algebraic_values == {}
    assert before == initial()
    modes = result.solver_diagnostic["residual"]["derivative_modes"]
    assert modes["rate"] == "analytic" and modes["state"] == "finite_difference"


@pytest.mark.parametrize("method", ["BDF", "Radau"])
@pytest.mark.parametrize("numerical_form", ["ode", "residual"])
def test_only_ode_path_constructs_numerical_jacobian_sparsity(monkeypatch, method, numerical_form):
    system = replace(
        shared_system(),
        numerical_form=numerical_form,
        policy=ODESolverPolicy("sparsity-routing-test", method=method),
    )
    before = initial()
    baseline = system.advance(before, 0.2, evaluation_times_s=[0, 0.1, 0.2])
    original = CoupledSystem.jacobian_sparsity
    calls = []

    def guarded(self):
        calls.append(self.numerical_form)
        if self.numerical_form == "residual":
            raise AssertionError("residual advance allocated an unused ODE sparsity mask")
        return original(self)

    monkeypatch.setattr(CoupledSystem, "jacobian_sparsity", guarded)
    actual = system.advance(before, 0.2, evaluation_times_s=[0, 0.1, 0.2])
    assert calls == (["ode"] if numerical_form == "ode" else [])
    assert actual.outcome == baseline.outcome == "success"
    assert actual.stopping_event == baseline.stopping_event is None
    assert actual.actual_duration_s == baseline.actual_duration_s == 0.2
    # Independent Radau calls need not be bitwise identical. Check all dynamic
    # fields to a last-bit envelope, far below the unchanged 1e-8 solver rtol;
    # the mask-routing guard above remains exact and catches the original bug.
    for point, expected in zip(
        (*actual.trajectory, actual.final_state),
        (*baseline.trajectory, baseline.final_state), strict=True,
    ):
        for item in fields(CoupledState):
            left, right = getattr(point, item.name), getattr(expected, item.name)
            if isinstance(left, Mapping):
                assert left.keys() == right.keys()
                if left:
                    np.testing.assert_array_max_ulp(
                        list(left.values()), [right[key] for key in left], maxulp=32)
            elif item.name in {"time_s", "volume_L"}:
                assert left == right
            else:
                np.testing.assert_array_max_ulp([left], [right], maxulp=32)
    # Balance residuals subtract physical totals, so their roundoff scale is the
    # initial total, not the (possibly zero) residual's own ULP.
    roundoff = 32 * np.finfo(float).eps
    assert actual.energy_balance_residual_J == pytest.approx(
        baseline.energy_balance_residual_J, rel=0,
        abs=roundoff * abs(system.thermodynamics.internal_energy_j(before)))
    assert actual.conservation_residuals == pytest.approx(
        baseline.conservation_residuals, rel=0,
        abs=roundoff * sum(before.amounts_mol.values()))


@pytest.mark.parametrize(
    "boundary",
    [DiffusionBoundary(), DiffusionBoundary("flux", 0.02), DiffusionBoundary("concentration", 1)],
)
def test_common_spatial_residual_assembly_preserves_real_boundary_receipts(boundary):
    grid = DiffusionMesh.planar(1, 5)
    ode, state = coupled(ConservativeDiffusion(grid, 0.1, right=boundary), 2 + grid.centers_m)
    result = replace(ode, numerical_form="residual").advance(
        state, 0.2, evaluation_times_s=[0, 0.2]
    )
    expected = ode.advance(state, 0.2, evaluation_times_s=[0, 0.2])
    assert result.final_state.amounts_mol == pytest.approx(
        expected.final_state.amounts_mol, abs=1e-9
    )
    assert result.final_state.exchanged_mol == pytest.approx(
        expected.final_state.exchanged_mol, abs=1e-9
    )
    for actual, wanted in zip(result.actual_exchanges, expected.actual_exchanges, strict=True):
        assert actual == pytest.approx(wanted, abs=1e-9)
    assert result.conservation_residuals["material_mol"] == pytest.approx(0, abs=1e-10)


def algebraic_reaction():
    algebraic = CoupledAlgebraic(
        (ResidualVariable("squared_A", "mol^2", 1, "algebraic", lower=0),),
        (ResidualEquation("square", "mol^2"),),
        lambda state: {"squared_A": 0.1},
        lambda state: [state.algebraic_values["squared_A"] - state.amounts_mol["A"] ** 2],
    )
    process = CoupledProcess(
        "conversion",
        lambda state: ProcessContribution(
            internal_transfers=(
                InternalTransfer("A-to-B", "A", "B", 0.4 * state.algebraic_values["squared_A"]),
            )
        ),
    )
    return CoupledSystem(
        ConstantCvThermodynamics({k: SpeciesInternalEnergy(0, 0) for k in ("A", "B")}, 1),
        (process,),
        isothermal=True,
        numerical_form="residual",
        algebraic=algebraic,
        inventory_scales_mol={"A": 1, "B": 1},
        conservation_laws={"material": {"A": 1, "B": 1}},
        events=(CoupledEvent("half-A", lambda state: state.algebraic_values["squared_A"] - 0.25),),
    )


def test_processes_and_events_consume_consistent_algebraic_values_without_persisting_them():
    system = algebraic_reaction()
    before = CoupledState({"A": 1, "B": 0}, 300, 1)
    result = system.advance(before, 5, evaluation_times_s=[0, 1, 4])
    assert result.stopping_event == "half-A"
    assert result.actual_duration_s == pytest.approx(2.5, rel=3e-7)
    assert result.final_state.amounts_mol == pytest.approx({"A": 0.5, "B": 0.5}, abs=1e-9)
    assert result.solver_diagnostic["algebraic_values"] == pytest.approx(
        {"squared_A": 0.25}, abs=1e-9
    )
    for point in result.trajectory:
        assert not point.algebraic_values
        assert point.amounts_mol["A"] == pytest.approx(1 / (1 + 0.4 * point.time_s), rel=3e-7)


@pytest.mark.parametrize("duration", [0, 1])
def test_initial_algebraic_stop_uses_consistency_and_existing_direction_rule(duration):
    system = algebraic_reaction()
    state = CoupledState({"A": 0.4, "B": 0.6}, 300, 1)
    result = system.advance(state, duration)
    assert result.actual_duration_s == 0
    assert result.solver_diagnostic["algebraic_values"]["squared_A"] == pytest.approx(0.16)
    assert result.final_state == state


def test_algebraic_constraints_cannot_be_silently_ignored_by_the_ode_path():
    with pytest.raises(ValueError, match="explicit residual"):
        replace(algebraic_reaction(), numerical_form="ode")
    with pytest.raises(ValueError, match="not persistent"):
        CoupledState({"A": 1}, 300, 1, algebraic_values={"hidden": 1})
