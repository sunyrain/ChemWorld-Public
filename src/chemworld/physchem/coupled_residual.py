"""Temporary residual packing for the existing coupled inventory/energy advance.

The caller retains its snapshot, conservation, thermal and resource checks.
The known rate matrix is analytic; physical state/time derivatives are explicitly
numerical until a domain supplies them. No solver state survives the interval.
"""

import numpy as np
from scipy import sparse

from chemworld.physchem.residual_solver import (
    ResidualEvent,
    ResidualSolvePolicy,
    mass_matrix_system,
)
from chemworld.physchem.residual_system import (
    ResidualDomainError,
    ResidualEquation,
    ResidualNumericalError,
    ResidualVariable,
    vector,
)
from chemworld.physchem.solver_backend import solve_residual


def solve_coupled_residual(system, state, initial, scales, rhs, view, callbacks, duration, grid):
    count = len(initial)
    material = system.thermodynamics.inventory_keys
    identities = (
        *material,
        "temperature",
        "boundary_heat",
        "chemical_energy",
        "heat_loss",
        "work",
        *(f"amount:{p.port_id}" for p in system.ports),
        *(f"energy:{p.port_id}" for p in (*system.ports, *system.energy_ports)),
        *(item.name for item in system.coordinates),
    )
    units = (
        *("mol" for _ in material),
        "K",
        *(["J"] * 4),
        *("mol" for _ in system.ports),
        *("J" for _ in (*system.ports, *system.energy_ports)),
        *(item.unit for item in system.coordinates),
    )
    # Inventory boundary-extension and accepted roundoff policy still belong
    # to CoupledSystem. This mathematical packing adds no physical clipping.
    variables = tuple(
        ResidualVariable(key, unit, scale)
        for key, unit, scale in zip(identities, units, scales, strict=True)
    )
    equations = tuple(
        ResidualEquation(key, unit + "/s", scale)
        for key, unit, scale in zip(identities, units, scales, strict=True)
    )
    algebraic = system.algebraic
    names = () if algebraic is None else tuple(v.name for v in algebraic.variables)

    def values(y):
        return {key: float(value) for key, value in zip(names, y[count:], strict=True)}

    if algebraic is None:
        guess = np.empty(0)
    else:
        guessed = algebraic.initial_values(state)
        if set(guessed) != set(names):
            raise ValueError("algebraic initial guesses must match declared identities")
        guess = vector([guessed[name] for name in names], len(names), "algebraic initial guess")
        variables += algebraic.variables
        equations += algebraic.equations

    def physical_rhs(t, y, p):
        if state.time_s + t < 0:
            raise ResidualDomainError("coupled physical time precedes zero")
        algebraic_values = values(y)
        rates = rhs(t, y[:count] / scales, algebraic_values) * scales
        if algebraic is None:
            return rates
        constraint = vector(
            algebraic.residual(view(t, y[:count], algebraic_values)),
            len(names),
            "coupled algebraic residual",
        )
        return np.r_[rates, -constraint]

    def physical_jacobian(t, y, p):
        jacobian = system.residual_rhs_jacobian(
            view(t, y[:count], values(y)), (*identities, *names)
        )
        if not system.isothermal:
            return jacobian
        # The fixed-temperature view ignores the numerical T coordinate. Apply
        # its chain rule to every RHS/constraint row, without mutating a domain
        # provider's matrix or changing the independent temperature mass row.
        temperature = len(material)
        if sparse.issparse(jacobian):
            projection = np.ones(len(variables))
            projection[temperature] = 0
            return jacobian.multiply(projection)
        projected = np.array(jacobian, copy=True)
        projected[:, temperature] = 0
        return projected

    residual = mass_matrix_system(
        "coupled-inventory-energy",
        variables,
        equations,
        sparse.diags(np.r_[np.ones(count), np.zeros(len(names))]),
        physical_rhs,
        derivatives="hybrid",
        rhs_jacobian=None if system.residual_rhs_jacobian is None else physical_jacobian,
    )
    events = tuple(
        ResidualEvent(
            spec.event_id,
            lambda t, y, p, callback=callback: callback(t, y[:count] / scales, values(y)),
            spec.direction,
            terminal=spec.terminal,
            initial_violation_stops=True,
        )
        for spec, callback in zip(system.all_events, callbacks, strict=True)
    )
    report = solve_residual(
        residual,
        np.r_[initial, guess],
        time_span_s=(0, duration),
        evaluation_times_s=None if not grid else sorted({0.0, duration, *grid}),
        policy=ResidualSolvePolicy(ode=system.policy),
        events=events,
    )
    if not report.success:
        raise ResidualNumericalError("coupled residual:" + report.failure_reason)
    diagnostics = {
        **report.diagnostics["ode"],
        "residual": report.diagnostics,
        "algebraic_values": values(report.final_point.state),
        "numerical_form": "residual",
        "solver_calls": int(bool(report.diagnostics["ode"])),
        "events": [
            {"name": name, "time_s": state.time_s + point.time_s} for name, point in report.events
        ],
    }
    return (
        [(p.time_s, p.state[:count]) for p in report.points],
        report.stopping_event,
        diagnostics,
    )
