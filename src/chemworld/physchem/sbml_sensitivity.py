"""Forward tangents of original smooth SBML equations after declared controls.

These are boundary analysis candidates, never an alternate live world. Initial
amounts are supplied by the actual initialized world; source initial assignments
are not repeated when a time-zero experimental control changes a parameter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from math import isfinite
from typing import Any, cast

import numpy as np

from chemworld.physchem.biochemical_analysis import _check_scope
from chemworld.physchem.caloric_math import FirstDerivative, scalar_value
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.physchem.sbml_variational import sbml_variational_rates
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def check_sensitivity_scope(model: SBMLSystemSpec, parameters: Sequence[str]) -> None:
    """First profile: constant external controls, autonomous smooth fixed geometry."""
    if not isinstance(model, SBMLSystemSpec):
        raise ValueError("sensitivity requires the original SBMLSystemSpec")
    _check_scope(model.frame)
    if model.numerical_form != "ode":
        raise ValueError("forward tangents require the declared explicit ODE profile")
    if not parameters or len(set(parameters)) != len(parameters):
        raise ValueError("sensitivity controls must be nonempty and unique")
    for name in parameters:
        if name not in model.controlled_parameters or not model.frame._parameters[name].get(
            "constant"
        ):
            raise ValueError("sensitivity requires declared constant global parameter controls")


def sbml_trajectory_sensitivities(
    model: SBMLSystemSpec,
    initial_values: Mapping[str, float],
    controls: Mapping[str, float],
    times_s: Sequence[float],
) -> dict[str, Any]:
    """Readouts and d(readout)/d(source control), holding initialized amounts fixed.

    The primal and tangent share original rates. Seeding the controlled P
    coordinates propagates both direct and dynamic responses by the chain rule,
    including source functions and smooth assignment parameters. Native units
    remain those of each existing observable; parameter derivatives use the
    declared source units. Log-normalized responses are undefined at zero output.
    """
    names = tuple(controls)
    check_sensitivity_scope(model, names)
    times = tuple(times_s)
    if (
        not times
        or any(isinstance(t, bool) or not isfinite(t) or t < 0 for t in times)
        or any(b <= a for a, b in pairwise(times))
    ):
        raise ValueError("sample times must be finite, nonnegative and strictly increasing")
    if set(initial_values) != set(model.frame.initial_values):
        raise ValueError("initial sensitivity coordinates must match the original complete state")
    initial = model.frame.controlled_values(initial_values, 0.0, controls)
    keys = tuple(initial)
    n, p = len(keys), len(names)
    declared_scales = cast(Mapping[str, float], model.coordinate_scales)
    scales = np.asarray([declared_scales.get(key, max(abs(initial[key]), 1e-6)) for key in keys])
    tangent = np.zeros((n, p))
    for column, name in enumerate(names):
        tangent[keys.index("P:" + name), column] = 1.0
    packed = np.concatenate(
        (np.asarray([initial[k] for k in keys]) / scales, (tangent / scales[:, None]).ravel())
    )
    policy = ODESolverPolicy(
        "chemworld_sbml_forward_tangent_v1",
        model.solver_method,
        rtol=model.rtol,
        atol=model.atol,
        max_step_s=model.maximum_step_s,
    )

    def unpack(vector: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        return dict(zip(keys, vector[:n] * scales, strict=True)), (
            vector[n:].reshape(n, p) * scales[:, None]
        )

    rhs_calls = 0

    def rhs(time_s: float, vector: np.ndarray) -> np.ndarray:
        nonlocal rhs_calls
        rhs_calls += 1
        values, slopes = unpack(vector)
        raw_primal, raw_derivatives = sbml_variational_rates(
            model.frame, values, keys, slopes, time_s
        )
        primal = raw_primal / scales
        derivatives = raw_derivatives / scales[:, None]
        return np.concatenate((primal, derivatives.ravel()))

    records: list[dict[str, Any]] = []
    solvers: list[dict[str, Any]] = []

    def point(vector: np.ndarray, previous: float, time_s: float) -> tuple[np.ndarray, dict]:
        if time_s > previous:
            solved = solve_ode(rhs, vector, time_span_s=(previous, time_s), policy=policy)
            solvers.append(solved.diagnostic.to_dict())
            if not solved.success:
                raise ValueError("SBML tangent solve failed: " + solved.message)
            # Signed solver trial coordinates are never installed. Every accepted
            # state remains in the original nonnegative formal-model domain.
            for index, node_time in enumerate(solved.t):
                values, _ = unpack(solved.y[:, index])
                model.frame.validate_values(values, float(node_time))
            vector = solved.y[:, -1]
        values, slopes = unpack(vector)
        jets = {key: FirstDerivative(values[key], slopes[i]) for i, key in enumerate(keys)}
        outputs = model.readout(jets, time_s)
        readings = {}
        for observable in model.observables:
            output = outputs[observable.channel_id]
            value = float(scalar_value(output))
            derivative = np.broadcast_to(
                output.slope if isinstance(output, FirstDerivative) else np.zeros(p), (p,)
            )
            if not np.all(np.isfinite(derivative)):
                raise ValueError("nonfinite original SBML readout sensitivity")
            readings[observable.channel_id] = {
                "value": value,
                "unit": observable.unit,
                "derivative": {key: float(derivative[i]) for i, key in enumerate(names)},
                "normalized": {
                    key: float(derivative[i] * controls[key] / value) if value else None
                    for i, key in enumerate(names)
                },
            }
        return vector, readings

    report = {
        "schema_version": "chemworld-sbml-forward-sensitivity-1",
        "initialization": "author_initial_state_then_time_zero_control",
        "parameter_basis": "declared_source_units",
        "controls": dict(controls),
        "initial_coordinates_after_control": initial,
        "records": records,
        "solvers": solvers,
        "status": "complete",
        "failure": None,
        "chemical_composition_scope": "unknown_formal_entities",
        "energy_scope": "not_declared_formal_kinetics",
    }
    previous = 0.0
    try:
        for time_s in times:
            packed, readings = point(packed, previous, time_s)
            records.append({"time_s": float(time_s), "readouts": readings})
            previous = time_s
    except (ValueError, ArithmeticError, RuntimeError) as error:
        report.update(status="failed", failure=f"{type(error).__name__}: {error}")
    report.update(
        rhs_evaluations=rhs_calls,
        solver_reported_rhs_evaluations=sum(int(d["nfev"]) for d in solvers),
    )
    return report


__all__ = ["sbml_trajectory_sensitivities"]
