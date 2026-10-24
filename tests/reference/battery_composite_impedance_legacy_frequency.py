# Historical regression fixture only; not an active scientific reference entry point.
# Exact source body: 252a1b8d:tests/reference/battery_frequency_pybamm.py.
# This fixture is used only by the full-return compatibility contract.
# ruff: noqa
"""Independent small-signal evaluation of released PyBaMM equations (validation only).

No ChemWorld solver, derivative, FV operator or constitutive law is used here.
The reference is the released physical model, not a claimed upstream EIS API.
"""

import traceback

import numpy as np
from battery_dfn_pybamm import build_dfn_reference
from battery_spme_pybamm import build_spme_reference
from scipy import sparse
from scipy.sparse.linalg import splu

OUTPUTS = (
    "Voltage [V]",
    "X-averaged negative electrolyte concentration [mol.m-3]",
    "X-averaged separator electrolyte concentration [mol.m-3]",
    "X-averaged positive electrolyte concentration [mol.m-3]",
)


def casadi_sparse(value):
    """Preserve the upstream sparse Jacobian, including explicit zeros and shape."""
    rows, columns = value.sparsity().get_triplet()
    return sparse.csc_matrix((value.nonzeros(), (rows, columns)), shape=value.shape)


def reference_frequency(
    spec, family, frequencies, *, refined=False, spme_radial_mesh="exponential-right",
    ocp_functions=None,
):
    if family in {"SPM", "SPMe"}:
        points = 240 if refined else 120
        simulation = build_spme_reference(spec, cells=points, radial_mesh=spme_radial_mesh,
                                          family=family, ocp_functions=ocp_functions)
        grid = {
            "spatial_cells_per_region": points,
            "radial_cells": points,
            "radial_mesh": spme_radial_mesh,
            **({"stretch": 2.3} if spme_radial_mesh == "exponential-right" else {}),
        }
    elif family == "DFN":
        nx, nr = (96, 200) if refined else (64, 100)
        simulation = build_dfn_reference(spec, spatial_cells=nx, radial_cells=nr,
                                         ocp_functions=ocp_functions)
        grid = {
            "spatial_cells_per_region": nx,
            "radial_cells": nr,
            "radial_mesh": "exponential-right",
            "stretch": 2.3,
        }
    else:
        raise ValueError("reference family must be SPM, SPMe or DFN")
    return frequency_from_simulation(simulation, frequencies, family=family, grid=grid)


def frequency_from_simulation(simulation, frequencies, *, family, grid, output_names=OUTPUTS):
    """Linearize a released model; preserve the original four-output default."""
    import casadi
    import pybamm

    simulation.build()
    model = simulation.built_model
    inputs = {"Current function [A]": 0.0}
    solver = simulation.solver
    solver.set_up(model, inputs=inputs)
    initial = np.asarray(solver.calculate_consistent_state(model, inputs=inputs)).ravel()
    exported = model.export_casadi_objects(output_names, input_parameter_order=list(inputs))
    states = casadi.vertcat(exported["x"], exported["z"])
    rhs = casadi.vertcat(exported["rhs"], exported["algebraic"])
    outputs = casadi.vertcat(*exported["variables"].values())
    function = casadi.Function(
        "frequency_coefficients",
        [exported["t"], states, exported["inputs"]],
        [
            rhs,
            casadi.jacobian(rhs, states),
            casadi.jacobian(rhs, exported["inputs"]),
            casadi.jacobian(outputs, states),
            casadi.jacobian(outputs, exported["inputs"]),
        ],
    )
    residual, jacobian, forcing, output_state, output_input = function(0, initial, [0])
    residual = np.asarray(residual).ravel()
    n_diff = int(model.concatenated_rhs.size)
    rhs_error = float(np.max(np.abs(residual[:n_diff]), initial=0))
    alg_error = float(np.max(np.abs(residual[n_diff:]), initial=0))
    metadata = {
        "model": family,
        "pybamm": pybamm.__version__,
        "casadi": casadi.__version__,
        "grid": grid,
        "state_count": len(initial),
        "differential_count": n_diff,
        "maximum_stationary_rhs": rhs_error,
        "maximum_stationary_algebraic": alg_error,
        "stationary_tolerance": 1e-6,
        "linear_tolerance": 1e-10,
        "response_order": (["Z_ohm", "negative_mol_m3_A", "separator_mol_m3_A", "positive_mol_m3_A"]
                           if output_names == OUTPUTS else ["Z_ohm"]),
        "definition": "Z=-delta_V/delta_I; positive discharge; exp(+i*2*pi*f*t)",
    }
    if not np.all(np.isfinite(residual)) or max(rhs_error, alg_error) > 1e-6:
        return {**metadata, "success": False, "failure": "reference_not_stationary", "points": []}
    mass = sparse.csc_matrix(model.mass_matrix.entries)
    jacobian, output_state = casadi_sparse(jacobian), casadi_sparse(output_state)
    forcing, output_input = np.asarray(forcing).ravel(), np.asarray(output_input).ravel()
    points = []
    for frequency in frequencies:
        point = {"frequency_Hz": float(frequency), "success": False}
        try:
            operator = (2j * np.pi * frequency * mass - jacobian).tocsc()
            response = splu(operator).solve(forcing.astype(complex))
            numerator = np.max(np.abs(operator @ response - forcing))
            denominator = np.max(np.abs(operator) @ np.abs(response) + np.abs(forcing))
            error = float(numerator / denominator) if denominator else float(numerator)
            values = output_state @ response + output_input
            values[0] *= -1
            finite = bool(np.all(np.isfinite(values)) and np.isfinite(error))
            point.update(
                success=finite and error <= 1e-10,
                relative_linear_residual=error if np.isfinite(error) else None,
                response_real=values.real.tolist() if finite else None,
                response_imag=values.imag.tolist() if finite else None,
            )
        except Exception:
            point["exception"] = traceback.format_exc()
        points.append(point)
    return {**metadata, "points": points, "success": all(p["success"] for p in points)}
