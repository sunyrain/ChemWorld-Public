"""Solve actual source Pyomo equations with optional IPOPT; never native seeds."""

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
from ro47close_source_canary import build_source
from scipy.optimize import least_squares


def solve_source(
    model,
    *,
    ipopt=False,
    initialize=True,
    ipopt_options=None,
    initialize_guess=None,
    source_continuation=False,
):
    import logging

    from idaes.core.util import scaling
    from idaes.core.util.model_statistics import degrees_of_freedom
    from pyomo.core.expr.sympy_tools import Pyomo2SympyVisitor, PyomoSympyBimap
    from pyomo.environ import Constraint, SolverFactory, value
    from sympy import lambdify

    logging.getLogger("idaes.core.util.scaling").setLevel(logging.ERROR)
    scaling.calculate_scaling_factors(model)
    termination = None
    continuation_steps = []
    if source_continuation:
        unit = model.fs.unit
        controls = [
            unit.inlet.temperature[0],
            unit.inlet.pressure[0],
            unit.area,
            unit.length,
            unit.inlet.flow_mass_phase_comp[0, "Liq", "H2O"],
            unit.inlet.flow_mass_phase_comp[0, "Liq", "TDS"],
        ]
        target = [value(v) for v in controls]
        warm = [298.15, 5e6, 10.0, 5.0, 0.965, 0.035]
        for step in range(5):
            fraction = step / 4
            for variable, start_value, target_value in zip(controls, warm, target, strict=True):
                variable.fix(start_value + fraction * (target_value - start_value))
            pre = solve_source(model, ipopt=step == 0, initialize=False)
            passed = pre["all_source_equalities_passed"] and pre["all_source_bounds_passed"]
            continuation_steps.append(
                {
                    "fraction": fraction,
                    "ipopt_termination": pre["ipopt_termination"],
                    "max_scaled_residual": pre["max_scaled_residual"],
                    "passed": passed,
                }
            )
            print(
                json.dumps({"stage": "source-only-continuation", **continuation_steps[-1]}),
                flush=True,
            )
            if not passed:
                raise RuntimeError(f"source-only continuation failed: {continuation_steps}")
        pre["source_continuation_steps"] = continuation_steps
        return pre
    if ipopt:
        print(json.dumps({"stage": "original-source-initialization-ipopt"}), flush=True)
        if initialize:
            model.fs.unit.initialize(
                solver="ipopt",
                optarg={"tol": 1e-10, "max_iter": 1000},
                initialize_guess=initialize_guess,
            )
        solved = SolverFactory("ipopt").solve(
            model, options=ipopt_options or {"tol": 1e-11, "max_iter": 1000}
        )
        termination = str(solved.solver.termination_condition)
    constraints = [c for c in model.component_data_objects(Constraint, active=True) if c.equality]
    mapping = PyomoSympyBimap()
    visitor = Pyomo2SympyVisitor(mapping, keep_mutable_parameters=False)
    expressions = [visitor.walk_expression(c.body - c.lower) for c in constraints]
    fixed = {
        s: value(v) for s, v in mapping.sympy2pyomo.items() if not v.is_variable_type() or v.fixed
    }
    symbols = [s for s in mapping.sympy2pyomo if s not in fixed]
    variables = [mapping.sympy2pyomo[s] for s in symbols]
    expressions = [e.xreplace(fixed) for e in expressions]
    index = {s: i for i, s in enumerate(symbols)}
    scales = np.array([1 / scaling.get_scaling_factor(v, default=1.0) for v in variables])
    start = np.array([value(v) if v.value is not None else 0.0 for v in variables]) / scales
    lo = np.array(
        [value(v.lb) / scales[i] if v.lb is not None else -np.inf for i, v in enumerate(variables)]
    )
    hi = np.array(
        [value(v.ub) / scales[i] if v.ub is not None else np.inf for i, v in enumerate(variables)]
    )
    start = np.maximum(np.minimum(start, hi), lo)
    print(
        json.dumps(
            {
                "stage": "compile-actual-source",
                "variables": len(variables),
                "equalities": len(expressions),
            }
        ),
        flush=True,
    )
    entries = [
        (r, index[s], e.diff(s))
        for r, e in enumerate(expressions)
        for s in e.free_symbols
        if s in index
    ]
    residual_function = lambdify(symbols, expressions, modules="numpy", cse=True)
    derivative_function = lambdify(symbols, [v for _, _, v in entries], modules="numpy", cse=True)
    ri, ci = np.array([r for r, _, _ in entries]), np.array([c for _, c, _ in entries])

    def raw_jacobian(x):
        out = np.zeros((len(expressions), len(symbols)))
        out[ri, ci] = np.asarray(derivative_function(*(x * scales)), dtype=float) * scales[ci]
        return out

    first_jac = raw_jacobian(start)
    row_scales = np.maximum(
        np.sum(np.abs(first_jac) * np.maximum(np.abs(start), 1e-9)[None, :], axis=1),
        1e-12,
    )
    counter, heartbeat = 0, time.monotonic()

    def residual(x):
        nonlocal counter, heartbeat
        counter += 1
        if time.monotonic() - heartbeat > 20:
            print(
                json.dumps({"stage": "solve-actual-source", "residual_calls": counter}), flush=True
            )
            heartbeat = time.monotonic()
        return np.asarray(residual_function(*(x * scales)), dtype=float) / row_scales

    def jac(x):
        return raw_jacobian(x) / row_scales[:, None]

    result = least_squares(
        residual,
        start,
        jac=jac,
        bounds=(lo, hi),
        max_nfev=500,
        xtol=1e-14,
        ftol=1e-14,
        gtol=1e-14,
        x_scale="jac",
    )
    # A tiny trace-ion residual can satisfy a least-squares stopping criterion
    # before the declared physical criterion. Polish the original square system
    # with bound-respecting Newton steps, never with a native-model seed.
    polished = result.x.copy()
    newton_steps = 0
    for _ in range(12):
        current_residual = residual(polished)
        derivative = jac(polished)
        correction = np.linalg.solve(derivative, -current_residual)
        alpha = 1.0
        norm = np.linalg.norm(current_residual)
        while alpha > 1e-10:
            # Project the solver iterate onto the original source bounds. This
            # avoids a roundoff-sized active-bound direction throttling every
            # unrelated trace-ion equation; verify every equality afterwards.
            candidate = np.clip(polished + alpha * correction, lo, hi)
            if np.linalg.norm(residual(candidate)) < norm:
                polished = candidate
                newton_steps += 1
                break
            alpha *= 0.5
        else:
            break
    final = polished * scales
    values = np.asarray(residual_function(*final), dtype=float)
    physical_scale = np.maximum(
        np.sum(
            np.abs(raw_jacobian(polished)) * np.maximum(np.abs(polished), 1e-9)[None, :], axis=1
        ),
        1e-12,
    )
    normalized = np.abs(values) / physical_scale
    violations = np.maximum(lo - polished, 0.0) + np.maximum(polished - hi, 0.0)
    for v, number in zip(variables, final, strict=True):
        v.set_value(float(number))
    worst = np.argsort(normalized)[-8:][::-1]
    details = [
        {
            "constraint": constraints[i].name,
            "residual": values[i],
            "scale": physical_scale[i],
            "normalized": normalized[i],
        }
        for i in worst
    ]
    return {
        "solver": "ipopt-and-scipy-original-equations" if ipopt else "scipy-original-equations",
        "ipopt_termination": termination,
        "source_initialization": initialize,
        "ipopt_options": ipopt_options or {"tol": 1e-11, "max_iter": 1000},
        "initialization_guess": initialize_guess,
        "source_continuation_steps": continuation_steps,
        "source_configuration": getattr(model, "_chemworld_reference_configuration", {}),
        "solver_success": bool(result.success),
        "message": result.message,
        "residual_calls": counter,
        "nfev": result.nfev,
        "newton_polish_steps": newton_steps,
        "max_scaled_residual": max(normalized),
        "all_source_equalities_passed": bool(np.all(normalized <= 1e-8)),
        "all_source_bounds_passed": bool(np.all(violations <= 1e-12)),
        "degrees_of_freedom": degrees_of_freedom(model),
        "original_equalities": len(expressions),
        "unknowns": len(variables),
        "worst_residuals": details,
        "values": {v.name: float(x) for v, x in zip(variables, final, strict=True)},
    }


def run(output, *, ipopt=False, initialize=True):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in ("seawater", "mcas"):
        start = time.monotonic()
        row = {"mode": mode}
        try:
            model = build_source(mode)
            row.update(solve_source(model, ipopt=ipopt, initialize=initialize))
        except Exception as exc:
            row.update(error=repr(exc), traceback=traceback.format_exc())
        row["elapsed_s"] = time.monotonic() - start
        (output / f"{mode}.json").write_text(json.dumps(row, indent=2), encoding="utf8")
        rows.append({k: v for k, v in row.items() if k not in ("values", "traceback")})
        print(json.dumps(rows[-1]), flush=True)
    (output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ipopt", action="store_true")
    parser.add_argument("--no-initialize", action="store_true")
    args = parser.parse_args()
    run(args.output, ipopt=args.ipopt, initialize=not args.no_initialize)
