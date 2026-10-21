"""Analytic finite CEF responses using the shared nested first-order jets."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose

import numpy as np

from chemworld.physchem.caloric_math import FirstDerivative, derivative_values
from chemworld.physchem.residual_system import readonly
from chemworld.physchem.tdb_expressions import scalar_value


def slope(value):
    return value.slope if isinstance(value, FirstDerivative) else 0.0


def value(value):
    return value.value if isinstance(value, FirstDerivative) else value


def analytic_response(function, coordinates):
    """Pack row/column seeds; all chain arithmetic remains FirstDerivative's.

    The inner slope is a row, the outer slope a column. Their existing
    multiplication broadcasts the complete cross derivative matrix. This
    changes seed layout only, avoiding O(n**2) repeated phase evaluations.
    """
    coordinates = tuple(float(x) for x in coordinates)
    n = len(coordinates)
    identity = np.eye(n)
    seeds = tuple(FirstDerivative(FirstDerivative(x, identity[k:k+1]),
        FirstDerivative(identity[k:k+1].T, np.zeros((n, n))))
        for k, x in enumerate(coordinates))
    outputs = tuple(function(seeds))
    values = np.array([scalar_value(v) for v in outputs])
    jacobian, hessian = np.zeros((len(values), n)), np.zeros((len(values), n, n))
    for row, output in enumerate(outputs):
        jacobian[row] = np.broadcast_to(slope(value(output)), (1, n))[0]
        hessian[row] = np.broadcast_to(slope(slope(output)), (n, n))
    if not all(np.all(np.isfinite(a)) for a in (values, jacobian, hessian)):
        raise ValueError("CEF requested analytic response is nonfinite")
    return values, jacobian, hessian


def site_labels(model):
    return tuple((i, species) for i, subl in enumerate(model.constituents) for species in subl)


def unpack_sites(model, coordinates):
    result, offset = [], 0
    for subl in model.constituents:
        result.append(dict(zip(subl, coordinates[offset:offset+len(subl)], strict=True)))
        offset += len(subl)
    if offset != len(coordinates):
        raise ValueError("CEF site coordinate count does not match its phase")
    return tuple(result)


def branch_diagnostics(model, fractions, temperature, pressure, *,
                       include_sites=True, include_pressure=False):
    model.evaluate(fractions, temperature, pressure, pressure_derivative=False)
    local_models = [(model, fractions)]
    views = model._ordering_views(fractions)
    if views is not None:
        disordered, disordered_fractions, randomized = views
        local_models.append((disordered, disordered_fractions))
        if not model._phase.model_hints.get("never_disorder"):
            local_models.append((model, randomized))
    diagnostics, curves = [], []
    for phase, sites in local_models:
        curves.extend(p.source.function for p in phase._parameters)
        hints = phase._phase.model_hints
        if "magnetic_structure_factor" in hints:
            tc = phase._sum({"TC"}, sites, temperature, pressure)
            afm = hints["magnetic_afm_factor"]
            criticals = ([tc / afm if tc < 0 else tc] if afm != 0 else
                         [tc, phase._sum({"NT"}, sites, temperature, pressure)])
            if any(c > 0 and isclose(temperature, c, rel_tol=1e-12) for c in criticals):
                diagnostics.append(f"magnetic critical point: {phase.phase_name}")
            moment = phase._sum({"BMAGN"}, sites, temperature, pressure)
            if afm != 0 and moment == 0 and tc != 0:
                n_potentials = 2 if include_pressure else 1
                coordinates = ((temperature, pressure) if include_pressure else (temperature,))
                if include_sites:
                    coordinates += tuple(sites[i][s] for i, s in site_labels(phase))

                def raw_moment(z, phase=phase, sites=sites, n_potentials=n_potentials):
                    y = unpack_sites(phase, z[n_potentials:]) if include_sites else sites
                    p = z[1] if include_pressure else pressure
                    return (phase._sum({"BMAGN"}, y, z[0], p),)

                _, first, second = analytic_response(raw_moment, coordinates)
                if np.any(first != 0) or np.any(second != 0):
                    diagnostics.append(f"magnetic parameter sign boundary: {phase.phase_name}")
    names = {s for c in curves for segment in c.segments for s in segment.expression.symbols}
    visited = set()
    while names - visited:
        name = sorted(names - visited)[0]
        visited.add(name)
        if name in model.database.functions:
            curve = model.database.functions[name]
            curves.append(curve)
            names.update(s for segment in curve.segments for s in segment.expression.symbols)
    for curve in curves:
        if any(isclose(temperature, s.lower_K, rel_tol=0, abs_tol=1e-10)
               for s in curve.segments[1:]):
            diagnostics.append(f"TDB temperature segment boundary: {curve.name}")
    return tuple(dict.fromkeys(diagnostics))


@dataclass(frozen=True)
class CefTemperatureResponse:
    temperature_K: float
    gibbs_J_mol: float
    entropy_J_mol_K: float
    enthalpy_J_mol: float
    fixed_constitution_heat_capacity_J_mol_K: float
    diagnostics: tuple[str, ...] = ()


def temperature_response(model, fractions, temperature_K, pressure_Pa=101325.0, *,
                         allow_one_sided=False):
    if type(allow_one_sided) is not bool:
        raise ValueError("CEF one-sided selection must be boolean")
    fractions = model._fractions(fractions)
    temperature, pressure = model._potentials(temperature_K, pressure_Pa)
    diagnostics = branch_diagnostics(model, fractions, temperature, pressure, include_sites=False)
    if any("parameter sign boundary" in d for d in diagnostics):
        raise ValueError(
            "CEF temperature derivative is undefined at a magnetic parameter sign boundary",
        )
    if diagnostics and not allow_one_sided:
        raise ValueError("CEF second derivative needs explicit one-sided branch acceptance")
    g, first, second = derivative_values(
        lambda t: sum(model._energies(fractions, t, pressure).values()), temperature, order=2,
    )
    return CefTemperatureResponse(temperature, g, -first, g-temperature*first,
                                  -temperature*second, diagnostics)


@dataclass(frozen=True)
class CefDerivativeResponse:
    coordinate_names: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    output_names: tuple[str, ...]
    output_units: tuple[str, ...]
    values: np.ndarray
    jacobian: np.ndarray
    hessian: np.ndarray
    diagnostics: tuple[str, ...]

    def __post_init__(self):
        for name in ("values", "jacobian", "hessian"):
            object.__setattr__(self, name, readonly(getattr(self, name)))


def derivatives(model, fractions, temperature_K, pressure_Pa=101325.0, *,
                include_pressure=True, allow_one_sided=False):
    if type(include_pressure) is not bool or type(allow_one_sided) is not bool:
        raise ValueError("CEF derivative selections must be boolean")
    fractions = model._fractions(fractions)
    temperature, pressure = model._potentials(temperature_K, pressure_Pa)
    model.evaluate(fractions, temperature, pressure, pressure_derivative=include_pressure)
    if any(y <= 0 for subl in fractions for y in subl.values()):
        raise ValueError(
            "CEF site Hessian requires an interior constitution with positive fractions",
        )
    diagnostics = branch_diagnostics(model, fractions, temperature, pressure,
                                     include_pressure=include_pressure)
    if any("parameter sign boundary" in d for d in diagnostics):
        raise ValueError("CEF site Hessian is undefined at a magnetic parameter sign boundary")
    if diagnostics and not allow_one_sided:
        raise ValueError("CEF Hessian needs explicit one-sided branch acceptance")
    labels = site_labels(model)
    elements = tuple(sorted(model._composition(fractions)[1]))
    n_potentials = 2 if include_pressure else 1
    coordinates = ((temperature, pressure) if include_pressure else (temperature,)) + tuple(
        fractions[i][s] for i, s in labels
    )

    def response(z):
        t, p = z[0], z[1] if include_pressure else pressure
        y = unpack_sites(model, z[n_potentials:])
        g = sum(model._energies(y, t, p).values())
        composition = model._composition(y)[1]
        return (g, *(composition[e] for e in elements), *model._constraints(y))

    values, jacobian, hessian = analytic_response(response, coordinates)
    coordinate_names = (("temperature", "pressure") if include_pressure else ("temperature",))
    coordinate_units = (("K", "Pa") if include_pressure else ("K",))
    return CefDerivativeResponse(
        (*coordinate_names, *(f"site:{i}:{s}" for i, s in labels)),
        (*coordinate_units, *("dimensionless" for _ in labels)),
        ("G", *(f"element:{e}" for e in elements), *model.constraint_labels),
        ("J/mol", *("dimensionless" for _ in elements),
         *("dimensionless" for _ in model.constraint_labels)),
        values, jacobian, hessian, diagnostics,
    )


def site_response(model, sites, temperature, pressure):
    """Internal trial response without requiring sums/charge before a KKT solve."""
    elements = tuple(sorted(model._composition(unpack_sites(model, sites))[1]))

    def response(z):
        return site_values(model, z, temperature, pressure, elements)

    values, jacobian, hessian = analytic_response(response, sites)
    return elements, values, jacobian, hessian


def site_values(model, sites, temperature, pressure, elements):
    y = unpack_sites(model, sites)
    composition = model._composition(y)[1]
    return (sum(model._energies(y, temperature, pressure).values()),
            *(composition[e] for e in elements), *model._constraints(y))
