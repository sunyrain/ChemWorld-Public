"""Liquid volume mixing from public equations, on the existing mixture state.

K, Pa, m³/mol and g/mol throughout. Saturation estimates are pseudo-fluid
correlations, not phase equilibrium. No upstream implementation is imported.
"""

from dataclasses import replace
from math import isclose, isfinite
from typing import Any

from chemworld.physchem.caloric_math import FirstDerivative, derivative_values, scalar_value
from chemworld.physchem.property_reports import R_J_PER_MOL_K, PropertyEvaluation
from chemworld.physchem.volume_kernels import _converted, _positive

MIXTURE_VOLUME_METHODS = {
    "Amgat": ("Vms",),
    "Rackett_mixture": ("MWs", "Tcs", "Pcs", "Zrs"),
    "COSTALD_mixture": ("Tcs", "Vcs", "omegas"),
    "COSTALD_mixture_compressed": ("Tcs", "Vcs", "omegas"),
}


def _real(value, name):
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"{name} must be a finite real number")


def _validate(method, T, xs, p):
    if method not in MIXTURE_VOLUME_METHODS:
        raise ValueError("unknown mixture volume method")
    _real(T, "T")
    _positive(T, "T")
    if not xs:
        raise ValueError("nonempty mixture required")
    for x in xs:
        _real(x, "fraction")
        if x < 0:
            raise ValueError("fractions must be nonnegative")
    if not isclose(sum(xs), 1, rel_tol=0, abs_tol=1e-9):
        raise ValueError("fractions must sum to one; no implicit normalization")
    required = set(MIXTURE_VOLUME_METHODS[method])
    optional = {"MWs"}
    compressed = method == "COSTALD_mixture_compressed"
    if compressed:
        required.add("P")
        optional.update(("Psat", "saturation_pressure_model"))
    if required - p.keys() or p.keys() - required - optional:
        raise ValueError(f"{method}: missing or unexpected parameters")
    for name, values in p.items():
        if name in {"P", "Psat"}:
            _real(values, name)
            _positive(values, name)
        elif name != "saturation_pressure_model":
            if len(values) != len(xs):
                raise ValueError("component vectors must be aligned with fractions")
            for value in values:
                _real(value, name)
                if name != "omegas":
                    _positive(value, name)
    if compressed:
        model = p.get("saturation_pressure_model")
        if ("Psat" in p and model is not None) or (
            "Psat" not in p and model != "generalized_riedel"
        ):
            raise ValueError("supply Psat or explicitly select generalized_riedel, exclusively")


def _costald_parameters(xs, p):
    v = p["Vcs"]
    vc = (sum(x * a for x, a in zip(xs, v, strict=True)) + 3 *
          sum(x * a**(2 / 3) for x, a in zip(xs, v, strict=True)) *
          sum(x * a**(1 / 3) for x, a in zip(xs, v, strict=True))) / 4
    tc = sum(x * (a * t)**0.5 for x, a, t in zip(xs, v, p["Tcs"], strict=True))**2 / vc
    omega = sum(x * w for x, w in zip(xs, p["omegas"], strict=True))
    return {"Tc": tc, "Vc": vc, "omega": omega}


def _reduced(method, xs, p):
    """Algebraic reduction into the same pure kernels, including derivatives."""
    if method == "Rackett_mixture":
        tc = sum(x * t for x, t in zip(xs, p["Tcs"], strict=True))
        mw = sum(x * m for x, m in zip(xs, p["MWs"], strict=True))
        scale = mw * sum(x * t / (m * pc) for x, t, m, pc in
                         zip(xs, p["Tcs"], p["MWs"], p["Pcs"], strict=True))
        return "Rackett", {"Tc": tc, "Pc": tc / scale,
                           "Zc": sum(x * z for x, z in zip(xs, p["Zrs"], strict=True))}
    values = _costald_parameters(xs, p)
    if method == "COSTALD_mixture":
        return "COSTALD", values
    values["Pc"] = _positive(0.291 - 0.080 * values["omega"], "pseudo-critical Z") * (
        R_J_PER_MOL_K * values["Tc"] / values["Vc"])
    values.update(P=p["P"], estimate_saturation=1)
    if "Psat" in p:
        values["Psat"] = p["Psat"]
    return "COSTALD_compressed", values


def _value(method, T, xs, p, output):
    if output not in {"molar_volume", "density", "specific_volume"}:
        raise ValueError("unknown volume output basis")
    if method == "Amgat":
        value = sum(x * v for x, v in zip(xs, p["Vms"], strict=True))
    else:
        pure, reduced = _reduced(method, xs, p)
        value = _converted(pure, T, reduced, "molar_volume")
    if output == "molar_volume":
        return value
    if "MWs" not in p:
        raise ValueError("mass/molar conversion requires explicit MWs")
    mw = sum(x * m for x, m in zip(xs, p["MWs"], strict=True)) / 1000
    return mw / value if output == "density" else value / mw


def mixture_volume_derivatives(method, T, xs, parameters, *, order=1,
                               variable="temperature", output="molar_volume"):
    """Up to third order; fixed composition and supplied reference parameters.

    Estimated saturation follows T. Supplied Psat/Vms are held fixed and
    require updates for physical temperature sweeps. No P derivative exists
    for a saturation curve; Amgat's supplied volumes have zero partial T slope.
    """
    _validate(method, T, xs, parameters)
    if variable == "temperature":
        return derivative_values(
            lambda t: _value(method, t, xs, parameters, output), T, order=order)
    if variable != "pressure" or method != "COSTALD_mixture_compressed":
        raise ValueError("pressure derivative requires an explicit compressed-liquid model")
    return derivative_values(
        lambda pressure: _value(method, T, xs, {**parameters, "P": pressure}, output),
        parameters["P"], order=order)


def mixture_volume_value(method, T, xs, *, output="molar_volume", **parameters):
    return mixture_volume_derivatives(method, T, xs, parameters, order=0, output=output)[0]


def mixture_volume_composition_jacobian(method, T, xs, parameters, *, output="molar_volume"):
    """n-1 independent mole fractions; x_last = 1 - sum(x_0..x_n-2).

    At simplex edges these are inward derivatives of the same smooth rule.
    No implicit renormalization or unconstrained/partial-molar reinterpretation.
    """
    _validate(method, T, xs, parameters)
    result = []
    for i in range(len(xs) - 1):
        dual: list[FirstDerivative[Any, float]] = [
                FirstDerivative(x, 1.0 if j == i else -1.0 if j == len(xs) - 1 else 0.0)
                for j, x in enumerate(xs)]
        value = _value(method, T, dual, parameters, output)
        result.append(scalar_value(value.slope))
    if any(not isfinite(v) for v in result):
        raise ValueError("nonfinite composition derivative")
    return tuple(result)


def _state_parameters(mixture, method, component_parameters, pressure_parameters):
    if mixture.phase_label != "liquid":
        raise ValueError("volume mixing requires a liquid MixtureSpec")
    if method not in MIXTURE_VOLUME_METHODS:
        raise ValueError("unknown mixture volume method")
    if set(component_parameters) != set(mixture.component_ids):
        raise ValueError("component parameters must match MixtureSpec identities exactly")
    names = set(MIXTURE_VOLUME_METHODS[method]) - {"MWs"}
    if any(set(row) != names for row in component_parameters.values()):
        raise ValueError("component parameter fields must match the selected method")
    p = {name: tuple(component_parameters[key][name] for key in mixture.component_ids)
         for name in names}
    p["MWs"] = mixture.molecular_weights_g_mol
    if method == "COSTALD_mixture_compressed":
        p["P"] = mixture.pressure_Pa
    p.update(pressure_parameters)
    if "P" in pressure_parameters:
        raise ValueError("pressure belongs to MixtureSpec, not a parameter override")
    _validate(method, mixture.temperature_K, mixture.mole_fractions, p)
    _real(mixture.pressure_Pa, "pressure")
    _positive(mixture.pressure_Pa, "pressure")
    mw = mixture.average_molecular_weight_g_mol
    if any(not isfinite(w) or not isclose(w, x * m / mw, rel_tol=1e-9, abs_tol=1e-12)
           for w, x, m in zip(mixture.mass_fractions, mixture.mole_fractions,
                              mixture.molecular_weights_g_mol, strict=True)):
        raise ValueError("mass and mole fractions must describe the same composition")
    return p


def mixture_volume_evaluation(mixture, *, method, component_parameters,
                              output="molar_volume", **pressure_parameters):
    """Actual T/P/composition consumption, returning the original evaluation type."""
    p = _state_parameters(mixture, method, component_parameters, pressure_parameters)
    if output not in {"molar_volume", "density"}:
        raise ValueError("state evaluation requires density or molar_volume")
    value = mixture_volume_value(method, mixture.temperature_K, mixture.mole_fractions,
                                 output=output, **p)
    return PropertyEvaluation(
        "mixture_" + output, method, method, value,
        "m^3/mol" if output == "molar_volume" else "kg/m^3",
        {"temperature": mixture.temperature_K, "pressure": mixture.pressure_Pa},
        warnings=("Empirical liquid mixing; no phase-equilibrium or excess-volume guarantee.",))


def mixture_volume_correlation(mixture, *, method, component_parameters,
                               temperature_range_K, source_note, pressure_range_Pa=None,
                               output="molar_volume", **pressure_parameters):
    """Compile a fixed composition to an existing PropertyCorrelation.

    Rebuild for a changed composition. Original inputs and component order are
    retained as metadata. Amgat consumes pointwise volumes through the ledger
    or evaluation API; it cannot invent a thermal curve from one set of volumes.
    """
    from chemworld.physchem.volume_correlations import volume_correlation

    p = _state_parameters(mixture, method, component_parameters, pressure_parameters)
    if method == "Amgat":
        raise ValueError("Amgat supplied point volumes do not define a temperature correlation")
    pure, reduced = _reduced(method, mixture.mole_fractions, p)
    reduced.pop("P", None)
    if output == "density":
        reduced["MW"] = mixture.average_molecular_weight_g_mol
    curve = volume_correlation(
        pure, parameters=reduced, temperature_range_K=temperature_range_K,
        pressure_range_Pa=pressure_range_Pa, output=output, source_note=source_note)
    return replace(curve, metadata={**curve.metadata, "mixture_method": method,
        "mixture": mixture.to_dict(), "component_parameters": component_parameters,
        "pressure_parameters": pressure_parameters,
        "composition_scope": "fixed; rebuild for changed composition"})
