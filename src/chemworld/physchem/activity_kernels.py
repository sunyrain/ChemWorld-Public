"""One excess-Gibbs equation per model, using the existing analytic arithmetic.

No parameter database or phase state lives here. Inputs are explicit model
parameters. The free-x extension is stated by these equations; mole derivatives
normalize amounts before differentiating the extensive Gibbs energy.
"""

from math import isfinite
from typing import Any

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    exponential,
    logarithm,
    scalar_value,
)

R = 8.31446261815324


def _positive(value: Any, name: str) -> Any:
    number = scalar_value(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def interaction(
    parameters: Any, prefix: str, left: str, right: str, temperature: Any, *, nrtl: bool = False
) -> Any:
    """Direct constants take precedence, matching the existing declaration."""
    pair = f":{left}|{right}"
    if prefix + pair in parameters:
        return parameters[prefix + pair]
    letters = "abefgh" if nrtl else "abcdef"
    a, b, c, d, e, f = (parameters.get(f"{prefix}_{v}{pair}", 0.0) for v in letters)
    result = a + b / temperature + c * logarithm(temperature) + d * temperature
    result += e / temperature**2 + f * temperature**2
    return result if nrtl else exponential(result)


def reduced_gibbs(spec: Any, xs: Any, temperature: Any, parameters: Any = None) -> Any:
    """gE/(RT), with independently variable x for composition partials."""
    p = spec.parameters if parameters is None else parameters
    ids = spec.component_ids
    n = len(ids)
    _positive(temperature, "temperature_K")
    if spec.model == "ideal":
        return 0.0
    if spec.model == "unifac":
        from chemworld.physchem.unifac import unifac_reduced_gibbs

        return unifac_reduced_gibbs(spec, xs, temperature, p)
    if spec.model in {"regular_solution", "flory_huggins", "hansen"}:
        from chemworld.physchem.regular_solution import regular_reduced_gibbs

        return regular_reduced_gibbs(spec, xs, temperature, p)
    if spec.model == "margules":
        result = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                ij, ji = f"A:{ids[i]}|{ids[j]}", f"A:{ids[j]}|{ids[i]}"
                if ij not in p and ji not in p:
                    raise ValueError(f"Margules requires an explicit A for pair {ids[i]}|{ids[j]}")
                a, b = p.get(ij, p.get(ji)), p.get(ji, p.get(ij))
                result += xs[i] * xs[j] * ((a + b) / 2 + (b - a) * (xs[i] - xs[j]) / 2)
        return result
    if spec.model == "wilson":
        matrix = [
            [
                1.0
                if i == j
                else _positive(
                    interaction(p, "lambda", ids[i], ids[j], temperature), "Wilson Lambda"
                )
                for j in range(n)
            ]
            for i in range(n)
        ]
        return -sum(
            xs[i]
            * logarithm(
                _positive(sum(xs[j] * matrix[i][j] for j in range(n)), "Wilson composition sum")
            )
            for i in range(n)
        )
    if spec.model == "nrtl":
        tau = [
            [
                0.0 if i == j else interaction(p, "tau", ids[i], ids[j], temperature, nrtl=True)
                for j in range(n)
            ]
            for i in range(n)
        ]
        g = []
        for i, left in enumerate(ids):
            row = []
            for j, right in enumerate(ids):
                key = f":{left}|{right}"
                alpha = p.get("alpha" + key)
                if alpha is None:
                    alpha = p.get("alpha_c" + key, 0.0) + p.get("alpha_d" + key, 0.0) * temperature
                row.append(
                    1.0 if i == j else exponential(-_positive(alpha, "NRTL alpha") * tau[i][j])
                )
            g.append(row)
        return sum(
            xs[i]
            * sum(xs[j] * tau[j][i] * g[j][i] for j in range(n))
            / _positive(sum(xs[j] * g[j][i] for j in range(n)), "NRTL denominator")
            for i in range(n)
        )
    if spec.model == "uniquac":
        r = [_positive(p[f"r:{key}"], "UNIQUAC r") for key in ids]
        q = [_positive(p[f"q:{key}"], "UNIQUAC q") for key in ids]
        z = _positive(p.get("z", 10.0), "UNIQUAC coordination number")
        rs, qs = (sum(x * v for x, v in zip(xs, values, strict=True)) for values in (r, q))
        _positive(rs, "UNIQUAC volume sum")
        _positive(qs, "UNIQUAC surface sum")
        theta = [xs[i] * q[i] / qs for i in range(n)]
        tau = [
            [
                1.0
                if i == j
                else _positive(interaction(p, "tau", ids[i], ids[j], temperature), "UNIQUAC tau")
                for j in range(n)
            ]
            for i in range(n)
        ]
        return sum(
            xs[i]
            * (
                logarithm(r[i] / rs)
                + z / 2 * q[i] * logarithm(q[i] * rs / (r[i] * qs))
                - q[i]
                * logarithm(
                    _positive(sum(theta[j] * tau[j][i] for j in range(n)), "UNIQUAC residual sum")
                )
            )
            for i in range(n)
        )
    raise ValueError(f"Unsupported excess Gibbs model: {spec.model}")


def log_activity(spec: Any, xs: Any, temperature: Any, parameters: Any = None) -> tuple[Any, ...]:
    """Partial molar gE/RT; derivatives commute with any outer parameter jets."""
    p = spec.parameters if parameters is None else parameters
    value = reduced_gibbs(spec, xs, temperature, p)
    gradient = []
    for i in range(len(xs)):
        result = reduced_gibbs(
            spec,
            [FirstDerivative(x, float(i == j)) for j, x in enumerate(xs)],
            FirstDerivative(temperature, 0.0),
            {key: FirstDerivative(v, 0.0) for key, v in p.items()},
        )
        gradient.append(result.slope if isinstance(result, FirstDerivative) else 0.0)
    center = sum(x * v for x, v in zip(xs, gradient, strict=True))
    return tuple(value + v - center for v in gradient)


def thermal_gibbs(
    spec: Any, xs: Any, temperature: Any, parameters: Any = None, *, order: int = 0
) -> Any:
    """T derivatives through order three with the same nested chain rules."""
    p = spec.parameters if parameters is None else parameters
    for _ in range(order):
        xs = [FirstDerivative(x, 0.0) for x in xs]
        p = {key: FirstDerivative(v, 0.0) for key, v in p.items()}
        temperature = FirstDerivative(temperature, 1.0)
    value = R * temperature * reduced_gibbs(spec, xs, temperature, p)
    for _ in range(order):
        value = value.slope if isinstance(value, FirstDerivative) else 0.0
    return value


def observable(spec: Any, xs: Any, temperature: Any, quantity: str, parameters: Any = None) -> Any:
    if quantity.startswith("ln_gamma:"):
        key = quantity.removeprefix("ln_gamma:")
        if key not in spec.component_ids:
            raise ValueError(f"unknown activity component: {key}")
        return log_activity(spec, xs, temperature, parameters)[spec.component_ids.index(key)]
    ge = thermal_gibbs(spec, xs, temperature, parameters)
    if quantity == "GE":
        return ge
    if quantity in {"HE", "SE"}:
        gt = thermal_gibbs(spec, xs, temperature, parameters, order=1)
        return ge - temperature * gt if quantity == "HE" else -gt
    if quantity == "CpE":
        return -temperature * thermal_gibbs(spec, xs, temperature, parameters, order=2)
    raise ValueError(f"Unsupported activity observable: {quantity}")
