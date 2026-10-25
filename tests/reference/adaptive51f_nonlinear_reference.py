"""Exact nonlinear flows and independent direct sampling, with no native imports."""

import math
from collections import Counter

import numpy as np
from adaptive51f_distribution_reference import ReferenceKernel


def riccati(value, stable, other, exponent):
    ratio = (value - stable) / (value - other) * math.exp(exponent)
    return (stable - ratio * other) / (1 - ratio)


class NonlinearReference(ReferenceKernel):
    def __init__(self, unit):
        super().__init__(unit)
        self.flow_types = Counter()

    def simulate(self, seed):
        self.flow_types.clear()
        row = super().simulate(seed)
        row["flow_types"] = dict(self.flow_types)
        return row

    def flow(self, values, active, dt):
        if all(active) or dt == 0:
            return values.copy()
        ode = {r["id"]: r for r, enabled in zip(self.reactions, active, strict=True) if not enabled}
        numbers = dict(zip(self.keys, map(float, values), strict=True))
        output = dict(numbers)
        if set(ode) == {"association"}:
            r = ode["association"]
            order = r["orders"]["A"]
            a, coefficient = numbers["A"], r["coefficient"]
            value = a * math.exp(
                -math.log1p(order * (order - 1) * coefficient * a ** (order - 1) * dt) / (order - 1)
            )
            output.update(A=value, B=numbers["B"] + (a - value) / order)
            self.flow_types["closed_power_" + str(order)] += 1
        elif set(ode) == {"association", "reverse"}:
            coefficient, reverse = ode["association"]["coefficient"], ode["reverse"]["coefficient"]
            a, total = numbers["A"], numbers["A"] + 2 * numbers["B"]
            delta = math.sqrt(reverse**2 + 8 * coefficient * reverse * total)
            stable = 2 * reverse * total / (reverse + delta)
            other = -(reverse + delta) / (4 * coefficient)
            value = riccati(a, stable, other, -delta * dt)
            output.update(A=value, B=numbers["B"] + (a - value) / 2)
            self.flow_types["closed_reversible_dimer"] += 1
        elif set(ode) == {"autocatalytic"}:
            a, b = numbers["A"], numbers["B"]
            total = a + b
            value = (
                total / (1 + b / a * math.exp(-ode["autocatalytic"]["coefficient"] * total * dt))
                if a
                else 0.0
            )
            output.update(A=value, B=total - value)
            self.flow_types["closed_logistic"] += 1
        elif set(ode) == {"binding", "unbinding"}:
            on, off = ode["binding"]["coefficient"], ode["unbinding"]["coefficient"]
            enzyme, substrate = numbers["E"] + numbers["ES"], numbers["S"] + numbers["ES"]
            center = on * (enzyme + substrate) + off
            delta = math.sqrt(center**2 - 4 * on**2 * enzyme * substrate)
            stable = 2 * on * enzyme * substrate / (center + delta)
            other = (center + delta) / (2 * on)
            value = riccati(numbers["ES"], stable, other, -delta * dt)
            output.update(ES=value, E=enzyme - value, S=substrate - value)
            self.flow_types["closed_enzyme_riccati"] += 1
        else:
            from scipy.integrate import solve_ivp

            def rhs(_time, vector):
                data = dict(zip(self.keys, vector, strict=True))
                rates = [
                    0.0
                    if enabled
                    else r["coefficient"] * math.prod(data[k] ** n for k, n in r["orders"].items())
                    for r, enabled in zip(self.reactions, active, strict=True)
                ]
                return self.changes @ rates

            result = solve_ivp(
                rhs,
                (0, dt),
                values,
                method="DOP853",
                rtol=1e-12,
                atol=1e-12,
                max_step=min(dt, 0.001),
                t_eval=[dt],
            )
            if not result.success:
                raise ArithmeticError("independent nonlinear fallback failed: " + result.message)
            self.flow_types["independent_dop853"] += 1
            return result.y[:, -1]
        return np.array([output[k] for k in self.keys])


def exact_endpoint(unit):
    model = NonlinearReference(unit)
    values = model.flow(
        np.array(list(unit["initial"].values())),
        [False] * len(unit["reactions"]),
        unit["duration_s"],
    )
    return dict(zip(model.keys, map(float, values), strict=True))
