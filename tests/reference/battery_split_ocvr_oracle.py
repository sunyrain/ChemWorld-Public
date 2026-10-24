"""Independent authored polynomial algebra, CV closed solutions and SOH roots."""

import numpy as np
from battery_split_ocvr_cases import F
from numpy.polynomial import Polynomial


def fractions(data, charge):
    return (
        data["initial_negative_fraction"] - charge / data["negative_capacity_C"],
        data["initial_positive_fraction"] + charge / data["positive_capacity_C"],
    )


def equilibrium_polynomial(data):
    xn = Polynomial([data["initial_negative_fraction"], -1.0 / data["negative_capacity_C"]])
    yp = Polynomial([data["initial_positive_fraction"], 1.0 / data["positive_capacity_C"]])
    return Polynomial(data["positive_polynomial_V"])(yp) - Polynomial(
        data["negative_polynomial_V"]
    )(xn)


def response(data, charge, current):
    x, y = fractions(data, charge)
    un, up = (
        Polynomial(data["negative_polynomial_V"])(x),
        Polynomial(data["positive_polynomial_V"])(y),
    )
    drop = data["resistance_ohm"] * current
    return [up - un - drop, current, charge, x, y, un, up, drop]


def inventories(data, charge):
    x, y = fractions(data, charge)
    return [data["negative_capacity_C"] * x / F, data["positive_capacity_C"] * y / F]


def cc_work(data, start_charge, final_charge, current):
    integral = equilibrium_polynomial(data).integ()
    return float(
        integral(final_charge)
        - integral(start_charge)
        - data["resistance_ohm"] * current * (final_charge - start_charge)
    )


def cv_at(data, start_charge, target, elapsed):
    ep = equilibrium_polynomial(data)
    if len(ep.coef) != 2 or data["resistance_ohm"] <= 0:
        raise ValueError("fixed CV oracle requires affine OCV and positive R")
    e0, slope = ep.coef
    a, r, limit = -slope, data["resistance_ohm"], data["maximum_current_A"]
    initial = float((ep(start_charge) - target) / r)
    direction = 1.0 if initial >= 0 else -1.0
    switch_time = 0.0
    if abs(initial) > limit:
        switch_charge = (e0 - target - direction * limit * r) / a
        switch_time = (switch_charge - start_charge) / (direction * limit)
        if elapsed <= switch_time:
            q = start_charge + direction * limit * elapsed
            return (
                q,
                direction * limit,
                cc_work(data, start_charge, q, direction * limit),
                switch_time,
            )
    else:
        switch_charge = start_charge
    qstar = (e0 - target) / a
    q = qstar + (switch_charge - qstar) * np.exp(-a * (elapsed - switch_time) / r)
    work = cc_work(data, start_charge, switch_charge, direction * limit) + target * (
        q - switch_charge
    )
    return float(q), float((ep(q) - target) / r), float(work), float(switch_time)


def stop_time(data, current):
    event = data["expected"]
    if event in ("minimum-voltage", "maximum-voltage"):
        target = data[event.replace("-", "_") + "_V"]
        roots = (equilibrium_polynomial(data) - current * data["resistance_ohm"] - target).roots()
        times = [
            float(q.real / current) for q in roots if abs(q.imag) < 1e-12 and q.real / current > 0
        ]
        return min(times)
    electrode, bound, _ = event.split("-")
    q = (
        data[electrode + "_" + bound + "_fraction"] - data["initial_" + electrode + "_fraction"]
    ) * data[electrode + "_capacity_C"]
    if electrode == "negative":
        q = -q
    return q / current


def eis(data, charge, frequencies):
    x, y = fractions(data, charge)
    slope = (
        Polynomial(data["negative_polynomial_V"]).deriv()(x) / data["negative_capacity_C"]
        + Polynomial(data["positive_polynomial_V"]).deriv()(y) / data["positive_capacity_C"]
    )
    return data["resistance_ohm"] - slope / (1j * 2 * np.pi * np.asarray(frequencies))


def soh(data):
    qn, qp = data["negative_capacity_C"], data["positive_capacity_C"]
    total = qn * data["initial_negative_fraction"] + qp * data["initial_positive_fraction"]
    yofx = Polynomial([total / qp, -qn / qp])
    voltage = Polynomial(data["positive_polynomial_V"])(yofx) - Polynomial(
        data["negative_polynomial_V"]
    )
    endpoints = {}
    for suffix, bound in (("0", "minimum"), ("100", "maximum")):
        roots = []
        for root in (voltage - data[bound + "_voltage_V"]).roots():
            if abs(root.imag) > 1e-12:
                continue
            x, y = float(root.real), float(yofx(root.real))
            if (
                data["negative_minimum_fraction"] <= x <= data["negative_maximum_fraction"]
                and data["positive_minimum_fraction"] <= y <= data["positive_maximum_fraction"]
            ):
                roots.append((x, y))
        if len(roots) != 1:
            raise ValueError("authored SOH requires exactly one physical polynomial root per bound")
        endpoints.update({"x_" + suffix: roots[0][0], "y_" + suffix: roots[0][1]})
    return {
        **endpoints,
        "cell_capacity_C": qn * (endpoints["x_100"] - endpoints["x_0"]),
        "cyclable_lithium_C": total,
    }
