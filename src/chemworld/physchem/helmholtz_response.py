"""Scalar second-order product rules for common residual Helmholtz terms.

The expression and coefficient source remain HelmholtzTerm. This evaluator
removes intermediate derivative objects from the state-response hot path;
uncommon terms retain the general analytic evaluator, including its critical
point limits. No finite differences, table interpolation or state cache.
"""

from math import exp, fsum

from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.second_order_pair import SecondOrderPair


def _power(x, p):
    if p == 0:
        return 1.0, 0.0, 0.0
    if p == 1:
        return x, 1.0, 0.0
    return x**p, p * x ** (p - 1), p * (p - 1) * x ** (p - 2)


def residual_response(term, delta, tau, reducing_temperature):
    """Return value, gradient and Hessian in reduced density/temperature."""
    kind = term.type
    if type(term) is not HelmholtzTerm or kind not in {
        "ResidualHelmholtzPower",
        "ResidualHelmholtzGaussian",
    }:
        return term.evaluate(
            SecondOrderPair(delta, dx=1), SecondOrderPair(tau, dy=1), reducing_temperature
        )
    c = term.coefficients
    rows = []
    for i, n in enumerate(c["n"]):
        dv, dd, ddd = _power(delta, c["d"][i])
        tv, tt, ttt = _power(tau, c["t"][i])
        # Preserve the original n * delta**d * tau**t product ordering.
        av, ax, axx = dv * n, dd * n, ddd * n
        value, dx, dy = av * tv, ax * tv, av * tt
        dxx, dxy, dyy = axx * tv, ax * tt, av * ttt
        ey = eyy = 0.0
        if kind == "ResidualHelmholtzPower":
            damping = c["l"][i]
            if damping:
                ev, ex, exx = _power(delta, damping)
                ev, ex, exx = -ev, -ex, -exx
            else:
                ev = ex = exx = 0.0
        else:
            x, y = delta - c["epsilon"][i], tau - c["gamma"][i]
            eta, beta = c["eta"][i], c["beta"][i]
            ev = -eta * x**2 - beta * y**2
            ex, exx = (2 * x) * -eta, 2 * -eta
            ey, eyy = -(2 * y * beta), -(2 * beta)
        ev = exp(ev)
        gx, gy = ev * ex, ev * ey
        gxx, gxy, gyy = ev * exx + ev * ex * ex, ev * ex * ey, ev * eyy + ev * ey * ey
        rows.append(
            (
                value * ev,
                dx * ev + value * gx,
                dy * ev + value * gy,
                dxx * ev + 2 * dx * gx + value * gxx,
                dxy * ev + dx * gy + dy * gx + value * gxy,
                dyy * ev + 2 * dy * gy + value * gyy,
            )
        )
    return SecondOrderPair(*(fsum(values) for values in zip(*rows, strict=True)))
