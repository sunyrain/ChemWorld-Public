"""Independent multiprecision Gray equations; optional source-defect reconstruction.

Only parameter declarations enter from the native object. Correlation coefficients
are read from the externally pinned source, and reference roots/gradients use
mpmath and explicit Gray equations rather than native implementations.
"""

from __future__ import annotations

import re
from itertools import product

import mpmath as mp
import numpy as np
from mie_reference_options_dense import diameter_series


def tables(source):
    text = source.read_text(encoding="utf-8")
    result = {}
    for family, order in product(("Luckas", "GubbinsTwu"), ("J", "K")):
        body = text.split(f"{family}_{order}_coeffs = {{", 1)[1].split("};", 1)[0]
        pattern = (
            r"\{\s*(\d+)\s*,\s*\{([^}]+)\}"
            if order == "J"
            else r"\{\{\s*(\d+)\s*,\s*(\d+)\s*\},\s*\{([^}]+)\}"
        )
        values = {}
        for match in re.findall(pattern, body):
            key = int(match[0]) if order == "J" else tuple(map(int, match[:2]))
            # std::map initializer preserves its first duplicate, as does the source.
            values.setdefault(
                key, tuple(mp.mpf(v.strip()) for v in match[-1].split(",") if v.strip())
            )
        result[family, order] = values
    return result


def integral(data, family, order, t, r):
    pair = isinstance(order, tuple)
    a = data[family, "K" if pair else "J"][order]
    if family == "GubbinsTwu":
        basis = (r * r * mp.log(t), r * r, r * mp.log(t), r, mp.log(t), 1)
        return (-1 if order == (334, 445) else 1) * mp.exp(mp.fdot(a, basis))
    width = 4 if pair else 3
    b = [mp.fsum(a[width * i + j] * r**i for i in range(4)) for j in range(width)]
    if pair:
        z = mp.exp((1 - r / mp.sqrt(2)) ** 4)
        return b[0] + b[1] * t + b[2] * z**2 + b[3] * z**3
    return (
        b[0] + b[1] * t ** (mp.mpf(".3") + mp.mpf(".05") * order) + b[2] * t ** (mp.mpf(1) / order)
    ) * mp.exp(1 / (t + 4 / abs(mp.log(r / mp.sqrt(2))) ** 3))


class PolarOracle:
    def __init__(self, model, temperature, source, *, source_replay=False):
        self.model = model
        self.spec = model.polar
        self.tables = tables(source)
        self.t0 = mp.mpf(temperature)
        self.source_replay = source_replay
        self.n = len(model.components)
        self.s = [mp.mpf(str(c.diameter_m)) * mp.mpf("1e10") for c in model.components]
        self.m = [mp.mpf(str(c.segment_count)) for c in model.components]
        self.e = [mp.mpf(str(c.well_depth_K)) for c in model.components]
        self.lr = [mp.mpf(str(c.repulsive_exponent)) for c in model.components]
        parameters = tuple(
            (c.component_id, c.segment_count, c.diameter_m, c.well_depth_K, c.repulsive_exponent)
            for c in model.components
        )
        self.diameters = [diameter_series(i, temperature, parameters) for i in range(self.n)]
        scale = mp.sqrt(1 / (4 * mp.pi * mp.mpf("8.8541878128e-12") * mp.mpf("1.380649e-23")))
        self.u0 = [mp.mpf(str(p.dipole_C_m)) * scale * mp.mpf("1e15") for p in self.spec.moments]
        self.q = [
            mp.mpf(str(p.quadrupole_C_m2)) * scale * mp.mpf("1e25") for p in self.spec.moments
        ]
        self.parallel = [
            mp.mpf(str(p.alpha_parallel_m3)) * mp.mpf("1e30") for p in self.spec.moments
        ]
        self.perpendicular = [
            mp.mpf(str(p.alpha_perpendicular_m3)) * mp.mpf("1e30") for p in self.spec.moments
        ]
        self.a = [(p + 2 * q) / 3 for p, q in zip(self.parallel, self.perpendicular, strict=True)]
        self.g = [p - q for p, q in zip(self.parallel, self.perpendicular, strict=True)]
        self.c3 = mp.mpf(str(self.spec.third_order_scale))
        self.c3b = mp.mpf(str(self.spec.cross_third_order_scale))
        self.sij = {
            (i, j): (self.s[i] + self.s[j]) / 2 for i, j in product(range(self.n), repeat=2)
        }
        self.eij = {
            (i, j): mp.sqrt(self.e[i] * self.e[j])
            * (1 - mp.mpf(str(model.interaction[i][j])))
            * (self.s[i] * self.s[j]) ** mp.mpf("1.5")
            / self.sij[i, j] ** 3
            for i, j in product(range(self.n), repeat=2)
        }
        nodes, weights = np.polynomial.legendre.leggauss(10)
        self.rule = [(mp.mpf(str(v)), mp.mpf(str(w))) for v, w in zip(nodes, weights, strict=True)]

    def diameter(self, i, t):
        if not self.source_replay:
            return mp.fsum(
                v * (t - self.t0) ** k / mp.factorial(k) for k, v in enumerate(self.diameters[i])
            )
        lr, la = self.lr[i], mp.mpf(6)
        k = self.e[i] * lr / (lr - la) * (lr / la) ** (la / (lr - la))
        eps = mp.mpf(str(np.finfo(float).eps))
        z = (-mp.log(eps) * t / k) ** (1 / lr)
        for _ in range(4):
            f = -k * (z**lr - z**la) / t - mp.log(eps)
            fp = -k * (lr * z**lr - la * z**la) / (t * z)
            fpp = -k * (lr * (lr - 1) * z**lr - la * (la - 1) * z**la) / (t * z * z)
            den = 2 * fp * fp - f * fpp
            if den < eps:
                break
            z -= 2 * f * fp / den
        low = 1 / z

        def integrand(node):
            radius = (1 + low) / 2 + (1 - low) * node / 2
            return -mp.expm1(-k * (radius**-lr - radius**-la) / t)

        return self.s[i] * (low + (1 - low) / 2 * mp.fsum(w * integrand(v) for v, w in self.rule))

    def integrals(self, t, rho, x):
        if self.spec.density_reduction == "packing_fraction":
            r = rho * mp.fsum(x[i] * self.m[i] * self.diameter(i, t) ** 3 for i in range(self.n))
        else:
            r = rho * mp.fsum(
                x[i] * x[j] * self.sij[i, j] ** 3 for i, j in product(range(self.n), repeat=2)
            )
        pair, triple, ks = {}, {}, {}
        orders = ((222, 333), (233, 344), (334, 445), (444, 555))
        for i, j in product(range(self.n), repeat=2):
            for o in (6, 8, 10, 11, 13, 15):
                pair[i, j, o] = (
                    4
                    * mp.pi
                    / self.sij[i, j] ** (o - 3)
                    * integral(self.tables, self.spec.integrals, o, t / self.eij[i, j], r)
                )
            for o in orders:
                ks[i, j, o] = integral(self.tables, self.spec.integrals, o, t / self.eij[i, j], r)
        for i, j, k in product(range(self.n), repeat=3):
            a, b, c = self.sij[i, j], self.sij[i, k], self.sij[j, k]
            factors = (
                64 * mp.pi**3 / 5 * mp.sqrt(14 * mp.pi / 5) / (a * b * c),
                2048 * mp.pi**3 / 7 * mp.sqrt(3 * mp.pi) / (a * (b * c) ** 2),
                -4096 * mp.pi**3 / 9 * mp.sqrt(22 * mp.pi / 7) / ((a * b) ** 2 * c**3),
                8192 * mp.pi**3 / 81 * mp.sqrt(2002 * mp.pi) / (a * b * c) ** 3,
            )
            for h, o in enumerate(orders):
                v = ks[i, j, o] * ks[i, k, o] * ks[j, k, o]
                triple[i, j, k, h] = factors[h] * mp.sign(v) * mp.root(abs(v), 3)
        return pair, triple

    def energy_grad(self, t, rho, x, u, pair, triple, *, source_gradient=False):
        n, q, g, a, c3, c3b = self.n, self.q, self.g, self.a, self.c3, self.c3b
        z = [u[i] ** 2 / (3 * t) + a[i] for i in range(n)]
        a2, a3 = mp.mpf(0), mp.mpf(0)
        for i, j in product(range(n), repeat=2):
            a2 -= (
                rho
                * x[i]
                * x[j]
                * (
                    mp.mpf("1.5") * (z[i] * z[j] - a[i] * a[j]) * pair[i, j, 6]
                    + mp.mpf("1.5") * z[i] * q[j] ** 2 / t * pair[i, j, 8]
                    + mp.mpf(".7") * q[i] ** 2 * q[j] ** 2 / t**2 * pair[i, j, 10]
                )
            )
            a3 += (
                c3
                * rho
                * x[i]
                * x[j]
                * (
                    (
                        mp.mpf(2) / 5 * u[i] ** 2 * u[j] ** 2 / t**2
                        + mp.mpf(4) / 5 * g[i] * u[j] ** 2 / t
                        + mp.mpf(4) / 25 * g[i] * g[j]
                    )
                    * q[i]
                    * q[j]
                    / t
                    * pair[i, j, 11]
                    + mp.mpf(12)
                    / 35
                    * (u[i] ** 2 / t + g[i])
                    * q[i]
                    * q[j] ** 3
                    / t**2
                    * pair[i, j, 13]
                    + mp.mpf(36) / 245 * q[i] ** 3 * q[j] ** 3 / t**3 * pair[i, j, 15]
                )
            )
            for k in range(n):
                a3 += (
                    c3
                    * rho
                    * rho
                    * x[i]
                    * x[j]
                    * x[k]
                    * (
                        (z[i] * z[j] * z[k] - a[i] * a[j] * a[k]) * triple[i, j, k, 0] / 2
                        + c3b
                        * (
                            mp.mpf(3) / 160 * z[i] * z[j] * q[k] ** 2 / t * triple[i, j, k, 1]
                            + mp.mpf(3)
                            / 640
                            * z[i]
                            * q[j] ** 2
                            * q[k] ** 2
                            / t**2
                            * triple[i, j, k, 2]
                        )
                        + mp.mpf(1)
                        / 6400
                        * q[i] ** 2
                        * q[j] ** 2
                        * q[k] ** 2
                        / t**3
                        * triple[i, j, k, 3]
                    )
                )
        gradients = []
        accum2, accum3 = mp.mpf(0), mp.mpf(0)
        for i in range(n):
            d2 = (
                -rho
                * u[i]
                / t
                * mp.fsum(
                    x[j]
                    * (
                        2 * z[i if source_gradient else j] * pair[i, j, 6]
                        + q[j] ** 2 / t * pair[i, j, 8]
                    )
                    for j in range(n)
                )
            )
            if not source_gradient:
                accum2, accum3 = mp.mpf(0), mp.mpf(0)
            for j in range(n):
                accum2 += (
                    rho
                    * x[j]
                    * (
                        mp.mpf(8) / 5 * (u[j] ** 2 / t + g[j]) * q[i] * q[j] / t * pair[i, j, 11]
                        + mp.mpf(24) / 35 * q[i] * q[j] ** 3 / t**2 * pair[i, j, 13]
                    )
                )
                for k in range(n):
                    accum3 += (
                        rho
                        * rho
                        * x[j]
                        * x[k]
                        * (
                            z[j] * z[k] * triple[i, j, k, 0]
                            + c3b / 40 * z[j] * q[k] ** 2 / t * triple[i, j, k, 1]
                            + (1 if source_gradient else c3b)
                            / 320
                            * q[j] ** 2
                            * q[k] ** 2
                            / t**2
                            * triple[i, j, k, 2]
                        )
                    )
            d3 = c3 * u[i] / t * (accum2 + accum3)
            grad = ((1 - 2 * a3 / a2) * d2 + d3) / (1 - a3 / a2) ** 2 if a2 else mp.mpf(0)
            gradients.append(grad * x[i] if source_gradient else grad)
        return (a2 * a2 / (a2 - a3) if a2 else mp.mpf(0)), gradients

    def evaluate(self, t, density, x, *, diagnostic=False):
        rho = density * mp.mpf("6.02214076e-7")
        pair, triple = self.integrals(t, rho, x)
        active = [i for i, p in enumerate(self.parallel) if p and self.u0[i]]

        def residual(*values):
            u = list(self.u0)
            for i, v in zip(active, values, strict=True):
                u[i] = v
            _, gradient = self.energy_grad(t, rho, x, u, pair, triple)
            return tuple(u[i] - self.u0[i] + t * self.parallel[i] * gradient[i] for i in active)

        u = list(self.u0)
        if self.source_replay:
            for _ in range(10):
                _, gradient = self.energy_grad(t, rho, x, u, pair, triple, source_gradient=True)
                u = [self.u0[i] - t * self.parallel[i] * gradient[i] for i in range(self.n)]
        elif active:
            root = mp.findroot(
                residual,
                tuple(u[i] for i in active),
                solver="mdnewton",
                tol=mp.eps * 1000,
                maxsteps=80,
            )
            for i, v in zip(active, root, strict=True):
                u[i] = v
        energy, gradient = self.energy_grad(
            t, rho, x, u, pair, triple, source_gradient=self.source_replay
        )
        cost = (
            mp.fsum(x[i] * t * self.parallel[i] * gradient[i] ** 2 / 2 for i in range(self.n))
            if self.source_replay
            else mp.fsum(
                x[i] * (u[i] - self.u0[i]) ** 2 / (2 * t * self.parallel[i])
                for i in range(self.n)
                if self.parallel[i]
            )
        )
        value = energy + cost
        if not diagnostic:
            return value
        _, correct = self.energy_grad(t, rho, x, u, pair, triple)
        stationarity = max(
            (
                abs(u[i] - self.u0[i] + t * self.parallel[i] * correct[i]) / max(abs(self.u0[i]), 1)
                for i in range(self.n)
            ),
            default=mp.mpf(0),
        )
        present = [i for i, p in enumerate(self.parallel) if p and x[i]]

        def total(*values):
            uu = list(u)
            for i, v in zip(present, values, strict=True):
                uu[i] = v
            perturb, _ = self.energy_grad(t, rho, x, uu, pair, triple)
            return perturb + mp.fsum(
                x[i] * (uu[i] - self.u0[i]) ** 2 / (2 * t * self.parallel[i]) for i in present
            )

        hessian = mp.matrix(len(present))
        for a, b in product(range(len(present)), repeat=2):
            orders = [0] * len(present)
            orders[a] += 1
            orders[b] += 1
            hessian[a, b] = mp.diff(total, tuple(u[i] for i in present), tuple(orders))
        minimum = mp.eigsy(hessian, eigvals_only=True)[0] if present else mp.mpf(1)
        return {
            "energy": float(value),
            "moments": list(map(float, u)),
            "normalized_stationarity": float(stationarity),
            "minimum_curvature": float(minimum),
        }
