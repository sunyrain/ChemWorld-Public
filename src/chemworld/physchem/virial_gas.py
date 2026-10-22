"""Finite virial expansions, explicit gas roots and composition responses.

Polynomial enumeration only supplies seeds. Accepted roots are refined with
the common residual solver and checked in the physical equation. Mechanical
stability is local; it is not a phase-equilibrium or global free-energy claim.
"""

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np

from chemworld.physchem.property_reports import R_J_PER_MOL_K as R
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root
from chemworld.physchem.virial_kernels import _real


def _coefficients(values):
    return tuple(_real(v, "virial coefficient") for v in values)


def virial_pressure(temperature_K, molar_volume_m3_mol, coefficients=()):
    """Density expansion P = RT/V (1 + B/V + C/V**2 + ...)."""
    t = _real(temperature_K, "temperature", positive=True)
    v = _real(molar_volume_m3_mol, "molar volume", positive=True)
    c = _coefficients(coefficients)
    p = R * t / v * (1 + sum(a / v**i for i, a in enumerate(c, 1)))
    if not isfinite(p) or p <= 0:
        raise ValueError("density expansion gives nonpositive or nonfinite pressure")
    return p


def pressure_virial_volume(temperature_K, pressure_Pa, coefficients=()):
    """Pressure expansion V = RT/P (1 + B'P + C'P**2 + ...)."""
    t = _real(temperature_K, "temperature", positive=True)
    p = _real(pressure_Pa, "pressure", positive=True)
    c = _coefficients(coefficients)
    v = R * t / p * (1 + sum(a * p**i for i, a in enumerate(c, 1)))
    if not isfinite(v) or v <= 0:
        raise ValueError("pressure expansion gives nonpositive or nonfinite volume")
    return v


def _multiply(a, b, size):
    return np.convolve(a, b)[:size]


def _compose(coefficients, argument, size):
    answer, power = np.zeros(size), np.zeros(size)
    power[0] = 1
    for coefficient in coefficients:
        answer += coefficient * power
        power = _multiply(power, argument, size)
    return answer


def density_to_pressure_virial(temperature_K, coefficients):
    """Same-order expansion about zero density; not equality of truncations."""
    t = _real(temperature_K, "temperature", positive=True)
    c = _coefficients(coefficients)
    size = len(c) + 2
    inverse = np.zeros(size)
    inverse[1] = 1
    for degree in range(2, size):
        inverse[degree] = -_compose((0, 1, *c), inverse, size)[degree]
    z = _compose((1, *c), inverse, size)
    return tuple(float(z[i] / (R * t) ** i) for i in range(1, size - 1))


def pressure_to_density_virial(temperature_K, coefficients):
    """Formal inverse conversion retaining exactly the supplied series order."""
    t = _real(temperature_K, "temperature", positive=True)
    c = _coefficients(coefficients)
    size = len(c) + 2
    pressure = np.zeros(size)
    pressure[1] = 1
    zc = (1, *(a * (R * t) ** i for i, a in enumerate(c, 1)))
    for degree in range(2, size):
        pressure[degree] = _compose(zc, pressure, size)[degree - 1]
    return tuple(float(x) for x in pressure[2:])


@dataclass(frozen=True)
class VirialGasState:
    temperature_K: float
    pressure_Pa: float
    molar_volume_m3_mol: float
    compressibility_factor: float
    pressure_volume_slope: float
    expansion: str
    candidates: tuple

    def to_dict(self):
        return asdict(self)


class VirialRootError(ValueError):
    def __init__(self, message, candidates):
        super().__init__(message)
        self.candidates = tuple(candidates)


def _roots(polynomial, function, jacobian, stability):
    """Dimensionless candidates; the common solver owns numerical refinement."""
    polynomial = np.trim_zeros(np.asarray(polynomial, dtype=float), "b")
    if len(polynomial) < 2 or not np.all(np.isfinite(polynomial)):
        raise ValueError("degenerate or nonfinite virial root polynomial")
    candidates, accepted = [], []
    seeds = np.polynomial.polynomial.polyroots(polynomial)
    for seed in sorted(seeds, key=lambda x: (x.real, x.imag)):
        if abs(seed.imag) > 1e-8 * max(1, abs(seed.real)) or seed.real <= 0:
            candidates.append(
                {
                    "seed_real": float(seed.real),
                    "seed_imag": float(seed.imag),
                    "accepted": False,
                    "reason": "not_positive_real",
                }
            )
            continue
        root = solve_root(
            lambda x: [function(float(x[0]))],
            [float(seed.real)],
            jacobian=lambda x: [[jacobian(float(x[0]))]],
            variables=(
                ResidualVariable(
                    "scaled_gas_root", "1", role="algebraic", lower=np.finfo(float).tiny
                ),
            ),
            residual_scales=[1.0],
            policy=RootPolicy(tolerance=1e-12),
        )
        q = float(root.value[0]) if root.success else float(seed.real)
        residual = abs(function(q))
        stiffness = stability(q)
        passed = root.success and residual <= 1e-10 and stiffness > 1e-10
        record = {
            "seed_real": float(seed.real),
            "seed_imag": float(seed.imag),
            "scaled_root": q,
            "relative_residual": residual,
            "mechanical_stiffness": stiffness,
            "accepted": passed,
            "reason": "stable_gas_candidate"
            if passed
            else (root.failure_reason if not root.success else "nonpositive_or_singular_stiffness"),
            "solver": root.diagnostics,
        }
        candidates.append(record)
        if passed and not any(abs(q - r) <= 1e-10 * max(1, abs(q)) for r in accepted):
            accepted.append(q)
    return accepted, tuple(candidates)


def density_virial_state(temperature_K, pressure_Pa, coefficients=()):
    """Largest-volume mechanically stable root; all candidates are reported."""
    t = _real(temperature_K, "temperature", positive=True)
    p = _real(pressure_Pa, "pressure", positive=True)
    c = _coefficients(coefficients)
    scale = R * t / p
    a = tuple(value / scale**i for i, value in enumerate(c, 1))
    n = len(c)
    polynomial = [0.0] * (n + 2)
    polynomial[n + 1], polynomial[n] = 1.0, -1.0
    for i, value in enumerate(a, 1):
        polynomial[n - i] -= value

    def f(q):
        return 1 / q + sum(value / q ** (i + 1) for i, value in enumerate(a, 1)) - 1

    def jac(q):
        return -1 / q**2 - sum((i + 1) * value / q ** (i + 2) for i, value in enumerate(a, 1))

    roots, candidates = _roots(polynomial, f, jac, lambda q: -jac(q) * q * q)
    if not roots:
        raise VirialRootError("no nonsingular mechanically stable gas root", candidates)
    q = max(roots)
    return VirialGasState(t, p, q * scale, q, p / scale * jac(q), "density", candidates)


def pressure_virial_state(temperature_K, molar_volume_m3_mol, coefficients=()):
    """Smallest-pressure stable root of the pressure-form expansion."""
    t = _real(temperature_K, "temperature", positive=True)
    v = _real(molar_volume_m3_mol, "molar volume", positive=True)
    c = _coefficients(coefficients)
    scale = R * t / v
    a = tuple(value * scale**i for i, value in enumerate(c, 1))
    polynomial = [-1.0, 1.0, *([0.0] * max(0, len(c) - 1))]
    for i, value in enumerate(a, 1):
        polynomial[i] -= value

    def f(q):
        return q - 1 - sum(value * q**i for i, value in enumerate(a, 1))

    def jac(q):
        return 1 - sum(i * value * q ** (i - 1) for i, value in enumerate(a, 1))

    def stiffness(q):
        return 1 - sum((i - 1) * value * q**i for i, value in enumerate(a, 1))

    roots, candidates = _roots(polynomial, f, jac, stiffness)
    if not roots:
        raise VirialRootError("no nonsingular mechanically stable pressure root", candidates)
    q = min(roots)
    pressure = q * scale
    slope = -(pressure**2) / (R * t * stiffness(q))
    return VirialGasState(t, pressure, v, q, slope, "pressure", candidates)


def second_virial_from_compressibility(temperature_K, pressure_Pa, compressibility_factor):
    """Effective pressure-linear B = (Z-1)RT/P; no hidden higher coefficients."""
    t = _real(temperature_K, "temperature", positive=True)
    p = _real(pressure_Pa, "pressure", positive=True)
    z = _real(compressibility_factor, "compressibility factor", positive=True)
    return (z - 1) * R * t / p


def compressibility_from_second_virial(temperature_K, pressure_Pa, coefficient):
    t = _real(temperature_K, "temperature", positive=True)
    p = _real(pressure_Pa, "pressure", positive=True)
    b = _real(coefficient, "second virial coefficient")
    z = 1 + b * p / (R * t)
    if z <= 0:
        raise ValueError("linear virial expression gives nonpositive compressibility")
    return z


def virial_volume_composition_response(
    temperature_K, pressure_Pa, second, third, *, expansion="density"
):
    """Fixed-T/P first and second volume derivatives in the supplied coordinates.

    Pressure form uses B'=B/RT and C'=(C-B**2)/(RT)**2; its finite truncation
    intentionally differs from the density cubic at nonzero pressure.
    """
    t = _real(temperature_K, "temperature", positive=True)
    p = _real(pressure_Pa, "pressure", positive=True)
    if second.coordinates != third.coordinates or len(second.gradient) != len(third.gradient):
        raise ValueError("B/C responses must use aligned composition coordinates")
    b, c = second.value, third.value
    bg, cg, bh, ch = second.gradient, third.gradient, second.hessian, third.hessian
    n = len(bg)
    if expansion == "density":
        state = density_virial_state(t, p, (b, c))
        v = state.molar_volume_m3_mol
        fv = 3 * p * v * v / (R * t) - 2 * v - b
        gradient = tuple((v * bg[i] + cg[i]) / fv for i in range(n))
        hessian = tuple(
            tuple(
                (
                    v * bh[i][j]
                    + ch[i][j]
                    + bg[i] * gradient[j]
                    + bg[j] * gradient[i]
                    - (6 * p * v / (R * t) - 2) * gradient[i] * gradient[j]
                )
                / fv
                for j in range(n)
            )
            for i in range(n)
        )
    elif expansion == "pressure":
        coefficients = density_to_pressure_virial(t, (b, c))
        v = pressure_virial_volume(t, p, coefficients)
        stiffness = 1 - coefficients[1] * p * p
        if stiffness <= 1e-10:
            raise VirialRootError("singular or unstable pressure-form volume", ())
        state = VirialGasState(
            t, p, v, p * v / (R * t), -p * p / (R * t * stiffness), "pressure", ()
        )
        gradient = tuple(bg[i] + p / (R * t) * (cg[i] - 2 * b * bg[i]) for i in range(n))
        hessian = tuple(
            tuple(
                bh[i][j] + p / (R * t) * (ch[i][j] - 2 * bg[i] * bg[j] - 2 * b * bh[i][j])
                for j in range(n)
            )
            for i in range(n)
        )
    else:
        raise ValueError("expansion must be density or pressure")
    return {
        "state": state.to_dict(),
        "gradient_m3_mol": gradient,
        "hessian_m3_mol": hessian,
        "coordinates": second.coordinates,
    }
