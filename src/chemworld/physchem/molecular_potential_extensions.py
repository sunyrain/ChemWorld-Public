"""Two-center and empirical/neural Mie residual models with explicit SI scales.

Equations were independently expressed after reading teqp 0.23.2 and its
published model references. Numerical coefficients retain upstream attribution
and licensing in the companion data module. No reference runtime is imported.
These pure homogeneous models use the shared residual Helmholtz consumers;
their declarations do not imply global stability or fitted material accuracy.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import cached_property
from itertools import product
from math import factorial, isfinite, log10, sqrt
from typing import Any

import numpy as np

from chemworld.physchem.caloric_math import (
    FirstDerivative,
    compensated_sum,
    derivative_depth,
    exponential,
    logarithm,
    nonnegative_power,
    scalar_value,
)
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.helmholtz_terms import HelmholtzTerm
from chemworld.physchem.molecular_potential_extensions_coefficients import FEANN, TWO_CENTER
from chemworld.physchem.pcsaft import AVOGADRO_PER_MOL
from chemworld.physchem.second_order_pair import SecondOrderPair


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _constant_zero(value: Any) -> bool:
    """A zero density with no active density direction stays zero at all T."""
    if isinstance(value, FirstDerivative):
        return _constant_zero(value.value) and _constant_zero(value.slope)
    if isinstance(value, SecondOrderPair):
        return all(
            _constant_zero(getattr(value, name))
            for name in ("value", "dx", "dy", "dxx", "dxy", "dyy")
        )
    return bool(value == 0)


@dataclass(frozen=True)
class _PurePotential:
    component_id: str
    diameter_m: float
    well_depth_K: float
    temperature_bounds_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in (self.component_id, self.source)):
            raise ValueError("component identity and parameter provenance are required")
        if any(
            not _finite(v) or v <= 0
            for v in (self.diameter_m, self.well_depth_K, self.gas_constant_J_mol_K)
        ):
            raise ValueError("SI diameter, energy scale and gas constant must be positive")
        bounds = tuple(self.temperature_bounds_K)
        if len(bounds) != 2 or any(not _finite(v) for v in bounds) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("explicit positive ordered temperature bounds are required")
        object.__setattr__(self, "temperature_bounds_K", bounds)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return (self.component_id,)

    def _reduced(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> tuple[Any, Any]:
        t, rho = scalar_value(temperature), scalar_value(density)
        if not _finite(t) or not self.temperature_bounds_K[0] <= t <= self.temperature_bounds_K[1]:
            raise ValueError("temperature outside the declared potential domain")
        if not _finite(rho) or rho < 0:
            raise ValueError("molar density must be finite and nonnegative")
        if (
            len(fractions) != 1
            or not _finite(scalar_value(fractions[0]))
            or abs(scalar_value(fractions[0]) - 1) > 1e-12
        ):
            raise ValueError("pure potential models require exactly one unit mole fraction")
        return temperature / self.well_depth_K, density * (AVOGADRO_PER_MOL * self.diameter_m**3)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Any:
        values = dict(value)
        if values.pop("kind", cls.__name__) != cls.__name__:
            raise ValueError("potential model kind mismatch")
        return cls(**values)


@dataclass(frozen=True)
class TwoCenterLJModel(_PurePotential):
    """Mecke/Lisal dimer EOS, with Saager dipole or quadrupole contribution.

    Elongation is L/sigma. Moment squared is the source dimensionless squared
    dipole or quadrupole (not a moment in Debye). At exactly L=0 the source
    uses the single-site LJ energy convention, Tred divided by four; its
    L->0+ continuation therefore differs. This convention is explicit here.
    Nonzero polar terms contain fractional density powers; vacuum second
    and third density derivatives need not exist.
    """

    author: str = "Lisal"
    elongation: float = 0.0
    polar: str = "none"
    reduced_moment_squared: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.author not in ("Mecke", "Lisal"):
            raise ValueError("two-center author must be Mecke or Lisal")
        if not _finite(self.elongation) or not 0 <= self.elongation <= 1:
            raise ValueError("two-center elongation must be between zero and one")
        if self.polar not in ("none", "dipole", "quadrupole"):
            raise ValueError("two-center polar term must be none, dipole or quadrupole")
        if not _finite(self.reduced_moment_squared) or self.reduced_moment_squared < 0:
            raise ValueError("reduced moment squared must be finite and nonnegative")
        if self.polar == "none" and self.reduced_moment_squared:
            raise ValueError("nonzero moment requires an explicit polar term")

    @cached_property
    def _reduction(self) -> tuple[float, float, float, float]:
        data = {k: v[self.author.upper()] for k, v in TWO_CENTER.items()}
        length = self.elongation
        a = data["p_alpha"]
        shape = a[0] + a[1] * length**2 + a[2] * length**3.5 + a[3] * length**4
        a = data["p_eta_rho"]
        packing = a[0] + a[1] * length**2 + a[2] * length**2.5 + a[3] * length**4
        a = data["p_rho"]
        density = (a[0] + a[1] * length) / (1 + a[2] * length + a[3] * length**2)
        a = data["p_t"]
        temperature = sqrt((a[0] + a[1] * length) / (1 + a[2] * length + a[3] * length**2))
        return shape, packing, density, temperature * (0.25 if length == 0 else 1)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        t, rho = self._reduced(temperature, density, fractions)
        if _constant_zero(rho):
            return 0.0 * (t + rho)
        shape, packing, red_rho, red_t = self._reduction
        tau, delta = t / red_t, rho / red_rho
        eta = rho * packing / (0.67793 + 0.32207 * tau**0.3674)
        if not 0 <= scalar_value(eta) < 1:
            raise ValueError("two-center hard-body packing must be below one")
        hard = (shape**2 - 1) * logarithm(1 - eta) + (
            (shape**2 + 3 * shape) * eta - 3 * shape * eta**2
        ) / (1 - eta) ** 2
        data = {k: v[self.author.upper()] for k, v in TWO_CENTER.items()}
        attraction = compensated_sum(
            c
            * tau**m
            * nonnegative_power(delta, n)
            * shape**o
            * exponential(p * nonnegative_power(delta, q))
            for c, m, n, o, p, q in zip(*(data[k] for k in "cmnopq"), strict=True)
        )
        moment = self.reduced_moment_squared
        pole: Any = 0.0
        if moment:
            suffix = "d" if self.polar == "dipole" else "q"
            pole = compensated_sum(
                c
                * tau ** (n / 2)
                * nonnegative_power(delta, m / 2)
                * moment ** (k / 4)
                * exponential(-o * delta**2)
                for c, n, m, k, o in zip(*(data[k + suffix] for k in "cnmko"), strict=True)
            )
        return hard + attraction + pole


@dataclass(frozen=True)
class PohlMieModel(_PurePotential):
    """Pohl et al. Mie(lambda_r,6) empirical EOS, lambda_r in [11,13]."""

    repulsive_exponent: float = 12.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not _finite(self.repulsive_exponent) or not 11 <= self.repulsive_exponent <= 13:
            raise ValueError("Pohl Mie repulsive exponent is bounded to 11 through 13")

    @cached_property
    def _terms(self) -> tuple[HelmholtzTerm, ...]:
        lr = self.repulsive_exponent
        n1 = tuple(
            a + b / lr
            for a, b in zip(
                (-0.0192944, 1.38, -2.2653, 1.6291, -1.974, 0.40412),
                (0.26021, -5.525, 8.329, -19.492, 25.8, -3.8133),
                strict=True,
            )
        )
        n2 = tuple(
            a + b / lr
            for a, b in zip(
                (0.1845, -0.3227, 1.1351, 2.232, -2.344, -0.4238),
                (-5.05, 2.7842, -9.523, -30.383, 17.902, 2.2264),
                strict=True,
            )
        )
        n3 = tuple(
            a + b / lr
            for a, b in zip(
                (-4.367, 0.0371, 1.3895, 2.835), (48.445, -5.506, -11.643, -24.36), strict=True
            )
        )
        return (
            HelmholtzTerm(
                "ResidualHelmholtzPower",
                {
                    "n": n1,
                    "d": (4, 1, 1, 2, 2, 3),
                    "t": (1, 0.236, 0.872, 0.313, 0.407, 0.703),
                    "l": (0,) * 6,
                },
                "Pohl2023",
            ),
            HelmholtzTerm(
                "ResidualHelmholtzPower",
                {
                    "n": n2,
                    "d": (1, 1, 3, 2, 2, 5),
                    "t": (1.78, 2.99, 2.866, 1.2, 3.06, 1.073),
                    "l": (1, 2, 2, 1, 2, 1),
                },
                "Pohl2023",
            ),
            HelmholtzTerm(
                "ResidualHelmholtzGaussian",
                {
                    "n": n3,
                    "d": (2, 3, 2, 2),
                    "t": (1.50, 1.03, 4.02, 1.57),
                    "eta": (0.362, 0.313, 1.17, 0.957),
                    "epsilon": (-1, -1, -0.195, -0.287),
                    "beta": (0.0761, 0.143, 0.63, 1.32),
                    "gamma": (1.55, -0.0826, 1.505, 1.07),
                },
                "Pohl2023",
            ),
        )

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        t, rho = self._reduced(temperature, density, fractions)
        lr = self.repulsive_exponent
        tc, rc = 0.668 + 6.84 / lr + 145 / lr**3, 0.2516 + 0.049 * log10(lr)
        return compensated_sum(
            term.evaluate(rho / rc, tc / t, tc * self.well_depth_K) for term in self._terms
        )


_WEIGHTS = tuple(
    np.asarray(FEANN[f"kernel_{k}"]).reshape(3 if k == 0 else 45, 45) for k in range(4)
)
_BIASES = tuple(np.asarray(FEANN[f"bias_{k}"]) for k in range(4))
_OUTPUT = np.asarray(FEANN["kernel_helmholtz"])
for _array in (*_WEIGHTS, *_BIASES, _OUTPUT):
    _array.setflags(write=False)


def _network_derivatives(alpha: float, rho: float, beta: float, order: int) -> tuple[Any, ...]:
    """Dense tanh network value and Cartesian derivatives in (rho*,1/T*).

    Only this fixed scientific network is evaluated here, not a generic ML
    runtime. Tensor chain rules avoid Python object matrices; the resulting
    local Taylor polynomial is lifted back onto ChemWorld's shared arithmetic.
    """
    value = np.asarray((alpha, rho, beta))
    gradient = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    hessian = np.zeros((3, 2, 2))
    third = np.zeros((3, 2, 2, 2))
    for weights, bias in zip(_WEIGHTS, _BIASES, strict=True):
        value = np.tanh(value @ weights + bias)
        if order >= 1:
            g = weights.T @ gradient
            first = 1 - value**2
            gradient = first[:, None] * g
        if order >= 2:
            h = np.einsum("nm,nij->mij", weights, hessian)
            second = -2 * value * first
            hessian = first[:, None, None] * h + second[:, None, None] * np.einsum(
                "ni,nj->nij", g, g
            )
        if order >= 3:
            d = np.einsum("nm,nijk->mijk", weights, third)
            third = (
                first[:, None, None, None] * d
                + second[:, None, None, None]
                * (
                    np.einsum("nij,nk->nijk", h, g)
                    + np.einsum("nik,nj->nijk", h, g)
                    + np.einsum("njk,ni->nijk", h, g)
                )
                + (first * (6 * value**2 - 2))[:, None, None, None]
                * np.einsum("ni,nj,nk->nijk", g, g, g)
            )
    result = [float(value @ _OUTPUT)]
    if order >= 1:
        result.append(np.einsum("n,ni->i", _OUTPUT, gradient))
    if order >= 2:
        result.append(np.einsum("n,nij->ij", _OUTPUT, hessian))
    if order >= 3:
        result.append(np.einsum("n,nijk->ijk", _OUTPUT, third))
    return tuple(result)


def _network(alpha: float, rho: Any, beta: Any) -> Any:
    depth = max(derivative_depth(rho), derivative_depth(beta))
    if depth > 3:
        raise ValueError("Chaparro network supports derivatives through total order three")
    r, b = scalar_value(rho), scalar_value(beta)
    tensors = _network_derivatives(alpha, r, b, depth)
    displacement = rho - r, beta - b
    terms = [tensors[0]]
    for degree in range(1, depth + 1):
        for axes in product(range(2), repeat=degree):
            term = float(tensors[degree][axes]) / factorial(degree)
            for axis in axes:
                term *= displacement[axis]
            terms.append(term)
    return compensated_sum(terms)


@dataclass(frozen=True)
class ChaparroMieModel(_PurePotential):
    """Complete published Chaparro/Mueller Mie free-energy neural network.

    Inputs are the integrated attraction alpha, molecular reduced density,
    and inverse reduced temperature. No simulation calibration is claimed
    outside the original training domain. Both exponents are explicit.
    """

    repulsive_exponent: float = 12.0
    attractive_exponent: float = 6.0

    def __post_init__(self) -> None:
        super().__post_init__()
        lr, la = self.repulsive_exponent, self.attractive_exponent
        if not _finite(lr) or not _finite(la) or not lr > la > 3:
            raise ValueError("Chaparro Mie exponents require repulsive > attractive > 3")

    @cached_property
    def attraction_integral(self) -> float:
        lr, la = self.repulsive_exponent, self.attractive_exponent
        prefactor = lr / (lr - la) * (lr / la) ** (la / (lr - la))
        return prefactor * (1 / (la - 3) - 1 / (lr - 3))

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        t, rho = self._reduced(temperature, density, fractions)
        beta = 1 / t
        return (
            _network(self.attraction_integral, rho, beta)
            - _network(self.attraction_integral, 0.0, beta)
        ) * beta
