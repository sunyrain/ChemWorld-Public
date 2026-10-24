"""Shared classical/quantum Mie reference expressions and explicit classical options.

Classical pairs carry no molecular mass or quantum constants. Their potential
and Barker--Henderson geometry feed the same contribution equations used by
quantum pairs. The original classic default path remains separately callable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from math import factorial, isfinite, pi, sqrt, tanh
from typing import Any

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential,
    exponential_minus_one,
    logarithm,
    partial_derivative,
    scalar_value,
)
from chemworld.physchem.pcsaft import (
    AVOGADRO_PER_MOL,
    hard_sphere_chain,
    hard_sphere_contact,
)
from chemworld.physchem.saftvrmie import (
    MieSegment,
    _perturbation_term,
    mie_correction_coefficients,
    mie_effective_diameter,
    mie_prefactor,
)


@dataclass(frozen=True)
class MieReferenceOptions:
    """Public classical Mie controls with explicit source conventions.

    ``additive-exact`` names the ADDITIVE / exact_binary_dhs=True preset.
    Optional matrices default to zero when bound to a model's component list.
    """

    hard_sphere_reference: str = "Lafitte"
    diameter_interaction: tuple[tuple[float, ...], ...] | None = None
    repulsion_interaction: tuple[tuple[float, ...], ...] | None = None
    exact_binary_dhs: bool | None = None
    enable_hs_extra: bool | None = None
    enable_hard_sphere: bool = True
    enable_a1: bool = True
    enable_a2: bool = True
    enable_a3: bool = True
    enable_chain: bool = True
    enable_truncation_correction: bool = False
    enable_shift_correction: bool = False
    reduced_radius_cut: float = 3.5
    enable_temperature_cache: bool = True

    def __post_init__(self) -> None:
        if self.hard_sphere_reference not in (
            "Lafitte",
            "additive",
            "additive-exact",
            "nonadditive-Santos",
        ):
            raise ValueError("unknown Mie hard-sphere reference")
        for name in ("diameter_interaction", "repulsion_interaction"):
            value = getattr(self, name)
            if value is not None:
                matrix = tuple(tuple(row) for row in value)
                n = len(matrix)
                if not n or any(len(row) != n for row in matrix):
                    raise ValueError("Mie option interactions require nonempty square matrices")
                for i, row in enumerate(matrix):
                    for j, entry in enumerate(row):
                        if (
                            isinstance(entry, bool)
                            or not isfinite(entry)
                            or entry >= 1
                            or entry != matrix[j][i]
                            or (i == j and entry != 0)
                        ):
                            raise ValueError("Mie option interactions symmetric <1, zero diagonal")
                object.__setattr__(self, name, matrix)
        flags = (
            self.enable_hard_sphere,
            self.enable_a1,
            self.enable_a2,
            self.enable_a3,
            self.enable_chain,
            self.enable_truncation_correction,
            self.enable_shift_correction,
            self.enable_temperature_cache,
        )
        if (
            any(type(value) is not bool for value in flags)
            or any(
                value is not None and type(value) is not bool
                for value in (self.exact_binary_dhs, self.enable_hs_extra)
            )
            or isinstance(self.reduced_radius_cut, bool)
            or not isfinite(self.reduced_radius_cut)
            or self.reduced_radius_cut <= 0
        ):
            raise ValueError("Mie controls require booleans and a positive cutoff")

    def bind(self, model: Any) -> ClassicalMieKernel:
        """Validate component dimensions and construct a mass-free expression input."""
        n = len(model.components)
        for matrix in (self.diameter_interaction, self.repulsion_interaction):
            if matrix is not None and len(matrix) != n:
                raise ValueError("Mie option matrix does not match component ordering")
        kernel = ClassicalMieKernel(model, self)
        for i in range(n):
            for j in range(i, n):
                kernel.pair(i, j)
        return kernel

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "mie-reference-options", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MieReferenceOptions:
        data = dict(payload)
        if data.pop("kind", None) != "mie-reference-options":
            raise ValueError("unknown Mie reference option encoding")
        return cls(**data)


@dataclass(frozen=True)
class ClassicalMiePair:
    """An explicit pure/unlike classical pair; no quantum mass is required."""

    diameter_m: float
    well_depth_K: float
    repulsive_exponent: float
    attractive_exponent: float

    def __post_init__(self) -> None:
        MieSegment(
            "pair",
            1.0,
            self.diameter_m,
            self.well_depth_K,
            "Explicit classical pair",
            self.repulsive_exponent,
            self.attractive_exponent,
        )

    def powers(self, temperature: Any) -> tuple[tuple[float, int], ...]:
        if not isfinite(scalar_value(temperature)) or scalar_value(temperature) <= 0:
            raise ValueError("Mie potential temperature must be positive")
        return ((self.repulsive_exponent, 1), (self.attractive_exponent, -1))

    def potential_K(  # noqa: N802 - explicit physical unit
        self, temperature: Any, reduced_radius: Any, radial_order: int = 0
    ) -> Any:
        if (
            not isfinite(scalar_value(reduced_radius))
            or scalar_value(reduced_radius) <= 0
            or type(radial_order) is not int
            or not 0 <= radial_order <= 2
        ):
            raise ValueError("potential radius positive; radial derivative order in 0..2")
        return (
            self.well_depth_K
            * mie_prefactor(self.repulsive_exponent, self.attractive_exponent)
            * compensated_sum(
                coefficient
                * (
                    1
                    if radial_order == 0
                    else -exponent
                    if radial_order == 1
                    else exponent * (exponent + 1)
                )
                * reduced_radius ** (-exponent - radial_order)
                for exponent, coefficient in self.powers(temperature)
            )
        )

    def geometry(self, temperature: Any, *, use_cache: bool = True) -> tuple[Any, Any, Any]:
        t, depth = scalar_value(temperature), derivative_depth(temperature)
        if not isfinite(t) or t <= 0 or depth > 3:
            raise ValueError("Mie geometry needs positive temperature and derivative order <=3")
        coefficients = (
            _classical_diameter_coefficients(self, float(t), depth)
            if use_cache
            else _classical_diameter_coefficients.__wrapped__(self, float(t), depth)
        )
        delta = temperature - t
        diameter = compensated_sum(
            value * delta**order / factorial(order) for order, value in enumerate(coefficients)
        )
        return 1.0, self.well_depth_K, diameter


@lru_cache(maxsize=4096)
def _classical_diameter_coefficients(
    pair: ClassicalMiePair, temperature: float, depth: int
) -> tuple[float, ...]:
    component = MieSegment(
        "pair",
        1.0,
        pair.diameter_m,
        pair.well_depth_K,
        "Explicit classical pair",
        pair.repulsive_exponent,
        pair.attractive_exponent,
    )
    return (
        float(mie_effective_diameter(component, temperature) / pair.diameter_m),
        *(
            float(
                partial_derivative(
                    lambda v: mie_effective_diameter(component, v["T"]) / pair.diameter_m,
                    {"T": temperature},
                    ("T",) * order,
                )
            )
            for order in range(1, depth + 1)
        ),
    )


@dataclass(frozen=True)
class ClassicalMieKernel:
    """Bind the common contribution equations to classical parameters and controls."""

    model: Any
    options: MieReferenceOptions

    def __getattr__(self, name: str) -> Any:
        if name in MieReferenceOptions.__dataclass_fields__:
            return getattr(self.options, name)
        return getattr(self.model, name)

    def pair(self, i: int, j: int) -> ClassicalMiePair:
        left, right = self.model.components[i], self.model.components[j]
        lij = self.options.diameter_interaction
        gamma = self.options.repulsion_interaction
        sigma = (left.diameter_m + right.diameter_m) / 2 * (1 - (lij[i][j] if lij else 0))
        epsilon = sqrt(left.well_depth_K * right.well_depth_K) * (1 - self.model.interaction[i][j])
        if self.model.energy_mixing == "Lafitte":
            epsilon *= (left.diameter_m * right.diameter_m) ** 1.5 / sigma**3
        repulsion = 3 + (1 - (gamma[i][j] if gamma else 0)) * sqrt(
            (left.repulsive_exponent - 3) * (right.repulsive_exponent - 3)
        )
        attraction = 3 + sqrt((left.attractive_exponent - 3) * (right.attractive_exponent - 3))
        return ClassicalMiePair(sigma, epsilon, repulsion, attraction)

    def _truncation(self, temperature: Any, density: Any) -> Any:
        if not self.options.enable_truncation_correction:
            return 0.0
        if len(self.model.components) != 1:
            raise ValueError("source Mie truncation EOS correction supports only pure fluids")
        pair = self.pair(0, 0)
        rc = self.options.reduced_radius_cut
        integral = compensated_sum(a * rc ** (3 - p) / (p - 3) for p, a in pair.powers(temperature))
        if self.options.enable_shift_correction:
            integral += compensated_sum(a * rc ** (3 - p) / 3 for p, a in pair.powers(temperature))
        return (
            -2
            * pi
            * density
            * AVOGADRO_PER_MOL
            * pair.diameter_m**3
            * pair.well_depth_K
            * mie_prefactor(pair.repulsive_exponent, pair.attractive_exponent)
            * integral
            / temperature
        )

    def potential_K(  # noqa: N802 - explicit physical unit
        self, i: int, j: int, temperature: Any, reduced_radius: Any, radial_order: int = 0
    ) -> Any:
        """Pair potential/kB; at a cutoff return the zero right-hand branch.

        Radius is r/sigma_ij; physical force/kB is minus the first derivative
        divided by sigma_ij. Mixture pair potentials support truncation even
        though the selected source's EOS tail correction is pure-fluid only.
        """
        pair = self.pair(i, j)
        value = pair.potential_K(temperature, reduced_radius, radial_order)
        if self.options.enable_truncation_correction:
            if scalar_value(reduced_radius) >= self.options.reduced_radius_cut:
                return 0.0 * value
            if self.options.enable_shift_correction and radial_order == 0:
                value -= pair.potential_K(temperature, self.options.reduced_radius_cut)
        return value

    def contributions(
        self, temperature: Any, density: Any, fractions: Sequence[Any]
    ) -> dict[str, Any]:
        return mie_reference_contributions(self, temperature, density, fractions)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        return compensated_sum(self.contributions(temperature, density, fractions).values())


def santos_hard_sphere(
    density: Any, fractions: Sequence[Any], diameters: Sequence[Sequence[Any]]
) -> Any:
    """Santos nonadditive HS from exact B2/B3 and a CS pure reference.

    Diameters and number density must use reciprocal units. The positive-part
    triangle extension is the declared source convention for strong nonadditivity.
    """
    n = len(fractions)
    moment = compensated_sum(fractions[i] * diameters[i][i] ** 3 for i in range(n))
    eta = pi * density * moment / 6
    if not 0 <= scalar_value(eta) < 1:
        raise ValueError("nonadditive hard-sphere packing outside [0,1)")
    b2 = 4 * compensated_sum(
        fractions[i] * fractions[j] * diameters[i][j] ** 3 for i in range(n) for j in range(n)
    )

    def triangle(k: int, i: int, j: int) -> Any:
        length = diameters[i][k] + diameters[j][k] - diameters[i][j]
        return length if scalar_value(length) > 0 else 0.0

    def contact(k: int, i: int, j: int) -> Any:
        length = triangle(k, i, j)
        return length**3 + 1.5 * length**2 * triangle(i, j, k) * triangle(j, i, k) / diameters[i][j]

    b3 = 4 * compensated_sum(
        fractions[i] * fractions[j] * fractions[k] * contact(k, i, j) * diameters[i][j] ** 3
        for i in range(n)
        for j in range(n)
        for k in range(n)
    )
    linear = (10 * moment * b2 - 4 * b3) / (6 * moment**2)
    pure = (b3 - moment * b2) / (6 * moment**2)
    return -linear * logarithm(1 - eta) + pure * (4 * eta - 3 * eta**2) / (1 - eta) ** 2


def mie_reference_contributions(
    kernel: Any, temperature: Any, density: Any, fractions: Sequence[Any]
) -> dict[str, Any]:
    t, rho = scalar_value(temperature), scalar_value(density)
    x = tuple(fractions)
    xs = tuple(scalar_value(v) for v in x)
    n = len(kernel.components)
    if (
        not isfinite(t)
        or not kernel.temperature_range_K[0] <= t <= kernel.temperature_range_K[1]
        or not isfinite(rho)
        or rho < 0
        or len(xs) != n
        or any(not isfinite(v) or v < 0 for v in xs)
        or abs(sum(xs) - 1) > 1e-10
    ):
        raise ValueError("Mie state outside declared T/rho/composition domain")
    pairs = {(i, j): kernel.pair(i, j) for i in range(n) for j in range(i, n)}
    geometry = {
        key: pair.geometry(temperature, use_cache=kernel.enable_temperature_cache)
        for key, pair in pairs.items()
    }
    exact_diameters = [
        [
            geometry[min(i, j), max(i, j)][2] * pairs[min(i, j), max(i, j)].diameter_m * 1e10
            for j in range(n)
        ]
        for i in range(n)
    ]
    exact = (
        kernel.hard_sphere_reference in ("additive-exact", "nonadditive-Santos")
        if kernel.exact_binary_dhs is None
        else kernel.exact_binary_dhs
    )
    extra = (
        kernel.hard_sphere_reference in ("additive", "additive-exact")
        if kernel.enable_hs_extra is None
        else kernel.enable_hs_extra
    )
    diameters = (
        exact_diameters
        if exact
        else [
            [(exact_diameters[i][i] + exact_diameters[j][j]) / 2 for j in range(n)]
            for i in range(n)
        ]
    )
    sigmas = [
        [
            geometry[min(i, j), max(i, j)][0] * pairs[min(i, j), max(i, j)].diameter_m * 1e10
            for j in range(n)
        ]
        for i in range(n)
    ]
    rhon = density * AVOGADRO_PER_MOL * 1e-30
    m = tuple(c.segment_count for c in kernel.components)
    mbar = compensated_sum(x[i] * m[i] for i in range(n))
    segments = tuple(x[i] * m[i] / mbar for i in range(n))
    rhos = rhon * mbar
    packing = (
        pi
        * rhos
        / 6
        * compensated_sum(
            segments[i] * segments[j] * diameters[i][j] ** 3 for i in range(n) for j in range(n)
        )
    )
    packing_bar = (
        pi
        * rhos
        / 6
        * compensated_sum(
            segments[i] * segments[j] * sigmas[i][j] ** 3 for i in range(n) for j in range(n)
        )
    )
    if not 0 <= scalar_value(packing) < 1:
        raise ValueError("Mie packing outside [0,1)")
    if kernel.hard_sphere_reference == "nonadditive-Santos":
        # The selected source's Santos implementation uses molecular
        # fractions/density and explicitly cancels the outer segment count.
        hard_sphere = santos_hard_sphere(rhon, x, diameters)
    else:
        # hard_sphere_chain includes its own hard-chain bond term; use only
        # the BMCSL segment reference here and the Mie bond contact below.
        hard_sphere, _ = hard_sphere_chain(
            rhos, segments, (1.0,) * n, tuple(diameters[i][i] for i in range(n))
        )
        hard_sphere *= mbar
    hard_sphere_extra: Any = 0.0
    if extra:
        eta = pi * rhos / 6 * compensated_sum(segments[i] * diameters[i][i] ** 3 for i in range(n))
        zeta2 = (
            pi * rhos / 6 * compensated_sum(segments[i] * diameters[i][i] ** 2 for i in range(n))
        )
        # Source Leonard extra term retains molecular n_i n_j, even for chains.
        hard_sphere_extra = (
            -2
            * pi
            * rhon
            * compensated_sum(
                x[i]
                * x[j]
                * ((diameters[i][i] + diameters[j][j]) / 2) ** 2
                * hard_sphere_contact(zeta2, eta, diameters[i][i], diameters[j][j])
                * ((diameters[i][i] + diameters[j][j]) / 2 - exact_diameters[i][j])
                for i in range(n)
                for j in range(n)
            )
        )
    kden = 1 + 4 * packing + 4 * packing**2 - 4 * packing**3 + packing**4
    khs = (1 - packing) ** 4 / kden
    rho_dk = -4 * (packing - 1) ** 3 * (packing**2 - 5 * packing - 2) * packing / kden**2
    first, second, third, chain = [], [], [], []
    for (i, j), pair in pairs.items():
        sigma_eff, eps_eff, _ = geometry[i, j]
        diameter = diameters[i][j] / (pair.diameter_m * 1e10)
        coefficients = pair.powers(temperature)
        c = mie_prefactor(pair.repulsive_exponent, pair.attractive_exponent)
        alpha = (
            -c
            * pair.well_depth_K
            / eps_eff
            * compensated_sum(
                a * sigma_eff ** (-exponent) / (exponent - 3) for exponent, a in coefficients
            )
        )
        fs = mie_correction_coefficients(alpha)
        weight = segments[i] * segments[j] * (1 if i == j else 2)
        scale = 2 * pi * rhos * diameters[i][j] ** 3
        powers = {exponent for exponent, _ in coefficients}
        powers.update(a + b for a, _ in coefficients for b, _ in coefficients)
        integral = {
            p: _perturbation_term(p, 1 / diameter, packing, integration_ratio=sigma_eff / diameter)
            for p in powers
        }
        first.append(
            -weight
            * scale
            * c
            * pair.well_depth_K
            * compensated_sum(a * integral[p][0] for p, a in coefficients)
            / temperature
        )
        chi = fs[0] * packing_bar + fs[1] * packing_bar**5 + fs[2] * packing_bar**8
        second.append(
            weight
            * scale
            * c
            * c
            * pair.well_depth_K**2
            * khs
            * (1 + chi)
            / 2
            * compensated_sum(
                a * b * integral[p + q][0] for p, a in coefficients for q, b in coefficients
            )
            / temperature**2
        )
        third.append(
            -weight
            * eps_eff**3
            * fs[3]
            * packing_bar
            * exponential(fs[4] * packing_bar + fs[5] * packing_bar**2)
            / temperature**3
        )
        if i == j and m[i] != 1 and kernel.enable_chain:
            # The public source uses original sigma/epsilon in g_d, explicit
            # g12/g22 and gamma, and quantum-corrected A1/A2 in g11/g21.
            ratio = 1 / diameter
            free = 1 - packing
            k0 = -logarithm(free) + (
                42 * packing - 39 * packing**2 + 9 * packing**3 - 2 * packing**4
            ) / (6 * free**3)
            k1 = (packing**4 + 6 * packing**2 - 12 * packing) / (2 * free**3)
            k2 = -3 * packing**2 / (8 * free**2)
            k3 = (-(packing**4) + 3 * packing**2 + 3 * packing) / (6 * free**3)
            log_gd = k0 + k1 * ratio + k2 * ratio**2 + k3 * ratio**3
            la, lr = pair.attractive_exponent, pair.repulsive_exponent
            # Source B_lambda is zero when the original sigma is below d.
            limit = ratio if scalar_value(ratio) > 1 else 1.0
            classical = {
                p: _perturbation_term(p, ratio, packing, integration_ratio=limit)[0]
                for p in (la, lr, 2 * la, la + lr, 2 * lr)
            }
            g1 = (
                (
                    -3 * c * compensated_sum(a * integral[p][1] for p, a in coefficients)
                    + c * (-la * classical[la] + lr * classical[lr])
                )
                if kernel.enable_a1
                else 0.0
            )
            g2 = (
                (
                    1.5
                    * c
                    * c
                    * compensated_sum(
                        a * b * (rho_dk * integral[p + q][0] + khs * integral[p + q][1])
                        for p, a in coefficients
                        for q, b in coefficients
                    )
                    + c
                    * c
                    * khs
                    * (
                        -la * classical[2 * la]
                        + (la + lr) * classical[la + lr]
                        - lr * classical[2 * lr]
                    )
                )
                if kernel.enable_a2
                else 0.0
            )
            be = pair.well_depth_K / temperature
            original_alpha = c * (1 / (la - 3) - 1 / (lr - 3))
            gamma = (
                10
                * (1 - tanh(10 * (0.57 - original_alpha)))
                * packing_bar
                * exponential_minus_one(be)
                * exponential(-6.7 * packing_bar - 8 * packing_bar**2)
            )
            log_g = log_gd + (be * g1 + be * be * (1 + gamma) * g2) * exponential(-log_gd)
            chain.append(-x[i] * (m[i] - 1) * log_g)
    return {
        "hard_sphere": hard_sphere if kernel.enable_hard_sphere else 0.0,
        "hard_sphere_extra": hard_sphere_extra,
        "dispersion1": mbar * compensated_sum(first) if kernel.enable_a1 else 0.0,
        "dispersion2": mbar * compensated_sum(second) if kernel.enable_a2 else 0.0,
        "dispersion3": mbar * compensated_sum(third) if kernel.enable_a3 else 0.0,
        "chain": compensated_sum(chain),
        "truncation": kernel._truncation(temperature, density),
    }
