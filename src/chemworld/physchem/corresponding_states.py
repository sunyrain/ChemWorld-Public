"""Lee--Kesler--Plocker and extended corresponding states on shared EOS jets.

Inputs are explicit SI parameters. LKP interpolates at common pressure, not
common density. Its local conformal root is not global phase selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from chemworld.physchem.caloric_math import (
    compensated_sum,
    derivative_depth,
    exponential,
    exponential_minus_one,
    logarithm,
    scalar_value,
)
from chemworld.physchem.cpa import CPAModel
from chemworld.physchem.cubic_alpha import cubic_alpha_value
from chemworld.physchem.cubic_mixing import (
    GibbsExcessMixingSpec,
    gibbs_excess_cubic_mixing,
    quadratic_cubic_mixing,
)
from chemworld.physchem.eos import R_J_PER_MOL_K
from chemworld.physchem.helmholtz import HelmholtzFluid
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root


def _positive(*values: float) -> bool:
    return all(not isinstance(v, bool) and isfinite(v) and v > 0 for v in values)


def _domain(domain: tuple[float, float], source: str, gas_constant: float) -> None:
    if (
        len(domain) != 2
        or not _positive(*domain, gas_constant)
        or domain[0] >= domain[1]
        or not source
    ):
        raise ValueError("corresponding states needs provenance, ordered positive T domain and R")


def _state(t: Any, rho: Any, xs: Sequence[Any], count: int, domain: tuple[float, float]) -> None:
    tv, rv = scalar_value(t), scalar_value(rho)
    values = tuple(scalar_value(x) for x in xs)
    if (
        not isfinite(tv)
        or not domain[0] <= tv <= domain[1]
        or not isfinite(rv)
        or rv < 0
        or len(values) != count
        or any(not isfinite(x) or x < 0 for x in values)
        or abs(sum(values) - 1) > 1e-10
    ):
        raise ValueError("corresponding state outside declared T, density or composition domain")


@dataclass(frozen=True)
class CorrespondingStatesComponent:
    component_id: str
    critical_temperature_K: float
    critical_molar_volume_m3_mol: float
    acentric_factor: float
    source: str

    def __post_init__(self) -> None:
        if (
            not self.component_id
            or not self.source
            or not _positive(self.critical_temperature_K, self.critical_molar_volume_m3_mol)
            or isinstance(self.acentric_factor, bool)
            or not isfinite(self.acentric_factor)
        ):
            raise ValueError("CSP components require identity, provenance and finite SI constants")


# Lee and Kesler AIChE J21 (1975), simple and reference-fluid constants.
_LK = (
    (
        0.1181193,
        0.265728,
        0.154790,
        0.030323,
        0.0236744,
        0.0186984,
        0.0,
        0.042724,
        0.0000155488,
        0.0000623689,
        0.65392,
        0.060167,
    ),
    (
        0.2026579,
        0.331511,
        0.027655,
        0.203488,
        0.0313385,
        0.0503618,
        0.016901,
        0.041577,
        0.000048736,
        0.00000740336,
        1.226,
        0.03754,
    ),
)


def lee_kesler_basis(
    temperature_reduced: Any,
    density_reduced: Any,
    reference: bool = False,
    *,
    coefficient_convention: str = "Lee-Kesler-1975",
) -> tuple[Any, Any, Any]:
    """Return alpha_r, Z and d(rho*Z)/d rho for one original LK basis fluid.

    Reduced density is rho*R*Tc/Pc (not rho/rhoc). Vacuum expression avoids
    subtracting two order-one exponentials, retaining the low-density limit.
    """
    if type(reference) is not bool:
        raise ValueError("reference selector must be boolean")
    t_value, rho_value = scalar_value(temperature_reduced), scalar_value(density_reduced)
    if not _positive(t_value) or not isfinite(rho_value) or rho_value < 0:
        raise ValueError("LK reduced temperature must be positive and density nonnegative")
    b1, b2, b3, b4, c1, c2, c3, c4, d1, d2, beta, gamma = _LK[int(reference)]
    if coefficient_convention not in ("Lee-Kesler-1975", "teqp-0.23.2"):
        raise ValueError("unknown Lee--Kesler coefficient convention")
    if coefficient_convention == "teqp-0.23.2" and not reference:
        d1 = 0.0000155428
    t, d = temperature_reduced, density_reduced
    b, c, e, dd = (
        b1 - b2 / t - b3 / t**2 - b4 / t**3,
        c1 - c2 / t + c3 / t**3,
        c4 / t**3,
        d1 + d2 / t,
    )
    q = gamma * d * d
    damp = exponential(-q)
    ar = (
        b * d
        + c * d * d / 2
        + dd * d**5 / 5
        + e / (2 * gamma) * (-(beta + 1) * exponential_minus_one(-q) - q * damp)
    )
    z = 1 + b * d + c * d * d + dd * d**5 + e * d * d * (beta + q) * damp
    stiffness = (
        1
        + 2 * b * d
        + 3 * c * d * d
        + 6 * dd * d**5
        + e * d * d * damp * (3 * beta + (5 - 2 * beta) * q - 2 * q * q)
    )
    return ar, z, stiffness


def _lk_conformal_ratios(tr: Any, dr: Any, weight: Any) -> tuple[Any, Any]:
    """Common-pressure volume roots; shared Newton solver then analytic lifting."""
    tp, dp, wp = scalar_value(tr), scalar_value(dr), scalar_value(weight)

    def equations(q: Sequence[Any] | np.ndarray, t: Any, d: Any, w: Any) -> tuple[Any, Any]:
        z0 = lee_kesler_basis(t, d / q[0])[1]
        z1 = lee_kesler_basis(t, d / q[1], True)[1]
        return z0 / q[0] - z1 / q[1], (1 - w) * q[0] + w * q[1] - 1

    def jacobian(q: Sequence[Any] | np.ndarray, t: Any, d: Any, w: Any) -> list[list[Any]]:
        s0 = lee_kesler_basis(t, d / q[0])[2]
        s1 = lee_kesler_basis(t, d / q[1], True)[2]
        return [[-s0 / q[0] ** 2, s1 / q[1] ** 2], [1 - w, w]]

    root = solve_root(
        lambda q: equations(q, tp, dp, wp),
        [1.0, 1.0],
        jacobian=lambda q: jacobian(q, tp, dp, wp),
        variables=tuple(
            ResidualVariable(
                f"LK_volume_ratio_{i}", "dimensionless", role="algebraic", lower=1e-5, upper=1e5
            )
            for i in range(2)
        ),
        residual_scales=(1.0, 1.0),
        policy=RootPolicy(tolerance=2e-13, maximum_iterations=60),
    )
    if not root.success:
        raise ValueError("LKP local conformal root failed: " + root.failure_reason)
    q0, q1 = root.value
    for _ in range(max(derivative_depth(tr), derivative_depth(dr), derivative_depth(weight))):
        f0, f1 = equations((q0, q1), tr, dr, weight)
        (a, b), (c, d) = jacobian((q0, q1), tr, dr, weight)
        det = a * d - b * c
        if abs(scalar_value(det)) < 1e-14:
            raise ValueError("singular LKP conformal volume response")
        q0, q1 = q0 - (d * f0 - b * f1) / det, q1 - (-c * f0 + a * f1) / det
    return q0, q1


@dataclass(frozen=True)
class LeeKeslerPlockerModel:
    components: tuple[CorrespondingStatesComponent, ...]
    temperature_interaction: tuple[tuple[float, ...], ...]
    temperature_range_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    interpolation: str = "common_pressure"
    coefficient_convention: str = "Lee-Kesler-1975"

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(
            self, "temperature_interaction", tuple(tuple(r) for r in self.temperature_interaction)
        )
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        _domain(self.temperature_range_K, self.source, self.gas_constant_J_mol_K)
        if self.interpolation not in ("common_pressure", "common_density"):
            raise ValueError("unknown LKP interpolation convention")
        if self.coefficient_convention not in ("Lee-Kesler-1975", "teqp-0.23.2") or (
            self.interpolation == "common_pressure"
            and self.coefficient_convention != "Lee-Kesler-1975"
        ):
            raise ValueError("unsupported LKP coefficient/interpolation combination")
        n = len(self.components)
        if not n or len(set(self.component_ids)) != n:
            raise ValueError("LKP requires distinct component identities")
        if len(self.temperature_interaction) != n or any(
            len(row) != n for row in self.temperature_interaction
        ):
            raise ValueError("LKP requires a complete temperature multiplier matrix")
        for i, row in enumerate(self.temperature_interaction):
            for j, v in enumerate(row):
                if (
                    not _positive(v)
                    or v != self.temperature_interaction[j][i]
                    or (i == j and v != 1)
                ):
                    raise ValueError(
                        "LKP multipliers must be positive symmetric with unit diagonal"
                    )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def reducing_parameters(self, fractions: Sequence[Any]) -> tuple[Any, Any, Any]:
        cs, xs = self.components, fractions
        volume = 0
        temperature_numerator = 0
        for i, a in enumerate(cs):
            for j, b in enumerate(cs):
                vij = (
                    (
                        a.critical_molar_volume_m3_mol ** (1 / 3)
                        + b.critical_molar_volume_m3_mol ** (1 / 3)
                    )
                    / 2
                ) ** 3
                tij = (a.critical_temperature_K * b.critical_temperature_K) ** 0.5
                volume += xs[i] * xs[j] * vij
                temperature_numerator += (
                    xs[i] * xs[j] * vij**0.25 * tij * self.temperature_interaction[i][j]
                )
        omega = sum(x * c.acentric_factor for x, c in zip(xs, cs, strict=True))
        temperature = temperature_numerator / volume**0.25
        zc = 0.2905 - 0.085 * omega
        if scalar_value(zc) <= 0:
            raise ValueError("LKP reducing compressibility must be positive")
        return temperature, volume / zc, omega / 0.3978

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _state(temperature, density, fractions, len(self.components), self.temperature_range_K)
        tc, density_scale, w = self.reducing_parameters(fractions)
        tr, dr = temperature / tc, density * density_scale
        if self.interpolation == "common_density":
            return (1 - w) * lee_kesler_basis(
                tr, dr, coefficient_convention=self.coefficient_convention
            )[0] + w * lee_kesler_basis(
                tr, dr, True, coefficient_convention=self.coefficient_convention
            )[0]
        q0, q1 = _lk_conformal_ratios(tr, dr, w)
        return (1 - w) * (lee_kesler_basis(tr, dr / q0)[0] - logarithm(q0)) + w * (
            lee_kesler_basis(tr, dr / q1, True)[0] - logarithm(q1)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-lkp-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LeeKeslerPlockerModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-lkp-v1":
            raise ValueError("unknown LKP encoding")
        data["components"] = tuple(CorrespondingStatesComponent(**c) for c in data["components"])
        return cls(**data)


_MBWR32_POWERS = (
    (2, 1),
    (2, 0.5),
    (2, 0),
    (2, -1),
    (2, -2),
    (3, 1),
    (3, 0),
    (3, -1),
    (3, -2),
    (4, 1),
    (4, 0),
    (4, -1),
    (5, 0),
    (6, -1),
    (6, -2),
    (7, -1),
    (8, -1),
    (8, -2),
    (9, -2),
    (3, -2),
    (3, -3),
    (5, -2),
    (5, -4),
    (7, -2),
    (7, -3),
    (9, -2),
    (9, -4),
    (11, -2),
    (11, -3),
    (13, -2),
    (13, -3),
    (13, -4),
)
_MBWR19_POWERS = (
    (2, 1),
    (2, 0),
    (2, -1),
    (2, -2),
    (2, -3),
    (3, 1),
    (3, 0),
    (3, -1),
    (4, 1),
    (4, 0),
    (5, 1),
    (5, 0),
    (6, 0),
    (3, -2),
    (3, -3),
    (3, -4),
    (5, -2),
    (5, -3),
    (5, -4),
)


def _damped_density_integral(density: Any, gamma: float, order: int) -> Any:
    """Integral from zero to d of s^(2m-1)*exp(-gamma*s^2) ds.

    An analytic power series near zero avoids cancellation of incomplete-gamma
    polynomials; both expressions are differentiated by the shared jet engine.
    """
    q = gamma * density * density
    if scalar_value(q) < 0.5:
        terms = []
        power: Any = 1.0
        for k in range(26):
            terms.append(power / (order + k))
            power *= -q / (k + 1)
        return density ** (2 * order) / 2 * compensated_sum(terms)
    value = -exponential_minus_one(-q)
    damp = exponential(-q)
    for k in range(1, order):
        value = k * value - q**k * damp
    return value / (2 * gamma**order)


@dataclass(frozen=True)
class MBWRReference:
    """Explicit MBWR19/Bender or MBWR32 reference with ThermoPack table units.

    MBWR32 coefficients use Pa,K,mol/L. Bender coefficients use kPa,K,mol/L
    and its historical negative signs on coefficients 2--5. Gamma is (L/mol)^2.
    These conventions are declared, never guessed from coefficient magnitudes.
    """

    component_id: str
    variant: int
    coefficients: tuple[float, ...]
    gamma_L2_mol2: float
    temperature_range_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        _domain(self.temperature_range_K, self.source, self.gas_constant_J_mol_K)
        if (
            type(self.variant) is not int
            or self.variant not in (19, 32)
            or len(self.coefficients) != self.variant
            or not self.component_id
            or not _positive(self.gamma_L2_mol2)
            or any(isinstance(v, bool) or not isfinite(v) for v in self.coefficients)
        ):
            raise ValueError("MBWR needs explicit 19/32 coefficients, positive gamma and identity")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return (self.component_id,)

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _state(temperature, density, fractions, 1, self.temperature_range_K)
        d = density / 1000
        powers, polynomial_count = (
            (_MBWR32_POWERS, 19) if self.variant == 32 else (_MBWR19_POWERS, 13)
        )
        terms = []
        for k, (c, (i, j)) in enumerate(zip(self.coefficients, powers, strict=True)):
            if self.variant == 19:
                c *= -1000 if 1 <= k <= 4 else 1000
            integral = (
                d ** (i - 1) / (i - 1)
                if k < polynomial_count
                else _damped_density_integral(d, self.gamma_L2_mol2, (i - 1) // 2)
            )
            terms.append(c * temperature ** (j - 1) * integral / (1000 * self.gas_constant_J_mol_K))
        return compensated_sum(terms)

    def to_dict(self) -> dict[str, Any]:
        return {"model": "chemworld-mbwr-reference-v1", **asdict(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MBWRReference:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-mbwr-reference-v1":
            raise ValueError("unknown MBWR encoding")
        return cls(**data)


def _reference_alpha(reference: HelmholtzFluid | CPAModel | MBWRReference, t: Any, rho: Any) -> Any:
    if isinstance(reference, HelmholtzFluid):
        if (
            not reference.minimum_temperature_K
            <= scalar_value(t)
            <= reference.maximum_temperature_K
        ):
            raise ValueError("conformal temperature outside reference domain")
        return reference.alpha(
            "residual",
            rho / reference.reducing_molar_density_mol_m3,
            reference.reducing_temperature_K / t,
        )
    if len(reference.component_ids) != 1:
        raise ValueError("CSP reference must be a pure fluid")
    return reference.alphar(t, rho, (1.0,))


def _reference_decode(payload: Mapping[str, Any]) -> HelmholtzFluid | CPAModel | MBWRReference:
    if payload.get("model") == "chemworld_pure_helmholtz_v1":
        return HelmholtzFluid.from_dict(payload)
    if payload.get("model") == "chemworld-mbwr-reference-v1":
        return MBWRReference.from_dict(payload)
    return CPAModel.from_dict(payload)


@dataclass(frozen=True)
class HuberElyModel:
    component: CorrespondingStatesComponent
    critical_compressibility: float
    reference_component: CorrespondingStatesComponent
    reference_critical_compressibility: float
    temperature_coefficients: tuple[float, float]
    volume_coefficients: tuple[float, float]
    reference: HelmholtzFluid | CPAModel | MBWRReference
    temperature_range_K: tuple[float, float]
    source: str
    gas_constant_J_mol_K: float = R_J_PER_MOL_K

    def __post_init__(self) -> None:
        for key in ("temperature_coefficients", "volume_coefficients", "temperature_range_K"):
            object.__setattr__(self, key, tuple(getattr(self, key)))
        _domain(self.temperature_range_K, self.source, self.gas_constant_J_mol_K)
        if not _positive(self.critical_compressibility, self.reference_critical_compressibility):
            raise ValueError("Huber--Ely critical compressibilities must be positive")
        for cs in (self.temperature_coefficients, self.volume_coefficients):
            if len(cs) != 2 or any(isinstance(c, bool) or not isfinite(c) for c in cs):
                raise ValueError(
                    "Huber--Ely requires two finite coefficients for each shape factor"
                )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return (self.component.component_id,)

    def shape_factors(self, temperature: Any) -> tuple[Any, Any]:
        c, ref = self.component, self.reference_component
        dw = c.acentric_factor - ref.acentric_factor
        logtr = logarithm(temperature / c.critical_temperature_K)
        theta = 1 + dw * (
            self.temperature_coefficients[0] + self.temperature_coefficients[1] * logtr
        )
        phi = (
            self.reference_critical_compressibility
            / self.critical_compressibility
            * (1 + dw * (self.volume_coefficients[0] + self.volume_coefficients[1] * logtr))
        )
        f = c.critical_temperature_K / ref.critical_temperature_K * theta
        h = c.critical_molar_volume_m3_mol / ref.critical_molar_volume_m3_mol * phi
        if scalar_value(f) <= 0 or scalar_value(h) <= 0:
            raise ValueError("Huber--Ely shape factor is nonpositive")
        return f, h

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _state(temperature, density, fractions, 1, self.temperature_range_K)
        f, h = self.shape_factors(temperature)
        return _reference_alpha(self.reference, temperature / f, density * h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-huber-ely-v1",
            "component": asdict(self.component),
            "reference_component": asdict(self.reference_component),
            "reference": self.reference.to_dict(),
            **{
                k: v
                for k, v in self.__dict__.items()
                if k not in ("component", "reference_component", "reference")
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HuberElyModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-huber-ely-v1":
            raise ValueError("unknown Huber--Ely encoding")
        for key in ("component", "reference_component"):
            data[key] = CorrespondingStatesComponent(**data[key])
        data["reference"] = _reference_decode(data["reference"])
        return cls(**data)


@dataclass(frozen=True)
class CubicShapeComponent:
    component_id: str
    attraction_Pa_m6_mol2: float
    covolume_m3_mol: float
    critical_temperature_K: float
    alpha_model: str
    alpha_coefficients: tuple[float, ...]
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "alpha_coefficients", tuple(self.alpha_coefficients))
        if (
            not self.component_id
            or not self.source
            or not _positive(
                self.attraction_Pa_m6_mol2, self.covolume_m3_mol, self.critical_temperature_K
            )
            or self.alpha_model not in ("soave", "twu91")
            or len(self.alpha_coefficients) != (1 if self.alpha_model == "soave" else 3)
            or any(isinstance(v, bool) or not isfinite(v) for v in self.alpha_coefficients)
        ):
            raise ValueError("cubic shape component requires explicit SI parameters and alpha")

    def attraction(self, temperature: Any) -> Any:
        return self.attraction_Pa_m6_mol2 * cubic_alpha_value(
            temperature,
            self.critical_temperature_K,
            model=self.alpha_model,
            coefficients=self.alpha_coefficients,
        )


@dataclass(frozen=True)
class ExtendedCorrespondingStatesModel:
    components: tuple[CubicShapeComponent, ...]
    interaction: tuple[tuple[float, ...], ...]
    reference_shape: CubicShapeComponent
    reference: HelmholtzFluid | CPAModel | MBWRReference
    temperature_range_K: tuple[float, float]
    source: str
    cubic_model: str = "srk"
    gas_constant_J_mol_K: float = R_J_PER_MOL_K
    ge_mixing: GibbsExcessMixingSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "interaction", tuple(tuple(r) for r in self.interaction))
        object.__setattr__(self, "temperature_range_K", tuple(self.temperature_range_K))
        _domain(self.temperature_range_K, self.source, self.gas_constant_J_mol_K)
        if (
            self.reference_shape.alpha_model == "soave"
            and self.reference_shape.alpha_coefficients[0] <= -1
        ):
            raise ValueError("ECS requires the positive Soave conformal branch with m > -1")
        n = len(self.components)
        if (
            not n
            or len(set(self.component_ids)) != n
            or self.cubic_model not in ("srk", "peng_robinson")
        ):
            raise ValueError("ECS requires distinct components and a supported cubic shape")
        if len(self.interaction) != n or any(len(r) != n for r in self.interaction):
            raise ValueError("ECS requires a complete interaction matrix")
        for i, row in enumerate(self.interaction):
            for j, v in enumerate(row):
                if (
                    isinstance(v, bool)
                    or not isfinite(v)
                    or v >= 1
                    or v != self.interaction[j][i]
                    or (i == j and v != 0)
                ):
                    raise ValueError("ECS interactions must be symmetric k<1 with zero diagonal")
        if self.ge_mixing and (
            self.ge_mixing.activity.component_ids != self.component_ids
            or any(v for row in self.interaction for v in row)
        ):
            raise ValueError("ECS GE component order must match and classical kij must be zero")

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(c.component_id for c in self.components)

    def shape_factors(self, temperature: Any, fractions: Sequence[Any]) -> tuple[Any, Any]:
        ai = tuple(c.attraction(temperature) for c in self.components)
        bi = tuple(c.covolume_m3_mol for c in self.components)
        a, b = (
            quadratic_cubic_mixing(fractions, ai, bi, self.interaction)
            if self.ge_mixing is None
            else gibbs_excess_cubic_mixing(
                self.ge_mixing,
                self.cubic_model,
                self.component_ids,
                fractions,
                temperature,
                ai,
                bi,
                self.gas_constant_J_mol_K,
            )
        )
        ref = self.reference_shape
        h = b / ref.covolume_m3_mol
        if scalar_value(a) <= 0 or scalar_value(h) <= 0:
            raise ValueError("ECS needs positive attraction and volume shape")
        target = a / (h * ref.attraction_Pa_m6_mol2)
        if ref.alpha_model == "soave":
            m = ref.alpha_coefficients[0]
            numerator = m * (temperature / ref.critical_temperature_K) ** 0.5 + target**0.5
            if scalar_value(numerator) <= 0:
                raise ValueError("ECS positive Soave conformal branch does not exist")
            f = (numerator / (1 + m)) ** 2
        else:
            f = _twu_temperature_shape(temperature, target, ref)
        if scalar_value(f) <= 0:
            raise ValueError("ECS temperature shape must be positive")
        return f, h

    def alphar(self, temperature: Any, density: Any, fractions: Sequence[Any]) -> Any:
        _state(temperature, density, fractions, len(self.components), self.temperature_range_K)
        f, h = self.shape_factors(temperature, fractions)
        return _reference_alpha(self.reference, temperature / f, density * h)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "chemworld-cubic-ecs-v1",
            "components": [asdict(c) for c in self.components],
            "reference_shape": asdict(self.reference_shape),
            "reference": self.reference.to_dict(),
            "interaction": self.interaction,
            "temperature_range_K": self.temperature_range_K,
            "source": self.source,
            "cubic_model": self.cubic_model,
            "gas_constant_J_mol_K": self.gas_constant_J_mol_K,
            "ge_mixing": self.ge_mixing.to_dict() if self.ge_mixing else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExtendedCorrespondingStatesModel:
        data = dict(payload)
        if data.pop("model", None) != "chemworld-cubic-ecs-v1":
            raise ValueError("unknown cubic ECS encoding")
        data["components"] = tuple(CubicShapeComponent(**c) for c in data["components"])
        data["reference_shape"] = CubicShapeComponent(**data["reference_shape"])
        data["reference"] = _reference_decode(data["reference"])
        if data.get("ge_mixing"):
            data["ge_mixing"] = GibbsExcessMixingSpec.from_dict(data["ge_mixing"])
        return cls(**data)


def _twu_temperature_shape(temperature: Any, target: Any, ref: CubicShapeComponent) -> Any:
    ell, m, n = ref.alpha_coefficients

    def equation(f: Any, t: Any) -> tuple[Any, Any]:
        tr = t / (f * ref.critical_temperature_K)
        alpha = cubic_alpha_value(
            t / f, ref.critical_temperature_K, model="twu91", coefficients=ref.alpha_coefficients
        )
        return f * alpha, alpha * (1 - n * (m - 1) + ell * m * n * tr ** (m * n))

    t, q = scalar_value(temperature), scalar_value(target)
    root = solve_root(
        lambda v: [equation(v[0], t)[0] - q],
        [1.0],
        jacobian=lambda v: [[equation(v[0], t)[1]]],
        variables=(
            ResidualVariable(
                "ECS_temperature_shape", "dimensionless", role="algebraic", lower=1e-5, upper=1e5
            ),
        ),
        residual_scales=(max(1.0, abs(q)),),
        policy=RootPolicy(tolerance=2e-13, maximum_iterations=60),
    )
    if not root.success:
        raise ValueError("ECS implicit Twu conformal root failed: " + root.failure_reason)
    f = root.value[0]
    for _ in range(max(derivative_depth(temperature), derivative_depth(target))):
        value, derivative = equation(f, temperature)
        f -= (value - target) / derivative
    return f
