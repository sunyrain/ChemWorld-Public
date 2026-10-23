"""Excess-Gibbs cubic mixing rules on the original EOS and analytic arithmetic.

Mixing changes a(T,x) and sometimes b(T,x). Fugacity and departures therefore
differentiate the same residual Helmholtz potential instead of reusing the
classical-quadratic composition or temperature formulas for another rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite, log, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from chemworld.physchem.activity_kernels import reduced_gibbs
from chemworld.physchem.caloric_math import (
    derivative_depth,
    logarithm,
    partial_derivative,
    scalar_value,
)
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.residual_system import ResidualVariable, RootPolicy, solve_root

if TYPE_CHECKING:
    from chemworld.physchem.eos import CubicEOSSpec

R = 8.31446261815324
MixingRule = Literal["huron_vidal", "mhv", "mhv1", "mhv2", "wong_sandler"]


@dataclass(frozen=True)
class GibbsExcessMixingSpec:
    rule: MixingRule
    activity: ActivityModelSpec
    coefficients: tuple[float, ...] = ()
    binary_interaction: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rule not in {"huron_vidal", "mhv", "mhv1", "mhv2", "wong_sandler"}:
            raise ValueError("unsupported excess-Gibbs cubic mixing rule")
        expected = 2 if self.rule == "mhv2" else 1 if self.rule == "mhv1" else 0
        if len(self.coefficients) != expected or any(
            isinstance(v, bool) or not isfinite(v) or v >= 0 for v in self.coefficients
        ):
            raise ValueError("MHV1/2 require explicitly selected negative q coefficients")
        object.__setattr__(self, "coefficients", tuple(self.coefficients))
        ids = self.activity.component_ids
        pairs = {"|".join(sorted((a, b))) for i, a in enumerate(ids) for b in ids[i+1:]}
        if self.rule == "wong_sandler":
            if set(self.binary_interaction) != pairs:
                raise ValueError("Wong-Sandler requires every explicit canonical binary pair")
        elif self.binary_interaction:
            raise ValueError("binary interaction belongs only to Wong-Sandler in GE mixing")
        if any(isinstance(v, bool) or not isfinite(v) for v in self.binary_interaction.values()):
            raise ValueError("Wong-Sandler binary interactions must be finite")
        object.__setattr__(self, "binary_interaction",
                           MappingProxyType(dict(self.binary_interaction)))

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "activity": self.activity.to_dict(),
                "coefficients": list(self.coefficients),
                "binary_interaction": dict(self.binary_interaction)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GibbsExcessMixingSpec:
        activity = dict(payload["activity"])
        activity["component_ids"] = tuple(activity["component_ids"])
        return cls(payload["rule"], ActivityModelSpec(**activity),
                   tuple(payload.get("coefficients", ())), payload.get("binary_interaction", {}))


def _shape(model: str) -> tuple[float, float]:
    return (1+sqrt(2), 1-sqrt(2)) if model == "peng_robinson" else (1., 0.)


def _zero_pressure_gibbs(alpha: Any, s1: float, s2: float) -> Any:
    discriminant = (alpha-s1-s2)**2-4*(alpha+s1*s2)
    if scalar_value(discriminant) <= 0:
        raise ValueError("exact MHV requires a differentiable zero-pressure liquid branch")
    # Rationalized small root avoids subtracting nearly equal large terms.
    volume = 2*(alpha+s1*s2)/(alpha-s1-s2+discriminant**0.5)
    return -1-logarithm(volume-1)-alpha/(s1-s2)*logarithm((volume+s1)/(volume+s2))


def _mhv_alpha(target: Any, initial: float, s1: float, s2: float) -> Any:
    lower = s1+s2+2+2*sqrt((1+s1)*(1+s2))
    lower += 1e-9*lower
    primal = float(scalar_value(target))
    if primal >= _zero_pressure_gibbs(lower, s1, s2):
        raise ValueError("exact MHV target has no differentiable liquid-branch solution")
    upper = max(2*initial, lower+2, 10*abs(primal)+lower)
    def objective(alpha: Any) -> Any:
        return _zero_pressure_gibbs(alpha, s1, s2)
    result = solve_root(
        lambda values: [objective(values[0])-primal], [max(lower, initial)],
        jacobian=lambda values: [[partial_derivative(
            lambda v: objective(v["a"]), {"a": float(values[0])}, ("a",))]],
        variables=(ResidualVariable("MHV_alpha", "dimensionless", 10., "algebraic", lower, upper),),
        residual_scales=[1.], policy=RootPolicy(tolerance=2e-12, maximum_iterations=60),
    )
    if not result.success or result.value is None:
        raise ValueError(f"exact MHV shared root solve failed: {result.failure_reason}")
    alpha: Any = float(result.value[0])
    # Newton lifting at the converged primal propagates the implicit analytic
    # derivatives, with no finite differences or extra root iteration policy.
    for _ in range(derivative_depth(target)):
        discriminant = (alpha-s1-s2)**2-4*(alpha+s1*s2)
        volume = 2*(alpha+s1*s2)/(alpha-s1-s2+discriminant**0.5)
        derivative = -logarithm((volume+s1)/(volume+s2))/(s1-s2)
        alpha -= (objective(alpha)-target)/derivative
    return alpha


def cubic_mixing_expression(
    spec: CubicEOSSpec, xs: Sequence[Any], temperature: Any,
) -> tuple[Any, Any]:
    """Original pure parameters plus declared mixing, preserving analytic jets."""
    from chemworld.physchem.cubic_alpha import cubic_alpha_value
    from chemworld.physchem.eos import _binary_interaction, cubic_pure_parameters
    from chemworld.physchem.ppr78 import ppr78_value

    pure = [cubic_pure_parameters(c, model=spec.model,
                                 temperature_K=float(scalar_value(temperature)),
                                 parameterization=spec.pure_parameterization)
            for c in spec.components]
    ai = [p.a_c*cubic_alpha_value(
        temperature, c.critical_temperature_K, model=c.alpha_model,
        coefficients=c.alpha_coefficients if c.alpha_model == "twu91" else (p.kappa,))
        for c, p in zip(spec.components, pure, strict=True)]
    bi = [p.b for p in pure]
    b_mix = sum(x*b for x, b in zip(xs, bi, strict=True))
    if spec.ge_mixing is None:
        a_mix = 0.0
        for i, left in enumerate(spec.component_ids):
            for j, right in enumerate(spec.component_ids):
                kij = _binary_interaction(spec, left, right)
                if i == j:
                    kij = 0.
                elif spec.predictive_interaction is not None:
                    kij = ppr78_value(spec.predictive_interaction, left, right, temperature,
                                      ai[i], ai[j], bi[i], bi[j])
                a_mix += xs[i]*xs[j]*(ai[i]*ai[j])**0.5*(1-kij)
        return a_mix, b_mix
    mixing = spec.ge_mixing
    excess = reduced_gibbs(mixing.activity, xs, temperature)
    alphas = [a/(b*R*temperature) for a, b in zip(ai, bi, strict=True)]
    mean = sum(x*alpha for x, alpha in zip(xs, alphas, strict=True))
    s1, s2 = _shape(spec.model)
    shape_c = -log((1+s1)/(1+s2))/(s1-s2)
    volume_term = sum(x*logarithm(b_mix/b) for x, b in zip(xs, bi, strict=True))
    if mixing.rule == "huron_vidal":
        alpha = mean+excess/shape_c
    elif mixing.rule == "mhv1":
        alpha = mean+(excess+volume_term)/mixing.coefficients[0]
    elif mixing.rule == "mhv2":
        q1, q2 = mixing.coefficients
        target = excess+volume_term+q1*mean+q2*sum(
            x*value**2 for x, value in zip(xs, alphas, strict=True))
        discriminant = q1*q1+4*q2*target
        if scalar_value(discriminant) <= 0:
            raise ValueError("MHV2 mixing has no differentiable positive branch")
        alpha = 2*target/(q1-discriminant**0.5)
    elif mixing.rule == "mhv":
        target = excess+volume_term+sum(
            x*_zero_pressure_gibbs(value, s1, s2)
            for x, value in zip(xs, alphas, strict=True))
        alpha = _mhv_alpha(target, float(scalar_value(mean)), s1, s2)
    else:
        alpha = mean+excess/shape_c
        q = 0.0
        for i, left in enumerate(spec.component_ids):
            for j, right in enumerate(spec.component_ids):
                kij = 0. if i == j else mixing.binary_interaction["|".join(sorted((left, right)))]
                q += xs[i]*xs[j]*((bi[i]-ai[i]/(R*temperature))
                                  +(bi[j]-ai[j]/(R*temperature)))/2*(1-kij)
        if abs(scalar_value(1-alpha)) < 1e-12:
            raise ValueError("Wong-Sandler mixture covolume denominator is singular")
        b_mix = q/(1-alpha)
    a_mix = alpha*b_mix*R*temperature
    if scalar_value(a_mix) < 0 or scalar_value(b_mix) <= 0:
        raise ValueError("GE mixing produced invalid attraction/covolume")
    return a_mix, b_mix


def cubic_residual_helmholtz(
    spec: CubicEOSSpec, xs: Sequence[Any], temperature: Any, molar_volume: Any,
) -> Any:
    """Molar residual A in J/mol relative to ideal gas at the same T,V,x."""
    a, b = cubic_mixing_expression(spec, xs, temperature)
    s1, s2 = _shape(spec.model)
    if scalar_value(molar_volume-b) <= 0:
        raise ValueError("cubic Helmholtz volume must exceed mixture covolume")
    return (-R*temperature*logarithm(1-b/molar_volume)
            -a/(b*(s1-s2))*logarithm((molar_volume+s1*b)/(molar_volume+s2*b)))


def cubic_helmholtz_fugacities(
    spec: CubicEOSSpec, xs: Sequence[float], temperature: float, volume: float, z: float,
) -> tuple[float, ...]:
    """ln(phi_i) from extensive residual A at fixed total volume and temperature."""
    ids = spec.component_ids
    amounts = dict(zip(ids, xs, strict=True))
    def extensive(values: Mapping[str, Any]) -> Any:
        total = sum(values.values())
        return total*cubic_residual_helmholtz(
            spec, [values[key]/total for key in ids], temperature, volume/total)
    return tuple(partial_derivative(extensive, amounts, (key,))/(R*temperature)-log(z)
                 for key in ids)
