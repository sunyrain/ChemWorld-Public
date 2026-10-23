"""Predictive PR78 interactions from explicit molecular groups and source rows.

No automatic molecular fragmentation or runtime reference package is used.
Source A/B values are in MPa; absent pairs are errors, not ideal interactions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.data_conflicts import DataSourceProvenance

if TYPE_CHECKING:
    from chemworld.physchem.eos import EOSComponentSpec

PR78_OMEGA_A = 0.4572355289213821893834601962251837888504
PR78_OMEGA_B = 0.0777960739038884559718447100373331839711
R = 8.31446261815324


def pr78_kappa(omega: float) -> float:
    """PR78's two acentric-factor branches; legacy PR remains separately selectable."""
    if omega <= 0.491:
        return 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    return 0.379642 + 1.48503 * omega - 0.164423 * omega**2 + 0.016666 * omega**3


@dataclass(frozen=True)
class PPR78Spec:
    variant: Literal["original", "extended"]
    groups: Mapping[str, Mapping[str, int]]
    interactions_MPa: Mapping[str, tuple[float, float]]
    source: DataSourceProvenance

    def __post_init__(self) -> None:
        if self.variant not in {"original", "extended"} or not self.groups:
            raise ValueError("PPR78 needs an original/extended variant and molecular groups")
        groups: dict[str, Mapping[str, int]] = {}
        for component, counts in self.groups.items():
            if not isinstance(component, str) or not component or not counts:
                raise ValueError("PPR78 groups require explicit component identities")
            if any(not isinstance(k, str) or not k or "|" in k for k in counts):
                raise ValueError("PPR78 group identities must be nonempty strings without '|'")
            if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0
                   for v in counts.values()):
                raise ValueError("PPR78 molecular group counts must be positive integers")
            groups[component] = MappingProxyType(dict(counts))
        rows = {}
        for key, pair in self.interactions_MPa.items():
            if len(key.split("|")) != 2 or any(not k for k in key.split("|")):
                raise ValueError("PPR78 interaction identities must be group|group")
            if len(pair) != 2 or any(isinstance(v, bool) or not isfinite(v) for v in pair):
                raise ValueError("PPR78 interactions require finite A and B in MPa")
            rows[key] = tuple(pair)
        ids = set().union(*(counts.keys() for counts in groups.values()))
        for left in ids:
            for right in ids:
                key = f"{left}|{right}"
                if key not in rows:
                    raise ValueError(f"missing PPR78 interaction {key}")
                if rows[key] != rows.get(f"{right}|{left}"):
                    raise ValueError("PPR78 interactions must be symmetric")
        object.__setattr__(self, "groups", MappingProxyType(groups))
        object.__setattr__(self, "interactions_MPa", MappingProxyType(rows))

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "groups": {k: dict(v) for k, v in self.groups.items()},
            "interactions_MPa": {k: list(v) for k, v in self.interactions_MPa.items()},
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PPR78Spec:
        return cls(payload["variant"], payload["groups"], payload["interactions_MPa"],
                   DataSourceProvenance.from_dict(payload["source"]))


def ppr78_value(
    spec: PPR78Spec, left: str, right: str, temperature: Any,
    a_left: Any, a_right: Any, b_left: float, b_right: float,
) -> Any:
    """Internal analytic expression using the caller's consistent PR78 parameters."""
    left_counts, right_counts = spec.groups[left], spec.groups[right]
    left_total, right_total = sum(left_counts.values()), sum(right_counts.values())
    ids = set(left_counts) | set(right_counts)
    delta = {k: left_counts.get(k, 0) / left_total - right_counts.get(k, 0) / right_total
             for k in ids}
    interaction = 0.0
    for k in sorted(ids):
        for ell in sorted(ids):
            a, b = spec.interactions_MPa[f"{k}|{ell}"]
            if a != 0.0:
                interaction += delta[k] * delta[ell] * a * (298.15 / temperature)**(b/a - 1)
    energy = -0.5e6 * interaction
    mismatch = (a_left**0.5 / b_left - a_right**0.5 / b_right)**2
    return (energy - mismatch) / (2 * (a_left * a_right)**0.5 / (b_left * b_right))


def ppr78_interaction_response(
    spec: PPR78Spec, left: EOSComponentSpec, right: EOSComponentSpec,
    *, temperature_K: float,
) -> dict[str, float]:
    """Predict k_ij and its first three temperature derivatives on standard PR78."""
    if not isfinite(temperature_K) or temperature_K <= 0:
        raise ValueError("PPR78 temperature must be positive and finite")
    if left.component_id not in spec.groups or right.component_id not in spec.groups:
        raise ValueError("PPR78 molecular groups missing for an EOS component")
    if left.alpha_model != "soave" or right.alpha_model != "soave":
        raise ValueError("PPR78 requires its standard PR78 Soave alpha parameterization")

    def parameters(component: EOSComponentSpec, temperature: Any) -> tuple[Any, float]:
        tc, pc = component.critical_temperature_K, component.critical_pressure_Pa
        alpha = (1 + pr78_kappa(component.acentric_factor) * (1 - (temperature/tc)**0.5))**2
        return PR78_OMEGA_A * R*R * tc*tc/pc * alpha, PR78_OMEGA_B * R * tc/pc

    def value(temperature: Any) -> Any:
        ai, bi = parameters(left, temperature)
        aj, bj = parameters(right, temperature)
        return ppr78_value(spec, left.component_id, right.component_id, temperature, ai, aj, bi, bj)

    values = derivative_values(value, temperature_K, order=3)
    return dict(zip(("k_ij", "dk_dT", "d2k_dT2", "d3k_dT3"), values, strict=True))
