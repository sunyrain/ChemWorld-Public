"""Explicit UNIFAC group declarations compiled into the existing GE model.

All six published temperature/combinatorial variants share one excess Gibbs
potential and the existing analytic derivative and calibration machinery.
Group identities are discrete input data; interaction coefficients are fitted
through the original activity calibration interface. There is no runtime
dependency on a reference library, implicit fragmentation, or missing-pair fill.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any

from chemworld.physchem.caloric_math import exponential, logarithm, scalar_value

if TYPE_CHECKING:
    from chemworld.physchem.equilibrium import ActivityModelSpec

UNIFAC_VARIANTS = {
    0: "original / LLE",
    1: "modified Dortmund / NIST",
    2: "PSRK",
    3: "VTPR",
    4: "Lyngby / Larsen",
    5: "NIST KT",
}


def _integer(value: Any, name: str, *, positive: bool = True) -> int:
    number = scalar_value(value)
    if isinstance(number, bool) or not isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer")
    if number < int(positive):
        raise ValueError(f"{name} is outside its integer domain")
    return int(number)


@dataclass(frozen=True)
class UNIFACSubgroup:
    group_id: int
    main_group_id: int
    R: float
    Q: float
    name: str = ""
    main_group_name: str = ""

    def __post_init__(self) -> None:
        _integer(self.group_id, "subgroup id")
        _integer(self.main_group_id, "main group id")
        if any(isinstance(v, bool) or not isfinite(v) or v < 0 for v in (self.R, self.Q)):
            raise ValueError("subgroup R and Q must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def unifac_rq(
    assignments: Mapping[int, int], subgroups: Mapping[int, UNIFACSubgroup]
) -> tuple[float, float]:
    """Bondi group volume/surface sums; explicit authored grouping only."""
    if not assignments:
        raise ValueError("molecular group assignment cannot be empty")
    r = q = 0.0
    for key, count in assignments.items():
        _integer(key, "subgroup id")
        _integer(count, "group count")
        if key not in subgroups or subgroups[key].group_id != key:
            raise ValueError(f"missing or inconsistent subgroup {key}")
        r += count * subgroups[key].R
        q += count * subgroups[key].Q
    return r, q


def unifac_activity_model(
    model_id: str,
    assignments: Mapping[str, Mapping[int, int]],
    *,
    subgroups: Mapping[int, UNIFACSubgroup],
    interactions: Mapping[tuple[int, int], tuple[float, float, float]],
    version: int,
    provenance: Mapping[str, object],
) -> ActivityModelSpec:
    """Compile an explicitly selected source revision into ActivityModelSpec.

    Every unlike main-group direction is required, including an explicit zero
    if that is the selected parameter. Same-main interactions are exactly zero
    by definition. Version 0 consumes only a; version 5 consumes a and b.
    """
    from chemworld.physchem.equilibrium import ActivityModelSpec

    if _integer(version, "UNIFAC version", positive=False) not in UNIFAC_VARIANTS:
        raise ValueError("UNIFAC version must be in 0..5")
    if not assignments or not provenance:
        raise ValueError("UNIFAC needs molecular assignments and source provenance")
    used = sorted({key for groups in assignments.values() for key in groups})
    for groups in assignments.values():
        unifac_rq(groups, subgroups)
    parameters = {"version": float(version)}
    for key in used:
        group = subgroups[key]
        parameters.update(
            {f"R:{key}": group.R, f"Q:{key}": group.Q, f"main:{key}": float(group.main_group_id)}
        )
        for component, groups in assignments.items():
            parameters[f"nu:{component}|{key}"] = float(groups.get(key, 0))
    mains = sorted({subgroups[key].main_group_id for key in used})
    letters = "a" if version == 0 else "ab" if version == 5 else "abc"
    for left in mains:
        for right in mains:
            if left == right:
                continue
            if (left, right) not in interactions:
                raise ValueError(f"missing directed UNIFAC interaction {left}|{right}")
            values = interactions[left, right]
            if len(values) != 3 or any(isinstance(v, bool) or not isfinite(v) for v in values):
                raise ValueError("UNIFAC interaction requires three finite coefficients")
            if (version == 0 and values[1:] != (0.0, 0.0)) or (version == 5 and values[2] != 0):
                raise ValueError("unused UNIFAC temperature coefficients must be explicitly zero")
            for letter, value in zip(letters, values, strict=False):
                parameters[f"psi_{letter}:{left}|{right}"] = value
    return ActivityModelSpec(
        model_id,
        tuple(assignments),
        "unifac",
        parameters,
        {
            "sources": dict(provenance),
            "variant": UNIFAC_VARIANTS[version],
            "subgroups": [subgroups[key].to_dict() for key in used],
            "assignments": {
                key: {str(g): v for g, v in groups.items()} for key, groups in assignments.items()
            },
            "group_count_unit": "dimensionless integer",
            "temperature_reference_K": 298.15 if version in {4, 5} else None,
            "missing_interactions": "rejected; no ideal-mixture fallback",
        },
    )


def validate_unifac_parameters(spec: ActivityModelSpec) -> None:
    p = spec.parameters
    if (
        "version" not in p
        or _integer(p["version"], "UNIFAC version", positive=False) not in UNIFAC_VARIANTS
    ):
        raise ValueError("UNIFAC version must be in 0..5")
    version = int(p["version"])
    groups = sorted(key.removeprefix("Q:") for key in p if key.startswith("Q:"))
    if not groups:
        raise ValueError("UNIFAC requires subgroup parameters")
    required = {"version"}
    for group in groups:
        _integer(float(group), "subgroup id")
        required.update(f"{name}:{group}" for name in ("R", "Q", "main"))
        required.update(f"nu:{key}|{group}" for key in spec.component_ids)
    if not required <= p.keys():
        raise ValueError(f"missing UNIFAC group parameters: {sorted(required - p.keys())}")
    mains = sorted({_integer(p[f"main:{g}"], "main group") for g in groups})
    letters = "a" if version == 0 else "ab" if version == 5 else "abc"
    required.update(
        f"psi_{letter}:{a}|{b}" for letter in letters for a in mains for b in mains if a != b
    )
    if set(p) != required:
        raise ValueError(
            f"UNIFAC parameters: missing={sorted(required - p.keys())}, "
            f"unknown={sorted(p.keys() - required)}"
        )
    for group in groups:
        if any(isinstance(p[f"{v}:{group}"], bool) or p[f"{v}:{group}"] < 0 for v in ("R", "Q")):
            raise ValueError("UNIFAC group R/Q must be nonnegative")
        counts = [p[f"nu:{key}|{group}"] for key in spec.component_ids]
        for count in counts:
            _integer(count, "group count", positive=False)
        if sum(counts) == 0:
            raise ValueError("unused UNIFAC subgroup")
    for key in spec.component_ids:
        r, q = (sum(p[f"nu:{key}|{g}"] * p[f"{v}:{g}"] for g in groups) for v in ("R", "Q"))
        if q <= 0 or (version != 3 and r <= 0):
            raise ValueError("UNIFAC molecules require positive surface and combinatorial volume")


def unifac_psi(version: int, temperature: Any, a: Any, b: Any = 0.0, c: Any = 0.0) -> Any:
    """Public interaction temperature functions, retaining analytic outer jets."""
    return exponential(_log_psi(version, temperature, a, b, c))


def _log_psi(version: int, temperature: Any, a: Any, b: Any = 0.0, c: Any = 0.0) -> Any:
    if version == 0:
        exponent = -a / temperature
    elif version in {1, 2, 3}:
        exponent = -a / temperature - b - c * temperature
    elif version in {4, 5}:
        exponent = (
            -a
            - b * (temperature - 298.15)
            - c * (temperature * logarithm(298.15 / temperature) + temperature - 298.15)
        ) / temperature
    else:
        raise ValueError("UNIFAC version must be in 0..5")
    return exponent


def unifac_reduced_gibbs(spec: ActivityModelSpec, xs: Any, temperature: Any, p: Any) -> Any:
    """Weighted public log-gamma equation, integrable on normalized composition."""
    # Discrete topology always comes from the validated base card, never a jet.
    version = int(spec.parameters["version"])
    groups = sorted(key.removeprefix("Q:") for key in spec.parameters if key.startswith("Q:"))
    ids = spec.component_ids
    counts = [[p[f"nu:{key}|{g}"] for g in groups] for key in ids]
    qg = [p[f"Q:{g}"] for g in groups]
    ri, qi = (
        [
            sum(counts[i][k] * p[f"{prefix}:{g}"] for k, g in enumerate(groups))
            for i in range(len(ids))
        ]
        for prefix in ("R", "Q")
    )
    mains = [int(spec.parameters[f"main:{g}"]) for g in groups]
    log_psi = [
        [
            0.0
            if a == b
            else _log_psi(
                version,
                temperature,
                p[f"psi_a:{a}|{b}"],
                p.get(f"psi_b:{a}|{b}", 0.0),
                p.get(f"psi_c:{a}|{b}", 0.0),
            )
            for b in mains
        ]
        for a in mains
    ]

    def group_gibbs(amounts: Any, active: list[int]) -> Any:
        surface = [v * q for v, q in zip(amounts, qg, strict=True)]
        total = sum(surface)
        if scalar_value(total) <= 0:
            raise ValueError("UNIFAC group surface sum must be positive")
        value = 0.0
        for k in active:
            # Preserve tiny interaction logarithms even when psi underflows.
            # Pure reference baselines include only that molecule's structural
            # groups; absent terms are exactly zero, not log(0) evaluations.
            shift = max(scalar_value(log_psi[j][k]) for j in active if scalar_value(surface[j]) > 0)
            weighted = sum(surface[j] * exponential(log_psi[j][k] - shift) for j in active)
            value -= surface[k] * (shift + logarithm(weighted / total))
        return value

    amount = [sum(xs[i] * counts[i][k] for i in range(len(ids))) for k in range(len(groups))]
    active = [k for k, g in enumerate(groups) if spec.parameters[f"Q:{g}"] > 0]
    residual = group_gibbs(amount, active) - sum(
        xs[i]
        * group_gibbs(
            counts[i], [k for k in active if spec.parameters[f"nu:{ids[i]}|{groups[k]}"] > 0]
        )
        for i in range(len(ids))
    )
    if version == 3:
        return residual
    rbar, qbar = (
        sum(x * r for x, r in zip(xs, ri, strict=True)),
        sum(x * q for x, q in zip(xs, qi, strict=True)),
    )
    power = 0.75 if version == 1 else 2 / 3 if version == 4 else 1.0
    modified = [r**power for r in ri]
    denominator = sum(x * r for x, r in zip(xs, modified, strict=True))
    combinatorial = 0.0
    for i, x in enumerate(xs):
        vprime = modified[i] / denominator
        value = 1 - vprime + logarithm(vprime)
        if version in {0, 1, 2, 5}:
            ratio = ri[i] * qbar / (qi[i] * rbar)
            value -= 5 * qi[i] * (1 - ratio + logarithm(ratio))
        combinatorial += x * value
    return residual + combinatorial
