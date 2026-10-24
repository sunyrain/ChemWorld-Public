"""Declarative gas thermo families sharing the native SpeciesThermoState contract.

Equations follow the public Cantera 3.2 model definitions. This module has no
Cantera runtime dependency; NASA7 remains in the original thermochemistry module.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import isfinite, log
from typing import Any

from chemworld.physchem.reaction_rate_laws import R_J_PER_MOL_K as R
from chemworld.physchem.thermochemistry import SpeciesThermoState

THERMO_MODELS = ("NASA7", "NASA9", "Shomate", "constant-cp", "piecewise-Gibbs")


def molar_quantity(value: Any, units: Mapping[str, str], *, entropy: bool = False) -> float:
    """Read a molar energy or entropy in J/mol or J/(mol K), respectively."""
    default = f"{units.get('energy', 'J')}/{units.get('quantity', 'kmol')}"
    if entropy:
        default += "/K"
    if isinstance(value, str):
        match = re.fullmatch(r"\s*([-+\d.eE]+)\s*(\S+)?\s*", value)
        if match is None:
            raise ValueError(f"invalid molar quantity: {value}")
        value, explicit = match.groups()
        default = explicit or default
    energy, amount, *temperature = default.split("/")
    if entropy != (temperature == ["K"]):
        raise ValueError(f"invalid molar quantity dimension: {default}")
    energies = {"J": 1.0, "kJ": 1000.0, "cal": 4.184, "kcal": 4184.0}
    amounts = {"mol": 1.0, "kmol": 1000.0}
    if energy not in energies or amount not in amounts:
        raise ValueError(f"unsupported molar quantity unit: {default}")
    return _finite(value) * energies[energy] / amounts[amount]


def temperature_quantity(value: Any) -> float:
    if isinstance(value, str):
        value = value.removesuffix("K").strip()
    return _finite(value)


def _finite(value: Any) -> float:
    result = float(value)
    if isinstance(value, bool) or not isfinite(result):
        raise ValueError("thermochemistry parameters must be finite numbers")
    return result


@dataclass(frozen=True)
class GasSpeciesThermo:
    """NASA9, Shomate, constant-Cp or piecewise constant-Cp Gibbs interpolation.

    NASA9 switches to the upper segment at an interior boundary; Shomate uses
    the lower one, as does NASA7. Evaluation rejects extrapolation outside an
    explicitly declared validity range. Source declarations round-trip intact.
    """

    species_id: str
    thermo: dict[str, Any]
    reference_pressure_Pa: float = 101325.0
    composition: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    _ranges: tuple[float, ...] = field(init=False, repr=False, default=())
    _rows: tuple[tuple[float, ...], ...] = field(init=False, repr=False, default=())
    _gibbs_nodes: tuple[tuple[float, float, float, float], ...] = field(
        init=False, repr=False, default=()
    )

    @property
    def min_temperature_K(self) -> float:  # noqa: N802 - shared unit-explicit contract
        return self._ranges[0]

    @property
    def max_temperature_K(self) -> float:  # noqa: N802 - shared unit-explicit contract
        return self._ranges[-1]

    def __post_init__(self) -> None:
        if not self.species_id.strip() or _finite(self.reference_pressure_Pa) <= 0:
            raise ValueError("species id and positive reference pressure are required")
        object.__setattr__(self, "thermo", copy.deepcopy(self.thermo))
        model = self.thermo.get("model")
        if model == "piecewise-Gibbs" and isinstance(self.thermo.get("data"), Mapping):
            self.thermo["data"] = {str(t): g for t, g in self.thermo["data"].items()}
        common = {"model", "reference-pressure", "note"}
        fields = {
            "NASA9": {"temperature-ranges", "data"},
            "Shomate": {"temperature-ranges", "data"},
            "constant-cp": {"T0", "h0", "s0", "cp0", "T-min", "T-max"},
            "piecewise-Gibbs": {"h0", "data", "dimensionless", "T-min", "T-max"},
        }
        if model not in fields or set(self.thermo) - (common | fields[model]):
            raise ValueError(f"unsupported thermochemistry declaration: {self.thermo}")
        if model in {"NASA9", "Shomate"}:
            ranges = tuple(temperature_quantity(v) for v in self.thermo["temperature-ranges"])
            rows = tuple(tuple(_finite(v) for v in row) for row in self.thermo["data"])
            count = 9 if model == "NASA9" else 7
            if (
                not rows
                or len(ranges) != len(rows) + 1
                or ranges[0] <= 0
                or any(b <= a for a, b in pairwise(ranges))
                or any(len(row) != count for row in rows)
                or (model == "Shomate" and len(rows) > 2)
            ):
                raise ValueError(f"invalid {model} segments")
            object.__setattr__(self, "_ranges", ranges)
            object.__setattr__(self, "_rows", rows)
        else:
            lo = temperature_quantity(self.thermo.get("T-min", 0))
            hi = temperature_quantity(self.thermo.get("T-max", 1e300))
            if lo < 0 or hi <= lo:
                raise ValueError("invalid thermochemical temperature range")
            object.__setattr__(self, "_ranges", (lo, hi))
            if model == "constant-cp":
                if temperature_quantity(self.thermo.get("T0", 298.15)) <= 0:
                    raise ValueError("constant-cp T0 must be positive")
                for key in ("h0", "s0", "cp0"):
                    molar_quantity(self.thermo.get(key, 0), self.units, entropy=key != "h0")
            else:
                object.__setattr__(self, "_gibbs_nodes", self._build_gibbs_nodes())

    def _build_gibbs_nodes(self) -> tuple[tuple[float, float, float, float], ...]:
        data = self.thermo["data"]
        if not isinstance(data, Mapping) or len(data) < 2:
            raise ValueError("piecewise-Gibbs needs at least two temperature/Gibbs nodes")
        points = sorted((temperature_quantity(t), g) for t, g in data.items())
        temperatures = [t for t, _ in points]
        if (
            temperatures[0] <= 0
            or 298.15 not in temperatures
            or len(set(temperatures)) != len(points)
        ):
            raise ValueError("piecewise-Gibbs needs positive unique nodes including 298.15 K")
        gibbs = [
            (
                _finite(g) * R * t
                if self.thermo.get("dimensionless", False)
                else molar_quantity(g, self.units)
            )
            for t, g in points
        ]
        h, s, cp = [0.0] * len(points), [0.0] * len(points), [0.0] * (len(points) - 1)
        anchor = temperatures.index(298.15)
        h[anchor] = molar_quantity(self.thermo.get("h0", 0), self.units)
        s[anchor] = (h[anchor] - gibbs[anchor]) / 298.15
        for i in range(anchor, len(points) - 1):
            t1, t2 = temperatures[i : i + 2]
            cp[i] = (gibbs[i + 1] - gibbs[i] + (t2 - t1) * s[i]) / (t2 - t1 - t2 * log(t2 / t1))
            h[i + 1] = h[i] + cp[i] * (t2 - t1)
            s[i + 1] = s[i] + cp[i] * log(t2 / t1)
        for i in range(anchor - 1, -1, -1):
            t1, t2 = temperatures[i : i + 2]
            cp[i] = (gibbs[i + 1] - gibbs[i] + (t2 - t1) * s[i + 1]) / (t2 - t1 - t1 * log(t2 / t1))
            h[i] = h[i + 1] - cp[i] * (t2 - t1)
            s[i] = s[i + 1] - cp[i] * log(t2 / t1)
        return tuple(
            (t, hi, si, cp[min(i, len(cp) - 1)])
            for i, (t, hi, si) in enumerate(zip(temperatures, h, s, strict=True))
        )

    def evaluate(self, temperature_K: float) -> SpeciesThermoState:
        t = _finite(temperature_K)
        if t <= 0 or not self._ranges[0] <= t <= self._ranges[-1]:
            raise ValueError("temperature outside species thermochemistry validity range")
        model, index = self.thermo["model"], 0
        if model in {"NASA9", "Shomate"}:
            index = min(
                sum(t >= edge if model == "NASA9" else t > edge for edge in self._ranges[1:-1]),
                len(self._rows) - 1,
            )
            a = self._rows[index]
            if model == "NASA9":
                cp_r = (
                    a[0] / t**2
                    + a[1] / t
                    + a[2]
                    + a[3] * t
                    + a[4] * t**2
                    + a[5] * t**3
                    + a[6] * t**4
                )
                h_rt = (
                    -a[0] / t**2
                    + a[1] * log(t) / t
                    + a[2]
                    + a[3] * t / 2
                    + a[4] * t**2 / 3
                    + a[5] * t**3 / 4
                    + a[6] * t**4 / 5
                    + a[7] / t
                )
                s_r = (
                    -a[0] / (2 * t**2)
                    - a[1] / t
                    + a[2] * log(t)
                    + a[3] * t
                    + a[4] * t**2 / 2
                    + a[5] * t**3 / 3
                    + a[6] * t**4 / 4
                    + a[8]
                )
                cp, h, s = R * cp_r, R * t * h_rt, R * s_r
            else:
                z = t / 1000
                cp = a[0] + a[1] * z + a[2] * z**2 + a[3] * z**3 + a[4] / z**2
                h = 1000 * (
                    a[0] * z + a[1] * z**2 / 2 + a[2] * z**3 / 3 + a[3] * z**4 / 4 - a[4] / z + a[5]
                )
                s = (
                    a[0] * log(z)
                    + a[1] * z
                    + a[2] * z**2 / 2
                    + a[3] * z**3 / 3
                    - a[4] / (2 * z**2)
                    + a[6]
                )
        else:
            if model == "constant-cp":
                t0 = temperature_quantity(self.thermo.get("T0", 298.15))
                h0 = molar_quantity(self.thermo.get("h0", 0), self.units)
                s0 = molar_quantity(self.thermo.get("s0", 0), self.units, entropy=True)
                cp = molar_quantity(self.thermo.get("cp0", 0), self.units, entropy=True)
            else:
                index = min(
                    sum(t > node[0] for node in self._gibbs_nodes[1:]), len(self._gibbs_nodes) - 2
                )
                t0, h0, s0, cp = self._gibbs_nodes[index]
            h, s = h0 + cp * (t - t0), s0 + cp * log(t / t0)
        return SpeciesThermoState(
            self.species_id,
            t,
            cp / R,
            h / (R * t),
            s / R,
            cp,
            h,
            s,
            h - t * s,
            f"{self.species_id}:segment-{index}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "species_id": self.species_id,
            "thermo": copy.deepcopy(self.thermo),
            "reference_pressure_Pa": self.reference_pressure_Pa,
            "composition": dict(self.composition),
            "units": dict(self.units),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GasSpeciesThermo:
        return cls(**dict(payload))
