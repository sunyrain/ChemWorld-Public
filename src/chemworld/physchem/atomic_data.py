"""Versioned factual atomic data; isotope tables load only on first request."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Protocol, TextIO, cast

ATOMIC_DATA_VERSION = "chemworld-atomic-data-1"


@dataclass(frozen=True)
class AtomicEstimate:
    value: float | None
    standard_uncertainty: float | None = None
    interval: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if self.value is not None and (not isfinite(self.value) or self.value < 0):
            raise ValueError("atomic estimate must be finite and nonnegative")
        if self.standard_uncertainty is not None and (
            self.value is None
            or not isfinite(self.standard_uncertainty)
            or self.standard_uncertainty < 0
        ):
            raise ValueError("uncertainty requires an available nominal value")
        if self.interval is not None:
            low, high = self.interval
            if not 0 <= low <= high or not isfinite(high):
                raise ValueError("atomic interval must be finite and ordered")
            object.__setattr__(self, "interval", (low, high))

    @property
    def available(self) -> bool:
        return self.value is not None or self.interval is not None


@dataclass(frozen=True)
class ElementData:
    atomic_number: int
    symbol: str
    name: str
    standard_weight: AtomicEstimate


@dataclass(frozen=True)
class IsotopeData:
    atomic_number: int
    symbol: str
    mass_number: int
    mass: AtomicEstimate
    estimated_mass: bool
    abundance: AtomicEstimate
    mass_source: str = "AME2020/periodictable-2.1.0"
    abundance_source: str = "CIAAW2021/periodictable-2.1.0"

    @property
    def label(self) -> str:
        return f"[{self.mass_number}{self.symbol}]"


class _AtomicResource(Protocol):
    # Resource files support text-open keywords; older Traversable stubs omit newline.
    def open(self, mode: Literal["r"], *, encoding: str, newline: str) -> TextIO: ...

    def read_text(self, encoding: str) -> str: ...


def _resource(name: str) -> _AtomicResource:
    return cast(_AtomicResource, files("chemworld.physchem").joinpath("atomic_data", name))


def atomic_data_provenance() -> dict[str, Any]:
    return json.loads(_resource("provenance.json").read_text(encoding="utf-8"))


def _number(text: str) -> float | None:
    return None if text == "" else float(text)


def _estimate(value: str, uncertainty: str, low: str = "", high: str = "") -> AtomicEstimate:
    if bool(low) != bool(high):
        raise ValueError("atomic interval needs both bounds")
    return AtomicEstimate(
        _number(value), _number(uncertainty), None if not low else (float(low), float(high))
    )


@cache
def element_data() -> Mapping[str, ElementData]:
    result: dict[str, ElementData] = {}
    with _resource("elements.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            item = ElementData(
                int(row["atomic_number"]),
                row["symbol"],
                row["name"],
                _estimate(
                    row["standard_weight_g_mol"],
                    row["weight_uncertainty_g_mol"],
                    row["weight_low_g_mol"],
                    row["weight_high_g_mol"],
                ),
            )
            if item.symbol in result:
                raise ValueError("duplicate atomic element record")
            result[item.symbol] = item
    return MappingProxyType(result)


@cache
def isotope_data() -> Mapping[str, IsotopeData]:
    result: dict[str, IsotopeData] = {}
    for_version = element_data()
    with _resource("isotopes.csv").open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            mass = _estimate(row["mass_u"], row["mass_uncertainty_u"])
            item = IsotopeData(
                int(row["atomic_number"]),
                row["symbol"],
                int(row["mass_number"]),
                mass,
                bool(int(row["estimated"])),
                _estimate(
                    row["abundance_fraction"],
                    row["abundance_uncertainty"],
                    row["abundance_low"],
                    row["abundance_high"],
                ),
            )
            if (
                item.mass.value is None
                or item.mass.value <= 0
                or item.mass_number <= 0
                or for_version[item.symbol].atomic_number != item.atomic_number
            ):
                raise ValueError("isotope needs a positive mass and its declared element")
            if item.label in result:
                raise ValueError("duplicate isotope record")
            result[item.label] = item
    return MappingProxyType(result)
