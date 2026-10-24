"""Sourced energy descriptors and an explicit bounded free-energy BEP profile."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance


def validate_source(source: DataSourceProvenance) -> None:
    if not source.version.strip() or not source.license.strip():
        raise ValueError("catalysis data require source version and license")


def finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value):
        raise ValueError(name + " must be a finite number")


@dataclass(frozen=True)
class DescriptorAxis:
    descriptor_id: str
    minimum_J_mol: float
    maximum_J_mol: float
    unit: str = "J/mol"

    def __post_init__(self) -> None:
        for name in ("minimum_J_mol", "maximum_J_mol"):
            finite_number(getattr(self, name), name)
        if (
            not self.descriptor_id
            or self.minimum_J_mol >= self.maximum_J_mol
            or self.unit != "J/mol"
        ):
            raise ValueError("descriptor needs identity, ordered domain and explicit J/mol")


@dataclass(frozen=True)
class LinearEnergyScaling:
    axes: tuple[DescriptorAxis, ...]
    coefficients: Mapping[str, float]
    intercept_J_mol: float
    source: DataSourceProvenance
    energy_unit: str = "J/mol"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        object.__setattr__(self, "coefficients", MappingProxyType(dict(self.coefficients)))
        validate_source(self.source)
        finite_number(self.intercept_J_mol, "energy intercept")
        keys = [axis.descriptor_id for axis in self.axes]
        if len(keys) != len(set(keys)) or set(keys) != set(self.coefficients):
            raise ValueError("each unique descriptor needs an explicit scaling coefficient")
        for value in self.coefficients.values():
            finite_number(value, "dimensionless scaling coefficient")
        if self.energy_unit != "J/mol":
            raise ValueError("scaling energies require J/mol")

    def evaluate(self, descriptors: Mapping[str, float]) -> float:
        if set(descriptors) != set(self.coefficients):
            raise ValueError("descriptor values must match all declared axes")
        for axis in self.axes:
            value = descriptors[axis.descriptor_id]
            finite_number(value, "descriptor value")
            if not axis.minimum_J_mol <= value <= axis.maximum_J_mol:
                raise ValueError("descriptor value outside declared domain: " + axis.descriptor_id)
        return self.intercept_J_mol + fsum(self.coefficients[k] * v for k, v in descriptors.items())

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": [asdict(a) for a in self.axes],
            "coefficients": dict(self.coefficients),
            "intercept_J_mol": self.intercept_J_mol,
            "energy_unit": self.energy_unit,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LinearEnergyScaling:
        values = dict(raw)
        values["axes"] = tuple(DescriptorAxis(**a) for a in values["axes"])
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)


@dataclass(frozen=True)
class FreeEnergyBEP:
    intercept_J_mol: float
    alpha: float
    minimum_delta_g_J_mol: float
    maximum_delta_g_J_mol: float
    source: DataSourceProvenance
    energy_unit: str = "J/mol"

    def __post_init__(self) -> None:
        validate_source(self.source)
        for name in ("intercept_J_mol", "alpha", "minimum_delta_g_J_mol", "maximum_delta_g_J_mol"):
            finite_number(getattr(self, name), name)
        if (
            self.intercept_J_mol < 0
            or not 0 <= self.alpha <= 1
            or self.minimum_delta_g_J_mol >= self.maximum_delta_g_J_mol
            or self.energy_unit != "J/mol"
        ):
            raise ValueError(
                "free-energy BEP requires nonnegative intercept, alpha in [0,1] and domain"
            )

    def barriers(self, delta_g_J_mol: float) -> tuple[float, float]:
        finite_number(delta_g_J_mol, "reaction free energy")
        if not self.minimum_delta_g_J_mol <= delta_g_J_mol <= self.maximum_delta_g_J_mol:
            raise ValueError("reaction free energy outside the declared BEP domain")
        forward = self.intercept_J_mol + self.alpha * delta_g_J_mol
        reverse = forward - delta_g_J_mol
        if min(forward, reverse) < 0:
            raise ValueError("negative free-energy BEP barrier; no implicit clipping")
        return forward, reverse

    def to_dict(self) -> dict[str, Any]:
        return {
            "intercept_J_mol": self.intercept_J_mol,
            "alpha": self.alpha,
            "minimum_delta_g_J_mol": self.minimum_delta_g_J_mol,
            "maximum_delta_g_J_mol": self.maximum_delta_g_J_mol,
            "energy_unit": self.energy_unit,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FreeEnergyBEP:
        values = dict(raw)
        values["source"] = DataSourceProvenance.from_dict(values["source"])
        return cls(**values)
