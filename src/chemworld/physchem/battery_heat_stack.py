"""Sourced pouch thickness FV geometry; no time solver or extra live state.

Boundary convention follows PyBaMM 25.4.2 OneDimensionalX: ideal thermally
conducting collectors are single finite-capacity nodes, adjacent to electrode
surface Dirichlet temperatures. Collector half-thickness is not a resistance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from math import isfinite
from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray

from chemworld.data.scientific_inputs import InputSource


def _numbers(
    obj: HeatStackLayer | HeatStackCollector | BatteryHeatStackSpec,
    positive: tuple[str, ...] = (),
    nonnegative: tuple[str, ...] = (),
) -> None:
    for name in (*positive, *nonnegative):
        value = getattr(obj, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            or (name in positive and value == 0)
        ):
            raise ValueError(f"{name} is outside its finite physical domain")
        object.__setattr__(obj, name, float(value))
    if not isinstance(obj.source, InputSource):
        raise ValueError("heat geometry and properties require an explicit InputSource")


def _array(values: ArrayLike) -> NDArray[np.float64]:
    result = np.array(values, dtype=float)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class HeatStackLayer:
    length_m: float
    heat_capacity_J_m3_K: float
    conductivity_W_m_K: float
    cells: int
    source: InputSource

    def __post_init__(self) -> None:
        _numbers(self, ("length_m", "heat_capacity_J_m3_K", "conductivity_W_m_K"))
        if type(self.cells) is not int or not 1 <= self.cells <= 256:
            raise ValueError("thermal layer requires 1 to 256 finite-volume cells")

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                k: getattr(self, k)
                for k in ("length_m", "heat_capacity_J_m3_K", "conductivity_W_m_K", "cells")
            },
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(**{**value, "source": InputSource.from_dict(value["source"])})


@dataclass(frozen=True)
class HeatStackCollector:
    length_m: float
    heat_capacity_J_m3_K: float
    face_heat_transfer_W_m2_K: float
    tab_heat_transfer_W_m2_K: float
    tab_width_m: float
    source: InputSource

    def __post_init__(self) -> None:
        _numbers(
            self,
            ("length_m", "heat_capacity_J_m3_K"),
            ("face_heat_transfer_W_m2_K", "tab_heat_transfer_W_m2_K", "tab_width_m"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{
                k: getattr(self, k)
                for k in (
                    "length_m",
                    "heat_capacity_J_m3_K",
                    "face_heat_transfer_W_m2_K",
                    "tab_heat_transfer_W_m2_K",
                    "tab_width_m",
                )
            },
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        return cls(**{**value, "source": InputSource.from_dict(value["source"])})


@dataclass(frozen=True)
class BatteryHeatStackSpec:
    negative: HeatStackLayer
    separator: HeatStackLayer
    positive: HeatStackLayer
    negative_collector: HeatStackCollector
    positive_collector: HeatStackCollector
    width_m: float
    height_m: float
    edge_heat_transfer_W_m2_K: float
    ambient_temperature_K: float
    source: InputSource
    minimum_temperature_K: float = 270.0
    maximum_temperature_K: float = 370.0

    def __post_init__(self) -> None:
        for name in (
            "negative",
            "separator",
            "positive",
            "negative_collector",
            "positive_collector",
        ):
            cls = HeatStackCollector if "collector" in name else HeatStackLayer
            value = getattr(self, name)
            if isinstance(value, dict):
                object.__setattr__(self, name, cls.from_dict(value))
            elif not isinstance(value, cls):
                raise ValueError("declare each thermal layer and collector explicitly")
        _numbers(
            self,
            (
                "width_m",
                "height_m",
                "ambient_temperature_K",
                "minimum_temperature_K",
                "maximum_temperature_K",
            ),
            ("edge_heat_transfer_W_m2_K",),
        )
        if not self.minimum_temperature_K < self.maximum_temperature_K:
            raise ValueError("local thermal stopping bounds must be ordered")
        if any(
            c.tab_width_m > 2 * (self.width_m + self.height_m)
            for c in (self.negative_collector, self.positive_collector)
        ):
            raise ValueError("tab area cannot exceed collector edge area")
        for values in (self.heat_capacities_J_K, self.volumes_m3, self.conductance_W_K):
            if not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError("derived finite thermal geometry must remain positive")
        if (
            not np.all(np.isfinite(self.loss_conductance_W_K))
            or not isfinite(self.total_heat_capacity_J_K)
            or not isfinite(float(np.sum(self.volumes_m3)))
        ):
            raise ValueError("derived thermal scale overflow")

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def layers(self) -> tuple[HeatStackLayer, HeatStackLayer, HeatStackLayer]:
        return (self.negative, self.separator, self.positive)

    @cached_property
    def node_ids(self) -> tuple[str, ...]:
        return (
            "negative-collector",
            *(
                f"{name}:{i}"
                for name, layer in zip(
                    ("negative", "separator", "positive"), self.layers, strict=True
                )
                for i in range(layer.cells)
            ),
            "positive-collector",
        )

    @property
    def coordinate_keys(self) -> tuple[str, ...]:
        return tuple(f"battery:thermal:delta:{i}_K" for i in range(len(self.node_ids) - 1))

    @cached_property
    def volumes_m3(self) -> NDArray[np.float64]:
        return _array(
            [
                self.negative_collector.length_m * self.area_m2,
                *(
                    self.area_m2 * layer.length_m / layer.cells
                    for layer in self.layers
                    for _ in range(layer.cells)
                ),
                self.positive_collector.length_m * self.area_m2,
            ]
        )

    @cached_property
    def heat_capacities_J_K(self) -> NDArray[np.float64]:  # noqa: N802 - physical units
        density = [
            self.negative_collector.heat_capacity_J_m3_K,
            *(layer.heat_capacity_J_m3_K for layer in self.layers for _ in range(layer.cells)),
            self.positive_collector.heat_capacity_J_m3_K,
        ]
        return _array(self.volumes_m3 * density)

    @property
    def total_heat_capacity_J_K(self) -> float:  # noqa: N802 - physical units
        return float(np.sum(self.heat_capacities_J_K))

    @cached_property
    def conductance_W_K(self) -> NDArray[np.float64]:  # noqa: N802 - physical units
        half = np.array(
            [
                layer.length_m / (2 * layer.cells * layer.conductivity_W_m_K * self.area_m2)
                for layer in self.layers
                for _ in range(layer.cells)
            ]
        )
        return _array(1 / np.r_[half[0], half[:-1] + half[1:], half[-1]])

    @cached_property
    def loss_conductance_W_K(self) -> NDArray[np.float64]:  # noqa: N802 - physical units
        perimeter = 2 * (self.width_m + self.height_m)
        result = [
            0.0,
            *(
                self.edge_heat_transfer_W_m2_K * perimeter * layer.length_m / layer.cells
                for layer in self.layers
                for _ in range(layer.cells)
            ),
            0.0,
        ]
        for index, c in ((0, self.negative_collector), (-1, self.positive_collector)):
            tab = c.tab_width_m * c.length_m
            result[index] = (
                c.face_heat_transfer_W_m2_K * self.area_m2
                + c.tab_heat_transfer_W_m2_K * tab
                + self.edge_heat_transfer_W_m2_K * (perimeter * c.length_m - tab)
            )
        return _array(result)

    @cached_property
    def conduction_matrix_W_K(self) -> NDArray[np.float64]:  # noqa: N802 - physical units
        n = len(self.node_ids)
        result = np.zeros((n, n))
        for i, value in enumerate(self.conductance_W_K):
            result[i, i] -= value
            result[i + 1, i + 1] -= value
            result[i, i + 1] += value
            result[i + 1, i] += value
        return _array(result)

    @cached_property
    def temperature_matrix(self) -> NDArray[np.float64]:
        # Columns: C-weighted mean followed by the independent temperature differences.
        n = len(self.node_ids)
        result = np.zeros((n, n))
        result[:, 0] = 1
        result[:-1, 1:] = np.eye(n - 1)
        result[-1, 1:] = -self.heat_capacities_J_K[:-1] / self.heat_capacities_J_K[-1]
        return _array(result)

    def temperatures(
        self, mean_temperature_K: float, coordinates: Mapping[str, float],
    ) -> NDArray[np.float64]:
        if not isfinite(mean_temperature_K):
            raise ValueError("thermal coordinates must remain finite")
        return mean_temperature_K + self.temperature_deviations(coordinates)

    def temperature_deviations(self, coordinates: Mapping[str, float]) -> NDArray[np.float64]:
        """Reconstruct relative temperatures without introducing the absolute mean."""
        first = np.array([coordinates[k] for k in self.coordinate_keys])
        if not np.all(np.isfinite(first)):
            raise ValueError("thermal coordinates must remain finite")
        last = -float(self.heat_capacities_J_K[:-1] @ first) / self.heat_capacities_J_K[-1]
        return np.r_[first, last]

    def conductive_heat_W(  # noqa: N802 - explicit physical units
        self, coordinates: Mapping[str, float],
    ) -> NDArray[np.float64]:
        """Same FV operator, evaluated as paired fluxes of mean-free differences."""
        relative = self.temperature_deviations(coordinates)
        flux = self.conductance_W_K * np.diff(relative)
        power = np.zeros(len(relative))
        power[:-1] += flux
        power[1:] -= flux
        return power

    def pack_temperatures(self, temperatures_K: ArrayLike) -> tuple[float, dict[str, float]]:
        values = np.asarray(temperatures_K, dtype=float)
        if values.shape != (len(self.node_ids),) or not np.all(np.isfinite(values)):
            raise ValueError("initial temperatures must declare every finite thermal node")
        # A uniform field packs to exactly zero differences, including when
        # capacities are unequal. Avoid summing absolute temperatures first.
        mean = float(
            values[-1]
            + self.heat_capacities_J_K @ (values - values[-1]) / self.total_heat_capacity_J_K
        )
        return mean, dict(zip(self.coordinate_keys, values[:-1] - mean, strict=True))

    def electrode_slice(self, electrode: str) -> slice:
        if electrode == "negative":
            return slice(1, 1 + self.negative.cells)
        if electrode == "positive":
            start = 1 + self.negative.cells + self.separator.cells
            return slice(start, start + self.positive.cells)
        raise ValueError("unknown heat-stack electrode")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "pouch-thickness-heat-stack-v1",
            **{
                k: getattr(self, k).to_dict()
                for k in (
                    "negative",
                    "separator",
                    "positive",
                    "negative_collector",
                    "positive_collector",
                    "source",
                )
            },
            **{
                k: getattr(self, k)
                for k in (
                    "width_m",
                    "height_m",
                    "edge_heat_transfer_W_m2_K",
                    "ambient_temperature_K",
                    "minimum_temperature_K",
                    "maximum_temperature_K",
                )
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        data = dict(value)
        if data.pop("schema") != "pouch-thickness-heat-stack-v1":
            raise ValueError("unknown heat-stack schema")
        data["source"] = InputSource.from_dict(data["source"])
        return cls(**data)
