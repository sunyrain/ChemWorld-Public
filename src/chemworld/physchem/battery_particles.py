"""Sourced particle-size quadrature, with explicit number/area/volume bases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite, log, sqrt
from numbers import Real
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from chemworld.data.scientific_inputs import InputSource, InputValue, ScientificTable

PSD_VERSION = "finite-particle-size-distribution-v1"


@dataclass(frozen=True)
class ParticleSizeDistribution:
    radii_m: tuple[float, ...]
    volume_weights: tuple[float, ...]
    source: InputSource
    parameter_rows: tuple[tuple[str, InputValue, InputValue], ...] = ()
    source_basis: str = "volume"
    distribution_parameters: tuple[tuple[str, InputValue], ...] = ()

    def __post_init__(self) -> None:
        radii, weights = tuple(self.radii_m), tuple(self.volume_weights)
        inputs: tuple[object, ...] = (*radii, *weights)
        if (
            not 1 <= len(radii) <= 64
            or len(weights) != len(radii)
            or any(
                isinstance(v, bool) or not isinstance(v, Real) or not isfinite(v) or v <= 0
                for v in inputs
            )
            or any(a >= b for a, b in pairwise(radii))
            or abs(sum(weights) - 1) > 1e-12
        ):
            raise ValueError("PSD requires ordered positive radii and normalized positive volumes")
        if not isinstance(self.source, InputSource) or self.source_basis not in {
            "number",
            "area",
            "volume",
        }:
            raise ValueError("PSD requires an InputSource and an explicit weighting basis")
        rows = tuple(self.parameter_rows)
        if (rows and len(rows) != len(radii)) or any(
            len(row) != 3
            or not isinstance(row[0], str)
            or not row[0]
            or not all(isinstance(cell, InputValue) for cell in row[1:])
            for row in rows
        ):
            raise ValueError("PSD parameter rows must retain each original radius/weight input")
        if len({row[0] for row in rows}) != len(rows):
            raise ValueError("PSD source row identities must be unique")
        object.__setattr__(self, "radii_m", tuple(float(v) for v in radii))
        object.__setattr__(self, "volume_weights", tuple(float(v) for v in weights))
        object.__setattr__(self, "parameter_rows", rows)
        parameters = tuple(self.distribution_parameters)
        if len(dict(parameters)) != len(parameters) or any(
            not isinstance(k, str) or not k or not isinstance(v, InputValue) for k, v in parameters
        ):
            raise ValueError("PSD distribution parameters need unique names and common InputValue")
        object.__setattr__(self, "distribution_parameters", parameters)

    def weights(self, basis: str) -> tuple[float, ...]:
        if basis not in {"number", "area", "volume"}:
            raise ValueError("unknown PSD weighting basis")
        exponent = {"number": -3, "area": -1, "volume": 0}[basis]
        # Normalize radius by the first point to avoid dimensional powers of
        # extreme SI radii. The common power cancels from normalized weights.
        weights = (
            np.array(self.volume_weights) * (np.array(self.radii_m) / self.radii_m[0]) ** exponent
        )
        return tuple(float(v) for v in weights / weights.sum())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": PSD_VERSION,
            "radii_m": list(self.radii_m),
            "volume_weights": list(self.volume_weights),
            "source": self.source.to_dict(),
            "source_basis": self.source_basis,
            "distribution_parameters": {k: v.to_dict() for k, v in self.distribution_parameters},
            "parameter_rows": [
                [name, radius.to_dict(), weight.to_dict()]
                for name, radius, weight in self.parameter_rows
            ],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if (
            set(value)
            != {
                "model_version",
                "radii_m",
                "volume_weights",
                "source",
                "source_basis",
                "parameter_rows",
                "distribution_parameters",
            }
            or value["model_version"] != PSD_VERSION
        ):
            raise ValueError("unsupported PSD fields or version")
        return cls(
            tuple(value["radii_m"]),
            tuple(value["volume_weights"]),
            InputSource.from_dict(value["source"]),
            tuple(
                (name, InputValue.from_dict(r), InputValue.from_dict(w))
                for name, r, w in value["parameter_rows"]
            ),
            value["source_basis"],
            tuple(
                (k, InputValue.from_dict(v)) for k, v in value["distribution_parameters"].items()
            ),
        )

    @classmethod
    def from_weights(
        cls,
        radii_m: Sequence[float] | NDArray[np.float64],
        weights: Sequence[float] | NDArray[np.float64],
        *,
        source: InputSource,
        basis: str,
        parameter_rows: tuple[tuple[str, InputValue, InputValue], ...] = (),
    ) -> Self:
        """Explicit conversion and normalization of positive relative input weights."""
        if any(isinstance(v, bool) for v in (*radii_m, *weights)):
            raise ValueError("PSD radius/weight inputs must be numeric, not booleans")
        radii = np.asarray(radii_m, dtype=float)
        input_weights = np.asarray(weights, dtype=float)
        if (
            basis not in {"number", "area", "volume"}
            or radii.ndim != 1
            or radii.shape != input_weights.shape
            or not len(radii)
            or not np.all(np.isfinite(radii))
            or not np.all(np.isfinite(input_weights))
            or np.any(radii <= 0)
            or np.any(input_weights <= 0)
        ):
            raise ValueError("PSD basis conversion requires positive finite radius/weight pairs")
        power = {"number": 3, "area": 1, "volume": 0}[basis]
        converted = input_weights * (radii / radii[0]) ** power
        converted /= converted.sum()
        return cls(tuple(radii), tuple(converted), source, parameter_rows, basis)

    @classmethod
    def from_parameter_table(
        cls, table: ScientificTable, *, radius_column: str, weight_column: str, basis: str,
    ) -> Self:
        if table.kind != "parameters" or not table.rows:
            raise ValueError("PSD requires a nonempty common parameter table")
        sources = {row.source_id for row in table.rows}
        if len(sources) != 1:
            raise ValueError("one PSD must have one explicit parameter-table source")
        pairs = tuple(
            (row.row_id, row.values[radius_column], row.values[weight_column]) for row in table.rows
        )
        if any(cell.quality != "valid" for _, radius, weight in pairs for cell in (radius, weight)):
            raise ValueError("PSD requires valid source parameters")
        return cls.from_weights(
            [cast(float, r.to("m").value) for _, r, _ in pairs],
            [cast(float, w.to("dimensionless").value) for _, _, w in pairs],
            basis=basis,
            source=next(s for s in table.sources if s.provenance.source_id in sources),
            parameter_rows=pairs,
        )


def lognormal_size_distribution(
    *,
    median_radius_m: float,
    log_standard_deviation: float,
    lower_radius_m: float,
    upper_radius_m: float,
    nodes: int,
    source: InputSource,
    basis: str = "area",
) -> ParticleSizeDistribution:
    """Gauss-Legendre quadrature of a declared truncated lognormal density.

    median and log-standard-deviation describe the INPUT basis, not an
    automatically inferred number distribution. Truncation is explicit.
    """
    if (
        type(nodes) is not int
        or not 1 <= nodes <= 64
        or basis not in {"number", "area", "volume"}
        or any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (median_radius_m, log_standard_deviation, lower_radius_m, upper_radius_m)
        )
        or lower_radius_m >= upper_radius_m
    ):
        raise ValueError("invalid declared truncated lognormal quadrature")
    x, w = np.polynomial.legendre.leggauss(nodes)
    radii = lower_radius_m + (x + 1) * (upper_radius_m - lower_radius_m) / 2
    density = np.exp(-0.5 * ((np.log(radii) - log(median_radius_m)) / log_standard_deviation) ** 2)
    density /= radii * log_standard_deviation * sqrt(2 * np.pi)
    quadrature = w * density * (upper_radius_m - lower_radius_m) / 2
    distribution = ParticleSizeDistribution.from_weights(
        radii, quadrature, source=source, basis=basis
    )
    from dataclasses import replace

    parameters = (
        ("median_radius", median_radius_m, "m"),
        ("log_standard_deviation", log_standard_deviation, "dimensionless"),
        ("lower_radius", lower_radius_m, "m"),
        ("upper_radius", upper_radius_m, "m"),
        ("quadrature_nodes", float(nodes), "dimensionless"),
    )
    return replace(
        distribution,
        distribution_parameters=tuple(
            (name, InputValue(value, unit, None, "Input uncertainty not supplied"))
            for name, value, unit in parameters
        ),
    )
