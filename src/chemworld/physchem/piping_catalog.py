"""Caller-owned pipe and wire schedules with explicit dimensional provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isclose, isfinite, sqrt
from typing import Any, cast

from chemworld.data.scientific_inputs import ScientificRow, ScientificTable
from chemworld.physchem.flow_meters import _positive
from chemworld.physchem.transport import FluidState, PipeSpec


def _cell(row: ScientificRow, key: str, unit: str) -> float:
    value = cast(float, row.values[key].to(unit).value)
    if isinstance(value, bool) or not isfinite(value):
        raise ValueError(f"numeric {key} required")
    return value


@dataclass(frozen=True)
class PipeDimensions:
    schedule: str
    nominal_size: float
    inner_diameter_m: float
    outer_diameter_m: float
    wall_thickness_m: float
    source_id: str
    row_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pipe_spec(
        self,
        *,
        length_m: float,
        roughness_m: float = 0.0,
        elevation_change_m: float = 0.0,
        fittings_loss_coefficient: float = 0.0,
    ) -> PipeSpec:
        """Use the selected internal bore in the existing actual flow consumer."""
        return PipeSpec(
            length_m,
            self.inner_diameter_m,
            roughness_m,
            elevation_change_m,
            fittings_loss_coefficient,
        )


@dataclass(frozen=True)
class PipeScheduleCatalog:
    """Rows group by schedule, columns nominal_size/inner/outer/wall.

    Nominal size is an identifier, not a measured length. Source dimensions
    are explicit and checked for closure, never inferred from schedule names.
    """

    table: ScientificTable

    def __post_init__(self) -> None:
        if self.table.kind != "parameters" or {c.quantity_id for c in self.table.columns} != {
            "nominal_size",
            "inner_diameter",
            "outer_diameter",
            "wall_thickness",
        }:
            raise ValueError("pipe catalog requires the four explicit parameter columns")
        seen = set()
        for row in self.table.rows:
            record = self._record(row)
            key = record.schedule, record.nominal_size
            if key in seen:
                raise ValueError("duplicate schedule/nominal size")
            seen.add(key)
            for name in (
                "nominal_size",
                "inner_diameter_m",
                "outer_diameter_m",
                "wall_thickness_m",
            ):
                _positive(getattr(record, name), name)
            if not isclose(
                record.outer_diameter_m - record.inner_diameter_m,
                2 * record.wall_thickness_m,
                rel_tol=2e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("published pipe inner/outer/wall dimensions do not close")

    @staticmethod
    def _record(row: ScientificRow) -> PipeDimensions:
        return PipeDimensions(
            row.group_id,
            _cell(row, "nominal_size", "dimensionless"),
            _cell(row, "inner_diameter", "m"),
            _cell(row, "outer_diameter", "m"),
            _cell(row, "wall_thickness", "m"),
            row.source_id,
            row.row_id,
        )

    @property
    def schedules(self) -> tuple[str, ...]:
        return tuple(sorted({r.group_id for r in self.table.rows}))

    def select(
        self,
        schedule: str,
        *,
        nominal_size: float | None = None,
        minimum_inner_diameter_m: float | None = None,
        minimum_outer_diameter_m: float | None = None,
    ) -> PipeDimensions:
        selectors = (nominal_size, minimum_inner_diameter_m, minimum_outer_diameter_m)
        if sum(v is not None for v in selectors) != 1:
            raise ValueError("select a pipe by exactly one nominal size or minimum diameter")
        records = [self._record(r) for r in self.table.rows if r.group_id == schedule]
        if not records:
            raise ValueError("pipe schedule not available in declared catalog")
        if nominal_size is not None:
            _positive(nominal_size, "nominal size")
            matches = [r for r in records if r.nominal_size == nominal_size]
        else:
            field = (
                "inner_diameter_m" if minimum_inner_diameter_m is not None else "outer_diameter_m"
            )
            minimum = cast(
                float,
                minimum_inner_diameter_m
                if minimum_inner_diameter_m is not None
                else minimum_outer_diameter_m,
            )
            _positive(minimum, "minimum diameter")
            matches = sorted(
                (r for r in records if getattr(r, field) >= minimum),
                key=lambda r: (getattr(r, field), r.nominal_size),
            )
        if not matches:
            raise ValueError("pipe size outside the declared schedule")
        return matches[0]

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PipeScheduleCatalog:
        return cls(ScientificTable.from_dict(value["table"]))


@dataclass(frozen=True)
class WireGaugeSelection:
    schedule: str
    gauge: float
    thickness_m: float
    nominal_thickness_m: float
    source_id: str
    row_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WireGaugeCatalog:
    """Gauge identifiers with both published SI and nominal thickness records.

    The BWG source rounds its SI column independently of the nominal inch
    column. Both remain explicit source values; selection uses nominal size.
    Caller may provide equal values when only one dimensional convention exists.
    """

    table: ScientificTable

    def __post_init__(self) -> None:
        if self.table.kind != "parameters" or {c.quantity_id for c in self.table.columns} != {
            "gauge",
            "thickness",
            "nominal_thickness",
        }:
            raise ValueError("wire catalog requires gauge/thickness/nominal_thickness")
        identities, sizes = set(), set()
        for row in self.table.rows:
            record = self._record(row)
            for value, name in (
                (record.thickness_m, "thickness"),
                (record.nominal_thickness_m, "nominal thickness"),
            ):
                _positive(value, name)
            if (record.schedule, record.gauge) in identities or (
                record.schedule,
                record.nominal_thickness_m,
            ) in sizes:
                raise ValueError("wire schedule gauge and nominal thickness must be unique")
            identities.add((record.schedule, record.gauge))
            sizes.add((record.schedule, record.nominal_thickness_m))

    @staticmethod
    def _record(row: ScientificRow) -> WireGaugeSelection:
        return WireGaugeSelection(
            row.group_id,
            _cell(row, "gauge", "dimensionless"),
            _cell(row, "thickness", "m"),
            _cell(row, "nominal_thickness", "m"),
            row.source_id,
            row.row_id,
        )

    @property
    def schedules(self) -> tuple[str, ...]:
        return tuple(sorted({r.group_id for r in self.table.rows}))

    def from_gauge(self, schedule: str, gauge: float) -> WireGaugeSelection:
        if isinstance(gauge, bool) or not isfinite(gauge):
            raise ValueError("gauge identifier must be finite")
        for row in self.table.rows:
            if row.group_id == schedule and _cell(row, "gauge", "dimensionless") == gauge:
                return self._record(row)
        raise ValueError("gauge not found in declared schedule")

    def from_thickness(
        self,
        schedule: str,
        thickness_m: float,
        *,
        undersize_gap_fraction: float = 0.1,
    ) -> WireGaugeSelection:
        """Select the next larger wire unless within a declared fraction of the gap.

        All comparisons are physical SI lengths. Unlike the source helper,
        below-minimum input rejects instead of returning a length as a gauge.
        """
        _positive(thickness_m, "thickness")
        if not isfinite(undersize_gap_fraction) or not 0 <= undersize_gap_fraction <= 1:
            raise ValueError("undersize gap fraction must be in [0,1]")
        records = sorted(
            (self._record(r) for r in self.table.rows if r.group_id == schedule),
            key=lambda r: r.nominal_thickness_m,
        )
        if not records:
            raise ValueError("wire schedule unavailable")
        for record in records:
            if isclose(record.nominal_thickness_m, thickness_m, rel_tol=2e-15, abs_tol=0):
                return record
        if not records[0].nominal_thickness_m <= thickness_m <= records[-1].nominal_thickness_m:
            raise ValueError("thickness outside wire schedule")
        for smaller, larger in pairwise(records):
            lo, hi = smaller.nominal_thickness_m, larger.nominal_thickness_m
            if lo < thickness_m < hi:
                return smaller if thickness_m - lo <= undersize_gap_fraction * (hi - lo) else larger
        raise ValueError("no wire interval contains requested thickness")

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict()}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WireGaugeCatalog:
        return cls(ScientificTable.from_dict(value["table"]))


def piping_erosional_velocity(fluid: FluidState, *, empirical_factor_imperial: float) -> float:
    """API RP14E speed in m/s; explicit source C in sqrt(lb/(ft*s²)).

    This is an engineering screen, not an erosion/corrosion rate or a solids
    qualification. C must be supplied for the actual service.
    """
    _positive(empirical_factor_imperial, "API erosional factor")
    density_lb_ft3 = fluid.density_kg_m3 / (0.45359237 / 0.3048**3)
    return 0.3048 * empirical_factor_imperial / sqrt(density_lb_ft3)
