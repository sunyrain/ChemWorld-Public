"""Explicit uniform-channel and free-weir engineering relations in SI.

Source: fluids 1.3.1 public documentation, Chow/Blevins/Shen formula families.
The two disputed SIA source expressions require an explicit reference-only
request. They do not become qualified flow measurement laws by being reproduced.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite, radians, sqrt, tan
from typing import Any

from chemworld.data.scientific_inputs import (
    InputSource,
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.physchem.transport import GRAVITY_M_S2

WEIR_METHODS = (
    "shen_v", "kindsvater_carter", "sia_contracted_reference", "ackers_full",
    "kindsvater_carter_full", "rehbock_full", "sia_full_reference",
)


def _positive(value: float, label: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")


def _slope(value: float) -> None:
    if isinstance(value, bool) or not isfinite(value) or value < 0:
        raise ValueError("uniform flow needs a finite nonnegative energy slope")


def manning_velocity(hydraulic_radius_m: float, slope: float, roughness: float) -> float:
    """Mean velocity [m/s]; n has units s/m^(1/3), with no US-unit factor."""
    _positive(hydraulic_radius_m, "hydraulic radius")
    _positive(roughness, "Manning roughness")
    _slope(slope)
    return hydraulic_radius_m**(2/3) * sqrt(slope) / roughness


def chezy_velocity(hydraulic_radius_m: float, slope: float, coefficient: float) -> float:
    _positive(hydraulic_radius_m, "hydraulic radius")
    _positive(coefficient, "Chezy coefficient")
    _slope(slope)
    return coefficient * sqrt(slope * hydraulic_radius_m)


def manning_to_chezy(roughness: float, hydraulic_radius_m: float) -> float:
    _positive(hydraulic_radius_m, "hydraulic radius")
    _positive(roughness, "Manning roughness")
    return hydraulic_radius_m**(1/6) / roughness


def chezy_to_manning(coefficient: float, hydraulic_radius_m: float) -> float:
    _positive(hydraulic_radius_m, "hydraulic radius")
    _positive(coefficient, "Chezy coefficient")
    return hydraulic_radius_m**(1/6) / coefficient


def uniform_channel_flow(
    area_m2: float, wetted_perimeter_m: float, slope: float, *, roughness: float,
) -> float:
    """Steady volume flow [m3/s] from explicit area/perimeter and SI Manning n."""
    _positive(area_m2, "cross-sectional area")
    _positive(wetted_perimeter_m, "wetted perimeter")
    return area_m2 * manning_velocity(area_m2/wetted_perimeter_m, slope, roughness)


@dataclass(frozen=True)
class WeirFlowReport:
    method: str
    flow_m3_s: float
    reference_only: bool
    warnings: tuple[str, ...]
    source: str = "fluids 1.3.1 / cited Chow, Blevins and Shen formula families"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def weir_flow(
    method: str, *, head_m: float, crest_height_m: float, width_m: float,
    channel_width_m: float | None = None, angle_degrees: float = 90,
    reference_only: bool = False,
) -> WeirFlowReport:
    """Free-discharge weirs within the declared geometric applicability ranges.

For a V notch width_m is the approach-channel width. Rectangular contracted
methods require channel_width_m separately. Full-width methods use width_m.
No submerged-flow, approach-velocity or installation correction is inferred.
"""
    if method not in WEIR_METHODS:
        raise ValueError("unknown weir method")
    if not isinstance(reference_only, bool):
        raise ValueError("reference_only must be boolean")
    h, p, b = head_m, crest_height_m, width_m
    for value, label in ((h, "head"), (p, "crest height"), (b, "width")):
        _positive(value, label)
    disputed = method in {"sia_contracted_reference", "sia_full_reference"}
    if disputed and not reference_only:
        raise ValueError("disputed SIA expression requires explicit reference_only=True")
    warnings: tuple[str, ...] = ()
    gravity = sqrt(GRAVITY_M_S2)
    if method == "shen_v":
        if not isfinite(angle_degrees) or not 20 <= angle_degrees <= 100:
            raise ValueError("Shen angle must lie within 20..100 degrees")
        if not (h >= .05 and p > .45 and h/p <= .4 and b > .9
                and h/b*tan(radians(angle_degrees)/2) < 2):
            raise ValueError("Shen weir geometry outside published domain")
        points = ((20., .59, .0028), (40., .58, .0017), (60., .575, .0012),
                  (80., .575, .001), (100., .58, .001))
        i = min(int((angle_degrees-20)//20), 3)
        lo, hi = points[i], points[i+1]
        fraction = (angle_degrees-lo[0])/20
        c, k = (lo[j]+fraction*(hi[j]-lo[j]) for j in (1, 2))
        q = c*tan(radians(angle_degrees)/2)*gravity*(h+k)**2.5
    else:
        if method == "ackers_full":
            valid = h > .02 and p > .15 and h/p <= 2.2
        elif method in {"rehbock_full", "sia_full_reference"}:
            lower, upper = (.03, .75) if method == "rehbock_full" else (.025, .8)
            valid = lower < h < upper and b > .3 and p > .3 and h/p < 1
        else:
            valid = h > .03 and p > .1 and b > .15 and h/p < 2
        if not valid:
            raise ValueError("rectangular weir geometry outside published domain")
        ratio = 1.0
        if method in {"kindsvater_carter", "sia_contracted_reference"}:
            if channel_width_m is None:
                raise ValueError("contracted weir requires approach channel width")
            _positive(channel_width_m, "approach channel width")
            ratio = b/channel_width_m
            if not .2 <= ratio < 1:
                raise ValueError("contracted width ratio must lie in [0.2, 1)")
        elif channel_width_m is not None and channel_width_m != b:
            raise ValueError("full-width weir must span the declared approach channel")
        if method == "kindsvater_carter":
            q = .554*(1-.0035*h/p)*(b+.0025)*gravity*(h+.0001)**1.5
        elif method == "ackers_full":
            q = .564*(1+.150*h/p)*b*gravity*(h+.001)**1.5
        elif method == "kindsvater_carter_full":
            q = 2*sqrt(2)/3*(.602+.075*h/p)*(b-.001)*gravity*(h+.001)**1.5
        elif method == "rehbock_full":
            q = 2*sqrt(2)/3*(.602+.0832*h/p)*b*gravity*(h+.00125)**1.5
        elif method == "sia_contracted_reference":
            q = (.544*(1+.064*ratio**2+(.00626-.00519*ratio**2)/(h+.0016))
                 *(1+.5*ratio**4*(h/(h+p))**2)*b*gravity*(h+p)**1.5)
            warnings = ("Source uses total height h+p; free-weir physical validity unqualified",)
        else:
            q = (2*sqrt(2)/3*(.615+.000615/(h+.0016))*b*gravity*h
                 +.5*(h/(h+p))**2*b*gravity*h**1.5)
            warnings = ("Source warns DO NOT USE; dimensionally inconsistent empirical expression",)
    return WeirFlowReport(method, q, disputed, warnings)


def manning_roughness_table(
    table_id: str, catalog: Mapping[str, Any], *, source: InputSource,
) -> ScientificTable:
    """Import a nested explicit source of (minimum, normal, maximum) n values.

    These three engineering statistics are not a standard uncertainty. Original
    paths are JSON row identities; source tables are not shipped in the runtime.
    """
    rows: list[ScientificRow] = []
    names = ("minimum", "normal", "maximum")

    def walk(data: Any, path: tuple[str, ...]) -> None:
        if isinstance(data, Mapping):
            if not data:
                raise ValueError("roughness source contains an empty category")
            for key, item in data.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("roughness source category must be named")
                walk(item, (*path, key))
            return
        if not path or not isinstance(data, (list, tuple)) or len(data) != 3:
            raise ValueError("roughness leaf requires minimum, normal and maximum")
        for value in data:
            _positive(value, "roughness source value")
        if not data[0] <= data[1] <= data[2]:
            raise ValueError("roughness statistics must be ordered")
        identity = json.dumps(path, ensure_ascii=False)
        rows.append(ScientificRow(identity, source.provenance.source_id, table_id, identity,
            "parameters", {name: InputValue(value, "s/m^(1/3)", None,
                "Published engineering statistic; no standard uncertainty supplied")
                for name, value in zip(names, data, strict=True)}))

    walk(catalog, ())
    columns = tuple(QuantityColumn(name, "s/m^(1/3)", "parameter", (0, 1e6),
        "SI Manning roughness statistic") for name in names)
    return ScientificTable(table_id, "parameters", columns, (source,), tuple(rows))


def select_manning_roughness(
    table: ScientificTable, path: tuple[str, ...], *, statistic: str = "normal",
) -> InputValue:
    if statistic not in {"minimum", "normal", "maximum"}:
        raise ValueError("unknown roughness statistic")
    identity = json.dumps(path, ensure_ascii=False)
    rows = [row for row in table.rows if row.row_id == identity]
    if len(rows) != 1:
        raise ValueError("roughness source path must resolve exactly once")
    value = rows[0].values[statistic].to("s/m^(1/3)")
    if value.quality != "valid" or value.value is None:
        raise ValueError("roughness source value is unavailable")
    _positive(value.value, "roughness source value")
    return value
