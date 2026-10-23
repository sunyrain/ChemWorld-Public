"""Sourced material roughness and smooth-pipe observations, in original tables.

Catalog numbers remain caller-owned ScientificTable data. No handbook charts
or original experimental observations are silently treated as native fits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import get_close_matches
from math import isfinite
from typing import Any, cast

from chemworld.data.scientific_inputs import ScientificTable
from chemworld.foundation.units import convert_value
from chemworld.physchem.flow_meters import _positive


def farshad_pipe_roughness(diameter_m: float, coefficient: float, exponent: float) -> float:
    """epsilon/inch = A*(D/inch)^(B+1), with the fit's explicit inch convention."""
    _positive(diameter_m, "diameter")
    _positive(coefficient, "Farshad coefficient")
    if not isfinite(exponent):
        raise ValueError("Farshad exponent must be finite")
    result = 0.0254 * coefficient * (diameter_m / 0.0254) ** (exponent + 1)
    _positive(result, "Farshad roughness")
    if result >= diameter_m:
        raise ValueError(
            "Farshad roughness reaches pipe diameter; microchannel extrapolation invalid"
        )
    return result


@dataclass(frozen=True)
class MaterialRoughnessResult:
    material: str
    roughness_m: float
    selected_statistic: str
    source_id: str
    fuzzy_match_used: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PipeRoughnessCatalog:
    """Consume material rows without adding another parameter storage format.

    Roughness table: group_id=material, unique sample_id ending :min/:max/:average,
    one parameter column ``roughness`` with length units. Missing statistics
    are absent rows, not zero observations. Optional Farshad table has
    group_id=material and columns ``average_roughness``, ``coefficient``,
    ``exponent``. Coefficients use the explicit inch-normalized fit above.
    """

    roughness_table: ScientificTable
    farshad_table: ScientificTable | None = None

    def __post_init__(self) -> None:
        if self.roughness_table.kind != "parameters":
            raise ValueError("roughness catalog requires a parameter table")
        names = {column.quantity_id for column in self.roughness_table.columns}
        if names != {"roughness"}:
            raise ValueError("roughness table requires exactly its roughness column")
        seen = set()
        for row in self.roughness_table.rows:
            statistic = row.sample_id.rsplit(":", 1)[-1]
            identity = (row.group_id, statistic)
            if identity in seen or statistic not in {"min", "max", "average"}:
                raise ValueError("one declared min/max/average row per material")
            seen.add(identity)
            value = row.values["roughness"]
            converted = convert_value(cast(float, value.value), value.unit, "m")
            if not isfinite(converted) or converted < 0:
                raise ValueError("roughness must be finite and nonnegative")
        if self.farshad_table is not None:
            if self.farshad_table.kind != "parameters" or {
                column.quantity_id for column in self.farshad_table.columns
            } != {"average_roughness", "coefficient", "exponent"}:
                raise ValueError(
                    "Farshad parameter table requires average_roughness/coefficient/exponent"
                )
            materials = [row.group_id for row in self.farshad_table.rows]
            if len(set(materials)) != len(materials):
                raise ValueError("one Farshad coefficient set per material")
            for row in self.farshad_table.rows:
                for name, unit in (
                    ("average_roughness", "m"),
                    ("coefficient", "dimensionless"),
                    ("exponent", "dimensionless"),
                ):
                    value = row.values[name]
                    converted = convert_value(cast(float, value.value), value.unit, unit)
                    if name != "exponent":
                        _positive(converted, name)

    @property
    def materials(self) -> tuple[str, ...]:
        names = {row.group_id for row in self.roughness_table.rows}
        if self.farshad_table is not None:
            names.update(row.group_id for row in self.farshad_table.rows)
        return tuple(sorted(names))

    def nearest_materials(self, query: str, *, count: int = 3) -> tuple[str, ...]:
        """Deterministic suggestions; a query never silently changes material."""
        if not query.strip() or count < 1:
            raise ValueError("provide a nonblank query and positive match count")
        folded = {name.casefold(): name for name in self.materials}
        return tuple(
            folded[name]
            for name in get_close_matches(query.casefold(), folded, n=count, cutoff=0.0)
        )

    def evaluate(
        self,
        material: str,
        *,
        diameter_m: float | None = None,
        optimism: bool | None = None,
        allow_fuzzy_match: bool = False,
    ) -> MaterialRoughnessResult:
        if diameter_m is not None:
            _positive(diameter_m, "diameter")
        fuzzy = material not in self.materials
        if fuzzy:
            if not allow_fuzzy_match:
                raise ValueError(
                    f"unknown exact material; suggestions={self.nearest_materials(material)}"
                )
            material = self.nearest_materials(material, count=1)[0]
        if self.farshad_table is not None:
            selected = next(
                (row for row in self.farshad_table.rows if row.group_id == material), None
            )
            if selected is not None:
                values = selected.values
                if diameter_m is None:
                    cell = values["average_roughness"]
                    value = convert_value(cast(float, cell.value), cell.unit, "m")
                    statistic = "average"
                else:
                    coefficient, exponent = values["coefficient"], values["exponent"]
                    value = farshad_pipe_roughness(
                        diameter_m,
                        convert_value(
                            cast(float, coefficient.value), coefficient.unit, "dimensionless"
                        ),
                        convert_value(cast(float, exponent.value), exponent.unit, "dimensionless"),
                    )
                    statistic = "Farshad diameter fit"
                return MaterialRoughnessResult(
                    material, value, statistic, selected.source_id, fuzzy
                )
        rows = {
            row.sample_id.rsplit(":", 1)[-1]: row
            for row in self.roughness_table.rows
            if row.group_id == material
        }
        order = (
            ("average", "max", "min")
            if optimism is None
            else ("min", "average", "max")
            if optimism
            else ("max", "average", "min")
        )
        statistic = next(name for name in order if name in rows)
        row = rows[statistic]
        cell = row.values["roughness"]
        result = convert_value(cast(float, cell.value), cell.unit, "m")
        if diameter_m is not None and result >= diameter_m:
            raise ValueError(
                "roughness exceeds pipe diameter; selected material geometry incompatible"
            )
        return MaterialRoughnessResult(material, result, statistic, row.source_id, fuzzy)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roughness_table": self.roughness_table.to_dict(),
            "farshad_table": self.farshad_table.to_dict()
            if self.farshad_table is not None
            else None,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PipeRoughnessCatalog:
        return cls(
            ScientificTable.from_dict(value["roughness_table"]),
            ScientificTable.from_dict(value["farshad_table"])
            if value["farshad_table"] is not None
            else None,
        )


def smooth_pipe_observation_pairs(table: ScientificTable) -> tuple[tuple[float, float], ...]:
    """Read all measured (Re, Darcy factor) pairs, preserving ordering and metadata.

    The original table remains the evidence object. This accessor neither fits
    its transitional observations nor substitutes a smooth model curve.
    """
    if table.kind != "measurements" or {column.quantity_id for column in table.columns} != {
        "reynolds",
        "darcy_factor",
    }:
        raise ValueError("smooth-pipe observations require reynolds/darcy_factor measurement table")
    result = []
    for row in table.rows:
        values = []
        for name in ("reynolds", "darcy_factor"):
            cell = row.values[name]
            if cell.value is None or cell.quality != "valid":
                raise ValueError(
                    "all observation pairs must be explicit valid values; no rows dropped"
                )
            value = convert_value(cell.value, cell.unit, "dimensionless")
            _positive(value, name)
            values.append(value)
        result.append((values[0], values[1]))
    return tuple(result)
