"""Caller-supplied sieve standards and series, retaining original source cells."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from chemworld.data.scientific_inputs import InputValue, ScientificTable


@dataclass(frozen=True)
class SieveSpecification:
    standard: str
    designation: str
    row_id: str
    source_id: str
    values: Mapping[str, InputValue]
    aliases: tuple[str, ...] = ()

    @property
    def opening_m(self) -> float:
        return self.values["opening"].to("m").value


class SieveCatalog:
    """One common table; series and aliases reference its existing row IDs.

    row.group_id identifies a standard; sample_id is 'standard:designation'.
    Opening is a positive length. Optional compliance fields retain caller
    units and missing-quality cells: their presence is not certification.
    Standards observations/arrays are not redistributed by this module.
    """

    def __init__(
        self,
        table: ScientificTable,
        *,
        series: Mapping[str, Sequence[str]] | None = None,
        aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.table = table
        self.rows = {row.row_id: row for row in table.rows}
        self.series = {name: tuple(ids) for name, ids in (series or {}).items()}
        self.aliases = {row_id: tuple(names) for row_id, names in (aliases or {}).items()}
        identities = set()
        for row in table.rows:
            prefix = row.group_id + ":"
            if not row.sample_id.startswith(prefix) or len(row.sample_id) == len(prefix):
                raise ValueError("sieve sample identity must be standard:designation")
            opening = row.values["opening"].to("m")
            if opening.quality != "valid" or opening.value <= 0:
                raise ValueError("sieve opening must be a valid positive length")
            identity = (row.group_id, row.sample_id)
            if identity in identities:
                raise ValueError("duplicate sieve designation")
            identities.add(identity)
        for ids in self.series.values():
            if len(set(ids)) != len(ids) or any(row_id not in self.rows for row_id in ids):
                raise ValueError("sieve series must reference unique existing rows")
        if any(row_id not in self.rows for row_id in self.aliases):
            raise ValueError("sieve alias references an unknown row")

    def by_row_id(self, row_id: str) -> SieveSpecification:
        row = self.rows[row_id]
        return SieveSpecification(
            row.group_id,
            row.sample_id[len(row.group_id) + 1 :],
            row_id,
            row.source_id,
            row.values,
            self.aliases.get(row_id, ()),
        )

    def lookup(self, standard: str, designation: str) -> SieveSpecification:
        matches = [
            self.by_row_id(row.row_id)
            for row in self.table.rows
            if row.group_id == standard
            and (
                row.sample_id == standard + ":" + designation
                or designation in self.aliases.get(row.row_id, ())
            )
        ]
        if len(matches) != 1:
            raise ValueError("unknown or ambiguous sieve designation within this source catalog")
        return matches[0]

    def spacing(
        self, series_or_standard: str, *, lower_m: float, upper_m: float
    ) -> tuple[float, ...]:
        if not 0 <= lower_m < upper_m:
            raise ValueError("increasing nonnegative sieve bounds required")
        ids = self.series.get(series_or_standard)
        if ids is None:
            ids = tuple(row.row_id for row in self.table.rows if row.group_id == series_or_standard)
        if not ids:
            raise ValueError("unknown or empty sieve series")
        return tuple(
            sorted(
                self.by_row_id(row_id).opening_m
                for row_id in ids
                if lower_m <= self.by_row_id(row_id).opening_m <= upper_m
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {"table": self.table.to_dict(), "series": self.series, "aliases": self.aliases}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SieveCatalog":
        return cls(
            ScientificTable.from_dict(value["table"]),
            series=value["series"],
            aliases=value["aliases"],
        )
