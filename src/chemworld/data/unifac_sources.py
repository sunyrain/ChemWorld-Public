"""Source-bearing UNIFAC tables and DDBST assignments without reference imports.

Input paths and revisions are explicit. Parsing is inert; a selected assignment
and selected directed interactions are compiled by the existing model builder.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.identifiers import cas_to_int, int_to_cas
from chemworld.physchem.unifac import UNIFACSubgroup, unifac_activity_model

ASSIGNMENT_METHODS = ("UNIFAC", "MODIFIED_UNIFAC", "PSRK")

if TYPE_CHECKING:
    from chemworld.physchem.equilibrium import ActivityModelSpec


@dataclass(frozen=True)
class UNIFACInteractionRecord:
    left: int
    right: int
    coefficients: tuple[float, float, float]
    source: DataSourceProvenance
    row: int
    temperature_range_K: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (self.left, self.right, self.row)
        ):
            raise ValueError("UNIFAC group ids and source row must be positive integers")
        if len(self.coefficients) != 3 or any(
            isinstance(v, bool) or not isfinite(v) for v in self.coefficients
        ):
            raise ValueError("UNIFAC interaction needs three finite coefficients")
        if self.temperature_range_K is not None:
            lower, upper = self.temperature_range_K
            if not isfinite(lower) or not isfinite(upper) or lower <= 0 or upper < lower:
                raise ValueError("UNIFAC source temperature range must be positive and ordered")

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "coefficients": list(self.coefficients),
            "source": self.source.to_dict(),
            "row": self.row,
            "temperature_range_K": None
            if self.temperature_range_K is None
            else list(self.temperature_range_K),
        }


def read_unifac_interactions(
    path: str | Path, *, source: DataSourceProvenance
) -> tuple[UNIFACInteractionRecord, ...]:
    """Read original, modified, or NIST temperature-range headerless TSV."""
    records = []
    seen = set()
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(fields) not in {3, 5, 7}:
                raise ValueError(f"UNIFAC source row {row} needs 3, 5 or 7 fields")
            left, right = map(int, fields[:2])
            values = tuple(map(float, fields[2:]))
            coefficients = (
                (values[0], 0.0, 0.0) if len(values) == 1 else (values[0], values[1], values[2])
            )
            if (left, right) in seen:
                raise ValueError(f"duplicate directed interaction in one source at row {row}")
            seen.add((left, right))
            temperature_range = (values[3], values[4]) if len(values) == 5 else None
            records.append(
                UNIFACInteractionRecord(left, right, coefficients, source, row, temperature_range)
            )
    if not records:
        raise ValueError("UNIFAC interaction source is empty")
    return tuple(records)


def select_unifac_interactions(
    revisions: Sequence[Sequence[UNIFACInteractionRecord]],
    *,
    later_revisions_override: bool = False,
) -> tuple[dict[tuple[int, int], tuple[float, float, float]], dict[str, Any]]:
    """Explicit ordered revision selection, retaining every overridden record."""
    chosen: dict[tuple[int, int], UNIFACInteractionRecord] = {}
    superseded = []
    for revision in revisions:
        for record in revision:
            pair = record.left, record.right
            if pair in chosen:
                if not later_revisions_override:
                    raise ValueError(
                        f"multiple UNIFAC sources for {pair}; select revision precedence"
                    )
                superseded.append(chosen[pair].to_dict())
            chosen[pair] = record
    return (
        {key: record.coefficients for key, record in chosen.items()},
        {
            "selection": "later explicit revision overrides"
            if later_revisions_override
            else "no overlapping source pairs",
            "selected": [record.to_dict() for record in chosen.values()],
            "superseded": superseded,
        },
    )


def read_unifac_subgroups(
    path: str | Path,
) -> tuple[dict[int, UNIFACSubgroup], DataSourceProvenance]:
    """Read a portable source card: {source: provenance, subgroups: [records]}."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"source", "subgroups"}:
        raise ValueError("UNIFAC subgroup card requires source and subgroups")
    source = DataSourceProvenance.from_dict(payload["source"])
    groups = {}
    for item in payload["subgroups"]:
        group = UNIFACSubgroup(**item)
        if group.group_id in groups:
            raise ValueError(f"duplicate UNIFAC subgroup {group.group_id}")
        groups[group.group_id] = group
    if not groups:
        raise ValueError("UNIFAC subgroup source cannot be empty")
    return groups, source


@dataclass(frozen=True)
class MolecularGroupAssignment:
    identity: str
    identity_kind: str
    method: str
    groups: Mapping[int, int]
    valid: bool
    source: DataSourceProvenance
    row: int

    def __post_init__(self) -> None:
        if not self.identity or self.identity_kind not in {"CAS", "InChIKey"}:
            raise ValueError("group assignments require an explicit CAS or InChIKey identity")
        if self.identity_kind == "CAS":
            cas_to_int(self.identity)
        if self.method not in ASSIGNMENT_METHODS or not isinstance(self.valid, bool):
            raise ValueError("unsupported group assignment method or validity flag")
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for pair in self.groups.items()
            for v in pair
        ):
            raise ValueError("molecular subgroup ids and counts must be integers")
        if any(count <= 0 or (self.valid and key <= 0) for key, count in self.groups.items()):
            raise ValueError("valid subgroup ids and all counts must be positive integers")
        if self.valid and not self.groups:
            raise ValueError("a valid molecular assignment cannot be empty")
        object.__setattr__(self, "groups", MappingProxyType(dict(self.groups)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "identity_kind": self.identity_kind,
            "method": self.method,
            "groups": {str(key): value for key, value in self.groups.items()},
            "valid": self.valid,
            "source": self.source.to_dict(),
            "row": self.row,
        }


def _group_pairs(fields: Sequence[str]) -> dict[int, int]:
    if len(fields) % 2:
        raise ValueError("molecular assignment requires subgroup/count pairs")
    pairs = [(int(fields[i]), int(fields[i + 1])) for i in range(0, len(fields), 2)]
    if len(dict(pairs)) != len(pairs):
        raise ValueError("duplicate subgroup within molecular assignment")
    return dict(pairs)


def iter_ddbst_text(
    path: str | Path, *, source: DataSourceProvenance
) -> Iterator[tuple[MolecularGroupAssignment, ...]]:
    """Preserve all three groupings and source validity flags for every row."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for row, fields in enumerate(csv.reader(stream, delimiter="\t"), 1):
            if len(fields) != 5:
                raise ValueError(f"DDBST row {row} needs five fields")
            flags = fields[1].split()
            if len(flags) != 3 or any(flag not in {"0", "1"} for flag in flags):
                raise ValueError(f"invalid DDBST validity flags at row {row}")
            yield tuple(
                MolecularGroupAssignment(
                    fields[0],
                    "InChIKey",
                    method,
                    _group_pairs(groups.split()),
                    flag == "1",
                    source,
                    row,
                )
                for method, flag, groups in zip(ASSIGNMENT_METHODS, flags, fields[2:], strict=True)
            )


def iter_ddbst_sqlite(
    path: str | Path, *, source: DataSourceProvenance, cas_number: str | None = None
) -> Iterator[tuple[MolecularGroupAssignment, ...]]:
    """Read-only original DDBST CAS lookup, with no persistent database copy."""
    path = Path(path).resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        sql = 'SELECT "index", UNIFAC, MODIFIED_UNIFAC, PSRK FROM DDBST'
        arguments = () if cas_number is None else (cas_to_int(cas_number),)
        sql += ' ORDER BY "index"' if cas_number is None else ' WHERE "index" = ?'
        for fields in connection.execute(sql, arguments):
            identity = int_to_cas(fields[0])
            yield tuple(
                MolecularGroupAssignment(
                    identity,
                    "CAS",
                    method,
                    _group_pairs(text.replace(":", ",").split(",")) if text else {},
                    bool(text),
                    source,
                    int(fields[0]),
                )
                for method, text in zip(ASSIGNMENT_METHODS, fields[1:], strict=True)
            )


def unifac_model_from_assignments(
    model_id: str,
    assignments: Mapping[str, MolecularGroupAssignment],
    *,
    subgroups: Mapping[int, UNIFACSubgroup],
    subgroup_source: DataSourceProvenance,
    revisions: Sequence[Sequence[UNIFACInteractionRecord]],
    version: int,
    later_revisions_override: bool = False,
) -> ActivityModelSpec:
    """Consume explicit identities and source selections through the original GE."""
    if not assignments or any(not record.valid for record in assignments.values()):
        raise ValueError("UNIFAC compilation requires valid molecular group assignments")
    if version not in {0, 1, 2}:
        raise ValueError(
            "DDBST assignments cover only original, modified UNIFAC and PSRK; "
            "other variants require explicit matching group declarations"
        )
    methods = {record.method for record in assignments.values()}
    required = "MODIFIED_UNIFAC" if version == 1 else "PSRK" if version == 2 else "UNIFAC"
    if methods != {required}:
        raise ValueError(f"UNIFAC version {version} requires {required} assignments")
    used_groups = {group for record in assignments.values() for group in record.groups}
    if not used_groups <= subgroups.keys():
        raise ValueError("molecular assignments reference missing UNIFAC subgroups")
    mains = {subgroups[group].main_group_id for group in used_groups}
    consumed = tuple(
        tuple(
            record
            for record in revision
            if record.left in mains and record.right in mains and record.left != record.right
        )
        for revision in revisions
    )
    interactions, selection = select_unifac_interactions(
        consumed, later_revisions_override=later_revisions_override
    )
    return unifac_activity_model(
        model_id,
        {key: record.groups for key, record in assignments.items()},
        subgroups=subgroups,
        interactions=interactions,
        version=version,
        provenance={
            "subgroup_source": subgroup_source.to_dict(),
            "interaction_selection": selection,
            "molecular_sources": {key: record.to_dict() for key, record in assignments.items()},
        },
    )
