"""Native, lazy readers for explicitly supplied chemical identifier data.

Source records are evidence, not simulator species. Conversion uses the
existing ComponentSpec/MixtureSpec contracts; no reference package is imported.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Integral
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypedDict, Unpack, cast, overload

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.elements import ElementSpec, get_element, serialize_formula
from chemworld.physchem.identifiers import check_cas_number, int_to_cas, validate_cas_number
from chemworld.physchem.specs import (
    ComponentFieldCandidate,
    ComponentProvenance,
    ComponentSpec,
    MixtureSpec,
)

IDENTIFIER_DATA_VERSION = "chemicals-identifiers-1.5.2"


class IdentifierSourceUnavailableError(FileNotFoundError):
    """A specifically requested external source is absent."""


class IdentifierSourceFormatError(ValueError):
    """A source does not have its declared tabular structure."""


class UnknownIdentifierMethodError(ValueError):
    pass


class UnknownIdentifierError(LookupError):
    pass


class AmbiguousIdentifierError(LookupError):
    def __init__(self, identity: object, candidates: Sequence[IdentifierRecord]) -> None:
        self.candidates = tuple(candidates)
        super().__init__(
            f"ambiguous identifier {identity!r}: {sorted({r.registry_key for r in candidates})}"
        )


class ConflictingIdentifierDataError(ValueError):
    def __init__(
        self, candidates: Sequence[IdentifierRecord], fields: Sequence[str],
    ) -> None:
        self.candidates, self.fields = tuple(candidates), tuple(fields)
        super().__init__(f"conflicting source fields {self.fields}; select a source explicitly")


class MissingIdentifierFieldError(ValueError):
    pass


class IdentifierConversionError(ValueError):
    pass


@dataclass(frozen=True)
class IdentifierSource:
    source_id: str
    filename: str
    kind: str
    priority: int
    version: str = IDENTIFIER_DATA_VERSION
    license: str = "Data redistribution permission not established; external read-only input"

    @property
    def provenance(self) -> DataSourceProvenance:
        return DataSourceProvenance(
            f"chemicals-identifiers:{self.source_id}",
            self.priority,
            f"chemicals 1.5.2 Identifiers/{self.filename}",
            self.version,
            self.license,
        )


IDENTIFIER_SOURCES = (
    IdentifierSource("pubchem-large", "chemical identifiers pubchem large.tsv", "compound", 60),
    IdentifierSource("pubchem-small", "chemical identifiers pubchem small.tsv", "compound", 50),
    IdentifierSource("example-user", "chemical identifiers example user db.tsv", "compound", 40),
    IdentifierSource("anion", "Anion db.tsv", "compound", 30),
    IdentifierSource("cation", "Cation db.tsv", "compound", 30),
    IdentifierSource("inorganic", "Inorganic db.tsv", "compound", 30),
    IdentifierSource("fake-registry", "Fake CAS Registry.tsv", "fake-registry", 100),
    IdentifierSource("mixtures", "Mixtures Compositions.tsv", "mixture", 50),
    IdentifierSource("dippr-2014", "dippr_2014.csv", "set", 50),
    IdentifierSource("anion-preferences", "anion_preferences.json", "preferences", 50),
    IdentifierSource("cation-preferences", "cation_preferences.json", "preferences", 50),
    IdentifierSource("inorganic-preferences", "inorganic_preferences.json", "preferences", 50),
    IdentifierSource("organic-preferences", "organic_preferences.json", "preferences", 50),
)
_SOURCES = {source.source_id: source for source in IDENTIFIER_SOURCES}
_COMPOUND_SOURCES = tuple(s.source_id for s in IDENTIFIER_SOURCES if s.kind == "compound")
_QUERY_KINDS = frozenset(
    {"name", "cas", "source_key", "pubchem", "formula", "smiles", "inchi", "inchikey"}
)
_CONFLICT_FIELDS = ("formula", "molecular_weight_g_mol", "smiles", "inchi", "inchi_key")


def _name(value: str) -> str:
    return " ".join(value.casefold().split())


def _inchi(value: str) -> str:
    return re.sub(r"^InChI=1S?/", "", value, flags=re.IGNORECASE)


def _formula(value: str) -> str:
    try:
        return serialize_formula(value)
    except ValueError:
        # Non-molecular or incomplete source formulas remain exactly searchable.
        return value


def _mass(value: str) -> float | None:
    try:
        answer = float(value)
    except (ValueError, TypeError):
        return None
    return answer if isfinite(answer) and answer > 0 else None


@dataclass(frozen=True)
class IdentifierRecord:
    source_id: str
    row_number: int
    raw_fields: tuple[str, ...]

    @property
    def record_id(self) -> str:
        return f"{self.source_id}:{self.row_number}"

    @property
    def registry_key(self) -> str:
        return self.raw_fields[1]

    @property
    def cas_number(self) -> str | None:
        return self.registry_key if check_cas_number(self.registry_key) else None

    @property
    def pubchem_id(self) -> int | None:
        text = self.raw_fields[0]
        return int(text) if text.isascii() and text.isdigit() else None

    @property
    def formula(self) -> str:
        return self.raw_fields[2]

    @property
    def molecular_weight_g_mol(self) -> float | None:
        return _mass(self.raw_fields[3])

    @property
    def smiles(self) -> str:
        return self.raw_fields[4]

    @property
    def inchi(self) -> str:
        return self.raw_fields[5]

    @property
    def inchi_key(self) -> str:
        return self.raw_fields[6]

    @property
    def iupac_name(self) -> str:
        return self.raw_fields[7]

    @property
    def common_name(self) -> str:
        return self.raw_fields[8]

    @property
    def synonyms(self) -> tuple[str, ...]:
        return self.raw_fields[9:]

    @property
    def source(self) -> IdentifierSource:
        return _SOURCES[self.source_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source.provenance.to_dict(),
            "row_number": self.row_number,
            "raw_fields": list(self.raw_fields),
            "registry_key": self.registry_key,
            "cas_number": self.cas_number,
            "pubchem_id": self.pubchem_id,
            "formula": self.formula,
            "molecular_weight_g_mol": self.molecular_weight_g_mol,
            "smiles": self.smiles,
            "inchi": self.inchi,
            "inchi_key": self.inchi_key,
            "iupac_name": self.iupac_name,
            "common_name": self.common_name,
            "synonyms": list(self.synonyms),
        }

    def to_component(
        self, *, identifier: str | None = None, phase: str = "unknown",
    ) -> ComponentSpec:
        if not self.formula:
            raise MissingIdentifierFieldError(f"{self.record_id}: missing molecular formula")
        if not self.registry_key:
            raise MissingIdentifierFieldError(f"{self.record_id}: missing CAS identity")
        if self.cas_number is None:
            raise IdentifierConversionError(
                f"{self.record_id}: source key {self.registry_key!r} is not a canonical CAS"
            )
        if self.raw_fields[3] and self.molecular_weight_g_mol is None:
            raise IdentifierConversionError(f"{self.record_id}: invalid reported molecular weight")
        names = tuple(
            dict.fromkeys(
                name.strip()
                for name in (self.common_name, self.iupac_name, *self.synonyms)
                if name.strip()
            )
        )
        # Bare source strings do not establish the Standard InChI (1S) flag.
        inchi = self.inchi
        if inchi and not inchi.startswith(("InChI=1/", "InChI=1S/")):
            inchi = "InChI=1/" + inchi
        try:
            return ComponentSpec(
                identifier=identifier or self.cas_number,
                formula=self.formula,
                molecular_weight_g_mol=self.molecular_weight_g_mol,
                default_phase=phase,
                cas_number=self.cas_number,
                inchi=inchi,
                inchi_key=self.inchi_key,
                aliases=names,
                provenance=(
                    ComponentProvenance(
                        self.source.provenance.source_id,
                        "chemicals identifier data",
                        self.source.filename,
                        self.record_id,
                        notes=(self.source.version, self.source.license),
                    ),
                ),
                metadata={"identifier_source_record": self.to_dict()},
            )
        except ValueError as exc:
            raise IdentifierConversionError(f"{self.record_id}: {exc}") from exc


@dataclass(frozen=True)
class FakeRegistryRecord:
    row_number: int
    raw_fields: tuple[str, ...]
    source_id: str = "fake-registry"

    @property
    def registry_key(self) -> str:
        return self.raw_fields[0]

    @property
    def record_id(self) -> str:
        return f"{self.source_id}:{self.row_number}"


@dataclass(frozen=True)
class IdentifierPreferences:
    source_id: str
    preferred_keys: tuple[str, ...]
    unpreferred_keys: tuple[str, ...]


@dataclass(frozen=True)
class IdentifierMixtureRecord:
    row_number: int
    raw_fields: tuple[str, ...]
    source_id: str = "mixtures"

    @property
    def name(self) -> str:
        return self.raw_fields[0].strip()

    @property
    def citation(self) -> str:
        return self.raw_fields[1]

    @property
    def component_count(self) -> int:
        return int(self.raw_fields[2])

    @property
    def registry_keys(self) -> tuple[str, ...]:
        return self.raw_fields[3 : 3 + self.component_count]

    @property
    def component_names(self) -> tuple[str, ...]:
        count = self.component_count
        return self.raw_fields[3 + count : 3 + 2 * count]

    @property
    def mass_fractions(self) -> tuple[float, ...]:
        count = self.component_count
        return tuple(float(v) for v in self.raw_fields[3 + 2 * count : 3 + 3 * count])

    @property
    def mole_fractions(self) -> tuple[float, ...]:
        count = self.component_count
        return tuple(float(v) for v in self.raw_fields[3 + 3 * count : 3 + 4 * count])

    @property
    def synonyms(self) -> tuple[str, ...]:
        return tuple(v for v in self.raw_fields[3 + 4 * self.component_count :] if v)


SourceRows = (
    tuple[IdentifierRecord, ...] | tuple[FakeRegistryRecord, ...]
    | tuple[IdentifierMixtureRecord, ...] | tuple[IdentifierPreferences, ...] | tuple[str, ...]
)
RecordIndex = dict[str, dict[str | int, list[IdentifierRecord]]]


class IdentifierSelection(TypedDict, total=False):
    kind: str
    source_ids: Iterable[str] | None
    preference_sources: Sequence[str]
    source_priority: Sequence[str]


def _index_record(index: RecordIndex, record: IdentifierRecord) -> None:
    keys: dict[str, tuple[str | int | None, ...]] = {
        "source_key": (record.registry_key,),
        "cas": (record.cas_number,),
        "pubchem": (record.pubchem_id,),
        "formula": (_formula(record.formula),),
        "smiles": (record.smiles,),
        "inchi": (_inchi(record.inchi),),
        "inchikey": (record.inchi_key,),
        "name": tuple(
            _name(name) for name in (record.iupac_name, record.common_name, *record.synonyms)
        ),
    }
    for kind, values in keys.items():
        for value in dict.fromkeys(values):
            if value is not None and value != "":
                index[kind].setdefault(value, []).append(record)


class IdentifierDatabase:
    """An explicit external data directory with per-source read-only lazy indices."""

    def __init__(self, source_directory: str | PathLike[str]) -> None:
        self.source_directory = Path(source_directory).resolve()
        self._records: dict[str, SourceRows] = {}
        self._indices: dict[str, RecordIndex] = {}

    @property
    def loaded_sources(self) -> tuple[str, ...]:
        return tuple(self._records)

    @property
    def catalog(self) -> tuple[IdentifierSource, ...]:
        return IDENTIFIER_SOURCES

    def source(self, source_id: str) -> IdentifierSource:
        try:
            return _SOURCES[source_id]
        except KeyError as exc:
            raise UnknownIdentifierMethodError(f"unknown identifier source {source_id!r}") from exc

    @overload
    def records(self, source_id: Literal["fake-registry"]) -> tuple[FakeRegistryRecord, ...]: ...

    @overload
    def records(self, source_id: Literal["mixtures"]) -> tuple[IdentifierMixtureRecord, ...]: ...

    @overload
    def records(self, source_id: Literal["dippr-2014"]) -> tuple[str, ...]: ...

    @overload
    def records(self, source_id: str) -> SourceRows: ...

    def records(self, source_id: str) -> SourceRows:
        source = self.source(source_id)
        if source_id in self._records:
            return self._records[source_id]
        path = self.source_directory / source.filename
        if not path.is_file():
            raise IdentifierSourceUnavailableError(f"{source_id}: source file unavailable: {path}")
        result: SourceRows
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                if source.kind == "preferences":
                    payload = json.load(stream)
                    if set(payload) != {"preferred_cas", "unpreferred_cas"} or any(
                        not isinstance(payload[key], list)
                        or any(not isinstance(item, str) for item in payload[key])
                        for key in payload
                    ):
                        raise ValueError("invalid preference keys or values")
                    result = (
                        IdentifierPreferences(
                            source_id,
                            tuple(payload["preferred_cas"]),
                            tuple(payload["unpreferred_cas"]),
                        ),
                    )
                else:
                    rows = tuple(tuple(row) for row in csv.reader(stream, delimiter="\t"))
                    if source.kind == "compound":
                        if any(len(row) < 9 for row in rows):
                            raise ValueError("compound row requires at least nine fields")
                        result = tuple(
                            IdentifierRecord(source_id, n, row) for n, row in enumerate(rows, 1)
                        )
                    elif source.kind == "fake-registry":
                        if any(len(row) < 3 for row in rows):
                            raise ValueError("fake registry requires key, kind, identifier")
                        result = tuple(FakeRegistryRecord(n, row) for n, row in enumerate(rows, 1))
                    elif source.kind == "mixture":
                        if not rows or rows[0][:3] != ("Primary Name", "Source", "N components"):
                            raise ValueError("unknown mixture table header")
                        mixture_rows = tuple(
                            IdentifierMixtureRecord(n, row) for n, row in enumerate(rows[1:], 1)
                        )
                        result = mixture_rows
                        if any(
                            r.component_count <= 0 or len(r.raw_fields) < 3 + 4 * r.component_count
                            for r in mixture_rows
                        ):
                            raise ValueError("mixture row does not match component count")
                    else:
                        if any(len(row) != 1 or not row[0] for row in rows):
                            raise ValueError("DIPPR collection requires one identity per row")
                        result = tuple(row[0] for row in rows)
        except (ValueError, TypeError, IndexError, csv.Error) as exc:
            raise IdentifierSourceFormatError(f"{source_id}: {exc}") from exc
        if source.kind == "compound":
            index: RecordIndex = {kind: {} for kind in _QUERY_KINDS}
            for record in cast(tuple[IdentifierRecord, ...], result):
                _index_record(index, record)
            self._indices[source_id] = index
        self._records[source_id] = result
        return result

    def element(self, identity: object) -> ElementSpec:
        return get_element(identity)

    def _query(self, identity: object, kind: str) -> tuple[str, str | int]:
        if kind == "auto":
            if not isinstance(identity, bool) and isinstance(identity, Integral):
                return "cas", int_to_cas(identity)
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("identifier requires nonempty text or an integer CAS")
            identity = identity.strip()
            for prefix, target in (
                ("smiles=", "smiles"),
                ("pubchem=", "pubchem"),
                ("inchikey=", "inchikey"),
                ("formula=", "formula"),
            ):
                if identity.casefold().startswith(prefix):
                    return self._query(identity[len(prefix) :], target)
            if identity.casefold().startswith("inchi="):
                return "inchi", _inchi(identity)
            if re.fullmatch(r"[0-9]+-[0-9]+-[0-9]+", identity):
                return "cas", validate_cas_number(identity)
            try:
                element = get_element(identity)
            except ValueError:
                pass
            else:
                return "formula", element.symbol
            return "name", _name(identity)
        if kind not in _QUERY_KINDS:
            raise UnknownIdentifierMethodError(f"unknown identifier query kind {kind!r}")
        if kind == "cas":
            if not isinstance(identity, bool) and isinstance(identity, Integral):
                identity = int_to_cas(identity)
            return kind, validate_cas_number(identity)
        if kind == "pubchem":
            text = str(identity)
            if isinstance(identity, bool) or not text.isascii() or not text.isdigit():
                raise ValueError("PubChem identity must be a nonnegative integer")
            return kind, int(text)
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("identifier requires nonempty text")
        identity = identity.strip()
        if kind == "name":
            identity = _name(identity)
        elif kind == "inchi":
            identity = _inchi(identity)
        elif kind == "inchikey":
            identity = re.sub(r"^InChIKey=", "", identity, flags=re.IGNORECASE)
        elif kind == "formula":
            identity = _formula(identity)
        return kind, identity

    def search(
        self, identity: object, *, kind: str = "auto", source_ids: Iterable[str] | None = None,
    ) -> tuple[IdentifierRecord, ...]:
        """Return every matching source row, including conflicts and duplicates."""
        automatic = kind == "auto"
        kind, key = self._query(identity, kind)
        source_ids = _COMPOUND_SOURCES if source_ids is None else tuple(source_ids)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("duplicate query source ids")
        result: list[IdentifierRecord] = []
        for source_id in source_ids:
            if self.source(source_id).kind != "compound":
                raise UnknownIdentifierMethodError(f"{source_id} is not a compound query source")
            self.records(source_id)
            result.extend(self._indices[source_id][kind].get(key, ()))
        if automatic and kind == "name" and not result:
            # Names take precedence; explicit kinds bypass these fallbacks.
            for fallback in ("formula", "smiles"):
                fallback_result = self.search(identity, kind=fallback, source_ids=source_ids)
                if fallback_result:
                    return fallback_result
        return tuple(result)

    def preferences(self, source_id: str) -> IdentifierPreferences:
        if self.source(source_id).kind != "preferences":
            raise UnknownIdentifierMethodError(f"{source_id} is not a preference source")
        return cast(tuple[IdentifierPreferences, ...], self.records(source_id))[0]

    def inspect_search(
        self, identity: object, *, kind: str = "auto", source_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return full candidates and separate missing/invalid source diagnostics.

        Use search's exact global name/formula/SMILES semantics over the
        available sources. A source failure never becomes proof of no match.
        """
        self._query(identity, kind)
        if isinstance(source_ids, str):
            raise TypeError("source_ids must be a sequence of source IDs")
        sources = _COMPOUND_SOURCES if source_ids is None else tuple(source_ids)
        if len(set(sources)) != len(sources):
            raise ValueError("duplicate query source ids")
        reports: list[dict[str, Any]] = []
        available: list[str] = []
        for source_id in sources:
            report: dict[str, Any] = {"source_id": source_id}
            reports.append(report)
            try:
                source = self.source(source_id)
                if source.kind != "compound":
                    raise UnknownIdentifierMethodError(
                        f"{source_id} is not a compound query source")
                report["source"] = source.provenance.to_dict()
                self.records(source_id)
            except (UnknownIdentifierMethodError, IdentifierSourceUnavailableError,
                    IdentifierSourceFormatError) as exc:
                report.update(status={UnknownIdentifierMethodError: "unknown_method",
                    IdentifierSourceUnavailableError: "source_unavailable",
                    IdentifierSourceFormatError: "invalid_source"}[type(exc)], error=str(exc))
                continue
            available.append(source_id)
        candidates = self.search(identity, kind=kind, source_ids=available)
        for report in reports:
            if report["source_id"] in available:
                report["record_ids"] = [r.record_id for r in candidates
                                        if r.source_id == report["source_id"]]
                report["status"] = "matched" if report["record_ids"] else "missing_identity"
        conflicts = [field for field in _CONFLICT_FIELDS if len({
            _formula(r.formula) if field == "formula" else getattr(r, field)
            for r in candidates if getattr(r, field) not in {None, ""}}) > 1]
        return {"identity": identity, "kind": kind, "sources": reports,
                "candidates": [r.to_dict() for r in candidates],
                "identity_conflict": len({r.registry_key for r in candidates}) > 1,
                "conflicting_fields": conflicts, "selected_record_id": None}

    def resolve(
        self, identity: object, *, kind: str = "auto", source_ids: Iterable[str] | None = None,
        preference_sources: Sequence[str] = (), source_priority: Sequence[str] = (),
    ) -> IdentifierRecord:
        candidates = self.search(identity, kind=kind, source_ids=source_ids)
        if not candidates:
            raise UnknownIdentifierError(f"no record for {identity!r}")
        original = candidates
        preferred: set[str] = set()
        unpreferred: set[str] = set()
        for source_id in preference_sources:
            item = self.preferences(source_id)
            preferred.update(item.preferred_keys)
            unpreferred.update(item.unpreferred_keys)
        if preferred & unpreferred:
            raise ValueError("selected preference files contradict one another")
        if preference_sources:
            hits = tuple(r for r in candidates if r.registry_key in preferred)
            candidates = hits or tuple(r for r in candidates if r.registry_key not in unpreferred)
            if not candidates:
                raise AmbiguousIdentifierError(identity, original)
        if len({r.registry_key for r in candidates}) != 1:
            raise AmbiguousIdentifierError(identity, candidates)
        if source_priority:
            for source_id in source_priority:
                if self.source(source_id).kind != "compound":
                    raise UnknownIdentifierMethodError(f"{source_id} is not a compound source")
            for source_id in source_priority:
                subset = tuple(r for r in candidates if r.source_id == source_id)
                if subset:
                    candidates = subset
                    break
        conflicts = tuple(
            field
            for field in _CONFLICT_FIELDS
            if len(
                {
                    _formula(r.formula) if field == "formula" else getattr(r, field)
                    for r in candidates
                    if getattr(r, field) not in {None, ""}
                }
            )
            > 1
        )
        if conflicts:
            raise ConflictingIdentifierDataError(candidates, conflicts)
        # Equal known scientific fields: prefer the most complete existing row,
        # without synthesizing a new record from multiple sources.
        return min(
            candidates,
            key=lambda r: (
                -sum(getattr(r, field) not in {None, ""} for field in _CONFLICT_FIELDS),
                r.source_id,
                r.row_number,
            ),
        )

    def field_candidates(
        self, identity: object, field_id: str, *, kind: str = "auto",
        source_ids: Iterable[str] | None = None,
    ) -> tuple[ComponentFieldCandidate, ...]:
        if field_id not in _CONFLICT_FIELDS:
            raise UnknownIdentifierMethodError(f"unknown identifier field {field_id!r}")
        return tuple(
            ComponentFieldCandidate(
                field_id,
                getattr(record, field_id),
                record.source.provenance.source_id,
                record.source.priority,
                note=record.record_id,
            )
            for record in self.search(identity, kind=kind, source_ids=source_ids)
            if getattr(record, field_id) not in {None, ""}
        )

    def component(
        self,
        identity: object,
        *,
        kind: str = "auto",
        source_ids: Iterable[str] | None = None,
        preference_sources: Sequence[str] = (),
        source_priority: Sequence[str] = (),
        identifier: str | None = None,
        phase: str = "unknown",
    ) -> ComponentSpec:
        return self.resolve(
            identity,
            kind=kind,
            source_ids=source_ids,
            preference_sources=preference_sources,
            source_priority=source_priority,
        ).to_component(identifier=identifier, phase=phase)

    def fake_records(self, registry_key: str) -> tuple[FakeRegistryRecord, ...]:
        return tuple(r for r in self.records("fake-registry") if r.registry_key == registry_key)

    def cas_from_any(self, identity: object, **selection: Unpack[IdentifierSelection]) -> str:
        record = self.resolve(identity, **selection)
        if record.cas_number is None:
            raise MissingIdentifierFieldError(f"{record.record_id}: no valid CAS number")
        return record.cas_number

    def ids_to_cas(
        self, identities: Iterable[object], **selection: Unpack[IdentifierSelection],
    ) -> tuple[str, ...]:
        return tuple(self.cas_from_any(identity, **selection) for identity in identities)

    def molecular_weight(
        self, identity: object, **selection: Unpack[IdentifierSelection],
    ) -> float:
        record = self.resolve(identity, **selection)
        if record.molecular_weight_g_mol is None:
            raise MissingIdentifierFieldError(f"{record.record_id}: no valid molecular weight")
        return record.molecular_weight_g_mol

    def dippr_compounds(self) -> frozenset[str]:
        return frozenset(self.records("dippr-2014"))

    def mixture(self, identity: object) -> IdentifierMixtureRecord:
        if isinstance(identity, (list, tuple)) and len(identity) == 1:
            identity = identity[0]
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("mixture identity requires text or a one-element list")

        def key(text: str) -> str:
            return "".join(text.casefold().split()).replace("-", "")

        matches = tuple(
            r
            for r in self.records("mixtures")
            if key(identity) in {key(name) for name in (r.name, *r.synonyms)}
        )
        if not matches:
            raise UnknownIdentifierError(f"no mixture for {identity!r}")
        if len(matches) != 1:
            raise ValueError(f"ambiguous mixture {identity!r}: {[r.name for r in matches]}")
        return matches[0]

    def mixture_spec(
        self,
        identity: object,
        *,
        basis: str,
        phase: str,
        temperature_K: float,
        pressure_Pa: float,
        source_ids: Iterable[str] | None = None,
        source_priority: Sequence[str] = (),
    ) -> MixtureSpec:
        record = self.mixture(identity)
        if basis not in {"mass", "mole"}:
            raise UnknownIdentifierMethodError("mixture basis must be mass or mole")
        components = tuple(
            self.component(
                key,
                kind="cas",
                source_ids=source_ids,
                source_priority=source_priority,
                phase=phase,
            )
            for key in record.registry_keys
        )
        fractions = record.mass_fractions if basis == "mass" else record.mole_fractions
        if any(not isfinite(value) or value < 0 for value in fractions):
            raise IdentifierConversionError("mixture fractions must be finite and nonnegative")
        construct = (
            MixtureSpec.from_mass_fractions if basis == "mass" else MixtureSpec.from_mole_fractions
        )
        return construct(
            components,
            fractions,
            phase_label=phase,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
        )
