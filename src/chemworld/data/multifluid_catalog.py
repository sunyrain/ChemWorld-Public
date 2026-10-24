"""Revisioned, caller-supplied multifluid sources on the existing native importer.

CAS identities and explicit source revisions select parameters. Catalogue array
positions only disambiguate a declared source record; they never identify a
chemical independently. This module performs no discovery, download or fitting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

from chemworld.data.multifluid import load_multifluid_model
from chemworld.data.scientific_inputs import strict_json_loads
from chemworld.physchem.multifluid import MultifluidModel

_TAG = "chemworld-multifluid-catalog-v1"
_FIELDS = {
    "model",
    "catalog_id",
    "revision",
    "source",
    "fluids",
    "binary_pairs",
    "departure_functions",
}


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} requires nonempty text")
    return value


def _records(values: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(values, (tuple, list)) or any(not isinstance(v, Mapping) for v in values):
        raise ValueError(f"{label} requires an explicit sequence of source records")
    return [dict(v) for v in values]


@dataclass(frozen=True, init=False)
class MultifluidSourceCatalog:
    """Immutable source envelope; model selection preserves original parameter direction."""

    _declaration_json: str

    def __init__(
        self,
        *,
        catalog_id: str,
        revision: str,
        source: str,
        fluids: Sequence[Mapping[str, Any]],
        binary_pairs: Sequence[Mapping[str, Any]],
        departure_functions: Sequence[Mapping[str, Any]],
    ) -> None:
        pure = _records(fluids, "fluids")
        if not pure:
            raise ValueError("catalogue requires at least one explicitly supplied pure source")
        cases, names = [], []
        for record in pure:
            info = record.get("INFO")
            if not isinstance(info, Mapping):
                raise ValueError("pure source requires INFO with NAME and CAS")
            cases.append(_text(info.get("CAS"), "pure CAS"))
            names.append(_text(info.get("NAME"), "pure NAME"))
            equations = record.get("EOS")
            if not isinstance(equations, list) or not equations:
                raise ValueError("pure source requires a nonempty EOS list")
        if len(set(cases)) != len(cases) or len(set(names)) != len(names):
            raise ValueError("ambiguous duplicate pure CAS or source NAME")
        data = {
            "model": _TAG,
            "catalog_id": _text(catalog_id, "catalog_id"),
            "revision": _text(revision, "revision"),
            "source": _text(source, "source"),
            "fluids": pure,
            "binary_pairs": _records(binary_pairs, "binary_pairs"),
            "departure_functions": _records(departure_functions, "departure_functions"),
        }
        try:
            encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("catalogue must be finite JSON data") from exc
        object.__setattr__(self, "_declaration_json", encoded)

    @property
    def catalog_id(self) -> str:
        return self.to_dict()["catalog_id"]

    @property
    def revision(self) -> str:
        return self.to_dict()["revision"]

    @property
    def declaration_sha256(self) -> str:
        return hashlib.sha256(self._declaration_json.encode("utf-8")).hexdigest()

    @property
    def components(self) -> tuple[dict[str, Any], ...]:
        """Source identities only; availability here does not certify every EOS form."""
        return tuple(
            {"cas": f["INFO"]["CAS"], "name": f["INFO"]["NAME"], "eos_count": len(f["EOS"])}
            for f in self.to_dict()["fluids"]
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._declaration_json)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MultifluidSourceCatalog:
        if not isinstance(data, Mapping) or set(data) != _FIELDS or data.get("model") != _TAG:
            raise ValueError("unknown or incomplete multifluid source catalogue encoding")
        return cls(**{k: v for k, v in data.items() if k != "model"})

    def select(
        self,
        component_cas: Sequence[str],
        *,
        gas_constant_J_mol_K: float,
        gas_constant_convention: str,
        eos_indexes: Mapping[str, int] | None = None,
        pair_indices: Sequence[int] | None = None,
        selection_reason: str | None = None,
    ) -> MultifluidModel:
        """Select a complete explicit mixture; never estimate missing interactions.

        ``component_cas`` establishes component order. Every selected pair must
        have exactly one source record, unless ``pair_indices`` names the entire
        intended set of binary records and ``selection_reason`` explains it.
        EOS selectors are keyed by CAS, not a fragile array position. Catalogue
        revision, CAS/name identities, exact indices, reason and digest remain
        in the returned model's normal source/provenance field and JSON codec.
        """
        cases = tuple(component_cas)
        if (
            not cases
            or any(not isinstance(c, str) or not c for c in cases)
            or len(set(cases)) != len(cases)
        ):
            raise ValueError("selection requires distinct explicit component CAS identities")
        data = self.to_dict()
        pure_by_cas = {f["INFO"]["CAS"]: f for f in data["fluids"]}
        if set(cases) - set(pure_by_cas):
            raise ValueError("selected CAS is absent from the declared source catalogue")
        if eos_indexes is not None and not isinstance(eos_indexes, Mapping):
            raise ValueError("EOS selectors must map selected CAS to explicit indices")
        indexes = dict(eos_indexes or {})
        if set(indexes) - set(cases):
            raise ValueError("EOS selector names an unselected CAS")
        if selection_reason is not None:
            _text(selection_reason, "selection reason")
        fluids = [pure_by_cas[c] for c in cases]
        required = {frozenset(pair) for pair in combinations(cases, 2)}
        candidates: dict[frozenset[str], list[int]] = {key: [] for key in required}
        for i, pair in enumerate(data["binary_pairs"]):
            c1, c2 = pair.get("CAS1"), pair.get("CAS2")
            if isinstance(c1, str) and isinstance(c2, str):
                key = frozenset((c1, c2))
                if key in candidates:
                    candidates[key].append(i)
        if pair_indices is None:
            if any(len(rows) != 1 for rows in candidates.values()):
                details = {
                    "/".join(sorted(key)): len(rows)
                    for key, rows in candidates.items()
                    if len(rows) != 1
                }
                raise ValueError(f"missing or ambiguous binary sources: {details}")
            selected = [candidates[frozenset(pair)][0] for pair in combinations(cases, 2)]
        else:
            _text(selection_reason, "explicit binary selection reason")
            selected = list(pair_indices)
            if any(
                type(i) is not int or not 0 <= i < len(data["binary_pairs"]) for i in selected
            ) or len(set(selected)) != len(selected):
                raise ValueError("explicit binary indices must be distinct valid source positions")
        pairs = [data["binary_pairs"][i] for i in selected]
        selected_keys = [
            frozenset((_text(p.get("CAS1"), "binary CAS1"), _text(p.get("CAS2"), "binary CAS2")))
            for p in pairs
        ]
        if len(set(selected_keys)) != len(selected_keys) or set(selected_keys) != required:
            raise ValueError("explicit selection must contain exactly one record for every pair")
        functions = set()
        for pair in pairs:
            name = pair.get("function", "")
            if name is not None and not isinstance(name, str):
                raise ValueError("departure source name must be text")
            if name:
                functions.add(_text(name, "departure source name"))
        departures = []
        for name in sorted(functions):
            matching = [d for d in data["departure_functions"] if d.get("Name") == name]
            if len(matching) != 1:
                raise ValueError(f"missing or ambiguous departure source: {name}")
            departures.append(matching[0])
        selection = {
            "catalog_id": data["catalog_id"],
            "revision": data["revision"],
            "catalog_sha256": self.declaration_sha256,
            "components": [
                {
                    "cas": c,
                    "name": pure_by_cas[c]["INFO"]["NAME"],
                    "eos_index": indexes.get(c, 0)
                    if len(pure_by_cas[c]["EOS"]) == 1
                    else indexes.get(c),
                }
                for c in cases
            ],
            "binary_records": [
                {
                    "index": i,
                    "CAS1": p.get("CAS1"),
                    "CAS2": p.get("CAS2"),
                    "Name1": p.get("Name1"),
                    "Name2": p.get("Name2"),
                    "citation": p.get("BibTeX", ""),
                    "departure": p.get("function", ""),
                }
                for i, p in zip(selected, pairs, strict=True)
            ],
            "selection_reason": selection_reason or "unique explicit source record per CAS pair",
        }
        provenance = data["source"] + "; selection=" + json.dumps(selection, sort_keys=True)
        return load_multifluid_model(
            {
                "fluids": fluids,
                "binary_pairs": pairs,
                "departure_functions": departures,
                "gas_constant_J_mol_K": gas_constant_J_mol_K,
                "gas_constant_convention": gas_constant_convention,
            },
            source=provenance,
            eos_indexes=tuple(indexes.get(c) for c in cases),
        )


def load_multifluid_catalog(data: Mapping[str, Any]) -> MultifluidSourceCatalog:
    """Load the explicit source envelope, without searching any external database."""
    return MultifluidSourceCatalog.from_dict(data)


def read_multifluid_catalog(path: str | Path) -> MultifluidSourceCatalog:
    """Read one unambiguous local JSON envelope with the existing32MiB input bound."""
    path = Path(path)
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("multifluid catalogue exceeds32MiB limit")
    return load_multifluid_catalog(strict_json_loads(path.read_text(encoding="utf-8")))
