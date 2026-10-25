"""Resolve declared Cantera gas sections without a source-library dependency.

Only explicit local files and caller-supplied search directories are used.
Selection precedes the existing physical converter; no reaction is discarded
unless the source explicitly requests a declared-species selection.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from chemworld.data.scientific_inputs import strict_json_loads

_HEADERS = {
    "description", "generator", "input-files", "cantera-version", "date",
    "units", "phases", "species", "reactions",
}


def read_mechanism_document(path: Path) -> dict[str, Any]:
    """Read one local data document, rejecting ambiguous repeated mapping keys."""
    from chemworld.physchem.mechanism_import import _MechanismLoader

    class SourceLoader(_MechanismLoader):
        def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
            keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
            if len(keys) != len(set(keys)):
                raise ValueError("duplicate mechanism mapping key")
            return super().construct_mapping(node, deep=deep)

    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("only declarative YAML/JSON files are supported")
    text = path.read_text(encoding="utf-8")
    raw = (strict_json_loads(text) if path.suffix.lower() == ".json"
           else yaml.load(text, Loader=SourceLoader))
    if not isinstance(raw, dict):
        raise ValueError("mechanism document must be a mapping")
    return raw


def local_document_reader(
    root_path: Path, search_paths: Sequence[str | Path],
) -> Callable[[str], Mapping[str, Any]]:
    """Make a deterministic, explicit local lookup for file/section selectors."""
    directories = (root_path.resolve().parent, *(Path(p).resolve() for p in search_paths))

    def read(name: str) -> Mapping[str, Any]:
        if "://" in name:
            raise ValueError("mechanism references must be local files")
        path = Path(name)
        candidates = (path,) if path.is_absolute() else tuple(p / path for p in directories)
        for candidate in candidates:
            if candidate.is_file():
                return read_mechanism_document(candidate)
        raise ValueError(f"mechanism source file not found in explicit search paths: {name}")

    return read


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(canonical).hexdigest()


def _section(raw: Mapping[str, Any], name: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    entries = raw.get(name)
    if not isinstance(entries, list):
        raise ValueError(f"missing or non-list mechanism section: {name}")
    units = dict(raw.get("units", {}))
    if entries and isinstance(entries[0], Mapping) and set(entries[0]) == {"units"}:
        units.update(entries[0]["units"])
        entries = entries[1:]
    if any(not isinstance(entry, Mapping) or set(entry) == {"units"} for entry in entries):
        raise ValueError(f"invalid mechanism section entries or misplaced units: {name}")
    return [dict(entry) for entry in entries], units


def _selection(value: Any, *, species: bool) -> list[tuple[str, Any]]:
    if isinstance(value, str):
        allowed = {"all"} if species else {"all", "declared-species", "none"}
        if value not in allowed:
            raise ValueError("invalid mechanism section selection")
        return [("species" if species else "reactions", value)]
    if not isinstance(value, list) or not value:
        raise ValueError("mechanism selection must be a nonempty list or a declared rule")
    if species and all(isinstance(item, str) for item in value):
        return [("species", value)]
    selections = []
    for item in value:
        if not species and isinstance(item, str):
            selections.append((item, "all"))
        elif isinstance(item, Mapping) and len(item) == 1:
            key, rule = next(iter(item.items()))
            if not isinstance(key, str) or not key:
                raise ValueError("section references must be nonempty strings")
            selections.append((key, rule))
        else:
            raise ValueError("invalid mechanism section selector")
    return selections


def resolve_gas_source(
    payload: Mapping[str, Any], *, phase_name: str | None,
    read_external: Callable[[str], Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Flatten one selected gas phase and bind each entry to its own unit scope."""
    from chemworld.physchem.mechanism_import import _side, _units

    raw = json.loads(json.dumps(payload, allow_nan=False))
    phases = raw.get("phases", [])
    if phase_name is None and len(phases) != 1:
        raise ValueError("multiple phases require explicit phase_name")
    selected = [p for p in phases if phase_name is None or p.get("name") == phase_name]
    if len(selected) != 1:
        raise ValueError("selected phase missing or duplicated")
    phase = selected[0]
    species_selection = phase.get("species", "all")
    reaction_selection = phase.get("reactions", "all")
    # Preserve the complete original codec/source metadata for ordinary imports.
    if (isinstance(species_selection, list)
            and all(isinstance(v, str) for v in species_selection)
            and reaction_selection == "all" and not set(raw) - _HEADERS
            and not any(k.startswith("skip-undeclared-") for k in phase)
            and not any(set(v) == {"units"} or "units" in v
                        or "units" in v.get("thermo", {}) for v in raw.get("species", []))
            and not any(set(v) == {"units"} for v in raw.get("reactions", []))):
        return raw, None

    documents: dict[str, Mapping[str, Any]] = {".": copy.deepcopy(raw)}
    provenance: dict[str, Any] = {
        "schema": "chemworld-mechanism-source-v1", "selected_phase": copy.deepcopy(phase),
        "species": [], "reactions": [], "omitted_reactions": [], "omitted_species": [],
    }

    def section(reference: str) -> tuple[str, str, list[dict[str, Any]], dict[str, str]]:
        if "/" in reference:
            file, key = reference.rsplit("/", 1)
            if not file or not key:
                raise ValueError("external selector requires file/section")
            if file not in documents:
                if read_external is None:
                    raise ValueError("external sections need import_mechanism or source_documents")
                documents[file] = json.loads(json.dumps(read_external(file), allow_nan=False))
        else:
            file, key = ".", reference
        entries, units = _section(documents[file], key)
        return file, key, entries, units

    flags = {}
    for flag in ("skip-undeclared-elements", "skip-undeclared-third-bodies"):
        flags[flag] = phase.pop(flag, False)
        if type(flags[flag]) is not bool:
            raise ValueError(f"{flag} must be boolean")
    elements = phase.get("elements")
    if elements is not None and (
        not isinstance(elements, list) or any(not isinstance(e, str) for e in elements)
    ):
        raise ValueError("only explicitly named standard elements are supported")
    if flags["skip-undeclared-elements"] and elements is None:
        raise ValueError("skip-undeclared-elements needs explicit phase elements")

    species_nodes, names = [], []
    for reference, rule in _selection(species_selection, species=True):
        file, key, entries, section_units = section(reference)
        indexed = {entry["name"]: (i, entry) for i, entry in enumerate(entries)}
        if len(indexed) != len(entries):
            raise ValueError("duplicate source species declaration")
        wanted = list(indexed) if rule == "all" else rule
        if (not isinstance(wanted, list) or any(not isinstance(n, str) for n in wanted)
                or len(wanted) != len(set(wanted))):
            raise ValueError("species rule must be all or a unique list of names")
        for name in wanted:
            if name not in indexed:
                raise ValueError(f"selected species missing from {reference}: {name}")
            position, entry = indexed[name]
            record = {"file": file, "section": key, "index": position, "name": name}
            if elements is not None and set(entry["composition"]) - set(elements):
                if not flags["skip-undeclared-elements"]:
                    raise ValueError(f"species {name} contains an undeclared element")
                provenance["omitted_species"].append(record)
                continue
            if name in names:
                raise ValueError(f"species selected more than once: {name}")
            entry = copy.deepcopy(entry)
            bound_units = {**section_units, **entry.get("units", {})}
            bound_units.update(entry["thermo"].pop("units", {}))
            entry["units"] = _units(bound_units)
            species_nodes.append(entry)
            names.append(name)
            provenance["species"].append(record)

    reactions = []
    for reference, rule in _selection(reaction_selection, species=False):
        if rule == "none" and reference == "reactions":
            continue
        if rule not in {"all", "declared-species"}:
            raise ValueError("reaction section rule must be all or declared-species")
        file, key, entries, section_units = section(reference)
        for position, raw_entry in enumerate(entries):
            entry = copy.deepcopy(raw_entry)
            record = {"file": file, "section": key, "index": position,
                      "equation": entry.get("equation")}
            if rule == "declared-species":
                equation = entry["equation"]
                colliders = set(re.findall(r"\(\+([^()]+)\)", equation)) - {"M"}
                clean = re.sub(r"\s*\(\+[^()]+\)", "", equation)
                arrow = "<=>" if "<=>" in clean else "=>" if "=>" in clean else "="
                sides = clean.split(arrow)
                if len(sides) != 2:
                    raise ValueError("reaction needs one declared equation arrow")
                participants = (set(_side(sides[0])) | set(_side(sides[1])) | colliders) - {"M"}
                if participants - set(names):
                    record["undeclared_species"] = sorted(participants - set(names))
                    provenance["omitted_reactions"].append(record)
                    continue
            if flags["skip-undeclared-third-bodies"] and "efficiencies" in entry:
                record["omitted_efficiencies"] = {
                    k: v for k, v in entry["efficiencies"].items() if k not in names
                }
                entry["efficiencies"] = {
                    k: v for k, v in entry["efficiencies"].items() if k in names
                }
            entry["units"] = _units({**section_units, **entry.get("units", {})})
            reactions.append(entry)
            provenance["reactions"].append(record)

    # Only recognized data sections may accompany the selected phase. Unknown
    # scalar/mapping top-level declarations must still reach strict conversion.
    result = {key: value for key, value in raw.items()
              if key in _HEADERS or not isinstance(value, list)}
    phase["species"] = names
    phase["reactions"] = "all" if reactions else "none"
    result.update(species=species_nodes, reactions=reactions)
    provenance["documents"] = [
        {"file": name, "sha256": _digest(document)} for name, document in documents.items()
    ]
    return result, provenance
