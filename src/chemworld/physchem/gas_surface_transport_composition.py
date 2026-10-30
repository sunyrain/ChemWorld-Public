"""Complete gas / ideal-surface / explicit-fit composition.

This module imports no provider. It does not create a world or apply source state.
The compatibility projection is private and has no independent physical meaning:
all gas kinetics and transport remain owned by the complete composition.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from itertools import pairwise
from math import isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.elements import get_element, molecular_weight
from chemworld.physchem.gas_network_surface import GasSurfaceAttachment, network_with_gas_surfaces
from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec
from chemworld.physchem.mechanism_import import _units, convert_mechanism
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.surface_mechanism_import import convert_surface_mechanism

MODEL = "imported-gas-surface-polynomial-composition-v1"
_HEADERS = {"description", "generator", "input-files", "cantera-version", "date"}
_INPUT_KEYS = {
    "model",
    "payload",
    "gas_phase",
    "surface_phase",
    "source",
    "transport",
    "minimum_temperature_K",
    "maximum_temperature_K",
    "source_documents",
}
_SOURCE_KEYS = {"source_id", "priority", "citation", "version", "license", "checksum"}


def _plain(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and isfinite(value):
        return value
    raise ValueError("composition requires finite plain JSON declarations")


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _keys(value, allowed, label):
    if not isinstance(value, Mapping) or set(value) - set(allowed):
        raise ValueError(f"unsupported {label} declaration")


def _number(value, label, *, positive=False):
    if type(value) not in (int, float) or not isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be a finite {'positive ' if positive else ''}number")
    return value


def _digest(value):
    return sha256(json.dumps(_plain(value), sort_keys=True, allow_nan=False).encode()).hexdigest()


def _references(selection, *, default_section):
    if isinstance(selection, str):
        if selection != "all":
            raise ValueError("only all or explicit selected source sections are supported")
        return [default_section]
    if not isinstance(selection, list):
        raise ValueError("invalid selected source sections")
    result = []
    for entry in selection:
        if isinstance(entry, Mapping) and len(entry) == 1:
            result.append(next(iter(entry)))
        elif isinstance(entry, str):
            result.append(entry)
        else:
            raise ValueError("invalid selected source section")
    return result


def _nasa(node):
    _keys(node, {"model", "temperature-ranges", "data", "reference-pressure", "note"}, "NASA7")
    ranges, rows = node.get("temperature-ranges"), node.get("data")
    if node.get("model") != "NASA7" or not isinstance(ranges, list) or not isinstance(rows, list):
        raise ValueError("composition first slice requires explicit NASA7")
    if len(rows) != len(ranges) - 1 or not rows:
        raise ValueError("NASA7 segment layout mismatch")
    for value in ranges:
        _number(value, "NASA7 temperature", positive=True)
    if any(a >= b for a, b in pairwise(ranges)):
        raise ValueError("NASA7 temperature ranges must increase")
    for row in rows:
        if not isinstance(row, list) or len(row) != 7:
            raise ValueError("NASA7 needs seven coefficients per branch")
        for value in row:
            _number(value, "NASA7 coefficient")


def _transport_entry(value):
    _keys(
        value,
        {
            "model",
            "geometry",
            "diameter",
            "well-depth",
            "dipole",
            "polarizability",
            "rotational-relaxation",
            "note",
        },
        "gas source transport",
    )
    if value.get("model") != "gas" or value.get("geometry") not in {"atom", "linear", "nonlinear"}:
        raise ValueError("unsupported source transport model/geometry")
    for key in ("diameter", "well-depth"):
        _number(value.get(key), key, positive=True)
    for key in ("dipole", "polarizability", "rotational-relaxation"):
        if _number(value.get(key, 0), key) < 0:
            raise ValueError("source transport parameter must be nonnegative")
    if "note" in value and not isinstance(value["note"], str):
        raise ValueError("source transport note must be text")


def _consume_elements(document):
    """Check explicit source weights before the older gas resolver sees the input."""
    result = deepcopy(document)
    records = result.pop("elements", None)
    if records is None:
        return result
    if not isinstance(records, list) or not records:
        raise ValueError("explicit element table must be nonempty")
    seen = set()
    for row in records:
        _keys(row, {"symbol", "atomic-weight"}, "element weight")
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or symbol in seen:
            raise ValueError("element weights require unique original symbols")
        weight = _number(row.get("atomic-weight"), "element weight", positive=True)
        if weight != get_element(symbol).atomic_weight_g_mol:
            raise ValueError("source element weight differs from original native table")
        seen.add(symbol)
    return result


def _validate_document(document, allowed_sections):
    _keys(
        document,
        _HEADERS | {"units", "phases", "elements", "species", "reactions"} | allowed_sections,
        "source document",
    )
    _units(document.get("units", {}))
    for key in allowed_sections | {"species", "reactions"}:
        section = document.get(key)
        if (
            isinstance(section, list)
            and section
            and isinstance(section[0], Mapping)
            and set(section[0]) == {"units"}
        ):
            _units({**document.get("units", {}), **section[0]["units"]})


def _signature(declaration):
    return {
        "composition": _plain(declaration["composition"]),
        "NASA7": {
            key: _plain(declaration["thermo"][key])
            for key in ("model", "temperature-ranges", "data")
        },
        "reference_pressure_Pa": declaration["reference_pressure_Pa"],
    }


def _leaf_receipts(documents, gas_index, surface_index, surface_names, surface_sections):
    records = []

    def walk(value, document, path=()):
        if isinstance(value, Mapping) and value:
            for key, item in value.items():
                walk(item, document, (*path, key))
            return
        if isinstance(value, list) and value:
            for index, item in enumerate(value):
                walk(item, document, (*path, str(index)))
            return
        owner = "complete-gas-source-resolver"
        representation = "resolved-or-original-selector-exclusion"
        if path and path[0] in _HEADERS:
            owner, representation = "source-metadata", "retained"
        elif path and path[0] == "elements":
            owner, representation = "native-element-weight-binding", "checked-and-retained"
        elif "units" in path:
            owner, representation = "explicit-unit-scope", "checked-and-retained"
        elif "transport" in path:
            owner, representation = "explicit-polynomial-transport", "checked-and-retained"
        elif (
            document == "."
            and path[:2]
            in {
                ("phases", str(gas_index)),
                ("phases", str(surface_index)),
            }
            and "state" in path
        ):
            owner, representation = "source-initial-state-metadata", "retained-never-applied"
        elif document == "." and path[:2] == ("phases", str(surface_index)):
            owner, representation = "complete-surface-kernel", "parsed-and-retained"
        elif document == "." and path and path[0] == "species" and len(path) > 1:
            entry = documents[document]["species"][int(path[1])]
            if entry.get("name") in surface_names:
                owner, representation = "complete-surface-kernel", "parsed-and-retained"
        elif document == "." and path and path[0] in surface_sections:
            owner, representation = "complete-surface-kernel", "parsed-and-retained"
        records.append(
            {
                "document": document,
                "path": "/" + "/".join(p.replace("~", "~0").replace("/", "~1") for p in path),
                "owner": owner,
                "representation": representation,
                "original_value_sha256": _digest(value),
            }
        )

    for name, document in documents.items():
        walk(document, name)
    return records


class ImportedGasSurfaceTransportComposition:
    """Detached declarations; each public native child is a fresh exact clone."""

    def __init__(self, inputs, gas, surface, transport, receipt, states):
        self._inputs = _freeze(_plain(inputs))
        self._gas = _freeze(_plain(gas.to_dict()))
        self._surface = _freeze(_plain(surface.to_dict()))
        self._transport = _freeze(_plain(transport.to_dict()))
        self._receipt = _freeze(_plain(receipt))
        self._states = _freeze(_plain(states))

    @property
    def gas_network(self):
        return ReactionNetworkSpec.from_dict(_plain(self._gas))

    @property
    def surface(self):
        return CatalyticMechanismSpec.from_dict(_plain(self._surface))

    @property
    def transport(self):
        return GasTransportPolynomialSpec.from_dict(_plain(self._transport))

    @property
    def composition_receipt(self):
        return _plain(self._receipt)

    @property
    def source_initial_states(self):
        return _plain(self._states)

    @property
    def source_documents(self):
        return {".": _plain(self._inputs["payload"]), **_plain(self._inputs["source_documents"])}

    def to_dict(self):
        return _plain(self._inputs)

    @classmethod
    def from_dict(cls, raw):
        _keys(raw, _INPUT_KEYS, "composition codec")
        if set(raw) != _INPUT_KEYS or raw["model"] != MODEL:
            raise ValueError("unknown or incomplete composition codec")
        values = _plain(raw)
        values.pop("model")
        source = values.pop("source")
        if set(source) != _SOURCE_KEYS or type(source["priority"]) is not int:
            raise ValueError("invalid composition source codec")
        for key in _SOURCE_KEYS - {"priority"}:
            if not isinstance(source[key], str):
                raise ValueError("invalid composition source text")
        values["source"] = DataSourceProvenance(**source)
        values["transport"] = GasTransportPolynomialSpec.from_dict(values["transport"])
        return convert_gas_surface_transport_composition(**values)


def convert_gas_surface_transport_composition(
    payload: Mapping[str, Any],
    *,
    gas_phase: str,
    surface_phase: str,
    source: DataSourceProvenance,
    transport: GasTransportPolynomialSpec,
    minimum_temperature_K: float,
    maximum_temperature_K: float,
    source_documents: Mapping[str, Mapping[str, Any]] | None = None,
) -> ImportedGasSurfaceTransportComposition:
    """Strict complete composition; only the declared first slice is accepted."""
    if source_documents is not None and not isinstance(source_documents, Mapping):
        raise ValueError("source_documents must be an explicit mapping")
    raw, external = _plain(payload), _plain({} if source_documents is None else source_documents)
    if not isinstance(raw, dict) or not isinstance(external, dict) or "." in external:
        raise ValueError("composition needs explicit main and named external dictionaries")
    if not isinstance(source, DataSourceProvenance) or not isinstance(
        transport, GasTransportPolynomialSpec
    ):
        raise ValueError("composition needs explicit source and polynomial transport")
    if (
        type(source.priority) is not int
        or source.priority < 0
        or any(
            not isinstance(getattr(source, key), str) or not getattr(source, key)
            for key in _SOURCE_KEYS - {"priority"}
        )
    ):
        raise ValueError("composition source needs complete text provenance and integer priority")
    lower = _number(minimum_temperature_K, "minimum T", positive=True)
    upper = _number(maximum_temperature_K, "maximum T", positive=True)
    if (
        lower >= upper
        or not isinstance(gas_phase, str)
        or not isinstance(surface_phase, str)
        or gas_phase == surface_phase
    ):
        raise ValueError("composition needs distinct phases and an ordered domain")
    phases = raw.get("phases")
    if not isinstance(phases, list) or len(phases) != 2:
        raise ValueError("composition requires exactly one gas and one surface phase")
    selected = {p.get("name"): p for p in phases if isinstance(p, Mapping)}
    if len(selected) != 2 or set(selected) != {gas_phase, surface_phase}:
        raise ValueError("selected phases missing or duplicated")
    gas_phase_raw, surface_phase_raw = selected[gas_phase], selected[surface_phase]
    _keys(
        gas_phase_raw,
        {
            "name",
            "thermo",
            "kinetics",
            "elements",
            "species",
            "reactions",
            "state",
            "transport",
            "skip-undeclared-elements",
            "skip-undeclared-third-bodies",
        },
        "gas phase",
    )
    _keys(
        surface_phase_raw,
        {
            "name",
            "thermo",
            "kinetics",
            "elements",
            "species",
            "reactions",
            "state",
            "adjacent-phases",
            "site-density",
            "Motz-Wise",
        },
        "surface phase",
    )
    if gas_phase_raw.get("transport") != "mixture-averaged" or surface_phase_raw.get(
        "adjacent-phases"
    ) != [gas_phase]:
        raise ValueError("composition requires mixture-averaged transport and one adjacent gas")
    if transport.thermal_diffusion != "off" or transport.soret is not None:
        raise ValueError("composition first slice requires explicit thermal diffusion off")
    root_units = _units(raw.get("units", {}))
    if "length" not in root_units or root_units["quantity"] not in {"mol", "kmol"}:
        raise ValueError("surface scope requires length and mol/kmol units")
    surface_names = surface_phase_raw.get("species")
    if (
        not isinstance(surface_names, list)
        or not surface_names
        or any(not isinstance(s, str) for s in surface_names)
        or len(set(surface_names)) != len(surface_names)
    ):
        raise ValueError("surface species must be complete unique local names")
    surface_sections = _references(
        surface_phase_raw.get("reactions", "all"), default_section="reactions"
    )
    if any("/" in section for section in surface_sections):
        raise ValueError("surface external sections are not supported in this slice")
    gas_selection = gas_phase_raw.get("species", "all")
    gas_refs = (
        []
        if isinstance(gas_selection, list) and all(isinstance(s, str) for s in gas_selection)
        else _references(gas_selection, default_section="species")
    )
    gas_refs += _references(gas_phase_raw.get("reactions", "all"), default_section="reactions")
    sections = {".": set(surface_sections)}
    for reference in gas_refs:
        document, section = reference.rsplit("/", 1) if "/" in reference else (".", reference)
        sections.setdefault(document, set()).add(section)
    if set(external) != set(sections) - {"."}:
        raise ValueError("external documents must exactly cover selected references")
    originals = {".": raw, **external}
    for name, document in originals.items():
        _validate_document(document, sections.get(name, set()))
    import_raw = _consume_elements(raw)
    import_external = {name: _consume_elements(document) for name, document in external.items()}
    gas = convert_mechanism(
        import_raw, phase_name=gas_phase, source=source.source_id, source_documents=import_external
    )
    if not gas.reactions or any(s.charge or s.phase != "gas" for s in gas.species):
        raise ValueError("composition needs complete neutral gas kinetics")
    if tuple(transport.species_ids) != tuple(gas.species_ids):
        raise ValueError("fit order must match original selected gas order")
    transport.validate_network(gas)
    if (
        not transport.fit_temperature_range_K[0]
        <= lower
        < upper
        <= transport.fit_temperature_range_K[1]
    ):
        raise ValueError("composition domain exceeds explicit fit domain")
    gas_thermo = _plain(gas.metadata["gas_thermochemistry"])
    projected_gas_species = []
    for species in gas.species:
        declaration = _plain(species.metadata["source_declaration"])
        if "equation-of-state" in declaration:
            raise ValueError("species equation-of-state override is outside ideal composition")
        _transport_entry(declaration.get("transport"))
        _nasa(gas_thermo[species.species_id]["thermo"])
        thermo = deepcopy(gas_thermo[species.species_id]["thermo"])
        thermo["reference-pressure"] = (
            f"{gas_thermo[species.species_id]['reference_pressure_Pa']} Pa"
        )
        projected_gas_species.append(
            {"name": species.species_id, "composition": dict(species.composition), "thermo": thermo}
        )
    source_species = raw.get("species")
    if not isinstance(source_species, list) or any(
        not isinstance(s, Mapping) for s in source_species
    ):
        raise ValueError("surface source species must be explicit local entries")
    by_name = {s.get("name"): s for s in source_species}
    if len(by_name) != len(source_species) or set(surface_names) - set(by_name):
        raise ValueError("missing or duplicated complete surface species")
    locally_selected = (
        set(gas_selection)
        if isinstance(gas_selection, list) and all(isinstance(s, str) for s in gas_selection)
        else set()
    )
    if set(by_name) - (set(surface_names) | locally_selected) or set(surface_names) & set(
        gas.species_ids
    ):
        raise ValueError("unconsumed or phase-ambiguous local species")
    surface_nodes = []
    for name in surface_names:
        node = by_name[name]
        _keys(node, {"name", "composition", "thermo", "sites", "note"}, "surface species")
        _nasa(node.get("thermo"))
        surface_nodes.append(deepcopy(node))
    surface_reactions = []
    for section in surface_sections:
        rows = raw.get(section)
        if not isinstance(rows, list) or any(not isinstance(r, Mapping) for r in rows):
            raise ValueError("surface section must contain complete reaction entries")
        if any("units" in row for row in rows):
            raise ValueError("surface local unit scopes are not supported in first slice")
        surface_reactions.extend(deepcopy(rows))
    projected_gas_phase = {
        "name": gas_phase,
        "thermo": "ideal-gas",
        "kinetics": "gas",
        "species": list(gas.species_ids),
        "reactions": "none",
        "transport": "none",
    }
    projected_surface_phase = {
        **deepcopy(surface_phase_raw),
        "reactions": ["surface-kernel-reactions"],
    }
    projection = {
        "units": deepcopy(raw.get("units", {})),
        "phases": [projected_gas_phase, projected_surface_phase],
        "species": projected_gas_species + surface_nodes,
        "surface-kernel-reactions": surface_reactions,
    }
    parsed = convert_surface_mechanism(
        projection,
        surface_phase=surface_phase,
        source=source,
        minimum_temperature_K=lower,
        maximum_temperature_K=upper,
    )
    # Compare the pre-binding parser output; never replace first and self-compare.
    for species_id in gas.species_ids:
        if _signature(parsed.surface.species_by_id[species_id].gas_declaration) != _signature(
            gas_thermo[species_id]
        ):
            raise ValueError("pre-binding gas NASA7/composition/reference-pressure mismatch")
    table = {s.species_id: s for s in gas.species}
    entries = tuple(
        CatalyticSpeciesThermo(
            table[e.species.species_id],
            e.reference_frame,
            e.source,
            gas_declaration=gas_thermo[e.species.species_id],
            caloric_boundary_policy="continuous-energy",
        )
        if e.species.phase == "gas"
        else replace(e, caloric_boundary_policy="continuous-energy")
        for e in parsed.surface.species
    )
    surface = replace(parsed.surface, species=entries)
    attachment = GasSurfaceAttachment(
        "composition-binding-only", "composition-binding-only", surface, 1.0
    )
    network_with_gas_surfaces(gas, (attachment,))
    all_elements = sorted({e for s in surface.species for e in s.species.composition})
    for name, document in originals.items():
        if "elements" in document:
            declared = {row["symbol"] for row in document["elements"]}
            used = (
                all_elements
                if name == "."
                else sorted({e for s in gas.species for e in s.composition})
            )
            if not set(used) <= declared:
                raise ValueError("explicit element table does not cover all used species")
    receipt = {
        "schema": MODEL + "-receipt",
        "source_documents": {name: _digest(doc) for name, doc in originals.items()},
        "counts": {
            "gas_species": len(gas.species),
            "gas_reactions": len(gas.reactions),
            "surface_species": len(surface.adsorbates),
            "surface_reactions": len(surface.reactions),
        },
        "ordered_gas_species": list(gas.species_ids),
        "ordered_surface_species": list(surface.adsorbates),
        "gas_source_resolution": deepcopy(gas.metadata.get("source_resolution")),
        "gas_source_resolution_scope": (
            "gas-import compatibility inputs; complete original documents retained separately"
        ),
        "gas_resolver_input_documents_sha256": {
            ".": _digest(import_raw),
            **{name: _digest(document) for name, document in import_external.items()},
        },
        "source_transport_units": {
            "diameter": "angstrom",
            "well-depth": "K",
            "dipole": "debye",
            "polarizability": "angstrom^3",
            "rotational-relaxation": "dimensionless",
        },
        "native_atomic_weights_g_mol": {
            e: get_element(e).atomic_weight_g_mol for e in all_elements
        },
        "molecular_weights_kg_mol": {
            e.species.species_id: molecular_weight(e.species.composition) / 1000
            for e in surface.species
        },
        "leaf_ownership": _leaf_receipts(
            originals,
            phases.index(gas_phase_raw),
            phases.index(surface_phase_raw),
            set(surface_names),
            set(surface_sections),
        ),
        "surface_projection": {
            "document": projection,
            "sha256": _digest(projection),
            "gas_kinetics_owner": "complete gas_network; projection is not a replacement mechanism",
            "transport_owner": "complete explicit transport",
            "pre_binding_NASA7_pressure_composition_checked": True,
        },
        "identity_binding": "canonical gas SpeciesSpec + original gas thermochemistry + native MW",
        "caloric_boundary_policy": "continuous-energy",
        "source_initial_states_applied": False,
        "provenance_is_qualification": False,
    }
    inputs = {
        "model": MODEL,
        "payload": raw,
        "gas_phase": gas_phase,
        "surface_phase": surface_phase,
        "source": source.to_dict(),
        "transport": transport.to_dict(),
        "minimum_temperature_K": lower,
        "maximum_temperature_K": upper,
        "source_documents": external,
    }
    states = {
        gas_phase: deepcopy(gas_phase_raw.get("state", {})),
        surface_phase: deepcopy(surface_phase_raw.get("state", {})),
    }
    return ImportedGasSurfaceTransportComposition(inputs, gas, surface, transport, receipt, states)
