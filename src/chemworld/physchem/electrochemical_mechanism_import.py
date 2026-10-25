"""Import the ChemWorld electrochemical extension of a Cantera mechanism.

The upstream Cantera YAML schema does not define a portable electrode-kinetics
contract.  ChemWorld therefore accepts an explicit, versioned extension rather
than guessing how a generic ``electrochemical`` phase should be executed.  The
importer converts every declared electrode reaction into the existing native
Nernst/Butler--Volmer consumer and keeps unsupported source boundaries visible.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from chemworld.physchem.electrochemistry import (
    ElectrodeReactionSpec,
    ElectrolyteResistanceSpec,
)
from chemworld.physchem.mechanism_import import _keys, _MechanismLoader, _number

ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION = "chemworld-cantera-electrochemical-v1"

_QUANTITY_FACTORS: dict[str, dict[str, float]] = {
    "potential": {"V": 1.0, "mV": 1.0e-3},
    "area": {"m^2": 1.0, "cm^2": 1.0e-4, "mm^2": 1.0e-6},
    "current_density": {
        "A/m^2": 1.0,
        "A/cm^2": 1.0e4,
        "mA/cm^2": 10.0,
        "mA/mm^2": 1.0e3,
    },
    "conductivity": {"S/m": 1.0, "S/cm": 100.0, "mS/cm": 0.1},
    "length": {"m": 1.0, "cm": 1.0e-2, "mm": 1.0e-3},
    "resistance": {"ohm": 1.0, "Ohm": 1.0, "Ω": 1.0},
}


def _quantity(value: Any, kind: str, label: str) -> float:
    """Parse a scalar in SI units or a scalar followed by one supported unit."""

    factors = _QUANTITY_FACTORS[kind]
    if isinstance(value, str):
        parts = value.split()
        if len(parts) != 2:
            raise ValueError(f"{label} requires an explicit '<value> <unit>' string")
        amount, unit = parts
        if unit not in factors:
            raise ValueError(f"unsupported {label} unit: {unit}")
        value = amount
    else:
        unit = next(iter(factors))
    result = _number(value, label) * factors[unit]
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _pick(node: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    found = [name for name in names if name in node]
    if len(found) > 1:
        raise ValueError(f"duplicate aliases declared for {names[0]}")
    return node[found[0]] if found else default


def _copy_mapping(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return copy.deepcopy(dict(value))


def _species_records(raw: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("electrochemical mechanism needs a non-empty species list")
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for item in raw:
        if isinstance(item, str):
            record = {"name": item}
        elif isinstance(item, Mapping):
            _keys(item, {"name", "charge", "composition", "note", "metadata"}, "species")
            record = copy.deepcopy(dict(item))
        else:
            raise ValueError("species declarations must be names or mappings")
        name = str(record.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"species names must be non-empty and unique: {name!r}")
        record["name"] = name
        if "charge" in record:
            record["charge"] = _number(record["charge"], f"charge[{name}]")
        if "composition" in record and not isinstance(record["composition"], Mapping):
            raise ValueError(f"composition[{name}] must be a mapping")
        if "metadata" in record:
            record["metadata"] = _copy_mapping(record["metadata"], f"metadata[{name}]")
        names.add(name)
        records.append(record)
    return tuple(records)


def _reaction(
    node: Mapping[str, Any],
    index: int,
    known_species: set[str],
) -> ElectrodeReactionSpec:
    _keys(
        node,
        {
            "id",
            "reaction_id",
            "electrons",
            "electrons-transferred",
            "electrons_transferred",
            "standard-potential",
            "standard_potential_V",
            "reaction-quotient",
            "reaction_quotient_exponents",
            "exchange-current-density",
            "exchange_current_density_A_m2",
            "exchange_current_A",
            "electrode-area",
            "electrode_area_m2",
            "alpha-anodic",
            "alpha_anodic",
            "alpha-cathodic",
            "alpha_cathodic",
            "faradaic-efficiency",
            "faradaic_efficiency_ref",
            "product-selectivity",
            "product_selectivity_ref",
            "overpotential-selectivity-sensitivity",
            "overpotential_selectivity_sensitivity_V_inv",
            "forward-current-sign",
            "forward_current_sign",
            "metadata",
        },
        "electrochemical reaction",
    )
    reaction_id = str(_pick(node, "id", "reaction_id", default=f"electrochemical-{index}")).strip()
    if not reaction_id:
        raise ValueError("electrochemical reaction id cannot be empty")
    quotient = _pick(
        node,
        "reaction-quotient",
        "reaction_quotient_exponents",
    )
    if not isinstance(quotient, Mapping) or not quotient:
        raise ValueError(f"{reaction_id}: reaction quotient must be a non-empty mapping")
    if any(not isinstance(key, str) or not key.strip() for key in quotient):
        raise ValueError(f"{reaction_id}: reaction quotient species ids must be non-empty strings")
    unknown = set(quotient) - known_species
    if unknown:
        raise ValueError(f"{reaction_id}: unknown reaction-quotient species: {sorted(unknown)}")
    exponents = {
        str(key): _number(value, f"reaction quotient[{key}]")
        for key, value in quotient.items()
    }
    forward_sign = _number(
        _pick(node, "forward-current-sign", "forward_current_sign", default=1),
        f"{reaction_id}.forward-current-sign",
    )
    if forward_sign not in {-1.0, 1.0}:
        raise ValueError(f"{reaction_id}.forward-current-sign must be either -1 or 1")
    return ElectrodeReactionSpec(
        reaction_id=reaction_id,
        electrons_transferred=_number(
            _pick(node, "electrons", "electrons-transferred", "electrons_transferred"),
            f"{reaction_id}.electrons",
        ),
        standard_potential_V=_quantity(
            _pick(node, "standard-potential", "standard_potential_V"),
            "potential",
            f"{reaction_id}.standard-potential",
        ),
        reaction_quotient_exponents=exponents,
        exchange_current_density_A_m2=_quantity(
            _pick(
                node,
                "exchange-current-density",
                "exchange_current_density_A_m2",
            ),
            "current_density",
            f"{reaction_id}.exchange-current-density",
        ),
        electrode_area_m2=_quantity(
            _pick(node, "electrode-area", "electrode_area_m2"),
            "area",
            f"{reaction_id}.electrode-area",
        ),
        alpha_anodic=_number(
            _pick(node, "alpha-anodic", "alpha_anodic", default=0.5),
            f"{reaction_id}.alpha-anodic",
        ),
        alpha_cathodic=_number(
            _pick(node, "alpha-cathodic", "alpha_cathodic", default=0.5),
            f"{reaction_id}.alpha-cathodic",
        ),
        faradaic_efficiency_ref=_number(
            _pick(node, "faradaic-efficiency", "faradaic_efficiency_ref", default=0.90),
            f"{reaction_id}.faradaic-efficiency",
        ),
        product_selectivity_ref=_number(
            _pick(node, "product-selectivity", "product_selectivity_ref", default=0.88),
            f"{reaction_id}.product-selectivity",
        ),
        overpotential_selectivity_sensitivity_V_inv=_number(
            _pick(
                node,
                "overpotential-selectivity-sensitivity",
                "overpotential_selectivity_sensitivity_V_inv",
                default=0.55,
            ),
            f"{reaction_id}.overpotential-selectivity-sensitivity",
        ),
        forward_current_sign=int(forward_sign),
        metadata=_copy_mapping(node.get("metadata"), f"{reaction_id}.metadata"),
    )


def _resistance(node: Any) -> ElectrolyteResistanceSpec | None:
    if node is None:
        return None
    if not isinstance(node, Mapping):
        raise ValueError("resistance must be a mapping")
    _keys(
        node,
        {
            "electrolyte-conductivity",
            "electrolyte_conductivity_S_m",
            "electrode-gap",
            "electrode_gap_m",
            "electrode-area",
            "electrode_area_m2",
            "electrolyte_resistance_ohm",
            "total_resistance_ohm",
            "contact-resistance",
            "contact_resistance_ohm",
            "voltage-window",
            "voltage_window_V",
            "metadata",
        },
        "electrolyte resistance",
    )
    return ElectrolyteResistanceSpec(
        electrolyte_conductivity_S_m=_quantity(
            _pick(node, "electrolyte-conductivity", "electrolyte_conductivity_S_m"),
            "conductivity",
            "electrolyte-conductivity",
        ),
        electrode_gap_m=_quantity(
            _pick(node, "electrode-gap", "electrode_gap_m"),
            "length",
            "electrode-gap",
        ),
        electrode_area_m2=_quantity(
            _pick(node, "electrode-area", "electrode_area_m2"),
            "area",
            "electrode-area",
        ),
        contact_resistance_ohm=_quantity(
            _pick(node, "contact-resistance", "contact_resistance_ohm", default=0.0),
            "resistance",
            "contact-resistance",
        ),
        voltage_window_V=(
            None
            if _pick(node, "voltage-window", "voltage_window_V") is None
            else _quantity(
                _pick(node, "voltage-window", "voltage_window_V"),
                "potential",
                "voltage-window",
            )
        ),
        metadata=_copy_mapping(node.get("metadata"), "electrolyte resistance metadata"),
    )


@dataclass(frozen=True)
class ImportedElectrochemicalMechanism:
    """Versioned, replayable native electrochemical mechanism bundle."""

    species: tuple[dict[str, Any], ...]
    reactions: tuple[ElectrodeReactionSpec, ...]
    resistance: ElectrolyteResistanceSpec | None
    source_digest: str
    source_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION,
            "species": [copy.deepcopy(item) for item in self.species],
            "reactions": [item.to_dict() for item in self.reactions],
            "resistance": None if self.resistance is None else self.resistance.to_dict(),
            "source_digest": self.source_digest,
            "source_metadata": copy.deepcopy(dict(self.source_metadata)),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ImportedElectrochemicalMechanism:
        if raw.get("model") != ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION:
            raise ValueError("unknown imported electrochemical mechanism")
        species = _species_records(raw.get("species"))
        known = {item["name"] for item in species}
        reactions_raw = raw.get("reactions")
        if not isinstance(reactions_raw, list) or not reactions_raw:
            raise ValueError("imported electrochemical mechanism needs reactions")
        reactions = tuple(_reaction(item, index, known) for index, item in enumerate(reactions_raw))
        ids = [item.reaction_id for item in reactions]
        if len(ids) != len(set(ids)):
            raise ValueError("electrochemical reaction ids must be unique")
        source_digest = str(raw.get("source_digest", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", source_digest):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        return cls(
            species=species,
            reactions=reactions,
            resistance=_resistance(raw.get("resistance")),
            source_digest=source_digest,
            source_metadata=_copy_mapping(raw.get("source_metadata"), "source_metadata"),
        )


def convert_electrochemical_mechanism(
    payload: Mapping[str, Any],
) -> ImportedElectrochemicalMechanism:
    """Convert a complete explicit electrochemical mechanism declaration."""

    if not isinstance(payload, Mapping):
        raise ValueError("electrochemical mechanism must be a mapping")
    payload = json.loads(json.dumps(payload, allow_nan=False))
    _keys(
        payload,
        {
            "schema_version",
            "description",
            "generator",
            "cantera-version",
            "phases",
            "species",
            "reactions",
            "electrochemical-reactions",
            "resistance",
            "metadata",
        },
        "electrochemical mechanism",
    )
    if payload.get("schema_version") != ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION:
        raise ValueError(
            "schema_version must be " + ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION
        )
    species = _species_records(payload.get("species"))
    known = {item["name"] for item in species}
    if "reactions" in payload and "electrochemical-reactions" in payload:
        raise ValueError("declare either reactions or electrochemical-reactions, not both")
    reactions_raw = payload.get("electrochemical-reactions", payload.get("reactions"))
    if not isinstance(reactions_raw, list) or not reactions_raw:
        raise ValueError("electrochemical mechanism needs a non-empty reaction list")
    reactions = tuple(_reaction(item, index, known) for index, item in enumerate(reactions_raw))
    ids = [item.reaction_id for item in reactions]
    if len(ids) != len(set(ids)):
        raise ValueError("electrochemical reaction ids must be unique")

    phases = payload.get("phases", [])
    if not isinstance(phases, list):
        raise ValueError("phases must be a list")
    normalized_phases: list[dict[str, Any]] = []
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise ValueError("phase declarations must be mappings")
        _keys(phase, {"name", "thermo", "kinetics", "species", "metadata"}, "phase")
        phase_species = phase.get("species", [])
        if not isinstance(phase_species, list) or any(item not in known for item in phase_species):
            raise ValueError("phase species must reference the top-level species declarations")
        if phase.get("kinetics", "electrochemical") != "electrochemical":
            raise ValueError("electrochemical phases must declare kinetics: electrochemical")
        normalized_phases.append(copy.deepcopy(dict(phase)))

    source_metadata = {
        "schema_version": ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION,
        "description": str(payload.get("description", "")),
        "generator": str(payload.get("generator", "")),
        "cantera-version": str(payload.get("cantera-version", "")),
        "phases": normalized_phases,
        "metadata": _copy_mapping(payload.get("metadata"), "metadata"),
    }
    source_digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return ImportedElectrochemicalMechanism(
        species=species,
        reactions=reactions,
        resistance=_resistance(payload.get("resistance")),
        source_digest=source_digest,
        source_metadata=source_metadata,
    )


def import_electrochemical_mechanism(
    path: str | Path,
) -> ImportedElectrochemicalMechanism:
    """Read one local YAML/JSON electrochemical extension file."""

    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("only declarative YAML/JSON files are supported")
    raw = path.read_text(encoding="utf-8")
    payload = (
        json.loads(raw)
        if path.suffix.lower() == ".json"
        else yaml.load(raw, Loader=_MechanismLoader)
    )
    return convert_electrochemical_mechanism(payload)


def electrochemical_mechanism_import_capabilities() -> dict[str, Any]:
    """Return the explicit source coverage and unsupported boundaries."""

    return {
        "source_specification": ELECTROCHEMICAL_MECHANISM_SCHEMA_VERSION,
        "formats": ["local YAML", "local JSON"],
        "phase_models": ["explicit electrochemical electrolyte/electrode extension"],
        "thermo_models": ["standard electrode potential", "activity reaction quotient"],
        "rate_models": ["Nernst", "Butler-Volmer", "finite Faradaic conversion"],
        "transport_models": ["lumped electrolyte/contact resistance"],
        "options": [
            "unit-bearing potential, area, current-density, conductivity and length",
            "duplicate reaction rejection",
            "explicit reaction-quotient species identities",
            "JSON replay with source digest",
        ],
        "unsupported": [
            "standard Cantera electrolyte/electrode YAML without the ChemWorld extension",
            "CHEMKIN input",
            "porous-electrode transport",
            "plasma/electron-collision kinetics",
            "coverage-dependent electrochemical thermochemistry",
        ],
        "whole_mechanism_policy": "all declared reactions or reject; no silent omission",
    }
