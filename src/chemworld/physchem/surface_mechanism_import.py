"""Strict self-contained Cantera ideal-surface declarations on native consumers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from math import isclose
from pathlib import Path
from typing import Any

import yaml

from chemworld.physchem.catalysis_network import CatalyticMechanismSpec
from chemworld.physchem.catalysis_thermochemistry import CatalyticSpeciesThermo
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.mechanism_import import (
    _keys,
    _MechanismLoader,
    _number,
    _pressure,
    _quantity,
    _side,
    _units,
    convert_mechanism,
)
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.surface_import_rates import ImportedSurfaceReaction

_LENGTH = {"m": 1.0, "cm": 0.01, "mm": 0.001}
_AMOUNT = {"mol": 1.0, "kmol": 1000.0}
_ENERGY = {
    "J/mol": 1,
    "J/kmol": 0.001,
    "kJ/mol": 1000,
    "kJ/kmol": 1,
    "cal/mol": 4.184,
    "cal/kmol": 0.004184,
    "kcal/mol": 4184,
    "kcal/kmol": 4.184,
    "K": R,
}


def _surface_coefficient(
    value: Any, units: Mapping[str, str], gas_order: float, surface_order: float
) -> float:
    dimension = (3 * gas_order + 2 * surface_order - 2, 1 - gas_order - surface_order, -1)
    if not isinstance(value, str) or len(value.split()) == 1:
        return (
            _number(value, "surface A")
            * _LENGTH[units["length"]] ** dimension[0]
            * _AMOUNT[units["quantity"]] ** dimension[1]
            / {"s": 1, "min": 60}[units["time"]]
        )
    number, expression = value.split(maxsplit=1)
    factors = {
        **{k: (v, (1, 0, 0)) for k, v in _LENGTH.items()},
        **{k: (v, (0, 1, 0)) for k, v in _AMOUNT.items()},
        "s": (1.0, (0, 0, 1)),
        "min": (60.0, (0, 0, 1)),
    }
    dims, factor, sign = [0.0] * 3, 1.0, 1.0
    for token in re.split(r"([*/])", expression):
        if token in {"*", "/"}:
            sign = -1 if token == "/" else 1
            continue
        match = re.fullmatch(r"([A-Za-z]+)(?:\^([-+\d.]+))?", token)
        if match is None or match[1] not in factors:
            raise ValueError("unsupported explicit surface coefficient unit")
        scale, shape = factors[match[1]]
        power = float(match[2] or 1) * sign
        factor *= scale**power
        dims = [a + power * b for a, b in zip(dims, shape, strict=True)]
    if any(not isclose(a, b, abs_tol=1e-10) for a, b in zip(dims, dimension, strict=True)):
        raise ValueError("surface coefficient units conflict with declared phase orders")
    return _number(number, "surface A") * factor


@dataclass(frozen=True)
class ImportedSurfaceMechanism:
    surface: CatalyticMechanismSpec
    gas_network: ReactionNetworkSpec
    source_initial_states: Mapping[str, Any]
    source_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "imported-gas-surface-v1",
            "surface": self.surface.to_dict(),
            "gas_network": self.gas_network.to_dict(),
            "source_initial_states": dict(self.source_initial_states),
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ImportedSurfaceMechanism:
        if raw.get("model") != "imported-gas-surface-v1":
            raise ValueError("unknown imported surface bundle")
        return cls(
            CatalyticMechanismSpec.from_dict(raw["surface"]),
            ReactionNetworkSpec.from_dict(raw["gas_network"]),
            dict(raw["source_initial_states"]),
            str(raw["source_digest"]),
        )


def _selected_reactions(phase: Mapping[str, Any], payload: Mapping[str, Any]) -> list[Any]:
    selection = phase.get("reactions", "all")
    if selection == "none":
        return []
    if selection == "all":
        selection = ["reactions"]
    if (
        not isinstance(selection, list)
        or not selection
        or any(not isinstance(k, str) or k not in payload or "/" in k for k in selection)
    ):
        raise ValueError("only explicit local reaction sections are supported")
    result = []
    for key in selection:
        if not isinstance(payload[key], list):
            raise ValueError("reaction section must be a list")
        result.extend(payload[key])
    return result


def convert_surface_mechanism(
    payload: Mapping[str, Any],
    *,
    surface_phase: str,
    source: DataSourceProvenance,
    minimum_temperature_K: float,
    maximum_temperature_K: float,
) -> ImportedSurfaceMechanism:
    """Import one adjacent ideal gas and a complete ideal surface (1/2/3-site species).

    Source initial states are returned explicitly as metadata; a world builder must
    receive its actual T, gas moles and coverages. No normalization or source-state
    fallback can silently refill an existing world.
    """
    payload = json.loads(json.dumps(payload, allow_nan=False))
    phases = payload.get("phases", [])
    selected = [p for p in phases if p.get("name") == surface_phase]
    if len(selected) != 1 or len(phases) != 2:
        raise ValueError("surface import needs exactly one gas and one selected surface")
    surface = selected[0]
    _keys(
        surface,
        {
            "name",
            "thermo",
            "kinetics",
            "elements",
            "species",
            "reactions",
            "adjacent-phases",
            "site-density",
            "state",
            "Motz-Wise",
        },
        "surface phase",
    )
    if surface.get("thermo") != "ideal-surface" or surface.get("kinetics") != "surface":
        raise ValueError("only ideal-surface thermochemistry with surface kinetics is supported")
    adjacent = surface.get("adjacent-phases")
    gas_list = [p for p in phases if p is not surface]
    gas = gas_list[0]
    if adjacent != [gas.get("name")]:
        raise ValueError("one explicit local adjacent gas phase is required")
    _keys(
        gas,
        {"name", "thermo", "kinetics", "elements", "species", "reactions", "state", "transport"},
        "adjacent gas phase",
    )
    if (
        gas.get("thermo") != "ideal-gas"
        or gas.get("kinetics", "gas") != "gas"
        or gas.get("transport", "none") != "none"
    ):
        raise ValueError("surface import requires ideal gas; declared gas transport is unsupported")
    sections = {"reactions"}
    for phase in phases:
        selection = phase.get("reactions")
        if isinstance(selection, list):
            sections.update(k for k in selection if isinstance(k, str))
    _keys(
        payload,
        {
            "description",
            "generator",
            "input-files",
            "cantera-version",
            "date",
            "units",
            "phases",
            "species",
            *sections,
        },
        "gas/surface mechanism",
    )
    units = _units(payload.get("units", {}))
    if "length" not in units:
        raise ValueError("surface mechanism requires explicit length units, not volume units")
    density = _quantity(
        surface["site-density"],
        units["quantity"] + "/" + units["length"] + "^2",
        {
            q + "/" + length + "^2": a / d**2
            for q, a in _AMOUNT.items()
            for length, d in _LENGTH.items()
        },
        "surface site density",
    )
    if density <= 0:
        raise ValueError("surface site density must be positive")
    gas_names, surface_names = gas["species"], surface["species"]
    for names in (gas_names, surface_names):
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(k, str) for k in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("phase species must be unique explicit local names; includes rejected")
    names = gas_names + surface_names
    raw_species = payload["species"]
    if (
        len(set(names)) != len(names)
        or len(raw_species) != len(names)
        or {s["name"] for s in raw_species} != set(names)
    ):
        raise ValueError("all declared species must be consumed exactly once by selected phases")
    entries, thermo, occupancies = [], {}, {}
    for name in names:
        raw = next(s for s in raw_species if s["name"] == name)
        _keys(raw, {"name", "composition", "thermo", "sites", "note"}, "surface/gas species")
        sites = _number(raw.get("sites", 1), "sites")
        if sites not in (1, 2, 3) or (name in gas_names and sites != 1):
            raise ValueError("surface sites must be integer 1/2/3; gas species cannot occupy sites")
        if name in surface_names:
            occupancies[name] = int(sites)
        composition = raw["composition"]
        if not composition or any(
            not re.fullmatch(r"[A-Z][a-z]?", k)
            or _number(v, "atom count") <= 0
            or float(v) != int(v)
            for k, v in composition.items()
        ):
            raise ValueError("species require positive integral atoms; electrons/ions unsupported")
        formula = "".join(k + (str(int(v)) if v != 1 else "") for k, v in composition.items())
        phase = "gas" if name in gas_names else "surface"
        native = SpeciesSpec(name, formula, phase=phase)
        node = raw["thermo"]
        if node.get("model") not in {"NASA7", "NASA9", "Shomate", "constant-cp"}:
            raise ValueError("unsupported source surface thermochemistry model")
        if node["model"] == "NASA7":
            _keys(
                node,
                {"model", "temperature-ranges", "data", "reference-pressure", "note"},
                "NASA7 thermochemistry",
            )
        declaration = {
            "thermo": node,
            "composition": composition,
            "reference_pressure_Pa": _pressure(node.get("reference-pressure", "1 atm"), units),
            "units": units,
        }
        thermo[name] = declaration
        entries.append(
            CatalyticSpeciesThermo(
                native,
                "imported:" + source.source_id,
                source,
                gas_declaration=declaration if phase == "gas" else None,
                surface_declaration=declaration if phase == "surface" else None,
            )
        )
    gas_nodes = _selected_reactions(gas, payload)
    if gas_nodes:
        gas_payload = {
            "units": payload.get("units", {}),
            "phases": [{k: v for k, v in {**gas, "reactions": "all"}.items() if k != "transport"}],
            "species": [s for s in raw_species if s["name"] in gas_names],
            "reactions": gas_nodes,
        }
        network = convert_mechanism(gas_payload, phase_name=gas["name"], source=source.source_id)
    else:
        network = ReactionNetworkSpec(
            str(gas["name"]),
            tuple(e.species for e in entries if e.species.phase == "gas"),
            (),
            metadata={"gas_thermochemistry": {k: thermo[k] for k in gas_names}},
        )
    reactions = []
    for index, raw in enumerate(_selected_reactions(surface, payload)):
        _keys(
            raw,
            {
                "equation",
                "id",
                "type",
                "rate-constant",
                "sticking-coefficient",
                "sticking-species",
                "Motz-Wise",
                "coverage-dependencies",
                "orders",
                "duplicate",
                "note",
            },
            "surface reaction",
        )
        sticking = "sticking-coefficient" in raw
        if ("rate-constant" in raw) == sticking or raw.get("type", "interface-Arrhenius") not in {
            "interface-Arrhenius",
            "sticking-Arrhenius",
            "elementary",
        }:
            raise ValueError("surface reaction needs one elementary Arrhenius/sticking rate")
        if (
            "type" in raw
            and raw["type"] != "elementary"
            and raw["type"] != ("sticking-Arrhenius" if sticking else "interface-Arrhenius")
        ):
            raise ValueError("surface reaction type contradicts its coefficient declaration")
        equation = raw["equation"]
        parts = re.split(r"\s*(<=>|=>|=)\s*", equation)
        if len(parts) != 3:
            raise ValueError("unsupported surface equation")
        sides = [_side(parts[0]), _side(parts[2])]
        if any(k not in names or int(v) != v for side in sides for k, v in side.items()):
            raise ValueError("surface reaction must use integral declared local species")
        reactants, products = ({k: int(v) for k, v in side.items()} for side in sides)
        orders = {**reactants, **raw.get("orders", {})}
        rate = raw["sticking-coefficient" if sticking else "rate-constant"]
        if isinstance(rate, list) and len(rate) == 3:
            rate = dict(zip(("A", "b", "Ea"), rate, strict=True))
        _keys(rate, {"A", "b", "Ea"}, "surface Arrhenius triple")
        coefficient = (
            _number(rate["A"], "sticking probability")
            if sticking
            else _surface_coefficient(
                rate["A"],
                units,
                sum(v for k, v in orders.items() if k in gas_names),
                sum(v for k, v in orders.items() if k in surface_names),
            )
        )
        energy = _quantity(
            rate.get("Ea", 0), units["activation-energy"], _ENERGY, "activation energy"
        )
        coverage = {}
        for key, values in raw.get("coverage-dependencies", {}).items():
            if isinstance(values, list) and len(values) == 3:
                values = dict(zip(("a", "m", "E"), values, strict=True))
            _keys(values, {"a", "m", "E"}, "surface coverage dependence")
            coverage[key] = (
                _number(values.get("a", 0), "coverage a"),
                _number(values.get("m", 0), "coverage m"),
                _quantity(values.get("E", 0), units["activation-energy"], _ENERGY, "coverage E"),
            )
        incident = None
        molar_mass = None
        if sticking:
            incidents = [k for k in reactants if k in gas_names]
            if len(incidents) != 1 or raw.get("sticking-species", incidents[0]) != incidents[0]:
                raise ValueError("sticking requires exactly one declared incident gas")
            incident = incidents[0]
            molar_mass = (
                molecular_weight(
                    next(e.species.composition for e in entries if e.species.species_id == incident)
                )
                / 1000
            )
        elif "sticking-species" in raw or "Motz-Wise" in raw:
            raise ValueError("sticking options cannot be applied to Arrhenius rates")
        if "duplicate" in raw and type(raw["duplicate"]) is not bool:
            raise ValueError("duplicate marker must be boolean")
        reactions.append(
            ImportedSurfaceReaction(
                str(raw.get("id", f"surface-{index + 1}")),
                reactants,
                products,
                surface_phase,
                source,
                coefficient,
                _number(rate.get("b", 0), "temperature exponent"),
                energy,
                orders,
                parts[1] != "=>",
                incident,
                molar_mass,
                raw.get("Motz-Wise", surface.get("Motz-Wise", False)) if sticking else False,
                coverage,
            )
        )
    spec = CatalyticMechanismSpec(
        surface_phase,
        tuple(entries),
        (surface_phase,),
        dict.fromkeys(surface_names, surface_phase),
        tuple(reactions),
        tuple(tuple(0.0 for _ in surface_names) for _ in surface_names),
        minimum_temperature_K,
        maximum_temperature_K,
        source,
        site_species_complete=True,
        site_densities_mol_m2={surface_phase: density},
        site_occupancies=occupancies,
    )
    return ImportedSurfaceMechanism(
        spec,
        network,
        {str(p["name"]): p.get("state", {}) for p in phases},
        sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
    )


def import_surface_mechanism(path: str | Path, **kwargs: Any) -> ImportedSurfaceMechanism:
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("only self-contained YAML/JSON is supported")
    raw = path.read_text(encoding="utf-8")
    payload = (
        json.loads(raw)
        if path.suffix.lower() == ".json"
        else yaml.load(raw, Loader=_MechanismLoader)
    )
    return convert_surface_mechanism(payload, **kwargs)


def surface_import_capabilities() -> dict[str, Any]:
    return {
        "source_specification": "Cantera 3.2 ideal surface YAML",
        "phases": ["one ideal-gas", "one complete ideal-surface; integer 1/2/3-site species"],
        "thermo": ["NASA7", "NASA9", "Shomate", "constant-cp"],
        "rates": [
            "Arrhenius",
            "sticking",
            "Motz-Wise",
            "coverage a/m/E",
            "thermochemical reverse",
            "irreversible positive reactant orders",
            "duplicates",
        ],
        "consumers": [
            "original catalytic slot/process",
            "original thermal world",
            "same-source U/G",
        ],
        "metadata_only": ["source initial state (returned explicitly)", "citation notes"],
        "unsupported": [
            "external includes",
            "gas transport",
            "noninteger or >3 site occupancy",
            "multidentate BEP/harmonic/hindered and nonideal standard states",
            "nonideal/electrochemical",
            "negative/nonreactant orders",
            "coverage-dependent thermochemistry",
        ],
    }
