"""Fail-closed audit and normalization for a small Reaktoro database subset.

Reaktoro database files are source material, not :class:`AqueousDatabase`
instances.  This module only validates a lossless, explicitly unit-labelled
subset of the public ``Elements``/``Species`` and
``StandardThermoModel``/``FormationReaction`` records.  It deliberately does
not infer standard states, fit equations, or construct a native database.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any

import yaml

REAKTORO_SOURCE_AUDIT_SCHEMA = "chemworld-reaktoro-database-audit-v1"
REAKTORO_SOURCE_JSON_SCHEMA = "chemworld-reaktoro-source-json-v1"

_SOURCE_KEYS = (
    "source_id",
    "citation",
    "source_url",
    "license_id",
    "license_url",
    "redistribution",
)
_REDISTRIBUTION = {"permitted", "restricted", "unknown", "synthetic"}
_TOP_KEYS = {"source", "elements", "Elements", "species", "Species"}
_UNSUPPORTED_FIELD_CATEGORIES = {
    "phases": "phase-model-metadata",
    "phasemodels": "phase-model-metadata",
    "phase_models": "phase-model-metadata",
    "activitymodels": "interaction-parameters",
    "activity_models": "interaction-parameters",
    "interactions": "interaction-parameters",
    "interactionparameters": "interaction-parameters",
    "equationofstate": "eos-pressure-corrections",
    "eos": "eos-pressure-corrections",
    "hkf": "eos-pressure-corrections",
    "waterhkf": "eos-pressure-corrections",
    "mineralhkf": "eos-pressure-corrections",
}
_ALIASES = {
    "name": ("name", "Name"),
    "symbol": ("symbol", "Symbol"),
    "atomic_weight": ("atomic_weight", "atomicWeight", "AtomicWeight", "MolarMass", "molarMass"),
    "formula": ("formula", "Formula", "composition", "Composition"),
    "elements": ("elements", "Elements"),
    "charge": ("charge", "Charge"),
    "phase": ("phase", "Phase", "aggregateState", "AggregateState"),
    "thermo": ("thermo", "standardThermoModel", "standard_thermo_model", "StandardThermoModel"),
    "model": ("model", "type", "Model", "Type"),
    "formation_reaction": ("formationReaction", "formation_reaction", "FormationReaction"),
    "reaction": ("reaction", "stoichiometry", "Reaction", "Stoichiometry"),
    "log_k": ("logK", "log_k", "logK_ref", "log_k_ref"),
    "delta_g": ("deltaG", "delta_g", "gibbs_energy", "GibbsEnergy"),
    "delta_h": ("deltaH", "delta_h", "enthalpy", "Enthalpy"),
    "delta_s": ("deltaS", "delta_s", "entropy", "Entropy"),
    "heat_capacity": ("heatCapacity", "heat_capacity", "Cp", "cp"),
    "molar_volume": ("molarVolume", "molar_volume", "Volume"),
}
_THERMO_ALLOWED = {
    "model",
    "type",
    "Model",
    "Type",
    "formationReaction",
    "formation_reaction",
    "FormationReaction",
    "reaction",
    "stoichiometry",
    "Reaction",
    "Stoichiometry",
    "logK",
    "log_k",
    "logK_ref",
    "log_k_ref",
    "deltaG",
    "delta_g",
    "gibbs_energy",
    "GibbsEnergy",
    "deltaH",
    "delta_h",
    "enthalpy",
    "Enthalpy",
    "deltaS",
    "delta_s",
    "entropy",
    "Entropy",
    "heatCapacity",
    "heat_capacity",
    "Cp",
    "cp",
    "molarVolume",
    "molar_volume",
    "Volume",
}
_ELEMENT_ALLOWED = set(_ALIASES["name"] + _ALIASES["symbol"] + _ALIASES["atomic_weight"])
_SPECIES_ALLOWED = set(
    _ALIASES["name"]
    + _ALIASES["formula"]
    + _ALIASES["elements"]
    + _ALIASES["charge"]
    + _ALIASES["phase"]
    + _ALIASES["thermo"]
)
_SI_UNITS = {
    "atomic_weight": {"kg/mol", "g/mol"},
    "log_k": {"1"},
    "delta_g": {"J/mol"},
    "delta_h": {"J/mol"},
    "delta_s": {"J/(mol*K)"},
    "heat_capacity": {"J/(mol*K)"},
    "molar_volume": {"m^3/mol"},
    "g0": {"J/mol"},
    "h0": {"J/mol"},
    "s0": {"J/(mol*K)"},
    "cp": {"J/(mol*K)"},
    "v0": {"m^3/mol"},
    "lgkr": {"1"},
    "temperature": {"K"},
    "pressure": {"Pa"},
}
_THERMO_PROPERTY_KEYS = {
    "G0": "g0",
    "H0": "h0",
    "S0": "s0",
    "Cp": "cp",
    "CP": "cp",
    "V0": "v0",
    "lgKr": "lgkr",
    "logK": "lgkr",
    "log_k": "lgkr",
    "temperature": "temperature",
    "temperatures": "temperature",
    "T": "temperature",
    "dHr": "delta_h",
    "Tr": "temperature",
    "Pr": "pressure",
}
_INTERP_PROPERTY_KEYS = {"G0": "g0", "H0": "h0", "S0": "s0", "Cp": "cp", "V0": "v0"}
# Reaktoro's PHREEQC/GEM models expose bare ``real`` coefficients, but the
# public formulae make their temperature dimensions explicit.  Keep those
# dimensions in the source audit instead of treating every coefficient as a
# dimensionless number.  A coefficient's unit is the inverse of the power of
# temperature multiplying it in the documented lg K(T) expression.
_LGK_COEFFICIENT_UNITS = {
    "PhreeqcLgK": {
        "A1": "1",
        "A2": "K^-1",
        "A3": "K",
        "A4": "1",
        "A5": "K^2",
        "A6": "K^-2",
    },
    "GemsLgK": {
        "A0": "1",
        "A1": "K^-1",
        "A2": "K",
        "A3": "1",
        "A4": "K^2",
        "A5": "K^-2",
        "A6": "K^0.5",
    },
}
_HKF_UNITS = {
    "Gf": "J/mol",
    "Hf": "J/mol",
    "Sr": "J/(mol*K)",
    "a1": "J/(mol*Pa)",
    "a2": "J/mol",
    "a3": "J*K/(mol*Pa)",
    "a4": "J*K/mol",
    "c1": "J/(mol*K)",
    "c2": "J*K/mol",
    "wref": "J/mol",
    "charge": "1",
    "Tmax": "K",
}
_WATER_HKF_UNITS = {
    "Ttr": "K",
    "Str": "J/(mol*K)",
    "Gtr": "J/mol",
    "Htr": "J/mol",
}
_MINERAL_HKF_UNITS = {
    "Gf": "J/mol",
    "Hf": "J/mol",
    "Sr": "J/(mol*K)",
    "Vr": "m^3/mol",
    "ntr": "1",
    "a": "J/(mol*K)",
    "b": "J/(mol*K^2)",
    "c": "J*K/mol",
    "Ttr": "K",
    "Htr": "J/mol",
    "Vtr": "m^3/mol",
    "dPdTtr": "Pa/K",
    "Tmax": "K",
}
_MINERAL_HKF_ARRAY_UNITS = {
    "a": "J/(mol*K)",
    "b": "J/(mol*K^2)",
    "c": "J*K/mol",
    "Ttr": "K",
    "Htr": "J/mol",
    "Vtr": "m^3/mol",
    "dPdTtr": "Pa/K",
}
_CONSTANT_UNITS = {
    "G0": "J/mol",
    "H0": "J/mol",
    "V0": "m^3/mol",
    # Reaktoro's public header documents both derivatives with this unit.
    # Preserve that published spelling rather than infer a pressure dimension.
    "VT0": "m^3/(mol*K)",
    "VP0": "m^3/(mol*K)",
    "Cp0": "J/(mol*K)",
}
_NASA_SCALAR_UNITS = {
    "dHf": "J/mol",
    "dH0": "J/mol",
    "H0": "J/mol",
    "T0": "K",
}
_NASA_POLYNOMIAL_UNITS = {
    "Tmin": "K",
    "Tmax": "K",
    "a1": "K^2",
    "a2": "K",
    "a3": "1",
    "a4": "K^-1",
    "a5": "K^-2",
    "a6": "K^-3",
    "a7": "K^-4",
    "b1": "K",
    "b2": "1",
}
_NASA_STATES = {
    "g",
    "Gas",
    "l",
    "Liquid",
    "s",
    "Solid",
    "pl",
    "Plasma",
    "cd",
    "CondensedPhase",
    "fl",
    "Fluid",
    "lc",
    "LiquidCrystal",
    "cr",
    "CrystallineSolid",
    "am",
    "AmorphousSolid",
    "vit",
    "Vitreous",
    "ads",
    "Adsorbed",
    "mon",
    "Monomeric",
    "pol",
    "Polymeric",
    "ss",
    "SolidSolution",
    "ex",
    "IonExchange",
    "aq",
    "Aqueous",
}
_MAIER_KELLEY_UNITS = {
    "Gf": "J/mol",
    "Hf": "J/mol",
    "Sr": "J/(mol*K)",
    "Vr": "m^3/mol",
    "a": "J/(mol*K)",
    "b": "J/(mol*K^2)",
    "c": "J*K/mol",
    "Tmax": "K",
}
_PRESSURE_CORRECTION_UNITS = {"Pr": "Pa"}
_REACTION_MODEL_VARIANTS = {
    "Constant",
    "Interpolation",
    "FormationReaction",
    "ReactionStandardThermoModel",
    "VantHoff",
    "PhreeqcLgK",
    "GemsLgK",
    "HKF",
    "Nasa",
    "MaierKelley",
    "WaterHKF",
    "MineralHKF",
}


def _one(mapping: Mapping[str, Any], key: str, *, required: bool = True) -> Any:
    present = [candidate for candidate in _ALIASES[key] if candidate in mapping]
    if len(present) > 1:
        raise ValueError(f"duplicate aliases for {key}: {', '.join(present)}")
    if not present:
        if required:
            raise ValueError(f"missing required field {key}")
        return None
    return mapping[present[0]]


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _unsupported_evidence(value: Any, path: str = "") -> list[dict[str, str]]:
    """Classify rejected format families without guessing their schema."""
    evidence: list[dict[str, str]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            category = _UNSUPPORTED_FIELD_CATEGORIES.get(key_text.replace("-", "").lower())
            if category:
                evidence.append({"path": f"{path}.{key_text}".strip("."), "category": category})
            evidence.extend(_unsupported_evidence(child, f"{path}.{key_text}".strip(".")))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            evidence.extend(_unsupported_evidence(child, f"{path}[{index}]"))
    return evidence


def _scalar(value: Any, key: str) -> dict[str, Any]:
    value = _mapping(value, key)
    unknown = set(value) - {"value", "unit"}
    if unknown or set(value) != {"value", "unit"}:
        raise ValueError(f"{key} requires exactly value and unit")
    number = value["value"]
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
    ):
        raise ValueError(f"{key}.value must be a finite number")
    unit = value["unit"]
    if unit not in _SI_UNITS[key]:
        raise ValueError(f"{key} has unsupported or non-SI unit {unit!r}")
    return {"value": number, "unit": unit}


def _entries(value: Any, label: str) -> list[tuple[str | None, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        return [
            (str(name), _mapping(record, f"{label}[{name!r}]")) for name, record in value.items()
        ]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [(None, _mapping(record, f"{label}[{index}]")) for index, record in enumerate(value)]
    raise ValueError(f"{label} must be a list or mapping")


def _source(source: Any) -> dict[str, Any]:
    source = _mapping(source, "source")
    unknown = sorted(set(source) - set(_SOURCE_KEYS))
    missing = [
        key
        for key in _SOURCE_KEYS
        if not isinstance(source.get(key), str) or not source[key].strip()
    ]
    if unknown or missing or source.get("redistribution") not in _REDISTRIBUTION:
        details = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unknown:
            details.append("unknown: " + ", ".join(unknown))
        if source.get("redistribution") not in _REDISTRIBUTION:
            details.append("unsupported redistribution policy")
        raise ValueError("invalid source metadata (" + "; ".join(details) + ")")
    return {key: source[key] for key in _SOURCE_KEYS}


def _normalize_elements(raw: Any) -> tuple[list[dict[str, Any]], set[str]]:
    records: list[dict[str, Any]] = []
    symbols: set[str] = set()
    for key, item in _entries(raw, "elements"):
        unknown = set(item) - _ELEMENT_ALLOWED
        if unknown:
            raise ValueError("element has unknown fields: " + ", ".join(sorted(unknown)))
        name = _one(item, "name", required=False) or key
        symbol = _one(item, "symbol", required=False) or key
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(symbol, str)
            or not symbol.strip()
        ):
            raise ValueError("element requires nonempty name and symbol")
        if symbol in symbols:
            raise ValueError(f"duplicate element symbol {symbol!r}")
        weight = _scalar(_one(item, "atomic_weight"), "atomic_weight")
        symbols.add(symbol)
        records.append({"name": name, "symbol": symbol, "atomic_weight": weight})
    if not records:
        raise ValueError("elements must contain at least one record")
    return records, symbols


def _normalize_species(raw: Any, elements: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names: set[str] = set()
    for key, item in _entries(raw, "species"):
        unknown = set(item) - _SPECIES_ALLOWED
        if unknown:
            raise ValueError("species has unknown fields: " + ", ".join(sorted(unknown)))
        name = _one(item, "name", required=False) or key
        if not isinstance(name, str) or not name.strip():
            raise ValueError("species requires a nonempty name")
        if name in names:
            raise ValueError(f"duplicate species {name!r}")
        formula = _one(item, "formula", required=False)
        elements_field = _one(item, "elements", required=False)
        if formula is None and elements_field is None:
            raise ValueError(f"species {name!r} requires Formula or Elements")
        if isinstance(formula, str):
            from .elements import parse_formula

            try:
                formula = parse_formula(formula)
            except (TypeError, ValueError) as exc:
                # Symbolic fixture elements (for example ``A+``) are legal in
                # Reaktoro source.  An explicit Elements field is authoritative
                # when Formula is not parseable by ChemWorld's periodic parser.
                if elements_field is None:
                    raise ValueError(f"species {name!r} has invalid formula") from exc
                formula = None
        if isinstance(elements_field, str):
            elements_field = _parse_elements_text(elements_field, name)
        if formula is None:
            formula = elements_field
        formula = _mapping(formula, f"species {name!r} formula")
        if elements_field is not None:
            elements_field = _mapping(elements_field, f"species {name!r} Elements")
            if dict(formula) != dict(elements_field):
                raise ValueError(f"species {name!r} Formula and Elements disagree")
        if not formula or any(
            not isinstance(atom, str) or atom not in elements for atom in formula
        ):
            missing = sorted(set(formula) - elements)
            raise ValueError(f"species {name!r} references missing elements: {', '.join(missing)}")
        if any(
            isinstance(count, bool)
            or not isinstance(count, (int, float))
            or not math.isfinite(count)
            or count <= 0
            for count in formula.values()
        ):
            raise ValueError(f"species {name!r} formula counts must be positive finite numbers")
        charge = _one(item, "charge")
        if isinstance(charge, bool) or not isinstance(charge, int):
            raise ValueError(f"species {name!r} charge must be an integer")
        phase = _one(item, "phase")
        if not isinstance(phase, str) or not phase.strip():
            raise ValueError(f"species {name!r} phase must be nonempty")
        thermo = _normalize_thermo(_one(item, "thermo"), name)
        names.add(name)
        records.append(
            {
                "name": name,
                "formula": dict(formula),
                "charge": charge,
                "phase": phase,
                "standard_thermo_model": thermo,
            }
        )
    if not records:
        raise ValueError("species must contain at least one record")
    identities = {record["name"] for record in records}
    for record in records:
        _validate_reaction_species_references(
            record["standard_thermo_model"], identities, record["name"]
        )
    return records


def _validate_reaction_species_references(
    value: Any, identities: set[str], species: str, path: str = "thermo"
) -> None:
    if not isinstance(value, Mapping):
        return
    for key, child in value.items():
        child_path = f"{path}.{key}"
        if str(key).lower() in {"reaction", "stoichiometry", "reactants"}:
            reaction = (
                _parse_reactants(child, species, child_path)
                if str(key).lower() == "reactants"
                else _mapping(child, f"species {species!r} {child_path}")
            )
            if not reaction:
                raise ValueError(f"species {species!r} {child_path} is empty")
            for identity, coefficient in reaction.items():
                if identity not in identities:
                    raise ValueError(
                        f"species {species!r} {child_path} references missing species {identity!r}"
                    )
                if (
                    isinstance(coefficient, bool)
                    or not isinstance(coefficient, (int, float))
                    or not math.isfinite(coefficient)
                    or coefficient == 0
                ):
                    raise ValueError(
                        f"species {species!r} {child_path} has invalid coefficient for {identity!r}"
                    )
        else:
            _validate_reaction_species_references(child, identities, species, child_path)


def _parse_reactants(value: Any, species: str, path: str) -> dict[str, float]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"species {species!r} {path} must be a nonempty token string")
    result: dict[str, float] = {}
    for token in value.split():
        if ":" not in token:
            raise ValueError(f"species {species!r} {path} has malformed token {token!r}")
        coefficient_text, identity = token.split(":", 1)
        try:
            coefficient = float(coefficient_text)
        except ValueError as exc:
            raise ValueError(f"species {species!r} {path} has malformed coefficient") from exc
        if not identity or not math.isfinite(coefficient) or coefficient == 0:
            raise ValueError(f"species {species!r} {path} has invalid reactant token")
        if identity in result:
            raise ValueError(f"species {species!r} {path} repeats {identity!r}")
        result[identity] = int(coefficient) if coefficient.is_integer() else coefficient
    return result


def _normalize_thermo(raw: Any, species: str) -> dict[str, Any]:
    model = _mapping(raw, f"species {species!r} thermo")
    # Official Reaktoro files wrap a model as
    # ``StandardThermoModel: {Constant: {...}}`` (and may use
    # ``FormationReaction``/``ReactionStandardThermoModel`` at the same
    # level).  Keep that subtree intact after checking every numeric leaf's
    # explicit SI unit.
    wrapper_key = next(
        (key for key in ("StandardThermoModel", "standardThermoModel") if key in model), None
    )
    if wrapper_key is None:
        variants = _REACTION_MODEL_VARIANTS
        direct = variants & set(model)
        if direct:
            if len(direct) != 1 or set(model) != direct:
                raise ValueError(f"species {species!r} thermo has unknown fields")
            variant = next(iter(direct))
            body = (
                _normalize_interpolation(model[variant], species)
                if variant == "Interpolation"
                else _normalize_vant_hoff(model[variant], species)
                if variant == "VantHoff"
                else _normalize_hkf(model[variant], species)
                if variant == "HKF"
                else _normalize_reaction_standard_thermo(model[variant], species)
                if variant == "ReactionStandardThermoModel"
                else _normalize_constant(model[variant], species)
                if variant == "Constant"
                else _normalize_nasa(model[variant], species)
                if variant == "Nasa"
                else _normalize_maier_kelley(model[variant], species)
                if variant == "MaierKelley"
                else _normalize_water_hkf(model[variant], species)
                if variant == "WaterHKF"
                else _normalize_mineral_hkf(model[variant], species)
                if variant == "MineralHKF"
                else _normalize_lgk_coefficients(model[variant], species, variant)
                if variant in {"PhreeqcLgK", "GemsLgK"}
                else _validate_thermo_tree(model[variant], species)
            )
            return {
                "model": "StandardThermoModel",
                variant: body,
            }
    if wrapper_key is not None:
        wrapper = _mapping(model[wrapper_key], f"species {species!r} StandardThermoModel")
        unknown = set(model) - {wrapper_key, "FormationReaction"}
        if unknown:
            raise ValueError(
                f"species {species!r} thermo has unknown fields: {', '.join(sorted(unknown))}"
            )
        if len(wrapper) != 1:
            raise ValueError(
                f"species {species!r} StandardThermoModel must select one explicit model"
            )
        variant, body = next(iter(wrapper.items()))
        if variant not in {
            "Constant",
            "Interpolation",
            "FormationReaction",
            "ReactionStandardThermoModel",
            "VantHoff",
            "PhreeqcLgK",
            "GemsLgK",
            "HKF",
            "Nasa",
            "MaierKelley",
            "WaterHKF",
            "MineralHKF",
        }:
            raise ValueError(f"species {species!r} has unsupported StandardThermoModel {variant!r}")
        normalized_body = (
            _normalize_interpolation(body, species)
            if variant == "Interpolation"
            else _normalize_vant_hoff(body, species)
            if variant == "VantHoff"
            else _normalize_hkf(body, species)
            if variant == "HKF"
            else _normalize_reaction_standard_thermo(body, species)
            if variant == "ReactionStandardThermoModel"
            else _normalize_constant(body, species)
            if variant == "Constant"
            else _normalize_nasa(body, species)
            if variant == "Nasa"
            else _normalize_maier_kelley(body, species)
            if variant == "MaierKelley"
            else _normalize_water_hkf(body, species)
            if variant == "WaterHKF"
            else _normalize_mineral_hkf(body, species)
            if variant == "MineralHKF"
            else _normalize_lgk_coefficients(body, species, variant)
            if variant in {"PhreeqcLgK", "GemsLgK"}
            else _validate_thermo_tree(body, species)
        )
        result = {"model": "StandardThermoModel", variant: normalized_body}
        if "FormationReaction" in model:
            result["FormationReaction"] = _validate_thermo_tree(model["FormationReaction"], species)
        return result
    if "FormationReaction" in model:
        if set(model) != {"FormationReaction"}:
            raise ValueError(f"species {species!r} thermo has unknown fields")
        return {
            "model": "FormationReaction",
            "FormationReaction": _validate_thermo_tree(model["FormationReaction"], species),
        }
    unknown = set(model) - _THERMO_ALLOWED
    if unknown:
        raise ValueError(
            f"species {species!r} thermo has unknown fields: {', '.join(sorted(unknown))}"
        )
    model_name = _one(model, "model")
    if model_name not in {"StandardThermoModel", "FormationReaction"}:
        raise ValueError(
            f"species {species!r} thermo must explicitly use "
            "StandardThermoModel or FormationReaction"
        )
    reaction_value = _one(model, "formation_reaction", required=False)
    # Reaktoro's older mapping puts FormationReaction fields directly under
    # StandardThermoModel; the newer mapping wraps them in formationReaction.
    reaction = _mapping(
        reaction_value if reaction_value is not None else model,
        f"species {species!r} FormationReaction",
    )
    allowed = set(
        _ALIASES["model"]
        + _ALIASES["reaction"]
        + _ALIASES["log_k"]
        + _ALIASES["delta_g"]
        + _ALIASES["delta_h"]
        + _ALIASES["delta_s"]
        + _ALIASES["heat_capacity"]
        + _ALIASES["molar_volume"]
    )
    unknown = set(reaction) - allowed
    if unknown:
        raise ValueError(
            f"species {species!r} FormationReaction has unknown fields: "
            f"{', '.join(sorted(unknown))}"
        )
    stoich = _mapping(_one(reaction, "reaction"), f"species {species!r} formation reaction")
    if not stoich or any(
        not isinstance(name, str)
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for name, value in stoich.items()
    ):
        raise ValueError(f"species {species!r} formation reaction must be finite stoichiometry")
    result: dict[str, Any] = {
        "model": "StandardThermoModel",
        "formation_reaction": {
            "reaction": dict(stoich),
            "log_k": _scalar(_one(reaction, "log_k"), "log_k"),
            "delta_g": _scalar(_one(reaction, "delta_g"), "delta_g"),
            "delta_h": _scalar(_one(reaction, "delta_h"), "delta_h"),
        },
    }
    for key in ("delta_s", "heat_capacity", "molar_volume"):
        value = _one(reaction, key, required=False)
        if value is not None:
            result["formation_reaction"][key] = _scalar(value, key)
    return result


def _parse_elements_text(text: str, species: str) -> dict[str, float]:
    """Parse Reaktoro's compact ``1:H 2:O`` element-count spelling."""
    result: dict[str, float] = {}
    for token in text.split():
        if ":" not in token:
            raise ValueError(f"species {species!r} Elements must use count:symbol tokens")
        count, symbol = token.split(":", 1)
        try:
            value = float(count)
        except ValueError as exc:
            raise ValueError(f"species {species!r} Elements has nonnumeric count") from exc
        if not symbol or not math.isfinite(value) or value <= 0:
            raise ValueError(f"species {species!r} Elements has invalid count")
        if symbol in result:
            raise ValueError(f"species {species!r} Elements repeats {symbol!r}")
        result[symbol] = int(value) if value.is_integer() else value
    if not result:
        raise ValueError(f"species {species!r} Elements is empty")
    return result


def _validate_thermo_tree(value: Any, species: str, *, field: str = "thermo") -> Any:
    """Validate the explicit-unit leaves used by official model mappings."""
    if isinstance(value, Mapping):
        if (
            field.rsplit(".", 1)[-1] == "ReactionStandardThermoModel"
            and "PressureCorrection" in value
        ):
            return _normalize_reaction_standard_thermo(value, species)
        if set(value) == {"value", "unit"}:
            unit = value["unit"]
            if (
                not isinstance(value["value"], (int, float))
                or isinstance(value["value"], bool)
                or not math.isfinite(value["value"])
            ):
                raise ValueError(f"species {species!r} {field}.value must be finite")
            expected = _THERMO_PROPERTY_KEYS.get(field.rsplit(".", 1)[-1])
            if expected is None:
                raise ValueError(
                    f"species {species!r} thermo field {field!r} has no declared SI unit"
                )
            if unit not in _SI_UNITS[expected]:
                raise ValueError(
                    f"species {species!r} {field} has unsupported or non-SI unit {unit!r}"
                )
            return {"value": value["value"], "unit": unit}
        if set(value) == {"VantHoff"}:
            return {"VantHoff": _normalize_vant_hoff(value["VantHoff"], species)}
        for variant in ("PhreeqcLgK", "GemsLgK"):
            if set(value) == {variant}:
                return {variant: _normalize_lgk_coefficients(value[variant], species, variant)}
        allowed = set(_THERMO_PROPERTY_KEYS) | {
            "ConstLgK",
            "ReactionStandardThermoModel",
            "FormationReaction",
            "Constant",
            "Interpolation",
            "reaction",
            "stoichiometry",
            "Reactants",
            "VantHoff",
            "PhreeqcLgK",
            "GemsLgK",
            "elements",
            "Elements",
            "temperature",
            "Temperature",
            "values",
            "units",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"species {species!r} thermo has unknown fields: {', '.join(sorted(unknown))}"
            )
        return {
            key: _validate_thermo_tree(item, species, field=f"{field}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_validate_thermo_tree(item, species, field=field) for item in value]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if field.rsplit(".", 1)[-1] in {"lgKr", "logK", "log_k"}:
            if not math.isfinite(value):
                raise ValueError(
                    f"species {species!r} thermo numeric field {field!r} is not finite"
                )
            return {"value": value, "unit": "1"}
        raise ValueError(f"species {species!r} thermo numeric field {field!r} lacks explicit unit")
    if isinstance(value, str):
        return value
    raise ValueError(f"species {species!r} thermo contains unsupported value")


def _normalize_hkf(value: Any, species: str) -> dict[str, Any]:
    """Audit the official HKF parameter record without evaluating it.

    Reaktoro exposes these fields as bare ``real`` members in
    ``StandardThermoModelParamsHKF``.  The source interchange contract keeps
    the API's published dimensions explicit and refuses aliases, omitted
    optional values, or additional fields.  Native HKF evaluation remains a
    separate, unsupported bridge capability.
    """

    mapping = _mapping(value, f"species {species!r} HKF")
    expected = set(_HKF_UNITS)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} HKF has noncanonical fields (" + "; ".join(details) + ")"
        )
    normalized: dict[str, Any] = {}
    for field, unit in _HKF_UNITS.items():
        item = _mapping(mapping[field], f"species {species!r} HKF.{field}")
        if set(item) != {"value", "unit"}:
            raise ValueError(
                f"species {species!r} HKF.{field} requires exactly value and unit"
            )
        number = item["value"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError(f"species {species!r} HKF.{field}.value must be finite")
        if item["unit"] != unit:
            raise ValueError(
                f"species {species!r} HKF.{field} requires unit {unit!r}"
            )
        normalized[field] = {"value": number, "unit": unit}
    return normalized


def _normalize_constant(value: Any, species: str) -> Any:
    """Audit the expanded official constant standard-thermo record.

    Existing sources use a small legacy ``Constant`` subset (for example,
    only ``G0``) in fixtures.  The VT0/VP0/Cp0 extension is enabled only when
    one of those new fields is present; then the complete six-field public
    Reaktoro record is required and every scalar must carry its exact unit.
    """

    mapping = _mapping(value, f"species {species!r} Constant")
    extension = {"VT0", "VP0", "Cp0"}
    if not set(mapping) & extension:
        return _validate_thermo_tree(mapping, species, field="Constant")
    expected = set(_CONSTANT_UNITS)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} Constant has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    normalized: dict[str, Any] = {}
    for field, unit in _CONSTANT_UNITS.items():
        item = _mapping(mapping[field], f"species {species!r} Constant.{field}")
        if set(item) != {"value", "unit"}:
            raise ValueError(
                f"species {species!r} Constant.{field} requires exactly value and unit"
            )
        number = item["value"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError(f"species {species!r} Constant.{field}.value must be finite")
        if item["unit"] != unit:
            raise ValueError(f"species {species!r} Constant.{field} requires unit {unit!r}")
        normalized[field] = {"value": number, "unit": unit}
    return normalized


def _normalize_nasa(value: Any, species: str) -> dict[str, Any]:
    """Audit the two official NASA serialization branches without evaluating.

    Reaktoro serializes either a polynomial branch (`dHf`, `dH0`,
    `Polynomials`) or a no-interval branch (`H0`, `T0`).  The polynomial
    coefficients are retained with dimensions derived from the published
    NASA expressions; no temperature extrapolation or property evaluation is
    performed here.
    """

    mapping = _mapping(value, f"species {species!r} Nasa")
    if "Polynomials" in mapping:
        expected = {"dHf", "dH0", "Polynomials"}
        missing = sorted(expected - set(mapping))
        unknown = sorted(set(mapping) - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ValueError(
                f"species {species!r} Nasa polynomial branch has noncanonical fields ("
                + "; ".join(details)
                + ")"
            )
        polynomials = mapping["Polynomials"]
        if not isinstance(polynomials, Sequence) or isinstance(
            polynomials, (str, bytes, bytearray)
        ):
            raise ValueError(f"species {species!r} Nasa Polynomials must be a sequence")
        if not polynomials:
            raise ValueError(f"species {species!r} Nasa Polynomials must not be empty")
        return {
            "dHf": _nasa_scalar(mapping["dHf"], species, "dHf"),
            "dH0": _nasa_scalar(mapping["dH0"], species, "dH0"),
            "Polynomials": [
                _normalize_nasa_polynomial(item, species) for item in polynomials
            ],
        }
    expected = {"H0", "T0"}
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} Nasa no-interval branch has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    return {
        "H0": _nasa_scalar(mapping["H0"], species, "H0"),
        "T0": _nasa_scalar(mapping["T0"], species, "T0"),
    }


def _nasa_scalar(value: Any, species: str, field: str) -> dict[str, Any]:
    unit = _NASA_SCALAR_UNITS[field]
    if not isinstance(value, Mapping):
        raise ValueError(f"species {species!r} Nasa.{field} requires explicit value/unit")
    return _explicit_unit_scalar(value, species, f"Nasa.{field}", unit)


def _normalize_nasa_polynomial(value: Any, species: str) -> dict[str, Any]:
    mapping = _mapping(value, f"species {species!r} Nasa Polynomial")
    expected = {"Label", "State", *_NASA_POLYNOMIAL_UNITS}
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} Nasa Polynomial has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    label = mapping["Label"]
    if not isinstance(label, str):
        raise ValueError(f"species {species!r} Nasa Polynomial.Label must be a string")
    state = mapping["State"]
    if not isinstance(state, str) or state not in _NASA_STATES:
        raise ValueError(
            f"species {species!r} Nasa Polynomial.State is not a supported AggregateState"
        )
    normalized: dict[str, Any] = {"Label": label, "State": state}
    for field, unit in _NASA_POLYNOMIAL_UNITS.items():
        normalized[field] = _explicit_unit_scalar(
            mapping[field], species, f"Nasa.Polynomial.{field}", unit
        )
    if normalized["Tmin"]["value"] >= normalized["Tmax"]["value"]:
        raise ValueError(
            f"species {species!r} Nasa Polynomial requires Tmin < Tmax"
        )
    return normalized


def _normalize_maier_kelley(value: Any, species: str) -> dict[str, Any]:
    """Audit the official Maier--Kelley parameter record without evaluating it."""

    mapping = _mapping(value, f"species {species!r} MaierKelley")
    required = set(_MAIER_KELLEY_UNITS) - {"Tmax"}
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - set(_MAIER_KELLEY_UNITS))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} MaierKelley has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    normalized: dict[str, Any] = {}
    for field, unit in _MAIER_KELLEY_UNITS.items():
        if field == "Tmax" and field not in mapping:
            continue
        normalized[field] = _explicit_unit_scalar(
            mapping[field], species, f"MaierKelley.{field}", unit
        )
    return normalized


def _normalize_water_hkf(value: Any, species: str) -> dict[str, Any]:
    """Audit the exact WaterHKF triple-point reference record."""

    mapping = _mapping(value, f"species {species!r} WaterHKF")
    expected = set(_WATER_HKF_UNITS)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} WaterHKF has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    return {
        field: _explicit_unit_scalar(mapping[field], species, f"WaterHKF.{field}", unit)
        for field, unit in _WATER_HKF_UNITS.items()
    }


def _normalize_mineral_hkf(value: Any, species: str) -> dict[str, Any]:
    """Audit MineralHKF serialization and its transition-array coupling.

    Reaktoro serializes ``ntr`` as the number of phase transitions.  The
    implementation consumes one transition record per transition and one
    ``a/b/c`` coefficient record per phase region, so a complete source
    record has ``len(Ttr)==len(Htr)==len(Vtr)==len(dPdTtr)==ntr`` and
    ``len(a)==len(b)==len(c)==ntr+1``.  This function only validates and
    preserves those explicit records; it does not evaluate the model.
    """

    mapping = _mapping(value, f"species {species!r} MineralHKF")
    required = set(_MINERAL_HKF_UNITS) - {"Tmax"}
    missing = sorted(required - set(mapping))
    unknown = sorted(set(mapping) - set(_MINERAL_HKF_UNITS))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} MineralHKF has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )

    normalized: dict[str, Any] = {}
    for field in ("Gf", "Hf", "Sr", "Vr"):
        normalized[field] = _explicit_unit_scalar(
            mapping[field], species, f"MineralHKF.{field}", _MINERAL_HKF_UNITS[field]
        )

    ntr = _explicit_unit_scalar(mapping["ntr"], species, "MineralHKF.ntr", "1")
    number = ntr["value"]
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ValueError(
            f"species {species!r} MineralHKF.ntr.value must be a nonnegative integer"
        )
    normalized["ntr"] = ntr

    for field, unit in _MINERAL_HKF_ARRAY_UNITS.items():
        raw = mapping[field]
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
            raise ValueError(f"species {species!r} MineralHKF.{field} must be a sequence")
        expected_length = number + 1 if field in {"a", "b", "c"} else number
        if len(raw) != expected_length:
            raise ValueError(
                f"species {species!r} MineralHKF.{field} length {len(raw)} "
                f"does not match expected {expected_length} for ntr={number}"
            )
        normalized[field] = [
            _explicit_unit_scalar(item, species, f"MineralHKF.{field}[{index}]", unit)
            for index, item in enumerate(raw)
        ]

    if "Tmax" in mapping:
        normalized["Tmax"] = _explicit_unit_scalar(
            mapping["Tmax"], species, "MineralHKF.Tmax", "K"
        )
    return normalized


def _normalize_pressure_correction(value: Any, species: str) -> dict[str, Any]:
    """Audit the official pressure-correction reference-pressure record.

    Reaktoro's public factory takes one reference pressure ``Pr`` in Pa.  The
    source audit preserves that input with an explicit unit and intentionally
    does not evaluate the pressure correction or claim native bridge support.
    """

    mapping = _mapping(value, f"species {species!r} PressureCorrection")
    expected = set(_PRESSURE_CORRECTION_UNITS)
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(
            f"species {species!r} PressureCorrection has noncanonical fields ("
            + "; ".join(details)
            + ")"
        )
    normalized: dict[str, Any] = {}
    for field, unit in _PRESSURE_CORRECTION_UNITS.items():
        item = _mapping(mapping[field], f"species {species!r} PressureCorrection.{field}")
        if set(item) != {"value", "unit"}:
            raise ValueError(
                f"species {species!r} PressureCorrection.{field} requires exactly value and unit"
            )
        number = item["value"]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
        ):
            raise ValueError(
                f"species {species!r} PressureCorrection.{field}.value must be finite"
            )
        if item["unit"] != unit:
            raise ValueError(
                f"species {species!r} PressureCorrection.{field} requires unit {unit!r}"
            )
        normalized[field] = {"value": number, "unit": unit}
    return normalized


def _normalize_reaction_standard_thermo(value: Any, species: str) -> Any:
    """Normalize a documented reaction standard-thermo subtree.

    ``PressureCorrection`` is accepted only as the sole explicit variant
    under ``ReactionStandardThermoModel``.  Other established variants retain
    the generic recursive audit path.
    """

    mapping = _mapping(value, f"species {species!r} ReactionStandardThermoModel")
    if "PressureCorrection" in mapping:
        if set(mapping) != {"PressureCorrection"}:
            raise ValueError(
                f"species {species!r} ReactionStandardThermoModel has unknown fields"
            )
        return {
            "PressureCorrection": _normalize_pressure_correction(
                mapping["PressureCorrection"], species
            )
        }
    return _validate_thermo_tree(mapping, species, field="ReactionStandardThermoModel")


def _normalize_interpolation(value: Any, species: str) -> dict[str, Any]:
    """Validate explicit temperature-node interpolation without fitting a basis."""
    mapping = _mapping(value, f"species {species!r} Interpolation")
    temp_keys = [key for key in ("temperature", "temperatures", "T") if key in mapping]
    if len(temp_keys) != 1:
        raise ValueError(f"species {species!r} Interpolation requires one temperature node field")
    unknown = set(mapping) - set(temp_keys) - set(_INTERP_PROPERTY_KEYS)
    if unknown:
        raise ValueError(
            f"species {species!r} Interpolation has unknown fields: {', '.join(sorted(unknown))}"
        )
    temperatures = mapping[temp_keys[0]]
    if not isinstance(temperatures, Sequence) or isinstance(temperatures, (str, bytes, bytearray)):
        raise ValueError(f"species {species!r} Interpolation temperatures must be a sequence")
    normalized_t = [
        _validate_thermo_tree(item, species, field="temperature") for item in temperatures
    ]
    numeric_t = [item["value"] for item in normalized_t]
    if len(numeric_t) < 2 or any(right <= left for left, right in pairwise(numeric_t)):
        raise ValueError(
            f"species {species!r} Interpolation temperatures must be strictly increasing"
        )
    result: dict[str, Any] = {"temperature": normalized_t}
    for key in _INTERP_PROPERTY_KEYS:
        if key not in mapping:
            continue
        values = mapping[key]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ValueError(f"species {species!r} Interpolation {key} values must be a sequence")
        normalized = [_validate_thermo_tree(item, species, field=key) for item in values]
        if len(normalized) != len(normalized_t):
            raise ValueError(
                f"species {species!r} Interpolation {key} length mismatches temperature"
            )
        result[key] = normalized
    if len(result) == 1:
        raise ValueError(f"species {species!r} Interpolation requires at least one property series")
    return result


def _normalize_vant_hoff(value: Any, species: str) -> dict[str, Any]:
    """Validate Reaktoro's documented VantHoff parameter names and units."""
    mapping = _mapping(value, f"species {species!r} VantHoff")
    required = {"lgKr", "dHr", "Tr", "Pr"}
    if set(mapping) != required:
        missing = sorted(required - set(mapping))
        unknown = sorted(set(mapping) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"species {species!r} VantHoff fields invalid ({'; '.join(details)})")
    units = {"lgKr": "1", "dHr": "J/mol", "Tr": "K", "Pr": "Pa"}
    result: dict[str, Any] = {}
    for key, unit in units.items():
        raw = mapping[key]
        if not isinstance(raw, Mapping):
            raise ValueError(f"species {species!r} VantHoff {key} requires explicit value/unit")
        result[key] = _validate_thermo_tree(raw, species, field=key)
        if result[key]["unit"] != unit:
            raise ValueError(f"species {species!r} VantHoff {key} requires unit {unit}")
    return result


def _normalize_lgk_coefficients(value: Any, species: str, variant: str) -> dict[str, Any]:
    """Audit documented PHREEQC/GEM coefficient sets without evaluating them."""
    mapping = _mapping(value, f"species {species!r} {variant}")
    coefficient_names = (
        ["A1", "A2", "A3", "A4", "A5", "A6"]
        if variant == "PhreeqcLgK"
        else ["A0", "A1", "A2", "A3", "A4", "A5", "A6"]
    )
    required = set(coefficient_names) | {"Pr"}
    if set(mapping) != required:
        missing = sorted(required - set(mapping))
        unknown = sorted(set(mapping) - required)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unknown " + ", ".join(unknown))
        raise ValueError(f"species {species!r} {variant} fields invalid ({'; '.join(details)})")
    result: dict[str, Any] = {}
    coefficient_units = _LGK_COEFFICIENT_UNITS[variant]
    for key in coefficient_names:
        raw = mapping[key]
        if not isinstance(raw, Mapping):
            raise ValueError(f"species {species!r} {variant} {key} requires explicit value/unit")
        result[key] = _explicit_unit_scalar(
            raw, species, f"{variant}.{key}", coefficient_units[key]
        )
    raw_pressure = mapping["Pr"]
    if not isinstance(raw_pressure, Mapping):
        raise ValueError(f"species {species!r} {variant} Pr requires explicit value/unit")
    result["Pr"] = _explicit_unit_scalar(raw_pressure, species, f"{variant}.Pr", "Pa")
    return result


def _explicit_unit_scalar(
    value: Mapping[str, Any], species: str, field: str, unit: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"species {species!r} {field} requires exactly value and unit")
    if set(value) != {"value", "unit"}:
        raise ValueError(f"species {species!r} {field} requires exactly value and unit")
    number = value["value"]
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
    ):
        raise ValueError(f"species {species!r} {field}.value must be finite")
    if value["unit"] != unit:
        raise ValueError(f"species {species!r} {field} requires unit {unit}")
    return {"value": number, "unit": unit}


def audit_reaktoro_source(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Audit JSON/YAML source and return a deterministic normalized report."""
    errors: list[str] = []
    unsupported: list[dict[str, str]] = []
    try:
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                payload = yaml.safe_load(payload)
        root = _mapping(payload, "Reaktoro source")
        unsupported = _unsupported_evidence(root)
        unknown = set(root) - _TOP_KEYS
        if unknown:
            raise ValueError("source has unknown fields: " + ", ".join(sorted(unknown)))
        source = _source(root.get("source"))
        element_key = "elements" if "elements" in root else "Elements"
        species_key = "species" if "species" in root else "Species"
        if element_key not in root or species_key not in root:
            raise ValueError("source requires Elements/elements and Species/species")
        elements, symbols = _normalize_elements(root[element_key])
        species = _normalize_species(root[species_key], symbols)
        normalized = {
            "schema_version": REAKTORO_SOURCE_JSON_SCHEMA,
            "provider": "reaktoro",
            "source": source,
            "elements": elements,
            "species": species,
            "native_conversion": {
                "eligible": False,
                "reason": "source audit is not AqueousDatabase construction",
            },
        }
        return {
            "schema_version": REAKTORO_SOURCE_AUDIT_SCHEMA,
            "valid": True,
            "errors": [],
            "unsupported": [],
            "normalized": normalized,
        }
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        errors.append(str(exc))
        if not unsupported:
            unsupported = _unsupported_evidence(payload)
        return {
            "schema_version": REAKTORO_SOURCE_AUDIT_SCHEMA,
            "valid": False,
            "errors": errors,
            "unsupported": unsupported,
            "normalized": None,
        }


def reaktoro_source_to_normalized_json(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    """Return normalized source JSON or reject an ineligible source."""
    report = audit_reaktoro_source(payload)
    if not report["valid"]:
        raise ValueError("Reaktoro source failed strict audit: " + "; ".join(report["errors"]))
    return report["normalized"]
