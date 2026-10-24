"""Declarative local Cantera gas YAML/JSON into the existing native network.

Every selected species and reaction is converted or the whole import fails.
External includes, nonideal/surface/plasma/linear-Burke models and executable
inputs remain explicit boundaries; transport declarations are metadata only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from math import isfinite, log10
from pathlib import Path
from typing import Any

import yaml

from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, ReactionSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import audit_reaction_rate_contract
from chemworld.physchem.reaction_rate_laws import R_J_PER_MOL_K
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


class _MechanismLoader(yaml.SafeLoader):
    """YAML 1.2 boolean spellings; the species NO is not the boolean False."""


_MechanismLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for _first, _resolvers in _MechanismLoader.yaml_implicit_resolvers.items():
    _MechanismLoader.yaml_implicit_resolvers[_first] = [
        (tag, expression) for tag, expression in _resolvers if tag != "tag:yaml.org,2002:bool"
    ]
_MechanismLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool", re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"), list("tTfF")
)


def _keys(node: Any, allowed: Iterable[str], label: str) -> None:
    if not isinstance(node, Mapping) or set(node) - set(allowed):
        raise ValueError(f"unsupported {label} declaration: {node}")


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _quantity(value: Any, default_unit: str, factors: Mapping[str, float], label: str) -> float:
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 2:
            value, default_unit = parts
        elif len(parts) != 1:
            raise ValueError(f"unsupported {label} units")
    if default_unit not in factors:
        raise ValueError(f"unsupported {label} unit: {default_unit}")
    return _number(value, label) * factors[default_unit]


def _pressure(value: Any, units: Mapping[str, str]) -> float:
    return _quantity(
        value,
        units.get("pressure", "Pa"),
        {"Pa": 1, "kPa": 1000, "bar": 100000, "atm": 101325},
        "pressure",
    )


def _units(values: Mapping[str, str]) -> dict[str, str]:
    result = {
        "length": "m",
        "quantity": "kmol",
        "time": "s",
        "activation-energy": "J/kmol",
        "pressure": "Pa",
        "temperature": "K",
        "energy": "J",
    }
    _keys(values, {*result, "volume"}, "unit system")
    if "activation-energy" not in values:
        result["activation-energy"] = (
            f"{values.get('energy', 'J')}/{values.get('quantity', 'kmol')}"
        )
    if "volume" in values:
        if "length" in values or values["volume"] not in {"L", "m^3", "cm^3"}:
            raise ValueError("declare either supported volume units or length units, not both")
        result.pop("length")
    result.update(values)
    for key, allowed in (
        ("length", {"m", "cm", "mm"}),
        ("quantity", {"mol", "kmol"}),
        ("energy", {"J", "kJ", "cal", "kcal"}),
        ("time", {"s", "min"}),
        ("temperature", {"K"}),
        ("pressure", {"Pa", "kPa", "bar", "atm"}),
        (
            "activation-energy",
            {
                "J/mol",
                "J/kmol",
                "kJ/mol",
                "kJ/kmol",
                "cal/mol",
                "cal/kmol",
                "kcal/mol",
                "kcal/kmol",
                "K",
            },
        ),
    ):
        if key in result and result[key] not in allowed:
            raise ValueError(f"unsupported {key} unit: {result[key]}")
    return result


def _coefficient_factor(units: Mapping[str, str], order: float) -> float:
    # A_native = A_source * (source-volume-in-L / source-amount-in-mol)^(n-1) / time-in-s.
    volume_L = (
        {"L": 1, "m^3": 1000, "cm^3": 0.001}[units["volume"]]
        if "volume" in units
        else {"m": 1000, "cm": 0.001, "mm": 1e-6}[units["length"]]
    )
    amount_mol = {"mol": 1, "kmol": 1000}[units["quantity"]]
    return (volume_L / amount_mol) ** (order - 1) / {"s": 1, "min": 60}[units["time"]]


def _pre_exponential(value: Any, units: Mapping[str, str], order: float) -> float:
    if not isinstance(value, str) or len(value.split()) == 1:
        return _number(value, "A") * _coefficient_factor(units, order)
    amount, expression = value.split(maxsplit=1)
    # Explicit source units are dimensional powers, e.g. cm^3/mol/s, m^6/kmol^2/s.
    # Fractional powers support noninteger and negative reaction orders.
    factors = {
        "m": (1000 ** (1 / 3), (1 / 3, 0, 0)),
        "cm": (0.1, (1 / 3, 0, 0)),
        "mm": (0.01, (1 / 3, 0, 0)),
        "L": (1, (1, 0, 0)),
        "mol": (1, (0, 1, 0)),
        "kmol": (1000, (0, 1, 0)),
        "s": (1, (0, 0, 1)),
        "min": (60, (0, 0, 1)),
    }
    dimensions, factor, sign = [0.0] * 3, 1.0, 1.0
    for token in re.split(r"([*/])", expression.replace(" ", "")):
        if token in {"*", "/"}:
            sign = -1.0 if token == "/" else 1.0
            continue
        match = re.fullmatch(r"([A-Za-z]+)(?:\^([-+\d.]+))?", token)
        if match is None or match[1] not in factors:
            raise ValueError(f"unsupported explicit A unit: {expression}")
        power = float(match[2] or 1) * sign
        scale, dims = factors[match[1]]
        factor *= scale**power
        dimensions = [v + power * d for v, d in zip(dimensions, dims, strict=True)]
    if any(abs(a - b) > 1e-10 for a, b in zip(dimensions, (order - 1, 1 - order, -1), strict=True)):
        raise ValueError(f"explicit A units do not match reaction order {order}")
    return _number(amount, "A") * factor


def _arrhenius(node: Any, units: Mapping[str, str], order: float) -> dict[str, float]:
    if isinstance(node, (list, tuple)):
        if len(node) != 3:
            raise ValueError("Arrhenius sequence requires A, b, Ea")
        node = dict(zip(("A", "b", "Ea"), node, strict=True))
    _keys(node, {"A", "b", "Ea"}, "Arrhenius")
    a, b = _pre_exponential(node["A"], units, order), _number(node.get("b", 0), "b")
    ea = _quantity(
        node.get("Ea", 0),
        units["activation-energy"],
        {
            "J/mol": 1,
            "J/kmol": 0.001,
            "kJ/mol": 1000,
            "kJ/kmol": 1,
            "cal/mol": 4.184,
            "cal/kmol": 0.004184,
            "kcal/mol": 4184,
            "kcal/kmol": 4.184,
            "K": R_J_PER_MOL_K,
        },
        "activation energy",
    )
    return {"A": a, "b": b, "Ea_J_per_mol": ea}


def _side(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for token in text.split(" + "):
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s+)?([A-Za-z][A-Za-z0-9_().+\-]*)", token.strip())
        if match is None:
            raise ValueError(f"unsupported finite equation term: {token}")
        amount, species = match.groups()
        result[species] = result.get(species, 0) + float(amount or 1)
    return result


def _reaction(
    node: Mapping[str, Any],
    index: int,
    units: Mapping[str, str],
    known: set[str],
    reference_pressure: float,
) -> ReactionSpec:
    kind = node.get("type", "elementary")
    allowed = {
        "equation",
        "type",
        "duplicate",
        "note",
        "id",
        "units",
        "orders",
        "negative-orders",
        "nonreactant-orders",
        "negative-A",
    }
    fields = {
        "elementary": {"rate-constant"},
        "three-body": {"rate-constant", "efficiencies", "default-efficiency"},
        "falloff": {
            "low-P-rate-constant",
            "high-P-rate-constant",
            "Troe",
            "SRI",
            "Tsang",
            "efficiencies",
            "default-efficiency",
        },
        "pressure-dependent-Arrhenius": {"rate-constants", "temperature-range"},
        "Chebyshev": {"temperature-range", "pressure-range", "data"},
        "Blowers-Masel": {"rate-constant"},
    }
    fields["chemically-activated"] = fields["falloff"]
    if kind not in fields:
        raise ValueError(f"unsupported reaction type: {kind}")
    _keys(node, allowed | fields[kind], "reaction")
    units = _units({**units, **node.get("units", {})})
    equation = str(node["equation"])
    collider = None
    clean = equation
    if kind in {"falloff", "chemically-activated"}:
        colliders = re.findall(r"\(\+([^()]+)\)", equation)
        if len(colliders) != 2 or colliders[0] != colliders[1]:
            raise ValueError("falloff needs the same (+collider) on both sides")
        collider = colliders[0]
        clean = re.sub(r"\s*\(\+[^()]+\)", "", equation)
    arrow = "<=>" if "<=>" in clean else "=>" if "=>" in clean else "="
    sides = clean.split(arrow)
    if len(sides) != 2:
        raise ValueError("reaction needs one => or <=> arrow")
    left, right = map(_side, sides)
    if kind == "elementary" and left.get("M") == right.get("M") == 1:
        kind = "three-body"
    if kind == "three-body":
        if left.get("M") == right.get("M") == 1:
            collider = "M"
        else:
            common = set(left) & set(right)
            if len(common) != 1:
                raise ValueError("named third body requires one shared collider")
            collider = next(iter(common))
        left[collider] -= 1
        right[collider] -= 1
        left = {s: n for s, n in left.items() if n}
        right = {s: n for s, n in right.items() if n}
    if (set(left) | set(right)) - known:
        raise ValueError("reaction references unknown species")
    reversible = arrow != "=>"
    orders = {**left, **{s: _number(n, "order") for s, n in node.get("orders", {}).items()}}
    if set(orders) - known or (set(orders) - set(left) and not node.get("nonreactant-orders")):
        raise ValueError("nonreactant orders need a known species and explicit permission")
    if "orders" in node and reversible:
        raise ValueError("custom reaction orders require an irreversible reaction")
    order = sum(orders.values())
    stoich = {key: right.get(key, 0) - left.get(key, 0) for key in left.keys() | right.keys()}
    stoich = {key: value for key, value in stoich.items() if value}
    params: dict[str, Any] = {
        "mechanism_schema": "cantera-gas-v2",
        "negative_A": bool(node.get("negative-A", False)),
    }
    if kind in {"elementary", "three-body"}:
        law = "arrhenius" if kind == "elementary" else "third_body_arrhenius"
        params.update(_arrhenius(node["rate-constant"], units, order + (kind == "three-body")))
    elif kind in {"falloff", "chemically-activated"}:
        broadenings = set(node) & {"Troe", "SRI", "Tsang"}
        if len(broadenings) > 1:
            raise ValueError("only one falloff broadening may be declared")
        broadening = next(iter(broadenings), "Lindemann")
        law = {
            "Troe": "troe_falloff",
            "SRI": "sri_falloff",
            "Tsang": "tsang_falloff",
            "Lindemann": "lindemann_falloff",
        }[broadening]
        activated = kind == "chemically-activated"
        params["chemically_activated"] = activated
        for prefix, field, extra in (
            ("low", "low-P-rate-constant", 1),
            ("high", "high-P-rate-constant", 0),
        ):
            params.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in _arrhenius(
                        node[field], units, order + extra - activated
                    ).items()
                }
            )
        if "Troe" in node:
            _keys(node["Troe"], {"A", "T1", "T2", "T3"}, "Troe")
            params.update(
                {
                    f"troe_{'a' if key == 'A' else key}": _number(value, key)
                    for key, value in node["Troe"].items()
                }
            )
        for family, required, optional in (
            ("SRI", {"A", "B", "C"}, {"D", "E"}),
            ("Tsang", {"A"}, {"B"}),
        ):
            if family in node:
                _keys(node[family], required | optional, family)
                if not required <= set(node[family]):
                    raise ValueError(f"missing {family} parameters")
                if family == "SRI" and (set(node[family]) & optional) not in (set(), optional):
                    raise ValueError("SRI requires both D and E or neither")
                params.update(
                    {
                        f"{family.lower()}_{key}": _number(value, key)
                        for key, value in node[family].items()
                    }
                )
    elif kind == "Blowers-Masel":
        law = "blowers_masel"
        raw = node["rate-constant"]
        _keys(raw, {"A", "b", "Ea0", "w"}, "Blowers-Masel")
        params.update(
            _arrhenius({"A": raw["A"], "b": raw.get("b", 0), "Ea": raw["Ea0"]}, units, order)
        )
        params["Ea0_J_per_mol"] = params.pop("Ea_J_per_mol")
        params["w_J_per_mol"] = _arrhenius({"A": 1, "Ea": raw["w"]}, units, order)["Ea_J_per_mol"]
    elif kind == "pressure-dependent-Arrhenius":
        law = "pressure_log"
        terms = []
        for term in node["rate-constants"]:
            _keys(term, {"P", "A", "b", "Ea"}, "P-log term")
            terms.append(
                {
                    "pressure_Pa": _pressure(term["P"], units),
                    **_arrhenius({k: v for k, v in term.items() if k != "P"}, units, order),
                }
            )
        params.update(
            rate_constants=terms,
            pressure_range_Pa=[
                min(t["pressure_Pa"] for t in terms),
                max(t["pressure_Pa"] for t in terms),
            ],
        )
    else:
        law = "chebyshev"
        rows = [[_number(v, "Chebyshev coefficient") for v in row] for row in node["data"]]
        if not rows or not rows[0]:
            raise ValueError("empty Chebyshev coefficients")
        rows[0][0] += log10(_coefficient_factor(units, order))
        params.update(
            coefficients=rows,
            pressure_range_Pa=[_pressure(p, units) for p in node["pressure-range"]],
        )
    if kind in {"three-body", "falloff", "chemically-activated"}:
        efficiencies = node.get("efficiencies", {})
        if set(efficiencies) - known:
            raise ValueError("third-body efficiencies reference unknown species")
        params.update(
            third_body_efficiencies={k: _number(v, "efficiency") for k, v in efficiencies.items()},
            default_efficiency=_number(node.get("default-efficiency", 1), "default efficiency"),
        )
        if collider != "M":
            if collider not in known or efficiencies or "default-efficiency" in node:
                raise ValueError(
                    "named collider must be known and cannot also declare efficiencies"
                )
            params.update(third_body_efficiencies={collider: 1.0}, default_efficiency=0.0)
    if law in {"pressure_log", "chebyshev"}:
        params.update(
            pressure_basis="ideal_gas",
            pressure_domain_policy="clamp" if law == "pressure_log" else "polynomial-extrapolation",
        )
        if "temperature-range" in node:
            params["temperature_range_K"] = [
                _number(v, "temperature") for v in node["temperature-range"]
            ]
    if reversible:
        params.update(reverse_basis="ideal_gas_thermo", reference_pressure_Pa=reference_pressure)
    reaction_id = str(node.get("id", f"reaction-{index + 1:03d}"))
    result = ReactionSpec(
        reaction_id,
        equation,
        stoich,
        reversible,
        RateLawSpec(reaction_id, law, params),
        metadata={
            "source_declaration": dict(node),
            "source_units": units,
            "negative_orders": bool(node.get("negative-orders", False)),
            "custom_orders": "orders" in node,
        },
        forward_orders=orders,
        reverse_orders=right if reversible else {},
    )
    report = audit_reaction_rate_contract(result)
    if not report.passed:
        raise ValueError(f"invalid {reaction_id}: {report.violations}")
    return result


def convert_mechanism(
    payload: Mapping, *, phase_name: str | None = None, source: str = "in-memory declarative input"
) -> ReactionNetworkSpec:
    """Return one balanced native network, with all selected thermo/source data."""
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
            "reactions",
        },
        "mechanism",
    )
    # Numeric Gibbs-temperature keys become strings in JSON. Canonicalize once
    # so network/source metadata are exactly replayable through the native codec.
    payload = json.loads(json.dumps(payload, allow_nan=False))
    phases = payload.get("phases", ())
    if phase_name is None and len(phases) != 1:
        raise ValueError("multiple phases require explicit phase_name")
    selected = [p for p in phases if phase_name is None or p["name"] == phase_name]
    if len(selected) != 1:
        raise ValueError("selected phase missing or duplicated")
    phase = selected[0]
    _keys(
        phase,
        {"name", "thermo", "elements", "species", "kinetics", "transport", "state", "reactions"},
        "phase",
    )
    if (
        phase.get("thermo") != "ideal-gas"
        or phase.get("kinetics") != "gas"
        or phase.get("reactions", "all") != "all"
    ):
        raise ValueError("only an ideal-gas phase using all local reactions is supported")
    names = phase["species"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(v, str) for v in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("phase needs unique locally declared species")
    units = _units(payload.get("units", {}))
    species, thermo_nodes, pressures = [], {}, set()
    source_species = payload["species"]
    if len({s["name"] for s in source_species}) != len(source_species) or set(names) - {
        s["name"] for s in source_species
    }:
        raise ValueError("selected local species must all exist without duplicate declarations")
    for name in names:
        raw = next(s for s in source_species if s["name"] == name)
        _keys(
            raw,
            {"name", "composition", "thermo", "transport", "equation-of-state", "note"},
            "species",
        )
        composition = raw["composition"]
        if not composition or any(
            not re.fullmatch(r"[A-Z][a-z]?", element)
            or _number(n, "atom count") <= 0
            or float(n) != int(n)
            for element, n in composition.items()
        ):
            raise ValueError("finite species need positive integer elemental compositions")
        node = raw["thermo"]
        if node.get("model") == "NASA7":
            _keys(
                node,
                {"model", "temperature-ranges", "data", "reference-pressure", "note"},
                "NASA7 thermochemistry",
            )
        pressure = _pressure(node.get("reference-pressure", "1 atm"), units)
        if pressure <= 0:
            raise ValueError("reference pressure must be positive")
        species_thermo_from_cantera_yaml(
            species_id=name,
            thermo=node,
            composition=composition,
            reference_pressure_Pa=pressure,
            units=units,
        )
        pressures.add(pressure)
        formula = "".join(
            f"{key}{int(value) if value != 1 else ''}" for key, value in composition.items()
        )
        species.append(
            SpeciesSpec(name, formula, phase="gas", metadata={"source_declaration": raw})
        )
        thermo_nodes[name] = {
            "thermo": node,
            "composition": composition,
            "reference_pressure_Pa": pressure,
            "units": units,
        }
    if len(pressures) != 1:
        raise ValueError("mixed thermochemical reference pressures are unsupported")
    reactions = payload["reactions"]
    if not isinstance(reactions, list) or not reactions:
        raise ValueError("gas mechanism needs a nonempty reaction list")
    converted = tuple(
        _reaction(node, i, units, set(names), next(iter(pressures)))
        for i, node in enumerate(reactions)
    )
    return ReactionNetworkSpec(
        str(phase["name"]),
        tuple(species),
        converted,
        metadata={
            "import_format": "cantera-gas-mechanism-v2",
            "source": source,
            "source_units": units,
            "gas_thermochemistry": thermo_nodes,
            "selected_phase": dict(phase),
            "unselected_phases": [p for p in phases if p is not phase],
            "source_description": payload.get("description", ""),
            "non_kinetic_metadata_only": ["transport", "equation-of-state", "state"],
            "source_sha256": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest(),
        },
    )


def import_mechanism(path: str | Path, *, phase_name: str | None = None) -> ReactionNetworkSpec:
    path = Path(path)
    if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
        raise ValueError("only declarative YAML/JSON files are supported")
    payload = (
        yaml.load(path.read_text(encoding="utf-8"), Loader=_MechanismLoader)
        if path.suffix.lower() != ".json"
        else json.loads(path.read_text(encoding="utf-8"))
    )
    return convert_mechanism(payload, phase_name=phase_name, source=path.name)


def mechanism_import_capabilities() -> dict[str, Any]:
    """Machine-readable source coverage; preserved metadata is not executed physics."""
    return {
        "source_specification": "Cantera 3.2 gas YAML",
        "formats": ["local YAML", "local JSON"],
        "phase_models": ["ideal-gas"],
        "thermo_models": ["NASA7", "NASA9", "Shomate", "constant-cp", "piecewise-Gibbs"],
        "gas_rate_models": [
            "Arrhenius",
            "third-body Arrhenius",
            "Lindemann",
            "Troe",
            "SRI",
            "Tsang",
            "chemically-activated (all four falloff variants)",
            "pressure-dependent-Arrhenius",
            "Chebyshev",
            "Blowers-Masel",
        ],
        "options": [
            "thermochemical reverse",
            "explicit reverse as separate irreversible reactions",
            "duplicate reactions",
            "negative-A opt-in",
            "custom/negative/nonreactant orders",
            "named colliders",
            "Arrhenius triples",
            "explicit coefficient units",
            "P-log signed sums at equal pressures",
            "JSON replay",
            "YAML 1.2 NO species",
        ],
        "metadata_only": ["transport", "state", "equation-of-state", "unselected phases"],
        "unsupported": [
            "CHEMKIN input",
            "external includes",
            "named reaction sections",
            "ionic/electron species",
            "surface/electrochemical/nonideal phases",
            "linear-Burke",
            "two-temperature-plasma",
            "electron-collision-plasma",
            "custom/extensible rate code",
            "species thermochemistry extrapolation",
            "negative orders at zero concentration",
            "Blowers-Masel outside 0 < Ea0 < w",
        ],
        "source_boundary_rules": {
            "NASA9_internal_join": "upper",
            "NASA7_Shomate_join": "lower",
            "P-log_pressure": "clamp endpoints",
            "Chebyshev": "polynomial extrapolation",
        },
        "whole_mechanism_policy": (
            "all selected reactions or reject; no omission or artificial size caps"
        ),
    }


def main() -> None:
    """Convert without an upstream dependency; refuse accidental output overwrite."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--phase")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    network = import_mechanism(args.input, phase_name=args.phase)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(network.to_dict(), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Converted {len(network.species)} species / {len(network.reactions)} reactions")


if __name__ == "__main__":
    main()
