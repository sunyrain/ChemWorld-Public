"""Finite declarative Cantera-style YAML/JSON import, not an upstream runtime.

Select one ideal-gas phase explicitly. NASA7 and all selected reactions are
converted, or the whole import fails. Transport and other phases are preserved
as source metadata, not claimed as implemented models. No executable input,
external species includes, negative-A fits, custom orders or auto-generation.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from math import isfinite, log10
from pathlib import Path

import yaml

from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.reaction_network_specs import RateLawSpec, ReactionSpec, SpeciesSpec
from chemworld.physchem.reaction_rate_contracts import audit_reaction_rate_contract
from chemworld.physchem.reaction_rate_laws import R_J_PER_MOL_K, validate_pressure_rate
from chemworld.physchem.thermochemistry import NASA7SpeciesThermo


def _keys(node, allowed, label):
    if not isinstance(node, Mapping) or set(node) - set(allowed):
        raise ValueError(f"unsupported {label} declaration: {node}")


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _quantity(value, default_unit, factors, label):
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 2:
            value, default_unit = parts
        elif len(parts) != 1:
            raise ValueError(f"unsupported {label} units")
    if default_unit not in factors:
        raise ValueError(f"unsupported {label} unit: {default_unit}")
    return _number(value, label) * factors[default_unit]


def _pressure(value, units):
    return _quantity(
        value,
        units.get("pressure", "Pa"),
        {"Pa": 1, "kPa": 1000, "bar": 100000, "atm": 101325},
        "pressure",
    )


def _units(values):
    result = {
        "length": "m",
        "quantity": "kmol",
        "time": "s",
        "activation-energy": "J/kmol",
        "pressure": "Pa",
        "temperature": "K",
    }
    _keys(values, {*result, "volume"}, "unit system")
    if "volume" in values:
        if "length" in values or values["volume"] not in {"L", "m^3", "cm^3"}:
            raise ValueError("declare either supported volume units or length units, not both")
        result.pop("length")
    result.update(values)
    for key, allowed in (
        ("length", {"m", "cm", "mm"}),
        ("quantity", {"mol", "kmol"}),
        ("time", {"s", "min"}),
        ("temperature", {"K"}),
        ("pressure", {"Pa", "kPa", "bar", "atm"}),
        ("activation-energy", {"J/mol", "J/kmol", "cal/mol", "kcal/mol", "K"}),
    ):
        if key in result and result[key] not in allowed:
            raise ValueError(f"unsupported {key} unit: {result[key]}")
    return result


def _coefficient_factor(units, order):
    # A_native = A_source * (source-volume-in-L / source-amount-in-mol)^(n-1) / time-in-s.
    volume_L = (
        {"L": 1, "m^3": 1000, "cm^3": 0.001}[units["volume"]]
        if "volume" in units
        else {"m": 1000, "cm": 0.001, "mm": 1e-6}[units["length"]]
    )
    amount_mol = {"mol": 1, "kmol": 1000}[units["quantity"]]
    return (volume_L / amount_mol) ** (order - 1) / {"s": 1, "min": 60}[units["time"]]


def _arrhenius(node, units, order):
    _keys(node, {"A", "b", "Ea"}, "Arrhenius")
    a, b = _number(node["A"], "A"), _number(node.get("b", 0), "b")
    if a <= 0:
        raise ValueError("finite import requires positive A")
    ea = _quantity(
        node.get("Ea", 0),
        units["activation-energy"],
        {"J/mol": 1, "J/kmol": 0.001, "cal/mol": 4.184, "kcal/mol": 4184, "K": R_J_PER_MOL_K},
        "activation energy",
    )
    return {"A": a * _coefficient_factor(units, order), "b": b, "Ea_J_per_mol": ea}


def _side(text):
    result = {}
    for token in text.split(" + "):
        match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)\s+)?([A-Za-z][A-Za-z0-9_-]*)", token.strip())
        if match is None:
            raise ValueError(f"unsupported finite equation term: {token}")
        amount, species = match.groups()
        result[species] = result.get(species, 0) + float(amount or 1)
    return result


def _reaction(node, index, units, known, reference_pressure):
    kind = node.get("type", "elementary")
    allowed = {"equation", "type", "duplicate", "note", "id", "units"}
    fields = {
        "elementary": {"rate-constant"},
        "three-body": {"rate-constant", "efficiencies", "default-efficiency"},
        "falloff": {
            "low-P-rate-constant",
            "high-P-rate-constant",
            "Troe",
            "efficiencies",
            "default-efficiency",
        },
        "pressure-dependent-Arrhenius": {"rate-constants", "temperature-range"},
        "Chebyshev": {"temperature-range", "pressure-range", "data"},
    }
    if kind not in fields:
        raise ValueError(f"unsupported reaction type: {kind}")
    _keys(node, allowed | fields[kind], "reaction")
    units = _units({**units, **node.get("units", {})})
    equation = str(node["equation"])
    clean = equation.replace(" (+M)", "") if kind == "falloff" else equation
    arrow = "<=>" if "<=>" in clean else "=>"
    sides = clean.split(arrow)
    if len(sides) != 2:
        raise ValueError("reaction needs one => or <=> arrow")
    left, right = map(_side, sides)
    if kind == "three-body" and (left.pop("M", None) != 1 or right.pop("M", None) != 1):
        raise ValueError("three-body reaction requires one M on each side")
    if kind == "falloff" and equation.count("(+M)") != 2:
        raise ValueError("falloff requires (+M) on both sides")
    if (set(left) | set(right)) - known:
        raise ValueError("reaction references unknown species")
    order, reversible = sum(left.values()), arrow == "<=>"
    stoich = {key: right.get(key, 0) - left.get(key, 0) for key in left.keys() | right.keys()}
    stoich = {key: value for key, value in stoich.items() if value}
    params = {}
    if kind in {"elementary", "three-body"}:
        law = "arrhenius" if kind == "elementary" else "third_body_arrhenius"
        params.update(_arrhenius(node["rate-constant"], units, order + (kind == "three-body")))
    elif kind == "falloff":
        law = "troe_falloff" if "Troe" in node else "lindemann_falloff"
        for prefix, field, extra in (
            ("low", "low-P-rate-constant", 1),
            ("high", "high-P-rate-constant", 0),
        ):
            params.update(
                {
                    f"{prefix}_{key}": value
                    for key, value in _arrhenius(node[field], units, order + extra).items()
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
    if kind in {"three-body", "falloff"}:
        efficiencies = node.get("efficiencies", {})
        if set(efficiencies) - known:
            raise ValueError("third-body efficiencies reference unknown species")
        params.update(
            third_body_efficiencies={k: _number(v, "efficiency") for k, v in efficiencies.items()},
            default_efficiency=_number(node.get("default-efficiency", 1), "default efficiency"),
        )
    if law in {"pressure_log", "chebyshev"}:
        if "temperature-range" not in node:
            raise ValueError("finite pressure import requires explicit temperature-range")
        params.update(
            temperature_range_K=[_number(v, "temperature") for v in node["temperature-range"]],
            pressure_basis="ideal_gas",
        )
        validate_pressure_rate(params, law)
    if reversible:
        params.update(reverse_basis="ideal_gas_nasa7", reference_pressure_Pa=reference_pressure)
    reaction_id = str(node.get("id", f"reaction-{index + 1:03d}"))
    result = ReactionSpec(
        reaction_id,
        equation,
        stoich,
        reversible,
        RateLawSpec(reaction_id, law, params),
        metadata={"source_declaration": dict(node), "source_units": units},
        forward_orders=left,
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
        or len(names) > 64
        or any(not isinstance(v, str) for v in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("finite phase needs 1-64 unique locally declared species")
    units = _units(payload.get("units", {}))
    species, thermo_nodes, pressures = [], {}, set()
    source_species = payload["species"]
    if len(source_species) != len(names) or {s["name"] for s in source_species} != set(names):
        raise ValueError(
            "finite import requires exactly the selected local species, without omissions"
        )
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
        _keys(
            node,
            {"model", "temperature-ranges", "data", "reference-pressure", "note"},
            "thermochemistry",
        )
        pressure = _pressure(node.get("reference-pressure", "1 atm"), units)
        if pressure <= 0:
            raise ValueError("reference pressure must be positive")
        NASA7SpeciesThermo.from_cantera_yaml_thermo(
            species_id=name, thermo=node, composition=composition, reference_pressure_Pa=pressure
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
        }
    if len(pressures) != 1:
        raise ValueError("mixed thermochemical reference pressures are unsupported")
    reactions = payload["reactions"]
    if not isinstance(reactions, list) or not 1 <= len(reactions) <= 256:
        raise ValueError("finite import requires 1-256 reactions")
    converted = tuple(
        _reaction(node, i, units, set(names), next(iter(pressures)))
        for i, node in enumerate(reactions)
    )
    return ReactionNetworkSpec(
        str(phase["name"]),
        tuple(species),
        converted,
        metadata={
            "import_format": "finite-gas-mechanism-v1",
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
    if path.stat().st_size > 1_000_000:
        raise ValueError("finite mechanism input exceeds 1 MB")
    payload = (
        yaml.safe_load(path.read_text())
        if path.suffix.lower() != ".json"
        else json.loads(path.read_text())
    )
    return convert_mechanism(payload, phase_name=phase_name, source=path.name)


def main():
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
