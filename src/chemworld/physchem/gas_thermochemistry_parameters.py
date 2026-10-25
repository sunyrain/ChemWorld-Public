"""Physical species enthalpy parameters through original source declarations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from typing import Any

from chemworld.physchem.mechanism_thermochemistry import molar_quantity
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R


@dataclass(frozen=True)
class GasThermochemistryParameter:
    """Species H at a stated temperature; additive steps have physical J/mol units.

    H and G shift together while S and Cp stay fixed. This is not a heat-transfer
    coefficient or an independently edited reaction heat. The initial physical
    n/T state stays fixed; its newly declared energy consequently changes.
    """

    species_id: str
    step_J_mol: float
    reference_temperature_K: float = 298.15

    def __post_init__(self) -> None:
        if (
            not isinstance(self.species_id, str)
            or not self.species_id
            or self.species_id.strip() != self.species_id
        ):
            raise ValueError("enthalpy parameter requires an explicit species")
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (self.step_J_mol, self.reference_temperature_K)
        ):
            raise ValueError("enthalpy step and reference temperature must be positive finite")

    @property
    def parameter_id(self) -> str:
        import json

        return json.dumps(
            ["species_enthalpy_J_mol", self.species_id, self.reference_temperature_K],
            separators=(",", ":"),
        )


def species_enthalpy_parameter(network: Any, parameter: GasThermochemistryParameter) -> float:
    thermo = network.imported_gas_thermochemistry
    if parameter.species_id not in thermo:
        raise ValueError("enthalpy parameter needs original imported species thermochemistry")
    return float(
        thermo[parameter.species_id].evaluate(parameter.reference_temperature_K).enthalpy_J_mol
    )


def with_species_enthalpy(
    network: Any, parameter: GasThermochemistryParameter, value_J_mol: float
) -> Any:
    """Change source integration constants; original evaluators consume every result."""
    if isinstance(value_J_mol, bool) or not isfinite(value_J_mol):
        raise ValueError("species enthalpy must be finite")
    initial = species_enthalpy_parameter(network, parameter)
    delta = value_J_mol - initial
    metadata = deepcopy(network.metadata)
    declaration = metadata["gas_thermochemistry"][parameter.species_id]
    thermo, units = declaration["thermo"], declaration["units"]
    model = thermo["model"]
    if model in {"NASA7", "NASA9", "Shomate"}:
        column, divisor = {"NASA7": (5, R), "NASA9": (7, R), "Shomate": (5, 1000.0)}[model]
        thermo["data"] = [list(row) for row in thermo["data"]]
        for row in thermo["data"]:
            row[column] += delta / divisor
    elif model in {"constant-cp", "piecewise-Gibbs"}:
        thermo["h0"] = f"{molar_quantity(thermo.get('h0', 0), units) + delta:.17g} J/mol"
        if model == "piecewise-Gibbs":
            thermo["data"] = {
                str(t): float(g) + delta / (R * float(t))
                if thermo.get("dimensionless", False)
                else f"{molar_quantity(g, units) + delta:.17g} J/mol"
                for t, g in thermo["data"].items()
            }
    else:
        raise ValueError("unsupported original species thermochemistry")
    metadata["analysis_enthalpy_parameter"] = {
        "species_id": parameter.species_id,
        "reference_temperature_K": parameter.reference_temperature_K,
        "initial_enthalpy_J_mol": initial,
        "enthalpy_J_mol": value_J_mol,
        "source_sha256_meaning": "original imported source before explicit analysis perturbation",
    }
    # Keep original source metadata as provenance. The detached authoritative
    # gas_thermochemistry declaration above is the candidate's physical input.
    result = replace(network, metadata=metadata)
    species_enthalpy_parameter(result, parameter)
    return result
