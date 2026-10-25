"""Original mechanism thermochemistry as an explicit pure ideal reference.

The source declaration remains authoritative. This adapter changes the ideal
standard state from p0 to the requested density; it does not fit a Helmholtz
polynomial or add ideal mixing entropy (the mixture already owns that term).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite, log
from typing import Any

from chemworld.physchem.elements import molecular_weight
from chemworld.physchem.eos import R_J_PER_MOL_K as R
from chemworld.physchem.gas_mechanism_batch import IdealGasThermodynamics
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml


def _source_encoding(source: Mapping[str, Any]) -> str:
    # YAML piecewise-Gibbs temperature keys become strings in a JSON snapshot.
    return json.dumps(json.loads(json.dumps(dict(source), allow_nan=False)), sort_keys=True)


@dataclass(frozen=True)
class MechanismIdealState:
    internal_energy_J_mol: float
    entropy_J_mol_K: float
    cv_J_mol_K: float
    cp_J_mol_K: float


@dataclass(frozen=True, init=False)
class MechanismIdealReference:
    """Immutable source snapshot; evaluations delegate to the original importer."""

    fluid_id: str
    _source_json: str
    _thermo: Any = field(compare=False, repr=False)
    molar_mass_kg_mol: float
    gas_constant_J_mol_K: float = field(default=R, init=False)
    residual_terms: tuple[()] = field(default=(), init=False)
    pseudo_pure: bool = field(default=False, init=False)
    ideal_prefactor: float = field(default=1.0, init=False)

    def __init__(self, fluid_id: str, source_declaration: Mapping[str, Any]) -> None:
        source_json = _source_encoding(source_declaration)
        source = json.loads(source_json)
        thermo = species_thermo_from_cantera_yaml(species_id=fluid_id, **source)
        object.__setattr__(self, "fluid_id", fluid_id)
        object.__setattr__(self, "_source_json", source_json)
        object.__setattr__(self, "_thermo", thermo)
        object.__setattr__(
            self, "molar_mass_kg_mol", molecular_weight(source["composition"]) / 1000
        )

    @property
    def source_declaration(self) -> dict[str, Any]:
        return json.loads(self._source_json)

    def matches_source(self, source_declaration: Mapping[str, Any]) -> bool:
        return self._source_json == _source_encoding(source_declaration)

    @property
    def minimum_temperature_K(self) -> float:  # noqa: N802
        return float(self._thermo.min_temperature_K)

    @property
    def maximum_temperature_K(self) -> float:  # noqa: N802
        return float(self._thermo.max_temperature_K)

    def validate_temperature_interval(self, lower_K: float, upper_K: float) -> None:
        IdealGasThermodynamics({self.fluid_id: self._thermo}).validate_temperature_interval(
            lower_K, upper_K
        )

    def state(self, *, temperature_K: float, molar_density_mol_m3: float) -> MechanismIdealState:
        if any(
            isinstance(v, bool) or not isfinite(v) or v <= 0
            for v in (temperature_K, molar_density_mol_m3)
        ):
            raise ValueError("mechanism ideal reference requires positive finite T and density")
        value = self._thermo.evaluate(temperature_K)
        return MechanismIdealState(
            value.enthalpy_J_mol - R * temperature_K,
            value.entropy_J_mol_K
            - R
            * log(molar_density_mol_m3 * R * temperature_K / self._thermo.reference_pressure_Pa),
            value.cp_J_mol_K - R,
            value.cp_J_mol_K,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "mechanism-ideal-reference-v1",
            "fluid_id": self.fluid_id,
            "source_declaration": self.source_declaration,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MechanismIdealReference:
        if data.get("model") != "mechanism-ideal-reference-v1":
            raise ValueError("unknown mechanism ideal reference encoding")
        return cls(str(data["fluid_id"]), data["source_declaration"])
