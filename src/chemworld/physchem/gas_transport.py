"""Aligned dilute-gas transport responses using existing species and 22E kernels.

This is an explicitly selected first-order nonpolar model, not Cantera's full
Stockmayer/high-order transport implementation. It owns no material or time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any

from chemworld.physchem.conductivity_kernels import conductivity_value
from chemworld.physchem.conductivity_mixtures import wassiljewa_herning_zipperer_conductivity
from chemworld.physchem.lennard_jones import (
    LennardJonesSpec,
    lennard_jones_diffusivity_report,
    lennard_jones_viscosity_report,
)
from chemworld.physchem.property_reports import _validated_fraction_mapping
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.viscosity_mixtures import wilke_viscosity


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(v) for key, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _frozen(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _frozen(v) for key, v in value.items()})
    return value


@dataclass(frozen=True)
class GasTransportSpec:
    parameters: tuple[LennardJonesSpec, ...]
    geometry: Mapping[str, str]
    thermal_diffusion: str
    collision_method: str = "Neufeld_Janzen_Aziz"
    conductivity_model: str = "Eucken"
    maximum_pressure_Pa: float = 200000.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "geometry", _frozen(self.geometry))
        ids = [item.component_id for item in self.parameters]
        if not ids or len(set(ids)) != len(ids) or set(ids) != set(self.geometry):
            raise ValueError("transport parameters and geometry must match unique original species")
        if any(v not in {"atom", "linear", "nonlinear"} for v in self.geometry.values()):
            raise ValueError("unsupported gas transport geometry")
        if self.thermal_diffusion != "off":
            raise ValueError("this transport model requires explicit thermal_diffusion='off'")
        if self.collision_method not in {"Neufeld_Janzen_Aziz", "Kim_Monroe"}:
            raise ValueError("unknown collision integral model")
        if self.conductivity_model not in {"Eucken", "Eucken_modified"}:
            raise ValueError("unknown pure gas conductivity model")
        _positive(self.maximum_pressure_Pa, "maximum pressure")
        if self.maximum_pressure_Pa > 200000:
            raise ValueError("this first dilute-gas model is restricted to at most 200000 Pa")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "dilute-gas-transport-v1",
            "parameters": [p.to_dict() for p in self.parameters],
            "geometry": dict(self.geometry),
            "thermal_diffusion": self.thermal_diffusion,
            "collision_method": self.collision_method,
            "conductivity_model": self.conductivity_model,
            "maximum_pressure_Pa": self.maximum_pressure_Pa,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasTransportSpec:
        payload = dict(value)
        if payload.pop("model") != "dilute-gas-transport-v1":
            raise ValueError("unknown gas transport declaration")
        payload["parameters"] = tuple(LennardJonesSpec.from_dict(p) for p in payload["parameters"])
        return cls(**payload)


def gas_transport_from_network(
    network: ReactionNetworkSpec,
    *,
    thermal_diffusion: str,
    collision_method: str = "Neufeld_Janzen_Aziz",
    conductivity_model: str = "Eucken",
) -> GasTransportSpec:
    """Consume preserved Cantera public transport declarations without executing them.

    Cantera gas declarations use diameter in angstrom, well-depth in kelvin,
    dipole in debye and polarizability in cubic angstrom. Nonzero dipole or a
    charged original species is rejected. Rotational relaxation/polarizability
    are preserved in the source note but are not inferred into Eucken models.
    """
    parameters, geometry = [], {}
    for species in network.species:
        if species.phase != "gas" or species.charge:
            raise ValueError("gas transport supports only original neutral gas species")
        source = species.metadata.get("source_declaration")
        if not isinstance(source, Mapping) or not isinstance(source.get("transport"), Mapping):
            raise ValueError(f"missing explicit transport declaration for {species.species_id}")
        transport = dict(source["transport"])
        allowed = {
            "model",
            "geometry",
            "diameter",
            "well-depth",
            "dipole",
            "polarizability",
            "rotational-relaxation",
            "acentric-factor",
            "dispersion-coefficient",
            "quadrupole-polarizability",
            "note",
        }
        if set(transport) - allowed or transport.get("model") != "gas":
            raise ValueError("unsupported gas transport source declaration")
        for key in (
            "dipole",
            "polarizability",
            "rotational-relaxation",
            "acentric-factor",
            "dispersion-coefficient",
            "quadrupole-polarizability",
        ):
            value = transport.get(key, 0)
            if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
                raise ValueError("transport molecular parameters must be finite declared numbers")
        if transport.get("dipole", 0) != 0:
            raise ValueError("polar Stockmayer transport is outside this nonpolar model")
        if transport.get("dispersion-coefficient", 0) or transport.get(
            "quadrupole-polarizability", 0
        ):
            raise ValueError("higher-order transport potentials are outside this LJ model")
        for name in ("diameter", "well-depth"):
            if not isinstance(transport.get(name), int | float):
                raise ValueError(
                    "Cantera gas diameter/well-depth must use explicit numeric source units"
                )
            _positive(transport[name], name)
        parameters.append(
            LennardJonesSpec(
                species.species_id,
                transport["well-depth"],
                transport["diameter"] * 1e-10,
                "Original gas species transport declaration; neutral nonpolar first-order LJ",
                sources={"source_declaration": transport},
            )
        )
        geometry[species.species_id] = transport["geometry"]
    return GasTransportSpec(
        tuple(parameters),
        geometry,
        thermal_diffusion,
        collision_method=collision_method,
        conductivity_model=conductivity_model,
    )


@dataclass(frozen=True)
class GasTransportResponse:
    species_ids: tuple[str, ...]
    temperature_K: float
    pressure_Pa: float
    mole_fractions: Mapping[str, float]
    mass_fractions: Mapping[str, float]
    molecular_weights_kg_mol: Mapping[str, float]
    species_viscosities_Pa_s: Mapping[str, float]
    species_conductivities_W_m_K: Mapping[str, float]
    binary_diffusivities_m2_s: Mapping[str, Mapping[str, float]]
    diffusion_coefficients_m2_s: Mapping[str, Mapping[str, float | None]]
    partial_enthalpies_J_mol: Mapping[str, float]
    molar_density_mol_m3: float
    density_kg_m3: float
    viscosity_Pa_s: float
    conductivity_W_m_K: float
    models: Mapping[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_ids", tuple(self.species_ids))
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, item.name, _frozen(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "dilute-gas-transport-response-v1",
            **{f.name: _plain(getattr(self, f.name)) for f in fields(self)},
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasTransportResponse:
        payload = dict(value)
        if payload.pop("model") != "dilute-gas-transport-response-v1":
            raise ValueError("unknown transport response")
        return cls(**payload)


def gas_transport_response(
    spec: GasTransportSpec,
    network: ReactionNetworkSpec,
    *,
    temperature_K: float,
    pressure_Pa: float,
    mole_fractions: Mapping[str, float],
) -> GasTransportResponse:
    """Full pair matrix and three explicitly named gradient/coefficient bases.

    Coefficient keys follow (flux quantity, composition gradient): molar_mole,
    mass_mole, mass_mass. At a pure vertex the populated species' mixture
    coefficient is direction-dependent and is None; the binary diagonal still
    reports its well-defined tracer self-diffusivity.
    """
    _positive(temperature_K, "temperature")
    _positive(pressure_Pa, "pressure")
    if pressure_Pa > spec.maximum_pressure_Pa:
        raise ValueError("pressure exceeds declared dilute-gas transport domain")
    ids = network.species_ids
    parameters = {p.component_id: p for p in spec.parameters}
    if set(ids) != set(parameters) or set(ids) != set(mole_fractions):
        raise ValueError("transport must use exactly the original network species")
    fractions = _validated_fraction_mapping(mole_fractions, "mole")
    # Only unit-sum roundoff is removed after the existing explicit validation.
    total = fsum(fractions.values())
    x = {s: fractions[s] / total for s in ids}
    thermo = network.imported_gas_thermochemistry
    if set(thermo) != set(ids) or any(s.phase != "gas" or s.charge for s in network.species):
        raise ValueError("neutral gas transport requires all original species thermochemistry")
    components = {
        s.species_id: ComponentSpec(s.species_id, s.formula, charge=s.charge, default_phase="gas")
        for s in network.species
    }
    masses = {}
    for s in ids:
        molecular_weight = components[s].molecular_weight_g_mol
        assert molecular_weight is not None
        masses[s] = molecular_weight / 1000
    mean_mass = fsum(x[s] * masses[s] for s in ids)
    y = {s: x[s] * masses[s] / mean_mass for s in ids}
    viscosity, conductivity, enthalpy = {}, {}, {}
    for s in ids:
        state = thermo[s].evaluate(temperature_K)
        if state.cp_J_mol_K <= R:
            raise ValueError("ideal gas must have positive finite Cv")
        viscosity[s] = lennard_jones_viscosity_report(
            components[s],
            parameters[s],
            temperature_K=temperature_K,
            collision_method=spec.collision_method,
        ).evaluation.value
        conductivity[s] = conductivity_value(
            spec.conductivity_model,
            temperature_K,
            MW=masses[s] * 1000,
            Cvm=state.cp_J_mol_K - R,
            mu=viscosity[s],
        )
        enthalpy[s] = state.enthalpy_J_mol
    binary: dict[str, dict[str, float]] = {s: {} for s in ids}
    for i, s in enumerate(ids):
        for target in ids[i:]:
            value = lennard_jones_diffusivity_report(
                components[s],
                parameters[s],
                components[target],
                parameters[target],
                temperature_K=temperature_K,
                pressure_Pa=pressure_Pa,
                collision_method=spec.collision_method,
                combining_rule="Lorentz_Berthelot",
            ).evaluation.value
            binary[s][target] = binary[target][s] = value
    mixture: dict[str, dict[str, float | None]] = {
        k: {} for k in ("molar_mole", "mass_mole", "mass_mass")
    }
    for s in ids:
        others = [j for j in ids if j != s]
        denominator = fsum(x[j] / binary[s][j] for j in others)
        other_x, other_y = fsum(x[j] for j in others), fsum(y[j] for j in others)
        if denominator == 0:
            for values in mixture.values():
                values[s] = None
            continue
        mixture["molar_mole"][s] = other_x / denominator
        mixture["mass_mole"][s] = other_y / denominator
        mass_resistance = denominator + x[s] * fsum(y[j] / binary[s][j] for j in others) / other_y
        mixture["mass_mass"][s] = 1 / mass_resistance
    weights = tuple(masses[s] * 1000 for s in ids)
    molar_density = pressure_Pa / (R * temperature_K)
    return GasTransportResponse(
        ids,
        temperature_K,
        pressure_Pa,
        x,
        y,
        masses,
        viscosity,
        conductivity,
        binary,
        mixture,
        enthalpy,
        molar_density,
        molar_density * mean_mass,
        wilke_viscosity(tuple(x.values()), tuple(viscosity.values()), weights),
        wassiljewa_herning_zipperer_conductivity(
            tuple(x.values()), tuple(conductivity.values()), weights
        ),
        {
            "collision": spec.collision_method,
            "pure_conductivity": spec.conductivity_model,
            "mixture_viscosity": "Wilke",
            "mixture_conductivity": "Wassiljewa-Herning-Zipperer",
            "thermal_diffusion": spec.thermal_diffusion,
            "binary_combining_rule": "Lorentz-Berthelot",
            "scope": "neutral nonpolar first-order dilute gas; no high-order kinetic transport",
        },
    )
