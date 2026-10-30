"""Aligned dilute-gas transport responses using existing species and 22E kernels.

This is an explicitly selected first-order nonpolar model, not Cantera's full
Stockmayer/high-order transport implementation. It owns no material or time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from math import fsum, isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_transport_number(record: Mapping[str, Any], key: str) -> float:
    value = record[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not isfinite(value):
        raise ValueError(f"CHEMKIN transport {key} must be finite numeric")
    return float(value)


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


def gas_transport_from_chemkin(
    network: ReactionNetworkSpec,
    *,
    thermal_diffusion: str,
    collision_method: str = "Neufeld_Janzen_Aziz",
    conductivity_model: str = "Eucken",
) -> GasTransportSpec:
    """Build the bounded neutral CHEMKIN transport consumer.

    ``convert_chemkin`` retains the fixed-width ``TRANSPORT`` records in
    network metadata because they are not Cantera declarations.  This adapter
    is the only consumer for that source shape: it requires one exact record
    for every original gas species, preserves all source fields in the
    resulting :class:`LennardJonesSpec`, and converts only epsilon/k (K) and
    sigma (angstrom) to the existing first-order Lennard-Jones model.

    The current Chapman--Enskog closure is neutral and non-polar.  CHEMKIN
    dipole values other than zero therefore fail closed; polarizability and
    rotational relaxation are retained as provenance but are not guessed into
    Eucken, Wilke, or binary-diffusion corrections.  Pressure, temperature,
    composition, and collision-integral domains remain enforced by
    :func:`gas_transport_response` and the underlying kernels.
    """

    if not isinstance(network, ReactionNetworkSpec):
        raise TypeError("CHEMKIN transport requires a ReactionNetworkSpec")
    if network.metadata.get("import_format") != "chemkin-gas-mechanism-v1":
        raise ValueError("network is not a strict CHEMKIN gas import")
    if network.metadata.get("transport_section_present") is not True:
        raise ValueError("CHEMKIN transport consumer requires a TRANSPORT section")
    records = network.metadata.get("transport_records")
    if not isinstance(records, list | tuple):
        raise ValueError("CHEMKIN transport metadata records are missing")

    expected = {
        "species",
        "geometry",
        "geometry_name",
        "epsilon_over_k_K",
        "sigma_A",
        "dipole_D",
        "polarizability_A3",
        "rotational_relaxation",
    }
    geometry_names = {0: "atom", 1: "linear", 2: "nonlinear"}
    species_by_id = {item.species_id: item for item in network.species}
    if len(species_by_id) != len(network.species) or not species_by_id:
        raise ValueError("CHEMKIN transport requires unique original species")
    if any(item.phase != "gas" or item.charge for item in network.species):
        raise ValueError("CHEMKIN transport supports only original neutral gas species")
    if len(records) != len(species_by_id):
        raise ValueError("CHEMKIN TRANSPORT must provide exactly one record per species")

    parameters: list[LennardJonesSpec] = []
    geometry: dict[str, str] = {}
    seen: set[str] = set()
    for raw in records:
        if not isinstance(raw, Mapping) or set(raw) != expected:
            raise ValueError("CHEMKIN transport record has missing or extra fields")
        species_id = raw["species"]
        if not isinstance(species_id, str) or species_id not in species_by_id:
            raise ValueError("CHEMKIN transport record references unknown species")
        if species_id in seen:
            raise ValueError(f"duplicate CHEMKIN transport record: {species_id}")
        geometry_value = raw["geometry"]
        if isinstance(geometry_value, bool) or not isinstance(geometry_value, int):
            raise ValueError("CHEMKIN transport geometry must be integer 0, 1, or 2")
        if geometry_value not in geometry_names:
            raise ValueError("CHEMKIN transport geometry must be integer 0, 1, or 2")
        if raw["geometry_name"] != geometry_names[geometry_value]:
            raise ValueError("CHEMKIN transport geometry name disagrees with geometry")

        epsilon = _finite_transport_number(raw, "epsilon_over_k_K")
        sigma = _finite_transport_number(raw, "sigma_A")
        dipole = _finite_transport_number(raw, "dipole_D")
        polarizability = _finite_transport_number(raw, "polarizability_A3")
        rotational_relaxation = _finite_transport_number(raw, "rotational_relaxation")
        if epsilon <= 0 or sigma <= 0:
            raise ValueError("CHEMKIN transport epsilon and sigma must be positive")
        if dipole < 0 or polarizability < 0 or rotational_relaxation < 0:
            raise ValueError("CHEMKIN transport optional properties must be nonnegative")
        if dipole != 0:
            raise ValueError("CHEMKIN transport dipole is outside the neutral nonpolar LJ consumer")
        record = dict(raw)
        parameters.append(
            LennardJonesSpec(
                species_id,
                epsilon,
                sigma * 1.0e-10,
                "Original CHEMKIN TRANSPORT record; neutral nonpolar first-order LJ",
                sources={
                    "source_declaration": record,
                    "source_units": {
                        "epsilon_over_k_K": "K",
                        "sigma_A": "angstrom",
                        "dipole_D": "debye",
                        "polarizability_A3": "angstrom^3",
                        "rotational_relaxation": "1",
                    },
                    "ignored_fields": ["polarizability_A3", "rotational_relaxation"],
                },
            )
        )
        geometry[species_id] = geometry_names[geometry_value]
        seen.add(species_id)
    if seen != set(species_by_id):
        raise ValueError("CHEMKIN TRANSPORT records do not cover every species")
    return GasTransportSpec(
        tuple(parameters),
        geometry,
        thermal_diffusion,
        collision_method=collision_method,
        conductivity_model=conductivity_model,
    )


def gas_transport_capabilities() -> dict[str, Any]:
    """Return the auditable model and source-consumer boundary.

    A CHEMKIN ``TRANSPORT`` record includes the inputs used by a Stockmayer
    potential, but the native coefficient table currently contains only the
    nonpolar Lennard-Jones fits.  In particular, no validated
    Monchick--Mason polar collision-integral table or fit is bundled here.
    Advertising that gap makes the nonzero-dipole rejection an explicit
    capability boundary instead of allowing callers to infer a correction.
    """

    from chemworld.physchem.gas_polynomial_transport import gas_polynomial_transport_capabilities

    return {
        "model": "dilute-gas-transport-v1",
        "source_consumers": {
            "cantera": {
                "consumer": "gas_transport_from_network",
                "status": "supported",
                "potential": "neutral-nonpolar Lennard-Jones 12-6",
            },
            "chemkin": {
                "consumer": "gas_transport_from_chemkin",
                "status": "supported",
                "potential": "neutral-nonpolar Lennard-Jones 12-6",
                "required_source_fields": [
                    "geometry",
                    "epsilon_over_k_K",
                    "sigma_A",
                    "dipole_D",
                    "polarizability_A3",
                    "rotational_relaxation",
                ],
            },
        },
        "supported": {
            "collision_integrals": ["Neufeld_Janzen_Aziz", "Kim_Monroe"],
            "mixture_viscosity": "Wilke",
            "mixture_conductivity": "Wassiljewa-Herning-Zipperer",
            "binary_combining_rule": "Lorentz-Berthelot",
            "thermal_diffusion": ["off"],
            "maximum_pressure_Pa": 200000.0,
        },
        "polar_stockmayer": {
            "status": "unsupported-fail-closed",
            "reason": (
                "validated Monchick-Mason polar collision-integral table or fit is not bundled"
            ),
            "required_for_enablement": [
                "validated Omega(1,1)* and Omega(2,2)* Stockmayer data or fit",
                "reproducible reduced-dipole and polarizability unit conversion",
                "independent polar mixture qualification",
            ],
            "policy": "reject any nonzero dipole_D; do not infer polar corrections",
        },
        "ignored_source_fields": ["polarizability_A3", "rotational_relaxation"],
        "scope": "neutral nonpolar first-order dilute gas; no high-order kinetic transport",
        "explicit_alternative_models": [gas_polynomial_transport_capabilities()],
    }


def gas_transport_from_dict(
    value: Mapping[str, Any],
) -> GasTransportSpec | GasTransportPolynomialSpec:
    """Select a declared codec; never fall back from a failed alternative."""
    if value.get("model") in ("gas-transport-polynomial-v1", "gas-transport-polynomial-soret-v2"):
        from chemworld.physchem.gas_polynomial_transport import GasTransportPolynomialSpec

        return GasTransportPolynomialSpec.from_dict(value)
    return GasTransportSpec.from_dict(value)


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
    thermal_diffusion_coefficients_kg_m_s: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "species_ids", tuple(self.species_ids))
        soret = self.thermal_diffusion_coefficients_kg_m_s
        enabled = self.models.get("thermal_diffusion") == "cantera-3.2-mixture-soret"
        if enabled != (soret is not None):
            raise ValueError("Soret response mode requires its explicit coefficient vector")
        if soret is not None and (
            not isinstance(soret, Mapping)
            or set(soret) != set(self.species_ids)
            or any(
                isinstance(v, bool) or not isinstance(v, int | float) or not isfinite(v)
                for v in soret.values()
            )
        ):
            raise ValueError(
                "Soret response coefficients must be finite signed values for every species"
            )
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Mapping):
                object.__setattr__(self, item.name, _frozen(value))

    def to_dict(self) -> dict[str, Any]:
        enabled = self.thermal_diffusion_coefficients_kg_m_s is not None
        return {
            "model": "dilute-gas-transport-response-v2"
            if enabled
            else "dilute-gas-transport-response-v1",
            **{
                f.name: _plain(getattr(self, f.name))
                for f in fields(self)
                if f.name != "thermal_diffusion_coefficients_kg_m_s" or enabled
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasTransportResponse:
        payload = dict(value)
        model = payload.pop("model")
        enabled = model == "dilute-gas-transport-response-v2"
        if model not in {"dilute-gas-transport-response-v1", "dilute-gas-transport-response-v2"}:
            raise ValueError("unknown transport response")
        if enabled != ("thermal_diffusion_coefficients_kg_m_s" in payload):
            raise ValueError("response schema and Soret field differ")
        if enabled and payload["thermal_diffusion_coefficients_kg_m_s"] is None:
            raise ValueError("enabled Soret response requires its coefficient vector")
        return cls(**payload)


def gas_transport_response(
    spec: GasTransportSpec | GasTransportPolynomialSpec,
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
    if not isinstance(spec, GasTransportSpec):
        from chemworld.physchem.gas_polynomial_transport import (
            GasTransportPolynomialSpec,
            gas_polynomial_transport_response,
        )

        if not isinstance(spec, GasTransportPolynomialSpec):
            raise ValueError("declare a supported gas transport model")
        return gas_polynomial_transport_response(
            spec,
            network,
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            mole_fractions=mole_fractions,
        )
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
        if not (molecular_weight is not None):
            raise ValueError("gas transport requires a molecular weight for every species")
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
