"""Explicit source-fit dilute-gas transport; no source-library execution.

The five ascending coefficients multiply powers of ln(T/K).  This module
implements the DEFAULT (non-CK) Cantera 3.2 parameterization, not its collision
theory or coefficient generation.  A provider declaration is not evidence that
the coefficients have passed an upstream comparison.  Material and time remain
owned by the original reaction-network and runtime consumers.

Formula reference: Cantera 3.2 GasTransport.cpp updateSpeciesViscosities,
updateDiff_T and MixTransport.cpp updateCond_T / thermalConductivity:
https://cantera.org/3.2/cxx/d5/de4/GasTransport_8cpp_source.html
https://cantera.org/3.2/cxx/d9/d17/classCantera_1_1MixTransport.html
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import fsum, isfinite, log, sqrt
from types import MappingProxyType
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.gas_soret_transport import GasMixtureSoretSpec
from chemworld.physchem.gas_transport import GasTransportResponse
from chemworld.physchem.property_reports import _validated_fraction_mapping
from chemworld.physchem.reaction_network import R_J_PER_MOL_K as R
from chemworld.physchem.reaction_network import ReactionNetworkSpec, SpeciesSpec
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml
from chemworld.physchem.viscosity_mixtures import wilke_viscosity

MODEL = "gas-transport-polynomial-v1"
SORET_MODEL = "gas-transport-polynomial-soret-v2"
SORET_MODE = "cantera-3.2-mixture-soret"
FORMULA_ID = "cantera-3.2-default-gas-transport-polynomials-v1"
COEFFICIENT_BASIS = "ascending powers of ln(T/K); binary coefficients normalized at 1 Pa"
UNITS = MappingProxyType(
    {
        "temperature": "K",
        "pressure": "Pa",
        "viscosity": "Pa*s",
        "conductivity": "W/(m*K)",
        "binary_diffusivity": "m^2/s",
    }
)
_SPECIES_FIELDS = {
    "species_id",
    "formula",
    "composition",
    "phase",
    "charge",
    "catalyst",
    "observable_aliases",
    "metadata",
}
_SOURCE_FIELDS = {"source_id", "priority", "citation", "version", "license", "checksum"}
_SPEC_FIELDS = {
    "species_ids",
    "species_declarations",
    "thermochemistry_declarations",
    "molecular_weights_kg_mol",
    "viscosity_coefficients",
    "conductivity_coefficients",
    "binary_diffusivity_coefficients",
    "source",
    "source_data_kind",
    "fit_temperature_range_K",
    "maximum_pressure_Pa",
    "thermal_diffusion",
    "mode",
    "formula_id",
}


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be nonempty text without surrounding whitespace")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the numeric domain") from exc
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _keys(value: Any, expected: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be an object with string keys")
    if set(value) != expected:
        raise ValueError(f"{name} has missing or unknown fields")


def _sequence(value: Any, name: str) -> tuple:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(value)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("source declarations require string object keys")
        return MappingProxyType({k: _freeze_json(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(v) for v in value)
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int | float):
        _number(value, "source declaration number")
        return value
    raise ValueError("source declarations must contain only finite JSON data")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _same_declaration(left: Any, right: Any) -> bool:
    # JSON spelling also distinguishes a numeric field from a boolean and
    # prevents source bindings being satisfied by Python's True == 1 equality.
    return json.dumps(_plain(left), sort_keys=True, allow_nan=False) == json.dumps(
        _plain(_freeze_json(right)), sort_keys=True, allow_nan=False
    )


def _coefficients(value: Any, name: str) -> tuple[float, ...]:
    values = _sequence(value, name)
    if len(values) != 5:
        raise ValueError(f"{name} needs exactly five DEFAULT coefficients")
    return tuple(_number(v, name) for v in values)


def _source(value: Any) -> DataSourceProvenance:
    if not isinstance(value, DataSourceProvenance):
        raise ValueError("source must be DataSourceProvenance")
    for name in _SOURCE_FIELDS - {"priority"}:
        _text(getattr(value, name), f"source.{name}")
    if type(value.priority) is not int or value.priority < 0:
        raise ValueError("source priority must be a nonnegative integer")
    return value


def _species(raw: Any, species_id: str) -> ComponentSpec:
    _keys(raw, _SPECIES_FIELDS, "species declaration")
    if raw["species_id"] != species_id or raw["phase"] != "gas":
        raise ValueError("source species must identify the original gas species")
    if type(raw["charge"]) is not int or raw["charge"] != 0:
        raise ValueError("source-fit transport supports only neutral gas species")
    if type(raw["catalyst"]) is not bool:
        raise ValueError("species catalyst must be a boolean")
    formula = _text(raw["formula"], "species formula")
    aliases = _sequence(raw["observable_aliases"], "observable_aliases")
    for alias in aliases:
        _text(alias, "observable alias")
    if not isinstance(raw["metadata"], Mapping) or not isinstance(raw["composition"], Mapping):
        raise ValueError("species metadata and composition must be objects")
    for atom, number in raw["composition"].items():
        _text(atom, "element identity")
        _number(number, "species atom count", positive=True)
    native = SpeciesSpec(
        species_id,
        formula,
        phase="gas",
        charge=0,
        catalyst=raw["catalyst"],
        observable_aliases=aliases,
        metadata=_plain(raw["metadata"]),
    )
    if not _same_declaration(raw, native.to_dict()):
        raise ValueError("source species declaration differs from SpeciesSpec.to_dict()")
    return ComponentSpec(species_id, formula, charge=0, default_phase="gas")


def _thermo(raw: Any, species_id: str, composition: Mapping[str, float]) -> Any:
    if (
        not isinstance(raw, Mapping)
        or "thermo" not in raw
        or set(raw)
        - {
            "thermo",
            "composition",
            "reference_pressure_Pa",
            "units",
        }
    ):
        raise ValueError("thermochemistry declaration has missing or unknown fields")
    if not isinstance(raw["thermo"], Mapping):
        raise ValueError("thermo must be an original thermochemistry object")
    if "composition" in raw:
        if not isinstance(raw["composition"], Mapping):
            raise ValueError("thermochemistry composition must be an object")
        for number in raw["composition"].values():
            _number(number, "thermochemistry atom count", positive=True)
        if raw["composition"] != composition:
            raise ValueError("thermochemistry composition differs from original species")
    if "reference_pressure_Pa" in raw:
        _number(raw["reference_pressure_Pa"], "reference pressure", positive=True)
    if "units" in raw and not isinstance(raw["units"], Mapping):
        raise ValueError("thermochemistry units must be an object")
    node = raw["thermo"]
    if node.get("model") == "NASA7" and set(node) - {
        "model",
        "temperature-ranges",
        "data",
        "reference-pressure",
        "note",
    }:
        raise ValueError("unknown NASA7 thermochemistry field")

    def reject_numeric_booleans(value: Any, key: str = "") -> None:
        if isinstance(value, bool) and key != "dimensionless":
            raise ValueError("boolean thermochemistry numbers are unsupported")
        if isinstance(value, Mapping):
            for k, v in value.items():
                reject_numeric_booleans(v, k)
        elif isinstance(value, tuple):
            for v in value:
                reject_numeric_booleans(v, key)

    reject_numeric_booleans(node)
    try:
        return species_thermo_from_cantera_yaml(
            species_id=species_id, boundary_policy="continuous-energy", **_plain(raw)
        )
    except (KeyError, TypeError, IndexError, ArithmeticError) as exc:
        raise ValueError("invalid original thermochemistry declaration") from exc


@dataclass(frozen=True)
class GasTransportPolynomialSpec:
    """Immutable coefficients bound to complete original native declarations.

    All species, including exact zero-fraction species, need both pure-property
    fits and every binary pair including the diagonal.  Source metadata records
    provenance only; it does not authenticate a provider or claim qualification.
    """

    species_ids: tuple[str, ...]
    species_declarations: Mapping[str, Mapping[str, Any]]
    thermochemistry_declarations: Mapping[str, Mapping[str, Any]]
    molecular_weights_kg_mol: Mapping[str, float]
    viscosity_coefficients: Mapping[str, tuple[float, ...]]
    conductivity_coefficients: Mapping[str, tuple[float, ...]]
    binary_diffusivity_coefficients: Mapping[str, Mapping[str, tuple[float, ...]]]
    source: DataSourceProvenance
    source_data_kind: str
    fit_temperature_range_K: tuple[float, float]
    maximum_pressure_Pa: float
    thermal_diffusion: str
    mode: str = "DEFAULT"
    formula_id: str = FORMULA_ID
    soret: GasMixtureSoretSpec | None = None

    def __post_init__(self) -> None:
        ids = _sequence(self.species_ids, "species_ids")
        for species_id in ids:
            _text(species_id, "species id")
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("species ids must be nonempty and unique")
        object.__setattr__(self, "species_ids", ids)
        if self.mode != "DEFAULT" or self.formula_id != FORMULA_ID:
            raise ValueError("only the Cantera 3.2 DEFAULT polynomial formula is supported")
        if self.thermal_diffusion not in ("off", SORET_MODE):
            raise ValueError("declare off or the explicit Cantera mixture-Soret model")
        if (self.thermal_diffusion == "off") != (self.soret is None):
            raise ValueError("Soret requires its explicit child declaration and matching mode")
        if self.soret is not None and not isinstance(self.soret, GasMixtureSoretSpec):
            raise ValueError("Soret must be GasMixtureSoretSpec")
        if not isinstance(self.source_data_kind, str) or self.source_data_kind not in {
            "authored-synthetic",
            "upstream-fit",
        }:
            raise ValueError("declare source-fit or authored synthetic coefficient origin")
        _source(self.source)
        maximum = _number(self.maximum_pressure_Pa, "maximum pressure", positive=True)
        if maximum > 200000:
            raise ValueError("source-fit dilute transport is restricted to at most 200000 Pa")
        object.__setattr__(self, "maximum_pressure_Pa", maximum)
        bounds = _sequence(self.fit_temperature_range_K, "fit temperature range")
        if len(bounds) != 2:
            raise ValueError("fit temperature range needs exactly two endpoints")
        bounds = tuple(_number(v, "fit temperature", positive=True) for v in bounds)
        if bounds[0] >= bounds[1]:
            raise ValueError("fit temperature range must be strictly increasing")
        object.__setattr__(self, "fit_temperature_range_K", bounds)
        for name in (
            "species_declarations",
            "thermochemistry_declarations",
            "molecular_weights_kg_mol",
            "viscosity_coefficients",
            "conductivity_coefficients",
            "binary_diffusivity_coefficients",
        ):
            _keys(getattr(self, name), set(ids), name)
        for name in ("species_declarations", "thermochemistry_declarations"):
            object.__setattr__(self, name, _freeze_json(getattr(self, name)))
        masses = {}
        for species_id in ids:
            component = _species(self.species_declarations[species_id], species_id)
            mass = _number(self.molecular_weights_kg_mol[species_id], "MW", positive=True)
            if mass != component.molecular_weight_g_mol / 1000:
                raise ValueError("transport MW differs from original native ComponentSpec")
            masses[species_id] = mass
            thermo = _thermo(
                self.thermochemistry_declarations[species_id], species_id, component.composition
            )
            if bounds[0] < thermo.min_temperature_K or bounds[1] > thermo.max_temperature_K:
                raise ValueError("fit temperature range exceeds original thermochemistry domain")
        object.__setattr__(self, "molecular_weights_kg_mol", MappingProxyType(masses))
        for name in ("viscosity_coefficients", "conductivity_coefficients"):
            object.__setattr__(
                self,
                name,
                MappingProxyType({s: _coefficients(getattr(self, name)[s], name) for s in ids}),
            )
        binary = {}
        for s in ids:
            _keys(self.binary_diffusivity_coefficients[s], set(ids), "binary pair row")
            binary[s] = {
                t: _coefficients(self.binary_diffusivity_coefficients[s][t], "binary pair")
                for t in ids
            }
        if any(binary[s][t] != binary[t][s] for s in ids for t in ids):
            raise ValueError("binary coefficients must be exactly symmetric by original ID")
        object.__setattr__(self, "binary_diffusivity_coefficients", _freeze_json(binary))
        if self.soret is not None:
            self.soret.validate_parent(self.base_transport_declaration())

    def validate_network(self, network: ReactionNetworkSpec) -> None:
        """Check real declarations, not a provider name, before consuming a fit."""
        if not isinstance(network, ReactionNetworkSpec) or set(network.species_ids) != set(
            self.species_ids
        ):
            raise ValueError("transport must bind exactly the original network species")
        for species in network.species:
            if not _same_declaration(
                self.species_declarations[species.species_id], species.to_dict()
            ):
                raise ValueError("transport species declaration binding mismatch")
        if not isinstance(network.metadata, Mapping) or not _same_declaration(
            self.thermochemistry_declarations, network.metadata.get("gas_thermochemistry")
        ):
            raise ValueError("transport thermochemistry declaration binding mismatch")

    def base_transport_declaration(self) -> dict[str, Any]:
        """The complete original off declaration bound by an optional Soret child."""
        payload = {
            "model": MODEL,
            "coefficient_basis": COEFFICIENT_BASIS,
            "units": dict(UNITS),
            **{
                name: self.source.to_dict() if name == "source" else _plain(getattr(self, name))
                for name in sorted(_SPEC_FIELDS)
            },
        }
        payload["thermal_diffusion"] = "off"
        return payload

    def to_dict(self) -> dict[str, Any]:
        payload = self.base_transport_declaration()
        if self.soret is not None:
            payload.update(
                model=SORET_MODEL,
                thermal_diffusion=SORET_MODE,
                soret=self.soret.to_dict(),
            )
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasTransportPolynomialSpec:
        enabled = isinstance(value, Mapping) and value.get("model") == SORET_MODEL
        extra = {"soret"} if enabled else set()
        _keys(
            value, _SPEC_FIELDS | {"model", "coefficient_basis", "units"} | extra, "transport codec"
        )
        if (
            value["model"] != (SORET_MODEL if enabled else MODEL)
            or value["coefficient_basis"] != COEFFICIENT_BASIS
            or value["units"] != UNITS
        ):
            raise ValueError("unknown source-fit transport schema, basis or units")
        if value["thermal_diffusion"] != (SORET_MODE if enabled else "off"):
            raise ValueError("transport codec and thermal-diffusion mode differ")
        _keys(value["source"], _SOURCE_FIELDS, "source")
        # Validate before the legacy provenance codec's string/int coercions.
        for name in _SOURCE_FIELDS - {"priority"}:
            _text(value["source"][name], f"source.{name}")
        if type(value["source"]["priority"]) is not int or value["source"]["priority"] < 0:
            raise ValueError("source priority must be a nonnegative integer")
        source = _source(DataSourceProvenance(**value["source"]))
        child = GasMixtureSoretSpec.from_dict(value["soret"]) if enabled else None
        return cls(**{k: value[k] for k in _SPEC_FIELDS - {"source"}}, source=source, soret=child)


def _polynomial(coefficients: tuple[float, ...], log_temperature: float) -> float:
    return fsum(c * log_temperature**i for i, c in enumerate(coefficients))


def gas_polynomial_transport_capabilities() -> dict[str, Any]:
    """Describe this coefficient consumer without authenticating input data."""
    return {
        "model": MODEL,
        "explicit_soret_model": SORET_MODEL,
        "formula_id": FORMULA_ID,
        "spec": "GasTransportPolynomialSpec",
        "response": "gas_polynomial_transport_response",
        "coefficient_basis": COEFFICIENT_BASIS,
        "units": dict(UNITS),
        "mode": "DEFAULT",
        "source_data_kinds": ["authored-synthetic", "upstream-fit"],
        "scope": "native evaluation of explicit source-fit coefficients",
        "maximum_pressure_Pa": 200000.0,
        "thermal_diffusion": ["off", SORET_MODE],
        "pure_vertex": "populated mixture D is None; exact zeros; no source Tiny",
        "spatial_consumer": "dynamic off-Soret with explicit fit-domain termination",
        "soret_consumers": ["explicit mixture flux", "premixed steady query"],
        "soret_dynamic_consumer": "unsupported; rejected explicitly",
        "not_implemented": ["CK", "independent collision theory", "MultiTransport"],
        "provenance_is_not_qualification": True,
    }


def gas_polynomial_transport_response(
    spec: GasTransportPolynomialSpec,
    network: ReactionNetworkSpec,
    *,
    temperature_K: float,
    pressure_Pa: float,
    mole_fractions: Mapping[str, float],
) -> GasTransportResponse:
    """Evaluate explicit source fits and the existing three physical D bases.

    Unlike the source Tiny convention, exact zero fractions stay zero.  At a
    pure vertex the populated species' directional mixture coefficients are
    None; the full binary matrix remains available to whole Stefan-Maxwell
    fluxes.  No trace inventory or source-library call is introduced.
    """
    if not isinstance(spec, GasTransportPolynomialSpec):
        raise ValueError("an explicit GasTransportPolynomialSpec is required")
    spec.validate_network(network)
    return _bound_polynomial_response(
        spec,
        network.species_ids,
        network.imported_gas_thermochemistry,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        mole_fractions=mole_fractions,
    )


def _bound_polynomial_response(
    spec, ids, native_thermo, *, temperature_K, pressure_Pa, mole_fractions
):
    """Private kernel for a query-owned, once-validated immutable material binding.

    Public calls above always validate original declarations. A BVP owns a
    deep network snapshot and binds its original thermo once; no global cache
    or bypass based on the identity of a caller-mutable network is introduced.
    All varying physical-state inputs are still validated on every evaluation.
    """
    t = _number(temperature_K, "temperature", positive=True)
    p = _number(pressure_Pa, "pressure", positive=True)
    if not spec.fit_temperature_range_K[0] <= t <= spec.fit_temperature_range_K[1]:
        raise ValueError("temperature outside declared source-fit domain")
    if p > spec.maximum_pressure_Pa:
        raise ValueError("pressure outside declared source-fit dilute domain")
    _keys(mole_fractions, set(spec.species_ids), "mole fractions")
    for fraction in mole_fractions.values():
        _number(fraction, "mole fraction")
    fractions = _validated_fraction_mapping(mole_fractions, "mole")
    masses = spec.molecular_weights_kg_mol
    try:
        total = fsum(fractions.values())
        x = {s: fractions[s] / total for s in ids}
        mean_mass = _number(fsum(x[s] * masses[s] for s in ids), "mean MW", positive=True)
        y = {s: x[s] * masses[s] / mean_mass for s in ids}
        logt, root_t = log(t), sqrt(t)
        viscosity, conductivity, enthalpy = {}, {}, {}
        for s in ids:
            # This fitted quantity is a square-root viscosity and must itself
            # be positive, not merely yield a positive result after squaring.
            amplitude = _number(
                _polynomial(spec.viscosity_coefficients[s], logt),
                "square-root viscosity fit",
                positive=True,
            )
            viscosity[s] = _number(root_t * amplitude * amplitude, "viscosity", positive=True)
            conductivity[s] = _number(
                root_t * _polynomial(spec.conductivity_coefficients[s], logt),
                "conductivity",
                positive=True,
            )
            state = native_thermo[s].evaluate(t)
            if _number(state.cp_J_mol_K, "native Cp") <= R:
                raise ValueError("ideal gas must have positive finite Cv")
            enthalpy[s] = _number(state.enthalpy_J_mol, "native species enthalpy")
        binary: dict[str, dict[str, float]] = {s: {} for s in ids}
        for i, s in enumerate(ids):
            for target in ids[i:]:
                value = _number(
                    t
                    * root_t
                    * _polynomial(spec.binary_diffusivity_coefficients[s][target], logt)
                    / p,
                    "binary diffusivity",
                    positive=True,
                )
                binary[s][target] = binary[target][s] = value
        mixture: dict[str, dict[str, float | None]] = {
            k: {} for k in ("molar_mole", "mass_mole", "mass_mass")
        }
        for s in ids:
            others = [j for j in ids if j != s]
            denominator = fsum(x[j] / binary[s][j] for j in others)
            other_x, other_y = fsum(x[j] for j in others), fsum(y[j] for j in others)
            if other_x == 0:
                for values in mixture.values():
                    values[s] = None
                continue
            _number(denominator, "mixture diffusion resistance", positive=True)
            _number(other_y, "other-species mass fraction", positive=True)
            mixture["molar_mole"][s] = other_x / denominator
            mixture["mass_mole"][s] = other_y / denominator
            resistance = denominator + x[s] * fsum(y[j] / binary[s][j] for j in others) / other_y
            mixture["mass_mass"][s] = 1 / resistance
            for values in mixture.values():
                _number(values[s], "mixture diffusivity", positive=True)
        mu = _number(
            wilke_viscosity(
                tuple(x.values()), tuple(viscosity.values()), tuple(masses[s] * 1000 for s in ids)
            ),
            "mixture viscosity",
            positive=True,
        )
        harmonic_resistance = _number(
            fsum(x[s] / conductivity[s] for s in ids),
            "mixture thermal resistance",
            positive=True,
        )
        lam = _number(
            0.5 * (fsum(x[s] * conductivity[s] for s in ids) + 1 / harmonic_resistance),
            "mixture conductivity",
            positive=True,
        )
        density = _number(p / (R * t), "molar density", positive=True)
        mass_density = _number(density * mean_mass, "mass density", positive=True)
    except ArithmeticError as exc:
        raise ValueError("source-fit transport exceeded its finite numeric domain") from exc
    response = GasTransportResponse(
        ids,
        t,
        p,
        x,
        y,
        masses,
        viscosity,
        conductivity,
        binary,
        mixture,
        enthalpy,
        density,
        mass_density,
        mu,
        lam,
        {
            "transport": MODEL,
            "formula": FORMULA_ID,
            "mode": spec.mode,
            "parameterization": "explicit source-fit coefficients; provenance is not qualification",
            "source_data_kind": spec.source_data_kind,
            "source_id": spec.source.source_id,
            "source_version": spec.source.version,
            "source_checksum": spec.source.checksum,
            "collision": "no independently implemented collision-integral or Stockmayer theory",
            "pure_viscosity": "sqrt(T)*P4(ln(T/K))^2",
            "pure_conductivity": "sqrt(T)*Q4(ln(T/K))",
            "binary_diffusivity": "T^(3/2)*R4(ln(T/K))/p(Pa)",
            "mixture_viscosity": "Wilke",
            "mixture_conductivity": "arithmetic-harmonic half-sum",
            "thermal_diffusion": "off",
            "pure_vertex": "populated mixture D is None; exact zeros; no source Tiny or trace fill",
            "enthalpy": "original native gas thermochemistry",
            "scope": "neutral ideal dilute gas source-fit consumer; no Soret or collision theory",
        },
    )
    if spec.soret is not None:
        from chemworld.physchem.gas_soret_transport import _bound_mixture_soret_coefficients

        soret = _bound_mixture_soret_coefficients(spec.soret, response)
        response = replace(
            response,
            thermal_diffusion_coefficients_kg_m_s=soret.thermal_diffusion_coefficients_kg_m_s,
            models={
                **response.models,
                "transport": SORET_MODEL,
                "thermal_diffusion": SORET_MODE,
                "thermal_diffusion_formula": spec.soret.formula_id,
                "scope": "neutral ideal dilute mixture-Soret source-fit consumer",
            },
        )
    return response
