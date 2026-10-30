"""Formula-only, explicit Cantera-3.2 mixture-Soret source-fit consumer.

No source library, coefficient generator, network solver or runtime is called.
The complete old-v1/off material declaration is bound as data; declaring an
upstream provider does not authenticate it. DEFAULT Cstar coefficients must
originate from fresh, unmodified reduced-temperature fits. The public getter
does not export the setter's actualT flag, so unknown construction is rejected.

Formula authority is the implementation (not the differing header equation):
https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/MixTransport.cpp#L46-L115
https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/GasTransport.cpp#L273-L458
Constants and customary-unit conversions:
https://github.com/Cantera/cantera/blob/v3.2.0/include/cantera/base/ct_defs.h
https://github.com/Cantera/cantera/blob/v3.2.0/src/transport/TransportData.cpp#L41-L55
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import fsum, isclose, isfinite, log, sqrt
from types import MappingProxyType
from typing import Any

from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.gas_transport import GasTransportResponse
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.thermochemistry import species_thermo_from_cantera_yaml
from chemworld.physchem.viscosity_mixtures import wilke_prefactors

MODEL = "gas-mixture-soret-source-fit-v1"
FORMULA_ID = "cantera-3.2.0-default-mixture-soret-cpp-v1"
CONSTRUCTION = "fresh-default-unmodified"
FIT_VARIABLE = "log-reduced-temperature"
CONSTANTS_ID = "cantera-3.2.0-ct_defs-CODATA2018"
BOLTZMANN_J_K = 1.380649e-23
LIGHT_SPEED_M_S = 299792458.0
DEBYE_C_M = 1e-21 / LIGHT_SPEED_M_S
_ELECTRON_CHARGE_C = 1.602176634e-19
_PLANCK_J_S = 6.62607015e-34
_FINE_STRUCTURE = 7.2973525693e-3
_PERMEABILITY = (
    2 * _FINE_STRUCTURE * _PLANCK_J_S / (_ELECTRON_CHARGE_C * _ELECTRON_CHARGE_C * LIGHT_SPEED_M_S)
)
VACUUM_PERMITTIVITY_F_M = 1 / (LIGHT_SPEED_M_S * LIGHT_SPEED_M_S * _PERMEABILITY)
_PI = 3.14159265358979323846
_BASE_MODEL = "gas-transport-polynomial-v1"
_BASE_FORMULA = "cantera-3.2-default-gas-transport-polynomials-v1"
_BASE_BASIS = "ascending powers of ln(T/K); binary coefficients normalized at 1 Pa"
_BASE_UNITS = {
    "temperature": "K",
    "pressure": "Pa",
    "viscosity": "Pa*s",
    "conductivity": "W/(m*K)",
    "binary_diffusivity": "m^2/s",
}
_SI_FIELDS = {"diameter_m", "well_depth_J", "dipole_C_m", "polarizability_m3"}
_SOURCE_FIELDS = {"source_id", "priority", "citation", "version", "license", "checksum"}
_PARENT_FIELDS = {
    "model",
    "formula_id",
    "coefficient_basis",
    "units",
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
}
_SPEC_FIELDS = {
    "parent_transport_declaration",
    "species_transport_SI",
    "cstar_coefficients",
    "fit_temperature_range_K",
    "source_data_kind",
    "source",
    "construction",
    "fit_variable",
    "formula_id",
}
_UNITS = {
    "diameter_m": "m",
    "well_depth_J": "J",
    "dipole_C_m": "C*m",
    "polarizability_m3": "m^3",
    "temperature": "K",
    "pressure": "Pa",
    "thermal_diffusion_coefficient": "kg/(m*s)",
    "Cstar": "dimensionless",
}


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} exceeds the finite numeric domain") from exc
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be nonempty unpadded text")
    return value


def _keys(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"{name} must be a string-keyed object")
    if set(value) != keys:
        raise ValueError(f"{name} has missing or unknown fields")


def _sequence(value: Any, name: str) -> tuple:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{name} must be a list or tuple")
    return tuple(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("declaration objects require string keys")
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int | float):
        _number(value, "declaration number")
        return value
    raise ValueError("declarations require finite plain JSON data")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


def _canonical(value: Any) -> str:
    return json.dumps(
        _plain(_freeze(value)), sort_keys=True, allow_nan=False, separators=(",", ":")
    )


def _coefficients(values: Any, count: int) -> tuple[float, ...]:
    row = _sequence(values, "coefficients")
    if len(row) != count:
        raise ValueError(f"exactly {count} coefficients are required")
    return tuple(_number(v, "coefficient") for v in row)


def _provenance(value: Any) -> DataSourceProvenance:
    raw = value.to_dict() if isinstance(value, DataSourceProvenance) else value
    _keys(raw, _SOURCE_FIELDS, "source")
    for name in _SOURCE_FIELDS - {"priority"}:
        _text(raw[name], "source." + name)
    if type(raw["priority"]) is not int or raw["priority"] < 0:
        raise ValueError("source priority must be a nonnegative integer")
    return DataSourceProvenance(**raw)


def _si_record(raw: Any) -> Mapping[str, float]:
    _keys(raw, _SI_FIELDS, "species transport SI")
    result = {k: _number(raw[k], k, positive=k in {"diameter_m", "well_depth_J"}) for k in raw}
    if result["dipole_C_m"] < 0 or result["polarizability_m3"] < 0:
        raise ValueError("dipole and polarizability must be nonnegative")
    return MappingProxyType(result)


def _parent_ids(parent: Any) -> tuple[str, ...]:
    _keys(parent, _PARENT_FIELDS, "complete parent transport declaration")
    if (
        parent["model"] != _BASE_MODEL
        or parent["formula_id"] != _BASE_FORMULA
        or parent["coefficient_basis"] != _BASE_BASIS
        or parent["units"] != _BASE_UNITS
        or parent["thermal_diffusion"] != "off"
        or parent["mode"] != "DEFAULT"
    ):
        raise ValueError("Soret requires the canonical old-v1 DEFAULT/off parent")
    ids = _sequence(parent["species_ids"], "parent species ids")
    for s in ids:
        _text(s, "species id")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("parent species ids must be nonempty and unique")
    for name in (
        "species_declarations",
        "thermochemistry_declarations",
        "molecular_weights_kg_mol",
        "viscosity_coefficients",
        "conductivity_coefficients",
        "binary_diffusivity_coefficients",
    ):
        _keys(parent[name], set(ids), "parent " + name)
    bounds = _sequence(parent["fit_temperature_range_K"], "parent temperature range")
    if len(bounds) != 2:
        raise ValueError("parent temperature range needs exactly two bounds")
    bounds = tuple(_number(v, "parent temperature bound", positive=True) for v in bounds)
    if bounds[0] >= bounds[1]:
        raise ValueError("parent temperature range must increase")
    for s in ids:
        _keys(parent["binary_diffusivity_coefficients"][s], set(ids), "parent binary row")
    for s in ids:
        species = parent["species_declarations"][s]
        _keys(
            species,
            {
                "species_id",
                "formula",
                "composition",
                "phase",
                "charge",
                "catalyst",
                "observable_aliases",
                "metadata",
            },
            "parent species declaration",
        )
        if species["species_id"] != s or species["phase"] != "gas":
            raise ValueError("parent must retain each original gas species")
        if type(species["charge"]) is not int or species["charge"] != 0:
            raise ValueError("charged species are outside mixture-Soret scope")
        if type(species["catalyst"]) is not bool or not isinstance(species["metadata"], Mapping):
            raise ValueError("invalid parent species declaration")
        for alias in _sequence(species["observable_aliases"], "observable aliases"):
            _text(alias, "observable alias")
        component = ComponentSpec(s, _text(species["formula"], "formula"), charge=0)
        if species["composition"] != component.composition:
            raise ValueError("parent composition differs from its original formula")
        for count in species["composition"].values():
            _number(count, "atom count", positive=True)
        mass = _number(parent["molecular_weights_kg_mol"][s], "parent MW", positive=True)
        if mass != component.molecular_weight_g_mol / 1000:
            raise ValueError("parent MW differs from original native ComponentSpec")
        thermo = parent["thermochemistry_declarations"][s]
        if (
            not isinstance(thermo, Mapping)
            or "thermo" not in thermo
            or not isinstance(thermo["thermo"], Mapping)
            or set(thermo) - {"thermo", "composition", "reference_pressure_Pa", "units"}
        ):
            raise ValueError("complete original thermochemistry declaration is required")
        if "composition" in thermo:
            if not isinstance(thermo["composition"], Mapping):
                raise ValueError("parent thermo composition must be an object")
            for count in thermo["composition"].values():
                _number(count, "thermo atom count", positive=True)
            if thermo["composition"] != component.composition:
                raise ValueError("parent thermo composition differs from original species")
        _validate_parent_thermo(thermo, s, bounds)
        for name in ("viscosity_coefficients", "conductivity_coefficients"):
            _coefficients(parent[name][s], 5)
        _keys(parent["binary_diffusivity_coefficients"][s], set(ids), "parent binary row")
        for t in ids:
            if _coefficients(parent["binary_diffusivity_coefficients"][s][t], 5) != _coefficients(
                parent["binary_diffusivity_coefficients"][t][s], 5
            ):
                raise ValueError("parent binary coefficients must be symmetric")
    maximum = _number(parent["maximum_pressure_Pa"], "parent maximum pressure", positive=True)
    if maximum > 200000:
        raise ValueError("parent dilute pressure domain exceeds 200000 Pa")
    _provenance(parent["source"])
    return ids


def _validate_parent_thermo(
    raw: Mapping[str, Any], species_id: str, bounds: tuple[float, float]
) -> None:
    """Reuse native data parsing only; no source object or state evaluation."""
    if "reference_pressure_Pa" in raw:
        _number(raw["reference_pressure_Pa"], "thermo reference pressure", positive=True)
    if "units" in raw and not isinstance(raw["units"], Mapping):
        raise ValueError("original thermo units must be an object")
    node = raw["thermo"]
    if node.get("model") == "NASA7" and set(node) - {
        "model",
        "temperature-ranges",
        "data",
        "reference-pressure",
        "note",
    }:
        raise ValueError("unknown original NASA7 field")

    def reject_numeric_booleans(value: Any, key: str = "") -> None:
        if isinstance(value, bool) and key != "dimensionless":
            raise ValueError("boolean thermo numbers are unsupported")
        if isinstance(value, Mapping):
            for k, v in value.items():
                reject_numeric_booleans(v, k)
        elif isinstance(value, tuple):
            for v in value:
                reject_numeric_booleans(v, key)

    reject_numeric_booleans(node)
    try:
        parsed = species_thermo_from_cantera_yaml(
            species_id=species_id, boundary_policy="source", **_plain(raw)
        )
        if bounds[0] < parsed.min_temperature_K or bounds[1] > parsed.max_temperature_K:
            raise ValueError("parent fit domain exceeds original thermo domain")
    except (TypeError, KeyError, IndexError, ArithmeticError) as exc:
        raise ValueError("invalid original parent thermochemistry declaration") from exc


def _source_si(species: Mapping[str, Any]) -> dict[str, float]:
    try:
        raw = species["metadata"]["source_declaration"]["transport"]
    except (KeyError, TypeError) as exc:
        raise ValueError("SI binding requires the original species transport declaration") from exc
    if not isinstance(raw, Mapping) or raw.get("model") != "gas":
        raise ValueError("only the declared Cantera gas transport format is supported")
    try:
        diameter = _number(raw["diameter"], "source diameter", positive=True)
        depth = _number(raw["well-depth"], "source well depth", positive=True)
    except KeyError as exc:
        raise ValueError("source diameter and well depth cannot be filled implicitly") from exc
    # These fields have fixed customary units in this particular source format;
    # a raw species/transport 'units' metadata object does not rescale them.
    return {
        "diameter_m": 1e-10 * diameter,
        "well_depth_J": BOLTZMANN_J_K * depth,
        "dipole_C_m": DEBYE_C_M * _number(raw.get("dipole", 0), "source dipole"),
        "polarizability_m3": 1e-30 * _number(raw.get("polarizability", 0), "source polarizability"),
    }


@dataclass(frozen=True)
class GasMixtureSoretSpec:
    parent_transport_declaration: Mapping[str, Any]
    species_transport_SI: Mapping[str, Mapping[str, float]]
    cstar_coefficients: Mapping[str, Mapping[str, tuple[float, ...]]]
    fit_temperature_range_K: tuple[float, float]
    source_data_kind: str
    source: DataSourceProvenance
    construction: str = CONSTRUCTION
    fit_variable: str = FIT_VARIABLE
    formula_id: str = FORMULA_ID
    _parent_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.construction != CONSTRUCTION
            or self.fit_variable != FIT_VARIABLE
            or self.formula_id != FORMULA_ID
        ):
            raise ValueError(
                "only fresh DEFAULT unmodified log-reduced-T collision fits are supported"
            )
        parent = _freeze(self.parent_transport_declaration)
        ids = _parent_ids(parent)
        if self.source_data_kind not in ("authored-synthetic", "upstream-fit"):
            raise ValueError("unsupported source data kind")
        source = _provenance(self.source)
        if (
            source.to_dict() != _plain(parent["source"])
            or self.source_data_kind != parent["source_data_kind"]
        ):
            raise ValueError("Soret and parent must retain the same complete source provenance")
        bounds = _sequence(self.fit_temperature_range_K, "Soret temperature range")
        if len(bounds) != 2:
            raise ValueError("Soret temperature range needs exactly two bounds")
        bounds = tuple(_number(v, "Soret temperature bound", positive=True) for v in bounds)
        if bounds[0] >= bounds[1] or bounds != tuple(parent["fit_temperature_range_K"]):
            raise ValueError("Soret temperature range must equal the original parent fit domain")
        _keys(self.species_transport_SI, set(ids), "SI species")
        si = {s: _si_record(self.species_transport_SI[s]) for s in ids}
        for s in ids:
            if dict(si[s]) != _source_si(parent["species_declarations"][s]):
                raise ValueError("SI parameters differ from the actual original source fields")
        _keys(self.cstar_coefficients, set(ids), "Cstar species")
        cstar = {}
        for s in ids:
            _keys(self.cstar_coefficients[s], set(ids), "Cstar pair row")
            cstar[s] = {t: _coefficients(self.cstar_coefficients[s][t], 9) for t in ids}
        if any(cstar[s][t] != cstar[t][s] for s in ids for t in ids):
            raise ValueError("Cstar coefficients must be exactly pair symmetric")
        object.__setattr__(self, "parent_transport_declaration", parent)
        object.__setattr__(self, "species_transport_SI", _freeze(si))
        object.__setattr__(self, "cstar_coefficients", _freeze(cstar))
        object.__setattr__(self, "fit_temperature_range_K", bounds)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_parent_json", _canonical(parent))

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(self.parent_transport_declaration["species_ids"])

    def validate_parent(self, mapping: Mapping[str, Any]) -> None:
        if _canonical(mapping) != self._parent_json:
            raise ValueError("complete parent transport declaration binding mismatch")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL,
            "constants_id": CONSTANTS_ID,
            "units": dict(_UNITS),
            **{
                name: self.source.to_dict() if name == "source" else _plain(getattr(self, name))
                for name in sorted(_SPEC_FIELDS)
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> GasMixtureSoretSpec:
        _keys(value, _SPEC_FIELDS | {"model", "constants_id", "units"}, "Soret codec")
        if (
            value["model"] != MODEL
            or value["constants_id"] != CONSTANTS_ID
            or value["units"] != _UNITS
        ):
            raise ValueError("unknown Soret schema, constants or units")
        return cls(
            **{k: value[k] for k in _SPEC_FIELDS - {"source"}}, source=_provenance(value["source"])
        )


def soret_pair_well_depths_J(  # noqa: N802 - explicit SI unit in the formula API
    species_transport_SI: Mapping[str, Mapping[str, float]],
) -> Mapping:
    """Symmetric pair epsilon [J], including the source polar/nonpolar correction."""
    if not isinstance(species_transport_SI, Mapping) or not species_transport_SI:
        raise ValueError("nonempty SI species parameters are required")
    si = {_text(s, "species id"): _si_record(v) for s, v in species_transport_SI.items()}
    pairs = {s: {} for s in si}
    try:
        for i, s in enumerate(si):
            for t in tuple(si)[i:]:
                a, b = si[s], si[t]
                epsilon = sqrt(a["well_depth_J"] * b["well_depth_J"])
                if (a["dipole_C_m"] > 0) != (b["dipole_C_m"] > 0):
                    polar, neutral = (a, b) if a["dipole_C_m"] > 0 else (b, a)
                    alpha_star = neutral["polarizability_m3"] / neutral["diameter_m"] ** 3
                    dipole_star = polar["dipole_C_m"] / sqrt(
                        4
                        * _PI
                        * VACUUM_PERMITTIVITY_F_M
                        * polar["diameter_m"] ** 3
                        * polar["well_depth_J"]
                    )
                    xi = 1 + 0.25 * alpha_star * dipole_star * dipole_star * sqrt(
                        polar["well_depth_J"] / neutral["well_depth_J"]
                    )
                    epsilon *= xi * xi
                pairs[s][t] = pairs[t][s] = _number(epsilon, "pair well depth", positive=True)
    except ArithmeticError as exc:
        raise ValueError("pair well depth exceeds its numeric domain") from exc
    return _freeze(pairs)


def soret_cstar(
    coefficients: tuple[float, ...], temperature_K: float, pair_well_depth_J: float
) -> float:
    """Evaluate nine ascending DEFAULT coefficients at ln(k_B T / epsilon)."""
    row = _coefficients(coefficients, 9)
    t = _number(temperature_K, "Cstar temperature", positive=True)
    epsilon = _number(pair_well_depth_J, "pair well depth", positive=True)
    try:
        z = log(BOLTZMANN_J_K * t / epsilon)
        value = row[-1]
        for c in reversed(row[:-1]):
            value = value * z + c
        return _number(value, "Cstar", positive=True)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError("Cstar outside its positive finite numeric domain") from exc


def midpoint_temperature_log_gradient(
    left_temperature_K: float, right_temperature_K: float, distance_m: float
) -> float:
    """Source midpoint force 2 delta(T)/[(T_left+T_right) distance], in 1/m."""
    left = _number(left_temperature_K, "left temperature", positive=True)
    right = _number(right_temperature_K, "right temperature", positive=True)
    distance = _number(distance_m, "distance", positive=True)
    try:
        denominator = _number(
            (left + right) * distance, "temperature-force denominator", positive=True
        )
        return _number(2 * (right - left) / denominator, "log-temperature gradient")
    except ArithmeticError as exc:
        raise ValueError("log-temperature gradient exceeds its numeric domain") from exc


def soret_a_coefficients(
    mole_fractions: Mapping[str, float],
    molecular_weights_kg_mol: Mapping[str, float],
    viscosities_Pa_s: Mapping[str, float],
) -> Mapping[str, float]:
    """Stable all-species a [mol/(m*s)], using the existing Wilke prefactors."""
    if not isinstance(mole_fractions, Mapping) or not mole_fractions:
        raise ValueError("nonempty original mole fractions are required")
    ids = tuple(_text(s, "species id") for s in mole_fractions)
    _keys(molecular_weights_kg_mol, set(ids), "a molecular weights")
    _keys(viscosities_Pa_s, set(ids), "a viscosities")
    x = {s: _number(mole_fractions[s], "mole fraction") for s in ids}
    if any(v < 0 for v in x.values()) or not isclose(fsum(x.values()), 1, rel_tol=0, abs_tol=2e-14):
        raise ValueError("a mole fractions must be nonnegative and already normalized")
    masses = {s: _number(molecular_weights_kg_mol[s], "MW", positive=True) for s in ids}
    mu = {s: _number(viscosities_Pa_s[s], "viscosity", positive=True) for s in ids}
    try:
        prefactors = wilke_prefactors(tuple(masses[s] for s in ids))
        a = {}
        for i, s in enumerate(ids):
            if x[s] == 0:
                a[s] = 0.0
                continue
            resistance = []
            for j, target in enumerate(ids):
                if j != i:
                    ratio = mu[s] / mu[target]
                    phi = (
                        prefactors[0][i][j] * ratio
                        + prefactors[1][i][j] * sqrt(ratio)
                        + prefactors[2][i][j]
                    )
                    resistance.append(x[target] * phi)
            denominator = _number(
                x[s] + 1.065 * fsum(resistance), "Soret a denominator", positive=True
            )
            a[s] = _number(
                (15 / 4) * mu[s] / masses[s] * x[s] / denominator, "Soret a", positive=True
            )
    except ArithmeticError as exc:
        raise ValueError("Soret a exceeds its finite numeric domain") from exc
    return _freeze(a)


@dataclass(frozen=True)
class GasMixtureSoretResult:
    thermal_diffusion_coefficients_kg_m_s: Mapping[str, float]
    receipt: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "thermal_diffusion_coefficients_kg_m_s",
            _freeze(self.thermal_diffusion_coefficients_kg_m_s),
        )
        object.__setattr__(self, "receipt", _freeze(self.receipt))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": MODEL + "-result",
            "thermal_diffusion_coefficients_kg_m_s": _plain(
                self.thermal_diffusion_coefficients_kg_m_s
            ),
            "receipt": _plain(self.receipt),
        }


def _matches(actual: float, expected: float, name: str) -> None:
    # This checks a previously evaluated base response, not a scientific error
    # threshold: allow only floating expression-order roundoff and no zero floor.
    if not isclose(actual, expected, rel_tol=2e-14, abs_tol=0):
        raise ValueError(name + " differs from the bound parent response")


def _validate_bound_response(spec: GasMixtureSoretSpec, response: GasTransportResponse) -> None:
    """Check live state without serializing declarations or evaluating parent fits."""
    if (
        not isinstance(response, GasTransportResponse)
        or set(response.species_ids) != set(spec.species_ids)
        or len(set(response.species_ids)) != len(response.species_ids)
    ):
        raise ValueError("base response must contain exactly the original species")
    ids, parent = response.species_ids, spec.parent_transport_declaration
    t = _number(response.temperature_K, "temperature", positive=True)
    p = _number(response.pressure_Pa, "pressure", positive=True)
    if (
        not spec.fit_temperature_range_K[0] <= t <= spec.fit_temperature_range_K[1]
        or p > parent["maximum_pressure_Pa"]
    ):
        raise ValueError("response outside bound Soret T/P domain")
    expected_models = {
        "transport": _BASE_MODEL,
        "formula": _BASE_FORMULA,
        "mode": "DEFAULT",
        "thermal_diffusion": "off",
        "source_data_kind": spec.source_data_kind,
        "source_id": spec.source.source_id,
        "source_version": spec.source.version,
        "source_checksum": spec.source.checksum,
    }
    if any(response.models.get(k) != v for k, v in expected_models.items()):
        raise ValueError("base response model/source declaration mismatch")
    for name in (
        "mole_fractions",
        "mass_fractions",
        "molecular_weights_kg_mol",
        "species_viscosities_Pa_s",
        "binary_diffusivities_m2_s",
    ):
        _keys(getattr(response, name), set(ids), "response " + name)
    x, y, masses = (
        response.mole_fractions,
        response.mass_fractions,
        response.molecular_weights_kg_mol,
    )
    for s in ids:
        if _number(x[s], "mole fraction") < 0 or _number(y[s], "mass fraction") < 0:
            raise ValueError("fractions must be nonnegative without Tiny filling")
        if _number(masses[s], "MW", positive=True) != parent["molecular_weights_kg_mol"][s]:
            raise ValueError("response MW differs from the real parent MW")
    if not isclose(fsum(x.values()), 1, rel_tol=0, abs_tol=2e-14) or not isclose(
        fsum(y.values()), 1, rel_tol=0, abs_tol=2e-14
    ):
        raise ValueError("base response fractions must already be normalized")
    mean_mass = _number(fsum(x[s] * masses[s] for s in ids), "mean MW", positive=True)
    for s in ids:
        _matches(y[s], x[s] * masses[s] / mean_mass, "mass fraction")
        _number(response.species_viscosities_Pa_s[s], "viscosity", positive=True)
        _keys(response.binary_diffusivities_m2_s[s], set(ids), "response binary row")
    for s in ids:
        for target in ids:
            value = _number(
                response.binary_diffusivities_m2_s[s][target], "binary D", positive=True
            )
            if value != response.binary_diffusivities_m2_s[target][s]:
                raise ValueError("response binary D must be symmetric")
    _keys(
        response.diffusion_coefficients_m2_s,
        {"molar_mole", "mass_mole", "mass_mass"},
        "response diffusion bases",
    )
    for values in response.diffusion_coefficients_m2_s.values():
        _keys(values, set(ids), "response diffusion basis")
    for s in ids:
        if fsum(x[j] for j in ids if j != s) == 0:
            if any(v[s] is not None for v in response.diffusion_coefficients_m2_s.values()):
                raise ValueError("pure populated mixture coefficients must retain None")
        else:
            _number(
                response.diffusion_coefficients_m2_s["mass_mole"][s],
                "mass-mole D",
                positive=True,
            )


def _validate_response(spec: GasMixtureSoretSpec, response: GasTransportResponse) -> None:
    _validate_bound_response(spec, response)
    ids, parent = response.species_ids, spec.parent_transport_declaration
    t, p = response.temperature_K, response.pressure_Pa
    x, y = response.mole_fractions, response.mass_fractions
    z = log(t)
    for s in ids:
        amplitude = fsum(c * z**i for i, c in enumerate(parent["viscosity_coefficients"][s]))
        _number(amplitude, "square-root viscosity fit", positive=True)
        _matches(
            _number(response.species_viscosities_Pa_s[s], "viscosity", positive=True),
            sqrt(t) * amplitude * amplitude,
            "viscosity",
        )
        for target in ids:
            value = _number(
                response.binary_diffusivities_m2_s[s][target], "binary D", positive=True
            )
            expected = (
                t
                * sqrt(t)
                * fsum(
                    c * z**i
                    for i, c in enumerate(parent["binary_diffusivity_coefficients"][s][target])
                )
                / p
            )
            _matches(value, _number(expected, "parent binary D", positive=True), "binary D")
    for s in ids:
        others = [j for j in ids if j != s]
        if fsum(x[j] for j in others) != 0:
            denominator = _number(
                fsum(x[j] / response.binary_diffusivities_m2_s[s][j] for j in others),
                "diffusion resistance",
                positive=True,
            )
            expected = fsum(y[j] for j in others) / denominator
            _matches(
                _number(
                    response.diffusion_coefficients_m2_s["mass_mole"][s],
                    "mass-mole D",
                    positive=True,
                ),
                expected,
                "mass-mole D",
            )


def gas_mixture_soret_coefficients(
    spec: GasMixtureSoretSpec,
    response: GasTransportResponse,
    *,
    parent_transport_declaration: Mapping[str, Any],
) -> GasMixtureSoretResult:
    """Evaluate signed all-species DT; the consuming model selects its Soret path."""
    if not isinstance(spec, GasMixtureSoretSpec):
        raise ValueError("explicit GasMixtureSoretSpec is required")
    spec.validate_parent(parent_transport_declaration)
    try:
        _validate_response(spec, response)
    except (ArithmeticError, TypeError, KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Soret formula exceeds its finite numeric domain") from exc
    return _bound_mixture_soret_coefficients(spec, response)


def _bound_mixture_soret_coefficients(
    spec: GasMixtureSoretSpec, response: GasTransportResponse
) -> GasMixtureSoretResult:
    """Query-owned core for a validated immutable spec and its parent kernel response.

    This internal entry point trusts the owner's completed declaration binding and
    base-fit evaluation. It still validates the current T/P, species, MW, fractions
    and finite response values; it does not authenticate a caller-provided fit.
    """
    if not isinstance(spec, GasMixtureSoretSpec):
        raise ValueError("explicit GasMixtureSoretSpec is required")
    try:
        _validate_bound_response(spec, response)
        ids = response.species_ids
        x, y, masses = (
            response.mole_fractions,
            response.mass_fractions,
            response.molecular_weights_kg_mol,
        )
        pure = next((s for s in ids if all(x[j] == 0 for j in ids if j != s)), None)
        normalization = 0.0
        if pure is not None:
            # The continuous limit is zero, even though the old composition-
            # diffusion coefficient of the populated species is directional.
            dt = dict.fromkeys(ids, 0.0)
        else:
            pairs = soret_pair_well_depths_J(spec.species_transport_SI)
            a = soret_a_coefficients(x, masses, response.species_viscosities_Pa_s)
            b = {s: [] for s in ids}
            for i, s in enumerate(ids):
                for target in ids[i + 1 :]:
                    cstar = soret_cstar(
                        spec.cstar_coefficients[s][target], response.temperature_K, pairs[s][target]
                    )
                    term = (
                        (1.2 * cstar - 1)
                        / response.binary_diffusivities_m2_s[s][target]
                        / (masses[s] + masses[target])
                        * (y[s] * a[target] - y[target] * a[s])
                    )
                    term = _number(term, "signed Soret pair term")
                    b[s].append(term)
                    b[target].append(-term)
            mean_mass = fsum(x[s] * masses[s] for s in ids)
            raw = {
                s: _number(
                    response.diffusion_coefficients_m2_s["mass_mole"][s]
                    * masses[s]
                    * mean_mass
                    * fsum(b[s]),
                    "unnormalized Soret DT",
                )
                for s in ids
            }
            normalization = _number(fsum(raw.values()), "Soret normalization")
            dt = {
                s: 0.0 if x[s] == 0 else _number(raw[s] - y[s] * normalization, "signed Soret DT")
                for s in ids
            }
    except (ArithmeticError, TypeError, KeyError, AttributeError, IndexError) as exc:
        raise ValueError("Soret formula exceeds its finite numeric domain") from exc
    return GasMixtureSoretResult(
        dt,
        {
            "formula_id": FORMULA_ID,
            "frame": "mass",
            "temperature_K": response.temperature_K,
            "pressure_Pa": response.pressure_Pa,
            "normalization_sum_kg_m_s": normalization,
            "mass_sum_kg_m_s": fsum(dt.values()),
            "pure_species": pure,
            "absent_species": tuple(s for s in ids if x[s] == 0),
            "source_data_kind": spec.source_data_kind,
            "source_id": spec.source.source_id,
            "source_qualification": "not inferred from provenance or construction declaration",
            "zero_policy": (
                "continuous exact-zero extension; no source Tiny clipping or inventory fill"
            ),
            "scope": ("formula-only mixture-averaged source-fit evaluation; not MultiTransport"),
        },
    )
