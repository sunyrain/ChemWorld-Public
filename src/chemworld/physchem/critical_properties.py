"""Source-bearing critical estimates and explicit native input consumers.

Reports describe derived configuration, never measured constants or runtime
state. Original ConstantValue sources and unknown uncertainties are retained.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from math import isfinite

from chemworld.data.constants import ConstantValue
from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.foundation.units import Quantity
from chemworld.physchem.critical_kernels import (
    acentric_factor_from_pressure,
    boiling_critical_temperature,
    critical_compressibility,
    critical_surface,
    hekayati_raeissi_critical,
    lee_kesler_acentric_factor,
    mersmann_kind_volume,
    mixture_critical_property,
    stiel_polar_factor,
)
from chemworld.physchem.elements import get_isotope
from chemworld.physchem.eos import EOSComponentSpec

_UNITS = {"Tc": "K", "Tb": "K", "Pc": "Pa", "Vc": "m^3/mol", "Psat": "Pa",
          "omega": "dimensionless", "Stiel": "dimensionless", "Zc": "dimensionless"}
_INPUT_UNITS = {**_UNITS, "MW": "g/mol", "V_sat": "m^3/mol",
                "saturation_temperature_K": "K"}
_DOCUMENTATION = "https://chemicals.readthedocs.io/chemicals.critical.html"
_ACENTRIC = "https://chemicals.readthedocs.io/chemicals.acentric.html"


@dataclass(frozen=True)
class CriticalEstimate:
    subject_ids: tuple[str, ...]
    property_id: str
    value: float
    method: str
    inputs: dict
    citation: str
    kind: str = "pure_estimate"

    def __post_init__(self):
        if self.property_id not in _UNITS or not isfinite(self.value):
            raise ValueError("critical estimate requires a known property and finite value")
        if self.property_id not in {"omega", "Stiel"} and self.value <= 0:
            raise ValueError("estimated temperature, pressure and volume must be positive")
        if not self.subject_ids or any(not isinstance(s, str) or not s for s in self.subject_ids):
            raise ValueError("critical estimate requires explicit subject identifiers")
        if len(set(self.subject_ids)) != len(self.subject_ids):
            raise ValueError("critical estimate subjects must be unique")
        if self.kind not in {"pure_estimate", "pseudocritical_estimate"}:
            raise ValueError("unsupported critical estimate kind")
        if self.kind == "pure_estimate" and len(self.subject_ids) != 1:
            raise ValueError("a pure estimate requires exactly one component")
        if not self.method or not self.citation:
            raise ValueError("critical estimate requires method and citation")
        object.__setattr__(self, "subject_ids", tuple(self.subject_ids))

    @property
    def unit(self):
        return _UNITS[self.property_id]

    def quantity(self, target_unit=None):
        return Quantity(self.value, self.unit).to(target_unit or self.unit)

    def input_value(self, target_unit=None):
        return InputValue(
            self.value, self.unit, None,
            f"{self.kind}; {self.method}; {self.citation}; uncertainty not established",
        ).to(target_unit or self.unit)

    def to_dict(self):
        return {"subject_ids": list(self.subject_ids), "property_id": self.property_id,
                "value": self.value, "method": self.method, "inputs": self.inputs,
                "citation": self.citation, "kind": self.kind}

    @classmethod
    def from_dict(cls, payload):
        return cls(**payload)


def _input(value, property_id, component_id, cas_number=""):
    """Convert an explicitly bound input; never rank sources or fill missing data."""
    if isinstance(value, CriticalEstimate):
        if (value.property_id != property_id or value.kind != "pure_estimate"
                or value.subject_ids != (component_id,)):
            raise ValueError("critical estimate property/identity does not match the consumer")
        return value.quantity(_INPUT_UNITS[property_id]).value
    if isinstance(value, ConstantValue):
        if value.field_id != value.record.source_field(property_id):
            raise ValueError("constant field does not match the requested property")
        identity = value.record.cas_number or value.record.source_key
        if identity not in {component_id, cas_number}:
            raise ValueError("constant identity does not match the consumer")
        return value.quantity(_INPUT_UNITS[property_id]).value
    if isinstance(value, InputValue):
        if value.quality != "valid":
            raise ValueError("critical input is missing, invalid or outside its domain")
        return value.to(_INPUT_UNITS[property_id]).value
    raise TypeError("critical consumer requires a CriticalEstimate, ConstantValue or InputValue")


def estimate_critical_property(component, *, method, parameters, output_property=None):
    """Estimate one property for an existing ComponentSpec, preserving every input.

    Parameters use documented Tc/Pc/Vc/Tb/Psat/omega keys and SI units.
    Scalars are explicit caller inputs. Source-bearing values are checked
    against this component's identity and converted using the native units.
    No source discovery, fallback, or table mutation occurs.

    Hekayati_Raeissi additionally accepts MW [g/mol], V_sat [m³/mol]
    at 298.15 K, and optional saturation_temperature_K. Select
    output_property='Tc', 'Pc' or 'Vc' when multiple fields are missing.
    An already supplied field cannot be relabelled as an estimate.
    """
    c, records = {}, {}
    for name, value in parameters.items():
        if isinstance(value, (CriticalEstimate, ConstantValue, InputValue)):
            if name not in _INPUT_UNITS:
                raise ValueError(f"no established critical input unit for {name}")
            c[name] = _input(value, name, component.identifier, component.cas_number)
            records[name] = value.to_dict()
        else:
            c[name] = value
            records[name] = dict(value) if isinstance(value, Mapping) else value
    citation = _DOCUMENTATION
    if method in {"Ihmels", "Meissner", "Meissner_MPa", "Grigoras"}:
        missing = [key for key in ("Tc", "Pc", "Vc") if c.get(key) is None]
        values = critical_surface(method=method, **c)
        field = missing[0]
        result = dict(zip(("Tc", "Pc", "Vc"), values, strict=True))[field]
        if method == "Meissner_MPa":
            citation = "https://doi.org/10.17230/ingciencia.15.30.4"
    elif method == "Hekayati_Raeissi":
        values = hekayati_raeissi_critical(**c)
        missing = [key for key in ("Tc", "Pc", "Vc") if c.get(key) is None]
        field = output_property or (missing[0] if len(missing) == 1 else None)
        if field is None:
            raise ValueError(
                "select output_property explicitly when several or no fields are missing")
        if field not in missing:
            raise ValueError(
                "output_property must name a missing Tc, Pc or Vc, not a supplied value")
        result = dict(zip(("Tc", "Pc", "Vc"), values, strict=True))[field]
        records["model_basis"] = "V_sat_at_298.15_K" if c.get("Vc") is None else "supplied_Vc"
        records["estimated_properties"] = missing
        records["supplied_Tc_Pc_policy"] = "preserve_without_reconciliation"
        if c.get("V_sat") is not None:
            records.setdefault("saturation_temperature_K", 298.15)
            records["unused_inputs"] = (["V_sat", "saturation_temperature_K"]
                                        if c.get("Vc") is not None else [])
        citation = ("https://doi.org/10.1002/aic.17004; equations 21-33 and coefficients "
                    "from chemicals 1.5.2 public Hekayati_Raeissi implementation; "
                    "original paper not directly verified")
    elif method == "Tb_Tc_relationship":
        Tb, Tc = boiling_critical_temperature(**c)
        field, result = ("Tc", Tc) if c.get("Tc") is None else ("Tb", Tb)
    elif method == "Mersmann_Kind":
        atoms = {}
        for atom, number in component.composition.items():
            element = get_isotope(atom).symbol if atom.startswith("[") else atom
            atoms[element] = atoms.get(element, 0) + number
        if component.charge:
            raise ValueError("Mersmann-Kind is not an ionic-species critical-point model")
        result = mersmann_kind_volume(atoms, **c)
        field = "Vc"
        records["formula"] = component.formula
        records["atoms"] = atoms
        citation = "https://doi.org/10.1021/acs.iecr.7b03171"
    elif method == "omega_definition":
        field, result = "omega", acentric_factor_from_pressure(**c)
        citation = _ACENTRIC
    elif method == "LK_omega":
        field, result = "omega", lee_kesler_acentric_factor(**c)
        citation = _ACENTRIC
    elif method == "Stiel_polar_factor":
        field, result = "Stiel", stiel_polar_factor(**c)
        citation = _ACENTRIC
    elif method == "Zc_definition":
        field, result = "Zc", critical_compressibility(**c)
    else:
        raise ValueError(f"unknown or not yet implemented critical estimate {method!r}")
    if output_property is not None and output_property != field:
        raise ValueError(f"method and inputs estimate {field}, not {output_property}")
    return CriticalEstimate((component.identifier,), field, result, method, records, citation)


def critical_correlation_input(
    component, *, family, method, parameters, temperature_range_K, source,
    output="molar_volume", pressure_range_Pa=None, polar=None, correlation_id=None,
):
    """Bind selected constants/estimates to an original volume or virial input card.

    The caller supplies an InputSource for this derived model; it is not a
    replacement for the sources retained with each parameter. No inference of
    missing values, parameter uncertainty, material validity or caloric datum.
    Noncritical equation coefficients may be explicit scalars as before.
    """
    from chemworld.physchem.property_inputs import PropertyInputCard
    from chemworld.physchem.property_packages import ComponentPropertyPackage
    from chemworld.physchem.specs import ComponentUncertainty
    from chemworld.physchem.virial_correlations import virial_correlation
    from chemworld.physchem.volume_correlations import volume_correlation

    if not isinstance(source, InputSource):
        raise TypeError("critical-derived input needs an explicit InputSource")
    values, origins = {}, {}
    for name, value in parameters.items():
        if isinstance(value, (CriticalEstimate, ConstantValue, InputValue)):
            if name not in _INPUT_UNITS:
                raise ValueError(f"no declared critical input unit for {name}")
            values[name] = _input(value, name, component.identifier, component.cas_number)
            origins[name] = value.to_dict()
        else:
            values[name] = value
            origins[name] = {"value": value, "basis": "explicit caller coefficient"}
    common = {"parameters": values, "temperature_range_K": temperature_range_K,
              "source_note": source.provenance.citation, "correlation_id": correlation_id}
    if family == "volume":
        if polar is not None:
            raise ValueError("volume equations require their own explicit polar coefficients")
        curve = volume_correlation(method, **common, output=output,
                                   pressure_range_Pa=pressure_range_Pa)
    elif family == "virial":
        if pressure_range_Pa is not None or output != "molar_volume":
            raise ValueError("virial coefficient basis is defined by its method, independent of P")
        curve = virial_correlation(method, **common, polar=polar)
    else:
        raise ValueError("critical-derived family must be volume or virial")
    curve = replace(curve, metadata={**curve.metadata, "parameter_sources": origins,
        "component_id": component.identifier, "source": source.provenance.to_dict(),
        "parameter_basis": "explicit measured, derived or caller inputs; see parameter_sources",
        "application_basis": "caller selected; no material validity inferred"})
    return PropertyInputCard(
        ComponentPropertyPackage(component, (curve,)), (source,),
        {curve.correlation_id: ComponentUncertainty(
            curve.correlation_id, unit=curve.output_unit, source_id=source.provenance.source_id,
            coverage="unspecified", note="Derived model uncertainty not quantified",
        )}, None, "Property model only; supply an explicit caloric model/reference separately",
    )


def acentric_estimate_from_correlation(component, curve, *, Tc, Pc, method, omega=None):
    """Evaluate the existing Psat curve at the exact required reduced temperature."""
    from chemworld.physchem.property_equations import evaluate_correlation

    if method not in {"omega_definition", "Stiel_polar_factor"}:
        raise ValueError("only omega_definition and Stiel_polar_factor use a Psat point")
    if curve.property_id != "vapor_pressure":
        raise ValueError("acentric estimate requires a vapor-pressure correlation")
    temperature = _input(Tc, "Tc", component.identifier, component.cas_number)
    pressure = _input(Pc, "Pc", component.identifier, component.cas_number)
    reduced = .7 if method == "omega_definition" else .6
    evaluated = evaluate_correlation(curve, temperature_K=reduced * temperature,
                                     validity_policy="raise")
    params = {"Psat": evaluated.to("Pa").value, "Pc": pressure,
              "reduced_temperature": reduced}
    if method == "Stiel_polar_factor":
        params["omega"] = _input(omega, "omega", component.identifier, component.cas_number)
    elif omega is not None:
        raise ValueError("omega definition does not accept an omega input")
    result = estimate_critical_property(component, method=method, parameters=params)
    records = {**result.inputs, "curve": curve.to_dict(), "Tc_source": Tc.to_dict(),
               "Pc_source": Pc.to_dict()}
    if omega is not None:
        records["omega_source"] = omega.to_dict()
    return CriticalEstimate(result.subject_ids, result.property_id, result.value,
                            result.method, records, result.citation)


def eos_component_from_critical_inputs(component, *, Tc, Pc, omega):
    """Bind selected measured/estimated inputs to the existing EOS component.

    All three parameters are required, including omega; the EOS's legacy zero
    default is not substituted for a missing measurement/estimate here.
    Original reports remain the caller's serializable configuration evidence.
    """
    values = {name: _input(value, name, component.identifier, component.cas_number)
              for name, value in (("Tc", Tc), ("Pc", Pc), ("omega", omega))}
    return EOSComponentSpec(component.identifier, values["Tc"], values["Pc"], values["omega"])


def mixture_critical_estimate(mixture, *, method, critical_inputs, interactions=None):
    """Consume the existing MixtureSpec and keyed Tc/Vc sources in its exact order.

    critical_inputs maps component IDs to property dictionaries. InputValue
    bindings explicitly associate caller data with those IDs; database inputs
    additionally require IDs equal to their CAS or source key. Interaction
    matrices use MixtureSpec.component_ids order, retained in the report.
    """
    if set(critical_inputs) != set(mixture.component_ids):
        raise ValueError("critical input identities must exactly match MixtureSpec")
    fields = ({"Tc", "Vc"} if method in {"Li", "Chueh_Prausnitz_Tc"}
              else {"Vc"} if method.endswith("_Vc") else {"Tc"})
    vectors = {name: [] for name in fields}
    records = {}
    for identity in mixture.component_ids:
        selected = critical_inputs[identity]
        if set(selected) != fields:
            raise ValueError(f"{method} requires exactly {sorted(fields)} for each component")
        records[identity] = {}
        for name in fields:
            vectors[name].append(_input(selected[name], name, identity))
            records[identity][name] = selected[name].to_dict()
    value = mixture_critical_property(
        method, mixture.mole_fractions, Tcs=vectors.get("Tc"), Vcs=vectors.get("Vc"),
        interactions=interactions,
    )
    return CriticalEstimate(
        mixture.component_ids, "Vc" if method.endswith("_Vc") else "Tc", value,
        method, {"mole_fractions": list(mixture.mole_fractions), "components": records,
                 "interactions": None if interactions is None else [list(r) for r in interactions]},
        _DOCUMENTATION, "pseudocritical_estimate",
    )
