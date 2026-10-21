"""Source-bearing critical estimates and explicit native input consumers.

Reports describe derived configuration, never measured constants or runtime
state. Original ConstantValue sources and unknown uncertainties are retained.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from chemworld.data.constants import ConstantValue
from chemworld.data.scientific_inputs import InputValue
from chemworld.foundation.units import Quantity
from chemworld.physchem.critical_kernels import (
    acentric_factor_from_pressure,
    boiling_critical_temperature,
    critical_surface,
    lee_kesler_acentric_factor,
    mersmann_kind_volume,
    mixture_critical_property,
    stiel_polar_factor,
)
from chemworld.physchem.elements import get_isotope
from chemworld.physchem.eos import EOSComponentSpec

_UNITS = {"Tc": "K", "Tb": "K", "Pc": "Pa", "Vc": "m^3/mol", "Psat": "Pa",
          "omega": "dimensionless", "Stiel": "dimensionless"}
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
        return value.quantity(_UNITS[property_id]).value
    if isinstance(value, ConstantValue):
        if value.field_id != property_id:
            raise ValueError("constant field does not match the requested property")
        identity = value.record.cas_number or value.record.source_key
        if identity not in {component_id, cas_number}:
            raise ValueError("constant identity does not match the consumer")
        return value.quantity(_UNITS[property_id]).value
    if isinstance(value, InputValue):
        if value.quality != "valid":
            raise ValueError("critical input is missing, invalid or outside its domain")
        return value.to(_UNITS[property_id]).value
    raise TypeError("critical consumer requires a CriticalEstimate, ConstantValue or InputValue")


def estimate_critical_property(component, *, method, parameters):
    """Estimate one property for an existing ComponentSpec, preserving every input.

    Parameters use documented Tc/Pc/Vc/Tb/Psat/omega keys and SI units.
    Scalars are explicit caller inputs. Source-bearing values are checked
    against this component's identity and converted using the native units.
    No source discovery, fallback, or table mutation occurs.
    """
    c, records = {}, {}
    for name, value in parameters.items():
        if isinstance(value, (CriticalEstimate, ConstantValue, InputValue)):
            if name not in _UNITS:
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
    else:
        raise ValueError(f"unknown or not yet implemented critical estimate {method!r}")
    return CriticalEstimate((component.identifier,), field, result, method, records, citation)


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
