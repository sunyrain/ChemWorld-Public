"""Immutable single-phase declarations/evaluations over existing native consumers.

This facade owns no evolving inventory, clock, phase split or solver. Caller
amounts are read once, and all physical equations remain in their original
property, EOS, activity and transport modules.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Any

from chemworld.foundation.units import convert_value
from chemworld.physchem.activity_thermodynamics import _freeze, _json
from chemworld.physchem.property_reports import (
    CaloricReference,
    PropertyEvaluation,
    StatePropertyReport,
    ValidityPolicy,
)
from chemworld.physchem.specs import (
    MixtureSpec,
    PropertyCorrelation,
    mass_fractions_from_mole_fractions,
    resolve_component_identifier,
)

_TRANSPORT_UNITS = {
    "viscosity": "Pa*s",
    "thermal_conductivity": "W/(m*K)",
    "surface_tension": "N/m",
}
_METHOD_PHASES: dict[str, dict[str, tuple[str, ...]]] = {
    "viscosity": {
        "Wilke": ("gas",),
        "Herning_Zipperer": ("gas",),
        "Brokaw": ("gas",),
        "LBC": ("gas", "liquid"),
        "log_mole_fraction": ("liquid",),
    },
    "thermal_conductivity": {
        "DIPPR9H": ("liquid",),
        "DIPPR9I": ("liquid",),
        "Filippov": ("liquid",),
        "Lindsay_Bromley": ("gas",),
        "Wassiljewa_Herning_Zipperer": ("gas",),
    },
    "surface_tension": {"Winterfeld_Scriven_Davis": ("liquid",), "Diguilio_Teja": ("liquid",)},
}


@dataclass(frozen=True)
class PhaseTransportSpec:
    property_id: str
    method: str
    source_note: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in _METHOD_PHASES.get(self.property_id, {}) or not self.source_note:
            raise ValueError("phase transport needs a supported explicit method and source note")
        derived = {
            "DIPPR9I": "molar_volumes_m3_mol",
            "Lindsay_Bromley": "viscosities_Pa_s",
            "Winterfeld_Scriven_Davis": "molar_densities_mol_m3",
        }
        if derived.get(self.method) in self.parameters:
            raise ValueError("phase transport consumes current pure curves for derived vectors")
        data = json.loads(json.dumps(_json(self.parameters), allow_nan=False))
        object.__setattr__(self, "parameters", _freeze(data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "phase_transport_v1",
            "property_id": self.property_id,
            "method": self.method,
            "source_note": self.source_note,
            "parameters": _json(self.parameters),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseTransportSpec:
        data = dict(payload)
        if data.pop("model") != "phase_transport_v1":
            raise ValueError("unknown phase transport schema")
        return cls(**data)


@dataclass(frozen=True)
class PhaseEvaluation:
    """Deeply immutable unit-bearing snapshot of an existing state report."""

    report: Mapping[str, Any]

    def __post_init__(self) -> None:
        data = json.loads(json.dumps(_json(self.report), allow_nan=False))
        amounts = data["component_amounts_mol"]
        if (
            not amounts
            or sum(amounts.values()) <= 0
            or any(v < 0 for v in amounts.values())
            or data["phase"] not in {"gas", "liquid", "solid"}
            or data["temperature_K"] <= 0
            or data["pressure_Pa"] <= 0
        ):
            raise ValueError("phase snapshot requires a valid inventory and physical state")
        object.__setattr__(self, "report", _freeze(data))

    def to_dict(self) -> dict[str, Any]:
        return _json(self.report)

    @property
    def phase(self) -> str:
        return str(self.report["phase"])

    @property
    def temperature_K(self) -> float:  # noqa: N802 - original state field with explicit unit
        return float(self.report["temperature_K"])

    @property
    def pressure_Pa(self) -> float:  # noqa: N802 - original state field with explicit unit
        return float(self.report["pressure_Pa"])

    @property
    def properties(self) -> Mapping[str, Any]:
        return self.report["properties"]

    @property
    def component_properties(self) -> Mapping[str, Any]:
        return self.report["component_properties"]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseEvaluation:
        return cls(payload)

    @property
    def composition(self) -> Mapping[str, float]:
        amounts = self.report["component_amounts_mol"]
        total = sum(amounts.values())
        return _freeze({k: v / total for k, v in amounts.items()})

    @property
    def valid(self) -> bool:
        return bool(self.report["valid"])

    def value(
        self, quantity: str, *, component: str | None = None, unit: str | None = None
    ) -> float:
        evaluation = (
            self.report["properties"][quantity]
            if component is None
            else self.report["component_properties"][quantity][component]
        )
        value = float(evaluation["value"])
        return value if unit is None else convert_value(value, evaluation["unit"], unit)


@dataclass(frozen=True, init=False)
class PhasePropertyModel:
    """Immutable serialized physical declaration; each evaluation uses caller inventory.

    Keeping a JSON declaration also prevents later edits to mutable coefficient
    dictionaries in older package dataclasses from changing an existing model.
    """

    _definition_json: str

    def __init__(
        self,
        packages: Sequence[Any],
        *,
        phase: str,
        references: Mapping[str, CaloricReference] | None = None,
        eos: Any = None,
        activity_model: Any = None,
        volume_model: Any = None,
        transport_models: Mapping[str, Any] | None = None,
        helmholtz_model: Any = None,
        water_model: Any = None,
        incompressible_model: Any = None,
        model_id: str = "native-phase-property-model",
    ) -> None:
        if phase not in {"gas", "liquid", "solid"} or not model_id or not packages:
            raise ValueError("phase model requires packages, phase and an identity")
        components = tuple(p.component for p in packages)
        ids = tuple(c.identifier for c in components)
        if len(set(ids)) != len(ids):
            raise ValueError("phase model package identities must be unique")
        if activity_model is not None and (phase != "liquid" or eos is not None):
            raise ValueError("activity phase requires liquid without a simultaneous EOS")
        if eos is not None and (phase == "solid" or set(eos.component_ids) != set(ids)):
            raise ValueError("fluid EOS identities/phase must match the packages")
        if activity_model is not None and set(activity_model.component_ids) != set(ids):
            raise ValueError("activity identities must match the packages")
        if volume_model is not None and (phase != "liquid" or eos is not None):
            raise ValueError("liquid volume model cannot replace a simultaneous fluid EOS")
        if helmholtz_model is not None and (
            len(ids) != 1
            or phase == "solid"
            or eos is not None
            or activity_model is not None
            or volume_model is not None
            or references
        ):
            raise ValueError("pure Helmholtz uses its own phase/caloric reference exclusively")
        if water_model is not None:
            from chemworld.physchem.water_if97_properties import IF97WaterModel

            if (not isinstance(water_model, IF97WaterModel) or len(ids) != 1
                    or phase == "solid" or any(value is not None for value in
                        (eos, activity_model, volume_model, helmholtz_model)) or references):
                raise ValueError("IF97 water uses its own pure phase/caloric reference exclusively")
            water_model.validate_component(components[0])
        if incompressible_model is not None:
            from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel

            if (not isinstance(incompressible_model, IncompressibleMaterialModel)
                    or phase != "liquid" or references or transport_models
                    or any(value is not None for value in
                           (eos, activity_model, volume_model, helmholtz_model, water_model))):
                raise ValueError("incompressible liquid requires an exclusive property declaration")
            incompressible_model.validate_components(components)
        ref: dict[str, Any] = {}
        for alias, value in (references or {}).items():
            key = resolve_component_identifier(components, alias)
            if key in ref and ref[key] != value.to_dict():
                raise ValueError("caloric reference aliases disagree")
            ref[key] = value.to_dict()
        transports = {}
        for quantity, declaration in (transport_models or {}).items():
            if quantity not in _TRANSPORT_UNITS:
                raise ValueError("unsupported phase transport quantity")
            if isinstance(declaration, PhaseTransportSpec):
                if (
                    declaration.property_id != quantity
                    or phase not in _METHOD_PHASES[quantity][declaration.method]
                ):
                    raise ValueError("transport method must match its property and phase")
            elif not isinstance(declaration, PropertyCorrelation):
                raise TypeError("transport needs an original correlation or PhaseTransportSpec")
            else:
                allowed = {quantity, f"{phase}_{quantity}", f"mixture_{quantity}"}
                if declaration.property_id not in allowed:
                    raise ValueError("transport correlation must declare the requested property")
                if declaration.metadata.get("phase", phase) != phase:
                    raise ValueError("transport correlation phase mismatch")
                if len(ids) > 1 and "mixture" not in declaration.metadata:
                    raise ValueError(
                        "fixed mixture transport correlation requires composition binding"
                    )
            transports[quantity] = declaration.to_dict()
        payload = {
            "model": "phase_property_model_v1",
            "model_id": model_id,
            "phase": phase,
            "packages": [p.to_dict() for p in packages],
            "references": ref,
            "eos": None if eos is None else eos.to_dict(),
            "activity_model": None if activity_model is None else activity_model.to_dict(),
            "volume_model": None if volume_model is None else volume_model.to_dict(),
            "transport_models": transports,
            "helmholtz_model": None if helmholtz_model is None else helmholtz_model.to_dict(),
        }
        # Preserve existing declarations/serialized evidence when water is not selected.
        if water_model is not None:
            payload["water_model"] = water_model.to_dict()
        if incompressible_model is not None:
            payload["incompressible_model"] = incompressible_model.to_dict()
        object.__setattr__(
            self, "_definition_json", json.dumps(payload, allow_nan=False, sort_keys=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._definition_json)

    @property
    def model_id(self) -> str:
        return str(self.to_dict()["model_id"])

    @property
    def phase(self) -> str:
        return str(self.to_dict()["phase"])

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(p["component"]["identifier"] for p in self.to_dict()["packages"])

    @staticmethod
    def _decode(payload: Mapping[str, Any]) -> dict[str, Any]:
        from chemworld.physchem.eos import fluid_eos_from_dict
        from chemworld.physchem.equilibrium import ActivityModelSpec
        from chemworld.physchem.helmholtz_solvers import HelmholtzPhaseModel
        from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
        from chemworld.physchem.property_packages import ComponentPropertyPackage
        from chemworld.physchem.water_if97_properties import IF97WaterModel

        values = dict(payload)
        water = values.get("water_model")
        values["water_model"] = None if water is None else IF97WaterModel.from_dict(water)
        incompressible = values.get("incompressible_model")
        values["incompressible_model"] = (None if incompressible is None
            else IncompressibleMaterialModel.from_dict(incompressible))
        if values.pop("model") != "phase_property_model_v1":
            raise ValueError("unknown phase property model schema")
        values["packages"] = tuple(
            ComponentPropertyPackage.from_dict(p) for p in values["packages"]
        )
        values["references"] = {k: CaloricReference(**v) for k, v in values["references"].items()}
        for name, constructor in (
            ("eos", fluid_eos_from_dict),
            ("volume_model", PropertyCorrelation.from_dict),
            ("helmholtz_model", HelmholtzPhaseModel.from_dict),
        ):
            if values[name] is not None:
                values[name] = constructor(values[name])
        if values["activity_model"] is not None:
            activity = dict(values["activity_model"])
            activity["component_ids"] = tuple(activity["component_ids"])
            values["activity_model"] = ActivityModelSpec(**activity)
        values["transport_models"] = {
            k: (
                PhaseTransportSpec.from_dict(v)
                if v.get("model") == "phase_transport_v1"
                else PropertyCorrelation.from_dict(v)
            )
            for k, v in values["transport_models"].items()
        }
        return values

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhasePropertyModel:
        return cls(**cls._decode(payload))

    def capabilities(self) -> Mapping[str, Mapping[str, Any]]:
        """Structural availability; validity and selected sources are checked at T/P."""
        d = self._decode(self.to_dict())
        packages, phase, refs = d["packages"], d["phase"], d["references"]
        if d["incompressible_model"] is not None:
            available = d["incompressible_model"].available_quantities()
            available.update(dict.fromkeys(("cv", "dU_dT_V", "speed_of_sound",
                "chemical_potential", "activity_coefficient", "fugacity_coefficient",
                "surface_tension"), False))
            return _freeze({key: {"supported": value,
                "condition": "Explicit incompressible material; source T/x/phase validity applies"
                if value else "Not supplied by the incompressible correlation model"}
                for key, value in available.items()})
        state = phase == "gas" or d["eos"] is not None
        base_phase = "gas" if d["eos"] is not None else phase
        internal = bool(refs) and all(r.energy_basis == "internal_energy" for r in refs.values())
        cp_id = (
            "isochoric_heat_capacity"
            if internal
            else "ideal_gas_heat_capacity"
            if base_phase == "gas"
            else base_phase + "_heat_capacity"
        )
        capacity = all(
            p.by_property(cp_id)
            or (
                not internal
                and any(
                    c.metadata.get("phase", p.component.default_phase) == base_phase
                    for c in p.by_property("heat_capacity")
                )
            )
            for p in packages
        )
        explicit_energy = all(p.component.identifier in refs for p in packages)
        entropy = explicit_energy and all(r.entropy_J_mol_K is not None for r in refs.values())
        volume = (
            state
            or d["volume_model"] is not None
            or (
                len(packages) == 1
                and any(
                    packages[0].by_property(phase + suffix)
                    for suffix in ("_density", "_molar_volume")
                )
            )
        )
        chemical = state or d["activity_model"] is not None or len(packages) == 1
        supported = {
            "cp": capacity and (not internal or state),
            "dH_dT_P": capacity and (not internal or state),
            "cv": capacity and (state or internal),
            "dU_dT_V": capacity and (state or internal),
            "enthalpy": capacity and explicit_energy and (not internal or state),
            "internal_energy": capacity and explicit_energy and (volume or internal),
            "entropy": capacity and entropy and chemical and (not internal or state),
            "chemical_potential": capacity and entropy and chemical and (not internal or state),
            "molar_volume": volume,
            "density": volume,
            "dV_dT_P": state,
            "dV_dP_T": state,
            "fugacity_coefficient": state,
            "activity_coefficient": d["activity_model"] is not None,
        }
        if d["helmholtz_model"] is not None:
            from chemworld.physchem.helmholtz_properties import _STATE_FIELDS

            supported = dict.fromkeys(
                (*_STATE_FIELDS, "molar_volume", "dV_dT_P", "dV_dP_T", "chemical_potential"), True
            )
        if d["water_model"] is not None:
            from chemworld.physchem.water_if97_properties import IF97_PROPERTY_QUANTITIES

            supported = dict.fromkeys(IF97_PROPERTY_QUANTITIES, True)
        for key in _TRANSPORT_UNITS:
            prop = phase + "_viscosity" if key == "viscosity" else key
            supported[key] = key in d["transport_models"] or (
                len(packages) == 1
                and (key != "surface_tension" or phase == "liquid")
                and any(
                    c.metadata.get("phase", phase) == phase for c in packages[0].by_property(prop)
                )
            )
            declaration = d["transport_models"].get(key)
            if isinstance(declaration, PhaseTransportSpec):
                method = declaration.method
                needs = [] if method == "LBC" else [prop]
                if method in {"DIPPR9I", "Winterfeld_Scriven_Davis"}:
                    needs.append("liquid_molar_volume")
                if method == "Lindsay_Bromley":
                    needs.append("gas_viscosity")
                supported[key] = (method != "LBC" or volume) and all(
                    package.by_property(required)
                    or (required == "liquid_molar_volume" and package.by_property("liquid_density"))
                    for package in packages
                    for required in needs
                )
        return _freeze(
            {
                key: {
                    "supported": bool(value),
                    "condition": (
                        "Declared sources available; T/P validity, composition and "
                        "reference basis still apply"
                        if value
                        else "Requires an additional explicit property/reference/model declaration"
                    ),
                }
                for key, value in supported.items()
            }
        )

    def evaluate(
        self,
        component_amounts_mol: Mapping[str, float],
        *,
        temperature_K: float,
        pressure_Pa: float,
        quantities: Sequence[str],
        validity_policy: ValidityPolicy = "raise",
    ) -> PhaseEvaluation:
        from chemworld.physchem.property_packages import evaluate_property_state

        if (
            not quantities
            or len(set(quantities)) != len(quantities)
            or validity_policy not in {"raise", "warn", "ignore"}
            or any(not isfinite(v) or v <= 0 for v in (temperature_K, pressure_Pa))
        ):
            raise ValueError("phase evaluation needs unique requested quantities and positive T/P")
        available = self.capabilities()
        if any(q not in available or not available[q]["supported"] for q in quantities):
            raise ValueError("requested quantity unavailable; inspect phase capabilities")
        d = self._decode(self.to_dict())
        packages = d["packages"]
        components = tuple(p.component for p in packages)
        amounts = dict.fromkeys((c.identifier for c in components), 0.0)
        for alias, value in component_amounts_mol.items():
            if isinstance(value, bool) or not isfinite(value) or value < 0:
                raise ValueError("phase amounts must be finite and nonnegative")
            key = resolve_component_identifier(components, alias)
            amounts[key] += value
        if not isfinite(sum(amounts.values())) or sum(amounts.values()) <= 0:
            raise ValueError("phase amounts require a positive finite total")
        fractions = tuple(value / sum(amounts.values()) for value in amounts.values())
        # The explicitly selected homogeneous phase governs this evaluation.
        # Stable component identity/default-phase metadata remains unchanged.
        mixture = MixtureSpec(
            tuple(c.identifier for c in components),
            tuple(c.molecular_weight_g_mol for c in components),
            fractions,
            mass_fractions_from_mole_fractions(components, fractions),
            d["phase"],
            temperature_K,
            pressure_Pa,
        )
        explicit = {q: d["transport_models"][q] for q in quantities if q in d["transport_models"]}
        core = [q for q in quantities if q not in explicit]
        if any(isinstance(v, PhaseTransportSpec) and v.method == "LBC" for v in explicit.values()):
            core = list(dict.fromkeys((*core, "molar_volume")))
        if d["incompressible_model"] is not None:
            report = evaluate_property_state(
                packages, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
                phase=d["phase"], quantities=core, validity_policy=validity_policy,
                incompressible_model=d["incompressible_model"])
        elif d["helmholtz_model"] is not None or d["water_model"] is not None:
            if validity_policy != "raise":
                raise ValueError("Helmholtz/IF97 phase evaluation requires strict source validity")
            report = (
                packages[0].evaluate_state(
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    phase=d["phase"],
                    quantities=core,
                    helmholtz_model=d["helmholtz_model"],
                    water_model=d["water_model"],
                )
                if core
                else StatePropertyReport(
                    d["model_id"], d["phase"], temperature_K, pressure_Pa, amounts, {}
                )
            )
            report = replace(report, component_amounts_mol=amounts)
        else:
            report = (
                evaluate_property_state(
                    packages,
                    amounts,
                    temperature_K=temperature_K,
                    pressure_Pa=pressure_Pa,
                    phase=d["phase"],
                    quantities=core,
                    references=d["references"],
                    eos=d["eos"],
                    activity_model=d["activity_model"],
                    volume_model=d["volume_model"],
                    validity_policy=validity_policy,
                )
                if core
                else StatePropertyReport(
                    d["model_id"], d["phase"], temperature_K, pressure_Pa, amounts, {}
                )
            )
        properties, details = dict(report.properties), dict(report.model_details)
        for quantity, declaration in explicit.items():
            evaluation, evidence = _transport_evaluation(
                declaration, quantity, packages, mixture, report, validity_policy
            )
            properties[quantity] = evaluation
            details.setdefault("transport", {})[quantity] = evidence
        details["phase_model_id"] = d["model_id"]
        report = replace(
            report,
            properties={k: properties[k] for k in quantities if k in properties},
            model_details=details,
        )
        return PhaseEvaluation(report.to_dict())


def _transport_evaluation(
    declaration: Any,
    quantity: str,
    packages: Sequence[Any],
    mixture: MixtureSpec,
    state: StatePropertyReport,
    policy: ValidityPolicy,
) -> tuple[PropertyEvaluation, dict[str, Any]]:
    from chemworld.physchem.conductivity_mixtures import mixture_conductivity_ledger
    from chemworld.physchem.interfaces import mixture_surface_tension
    from chemworld.physchem.property_equations import evaluate_correlation
    from chemworld.physchem.transport_properties import (
        MixtureTransportLedger,
        mixture_viscosity_log_rule,
    )
    from chemworld.physchem.viscosity_mixtures import (
        dense_mixture_viscosity_ledger,
        gas_mixture_viscosity_ledger,
    )

    T, P, phase = mixture.temperature_K, mixture.pressure_Pa, mixture.phase_label
    if isinstance(declaration, PropertyCorrelation):
        binding = declaration.metadata.get("mixture")
        if binding is not None:
            if not isinstance(binding, dict):
                raise ValueError("transport composition binding must be an original MixtureSpec")
            original = MixtureSpec.from_dict(binding)
            if (
                original.component_ids != mixture.component_ids
                or any(
                    abs(a - b) > 1e-9
                    for a, b in zip(original.mole_fractions, mixture.mole_fractions, strict=True)
                )
                or original.molecular_weights_g_mol != mixture.molecular_weights_g_mol
            ):
                raise ValueError("fixed transport correlation composition/MW differs from phase")
        evaluation = evaluate_correlation(
            declaration,
            temperature_K=T,
            pressure_Pa=P,
            validity_policy="warn" if policy == "ignore" else policy,
        ).to(_TRANSPORT_UNITS[quantity])
        return replace(evaluation, property_id=quantity), declaration.to_dict()
    parameters = _json(declaration.parameters)
    pure_evidence: list[dict[str, Any]] = []

    def values(prop: str, unit: str, temperatures: Sequence[float] | None = None) -> list[float]:
        output = []
        for i, package in enumerate(packages):
            temperature = T if temperatures is None else temperatures[i]
            evaluation_policy = "warn" if policy == "ignore" else policy
            if package.by_property(prop):
                candidates = tuple(
                    c for c in package.by_property(prop) if c.metadata.get("phase", phase) == phase
                )
                if not candidates:
                    raise ValueError(f"No pure {prop} correlation on the selected phase")
                curve = package.select_correlation(
                    prop, candidates=candidates, temperature_K=temperature, pressure_Pa=P
                )
                value = evaluate_correlation(
                    curve,
                    temperature_K=temperature,
                    pressure_Pa=P,
                    molecular_weight_g_mol=package.component.molecular_weight_g_mol,
                    validity_policy=evaluation_policy,
                ).to(unit)
            else:
                value = package.evaluate(
                    prop,
                    temperature_K=temperature,
                    pressure_Pa=P,
                    validity_policy=evaluation_policy,
                ).to(unit)
            output.append(value.value)
            pure_evidence.append(value.to_dict())
        return output

    method = declaration.method
    if quantity == "viscosity":
        if method == "LBC":
            result = dense_mixture_viscosity_ledger(
                mixture, state.properties["molar_volume"].value, **parameters
            )
        else:
            viscosities = values(phase + "_viscosity", "Pa*s")
            if method == "log_mole_fraction":
                if parameters:
                    raise ValueError("logarithmic viscosity has no extra parameters")
                result = mixture_viscosity_log_rule(mixture, tuple(viscosities))
            else:
                result = gas_mixture_viscosity_ledger(
                    mixture, viscosities, method=method, **parameters
                )
    elif quantity == "thermal_conductivity":
        if method == "DIPPR9I":
            parameters["molar_volumes_m3_mol"] = values("liquid_molar_volume", "m^3/mol")
        elif method == "Lindsay_Bromley":
            parameters["viscosities_Pa_s"] = values("gas_viscosity", "Pa*s")
        result = mixture_conductivity_ledger(
            mixture, values(quantity, "W/(m*K)"), method=method, **parameters
        )
    else:
        if method == "Winterfeld_Scriven_Davis":
            parameters["molar_densities_mol_m3"] = [
                1 / v for v in values("liquid_molar_volume", "m^3/mol")
            ]
        temperatures = (
            parameters.get("boiling_temperatures_K") if method == "Diguilio_Teja" else None
        )
        result = mixture_surface_tension(
            mixture,
            method=method,
            component_tensions_N_m=values(quantity, "N/m", temperatures),
            **parameters,
        )
    evidence = {
        "declaration": declaration.to_dict(),
        "mixture": mixture.to_dict(),
        "pure_evaluations": pure_evidence,
        "result": result.to_dict(),
    }
    if isinstance(result, MixtureTransportLedger):
        evaluation = PropertyEvaluation(
            quantity,
            result.ledger_id,
            result.method_family,
            result.mixture_value,
            result.unit,
            {"temperature": T, "pressure": P},
            result.warnings,
        )
    else:
        evaluation = replace(result, property_id=quantity)
    warnings = tuple(
        dict.fromkeys(
            (
                *evaluation.warnings,
                *(warning for item in pure_evidence for warning in item["warnings"]),
            )
        )
    )
    return replace(evaluation, warnings=warnings), evidence
