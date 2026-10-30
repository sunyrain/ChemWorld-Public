"""Viscosity, conductivity, diffusivity, and mixture transport helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt
from typing import TYPE_CHECKING, cast

from chemworld.physchem.property_cards import _TRANSPORT_REFERENCE_READING
from chemworld.physchem.property_equations import (
    _transport_method_family,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import (
    CaloricReference,
    PropertyEvaluation,
    ValidityPolicy,
    _require_same_keys,
    _validate_phase,
    _validated_fraction_mapping,
)
from chemworld.physchem.specs import MixtureSpec, PropertyCorrelation

if TYPE_CHECKING:
    from chemworld.physchem.equilibrium import ActivityModelSpec
    from chemworld.physchem.incompressible_properties import IncompressibleMaterialModel
    from chemworld.physchem.property_packages import ComponentPropertyPackage
    from chemworld.physchem.transport import FluidState


def ideal_gas_transport_state(
    packages: Sequence[ComponentPropertyPackage], amounts: Mapping[str, float],
    temperature_K: float, pressure_Pa: float,
    references: Mapping[str, CaloricReference] | None,
) -> tuple[FluidState, float, float]:
    """Explicit ideal-gas volume, Wilke viscosity and WHZ conductivity closure.

    Return the existing FluidState, molar enthalpy and total volume, as for
    liquid_transport_state. This is a local dilute-gas property closure for
    pipe/heat-transfer consumers, not a compressible-flow or dense-fluid EOS.
    All properties and caloric datums are evaluated by the original package.
    """
    from chemworld.physchem.conductivity_mixtures import (
        wassiljewa_herning_zipperer_conductivity,
    )
    from chemworld.physchem.property_packages import evaluate_property_state
    from chemworld.physchem.transport import FluidState
    from chemworld.physchem.viscosity_mixtures import wilke_viscosity

    ids = tuple(p.component.identifier for p in packages)
    _require_same_keys(dict.fromkeys(ids), amounts, "component amounts")
    if references is None:
        raise ValueError("gas transport needs explicit caloric references")
    _require_same_keys(dict.fromkeys(ids), references, "explicit caloric references")
    report = evaluate_property_state(
        packages, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
        phase="gas", quantities=("cp", "enthalpy", "density", "molar_volume"),
        references=references,
    )
    total = sum(amounts.values())
    fractions = tuple(amounts[k] / total for k in ids)
    weights = cast(tuple[float, ...], tuple(p.component.molecular_weight_g_mol for p in packages))
    pure = [p.evaluate_state(
        temperature_K=temperature_K, pressure_Pa=pressure_Pa, phase="gas",
        quantities=("viscosity", "thermal_conductivity"),
    ) for p in packages]
    values = {key: value.value for key, value in report.properties.items()}
    molecular_weight = sum(x * mw for x, mw in zip(fractions, weights, strict=True)) / 1000
    fluid = FluidState(
        values["density"],
        wilke_viscosity(fractions, tuple(p.properties["viscosity"].value for p in pure), weights),
        values["cp"] / molecular_weight,
        wassiljewa_herning_zipperer_conductivity(
            fractions, tuple(p.properties["thermal_conductivity"].value for p in pure), weights,
        ),
    )
    return fluid, values["enthalpy"], total * values["molar_volume"]


def liquid_transport_state(
    packages: Sequence[ComponentPropertyPackage], amounts: Mapping[str, float],
    temperature_K: float, pressure_Pa: float,
    references: Mapping[str, CaloricReference] | None, *,
    viscosity_pressure_parameters: Mapping[str, Mapping[str, float | str]] | None = None,
    volume_mixing_method: str | None = None,
    volume_mixing_parameters: Mapping[str, Mapping[str, float]] | None = None,
    volume_pressure_parameters: Mapping[str, float | str] | None = None,
    activity_model: ActivityModelSpec | None = None,
    incompressible_model: IncompressibleMaterialModel | None = None,
) -> tuple[FluidState, float, float]:
    """Declared volume, log-viscosity and DIPPR9H mixture closure.

    Caloric and pure transport values come from the common property evaluator;
    this helper selects mixing laws without adding a second property database.
    Amgat uses current pure volumes; an explicit Rackett/COSTALD selection
    consumes the current MixtureSpec T/P/composition at every evaluation.
    Liquid calorics use the declared pure references and optional same-GE
    excess enthalpy/heat capacity. This remains a declared single-liquid path;
    no liquid-liquid split or implicit excess-volume law is supplied.
    """
    from chemworld.physchem.property_packages import evaluate_property_state
    from chemworld.physchem.transport import FluidState

    if incompressible_model is not None:
        if (references or activity_model is not None or viscosity_pressure_parameters
                or volume_mixing_method is not None or volume_mixing_parameters
                or volume_pressure_parameters):
            raise ValueError("incompressible transport uses its own caloric/volume/transport fits")
        report = evaluate_property_state(
            packages, amounts, temperature_K=temperature_K, pressure_Pa=pressure_Pa,
            phase="liquid", quantities=("cp", "enthalpy", "density", "molar_volume",
                                       "viscosity", "thermal_conductivity"),
            incompressible_model=incompressible_model)
        values = {key: value.value for key, value in report.properties.items()}
        mass_per_mole = cast(float, report.model_details["molar_mass_kg_mol"])
        return (FluidState(values["density"], values["viscosity"], values["cp"]/mass_per_mole,
                           values["thermal_conductivity"]), values["enthalpy"],
                sum(report.component_amounts_mol.values())*values["molar_volume"])

    report = evaluate_property_state(
        packages,
        amounts,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase="liquid",
        quantities=("cp", "enthalpy"),
        references=references,
        activity_model=activity_model,
    )
    ids = tuple(p.component.identifier for p in packages)
    weights = cast(tuple[float, ...], tuple(p.component.molecular_weight_g_mol for p in packages))
    total, mass = (
        sum(amounts.values()),
        sum(amounts[k] * mw for k, mw in zip(ids, weights, strict=True)),
    )
    mixture = MixtureSpec(
        ids,
        weights,
        tuple(amounts[k] / total for k in ids),
        tuple(amounts[k] * mw / mass for k, mw in zip(ids, weights, strict=True)),
        "liquid",
        temperature_K,
        pressure_Pa,
    )
    pure = [
        p.evaluate_state(
            temperature_K=temperature_K,
            pressure_Pa=pressure_Pa,
            phase="liquid",
            quantities=("density", "viscosity", "thermal_conductivity"),
        )
        for p in packages
    ]
    viscosities = [p.properties["viscosity"].value for p in pure]
    if viscosity_pressure_parameters:
        from chemworld.physchem.dense_viscosity import lucas_liquid_viscosity

        if set(viscosity_pressure_parameters) - set(ids):
            raise ValueError("pressure correction names an undeclared liquid component")
        for i, package in enumerate(packages):
            viscosity_parameters = viscosity_pressure_parameters.get(ids[i])
            if viscosity_parameters is not None:
                psat = package.evaluate(
                    "vapor_pressure", temperature_K=temperature_K, validity_policy="raise",
                ).to("Pa").value
                viscosities[i] = lucas_liquid_viscosity(
                    temperature_K, pressure_Pa, saturation_pressure_Pa=psat,
                    saturation_viscosity_Pa_s=viscosities[i], **viscosity_parameters,
                )
    mu = mixture_viscosity_log_rule(mixture, tuple(viscosities))
    conductivity = liquid_mixture_thermal_conductivity_dippr9h_ledger(
        component_mass_fractions=dict(zip(ids, mixture.mass_fractions, strict=True)),
        component_thermal_conductivities_W_m_K={
            k: p.properties["thermal_conductivity"].value for k, p in zip(ids, pure, strict=True)
        },
    )
    values = {key: value.value for key, value in report.properties.items()}
    from chemworld.physchem.volume_mixtures import mixture_volume_evaluation

    method = volume_mixing_method or "Amgat"
    parameters = volume_mixing_parameters or {}
    if method == "Amgat":
        if parameters or volume_pressure_parameters:
            raise ValueError("Amgat transport uses current pure volumes, not frozen parameters")
        parameters = {
            k: {"Vms": mw / 1000 / p.properties["density"].value}
            for k, mw, p in zip(ids, weights, pure, strict=True)
        }
    volume = total * mixture_volume_evaluation(
        mixture, method=method, component_parameters=parameters,
        **(volume_pressure_parameters or {}),
    ).value
    fluid = FluidState(
        mass / 1000 / volume,
        mu.value,
        values["cp"] / (mass / total / 1000),
        conductivity.mixture_value,
    )
    return fluid, values["enthalpy"], mass / 1000 / fluid.density_kg_m3


@dataclass(frozen=True)
class TransportPropertyReport:
    """Transport property value with validity and uncertainty metadata."""

    evaluation: PropertyEvaluation
    phase: str
    method_family: str
    validity_status: str
    relative_uncertainty: float | None = None
    uncertainty_note: str = ""
    reference_reading: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_phase(self.phase)
        if self.evaluation.property_id not in {
            "liquid_viscosity",
            "gas_viscosity",
            "mixture_viscosity",
            "thermal_conductivity",
            "mixture_thermal_conductivity",
            "binary_gas_diffusivity",
            "effective_gas_diffusivity",
            "thermal_diffusivity",
        }:
            raise ValueError("TransportPropertyReport requires a transport property")
        if self.evaluation.value <= 0 or not isfinite(self.evaluation.value):
            raise ValueError("transport property value must be positive and finite")
        if self.validity_status not in {"valid", "out_of_range", "estimated"}:
            raise ValueError("invalid transport validity_status")
        if self.relative_uncertainty is not None and (
            self.relative_uncertainty < 0 or not isfinite(self.relative_uncertainty)
        ):
            raise ValueError("relative_uncertainty must be nonnegative when provided")
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "property": self.evaluation.to_dict(),
            "phase": self.phase,
            "method_family": self.method_family,
            "validity_status": self.validity_status,
            "relative_uncertainty": self.relative_uncertainty,
            "uncertainty_note": self.uncertainty_note,
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class MixtureTransportLedger:
    """Mixture transport-property ledger with method-specific contributions."""

    ledger_id: str
    phase: str
    property_id: str
    method_family: str
    unit: str
    mixture_value: float
    contributions: dict[str, dict[str, float]]
    warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise ValueError("ledger_id cannot be empty")
        _validate_phase(self.phase)
        if self.property_id not in {
            "mixture_viscosity",
            "mixture_thermal_conductivity",
            "effective_gas_diffusivity",
        }:
            raise ValueError("MixtureTransportLedger requires a mixture transport property")
        if self.mixture_value <= 0 or not isfinite(self.mixture_value):
            raise ValueError("mixture_value must be positive and finite")
        if not self.contributions:
            raise ValueError("MixtureTransportLedger requires contributions")
        object.__setattr__(
            self,
            "contributions",
            {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_id": self.ledger_id,
            "phase": self.phase,
            "property_id": self.property_id,
            "method_family": self.method_family,
            "unit": self.unit,
            "mixture_value": self.mixture_value,
            "contributions": {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
            "warnings": list(self.warnings),
            "reference_reading": list(self.reference_reading),
        }


def mixture_viscosity_log_rule(
    mixture: MixtureSpec,
    component_viscosities_Pa_s: tuple[float, ...],
) -> PropertyEvaluation:
    """Logarithmic liquid-mixture viscosity rule."""

    if len(component_viscosities_Pa_s) != len(mixture.component_ids):
        raise ValueError("Viscosity vector must match mixture components")
    if any(mu <= 0 for mu in component_viscosities_Pa_s):
        raise ValueError("Component viscosities must be positive")
    log_mu = sum(
        z * log(mu)
        for z, mu in zip(
            mixture.mole_fractions,
            component_viscosities_Pa_s,
            strict=True,
        )
    )
    return PropertyEvaluation(
        property_id="mixture_viscosity",
        correlation_id="log_mole_fraction_mixing",
        equation_id="log_mole_fraction_mixing",
        value=exp(log_mu),
        unit="Pa*s",
        inputs={"temperature": mixture.temperature_K, "pressure": mixture.pressure_Pa},
    )


def transport_property_report(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    phase: str,
    pressure_Pa: float | None = None,
    molecular_weight_g_mol: float | None = None,
    validity_policy: ValidityPolicy = "warn",
    relative_uncertainty: float | None = None,
    uncertainty_note: str = "",
) -> TransportPropertyReport:
    """Evaluate a pure-component transport property with method metadata."""

    _validate_phase(phase)
    if correlation.property_id not in {
        "liquid_viscosity",
        "gas_viscosity",
        "thermal_conductivity",
    }:
        raise ValueError("transport_property_report requires viscosity or thermal_conductivity")
    result = evaluate_correlation(
        correlation,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        molecular_weight_g_mol=molecular_weight_g_mol,
        validity_policy=validity_policy,
    )
    return TransportPropertyReport(
        evaluation=result,
        phase=phase,
        method_family=_transport_method_family(correlation.equation_id),
        validity_status="out_of_range" if result.warnings else "valid",
        relative_uncertainty=relative_uncertainty,
        uncertainty_note=uncertainty_note,
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def gas_thermal_conductivity_dippr9b_report(
    *,
    temperature_K: float,
    molecular_weight_g_mol: float,
    molar_cv_J_mol_K: float,
    viscosity_Pa_s: float,
    critical_temperature_K: float | None = None,
    molecule_type: str = "linear",
    relative_uncertainty: float = 0.2,
) -> TransportPropertyReport:
    """DIPPR9B-style gas thermal conductivity from Cv and viscosity."""

    from chemworld.physchem.conductivity_kernels import conductivity_value

    if molecule_type not in {"monoatomic", "nonlinear", "linear"}:
        raise ValueError("molecule_type must be monoatomic, linear, or nonlinear")
    parameters = {"MW": molecular_weight_g_mol, "Cvm": molar_cv_J_mol_K, "mu": viscosity_Pa_s}
    if molecule_type == "linear":
        if critical_temperature_K is None or critical_temperature_K <= 0:
            raise ValueError("linear DIPPR9B gas conductivity requires positive Tc")
        parameters["Tc"] = critical_temperature_K
    conductivity = conductivity_value("DIPPR9B_" + molecule_type, temperature_K, **parameters)
    return TransportPropertyReport(
        evaluation=PropertyEvaluation(
            property_id="thermal_conductivity",
            correlation_id="dippr9b_gas_thermal_conductivity",
            equation_id="dippr9b_gas_thermal_conductivity",
            value=conductivity,
            unit="W/(m*K)",
            inputs={
                "temperature": temperature_K,
                "molecular_weight_g_mol": molecular_weight_g_mol,
                "molar_cv_J_mol_K": molar_cv_J_mol_K,
                "viscosity_Pa_s": viscosity_Pa_s,
                "critical_temperature_K": critical_temperature_K or 0.0,
            },
            warnings=(
                "DIPPR9B is an empirical gas thermal-conductivity estimate; "
                "accuracy depends on pure-gas viscosity and molecule type.",
            ),
        ),
        phase="gas",
        method_family="DIPPR9B",
        validity_status="estimated",
        relative_uncertainty=relative_uncertainty,
        uncertainty_note="DIPPR-style empirical gas conductivity estimate.",
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def wilke_gas_mixture_viscosity_ledger(
    *,
    component_mole_fractions: Mapping[str, float],
    component_viscosities_Pa_s: Mapping[str, float],
    component_molecular_weights_g_mol: Mapping[str, float],
    ledger_id: str = "wilke_gas_mixture_viscosity",
) -> MixtureTransportLedger:
    """Wilke low-pressure gas-mixture viscosity ledger."""

    from chemworld.physchem.viscosity_mixtures import wilke_denominators

    fractions = _validated_fraction_mapping(component_mole_fractions, "mole")
    _require_same_keys(fractions, component_viscosities_Pa_s, "viscosity")
    _require_same_keys(fractions, component_molecular_weights_g_mol, "molecular weight")
    component_ids = tuple(fractions)
    denominators = wilke_denominators(
        tuple(fractions.values()),
        tuple(component_viscosities_Pa_s[k] for k in component_ids),
        tuple(component_molecular_weights_g_mol[k] for k in component_ids),
    )
    contributions: dict[str, dict[str, float]] = {}
    mixture_viscosity = 0.0
    for i, denominator in zip(component_ids, denominators, strict=True):
        yi = fractions[i]
        mui = component_viscosities_Pa_s[i]
        mwi = component_molecular_weights_g_mol[i]
        partial = yi * mui / denominator
        mixture_viscosity += partial
        contributions[i] = {
            "mole_fraction": yi,
            "viscosity_Pa_s": mui,
            "molecular_weight_g_mol": mwi,
            "wilke_denominator": denominator,
            "partial_viscosity_Pa_s": partial,
        }
    return MixtureTransportLedger(
        ledger_id=ledger_id,
        phase="gas",
        property_id="mixture_viscosity",
        method_family="Wilke",
        unit="Pa*s",
        mixture_value=mixture_viscosity,
        contributions=contributions,
        warnings=(
            "Wilke low-pressure gas-mixture rule; hydrogen-rich systems can "
            "have larger errors and pure-component viscosity errors propagate.",
        ),
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def liquid_mixture_thermal_conductivity_dippr9h_ledger(
    *,
    component_mass_fractions: Mapping[str, float],
    component_thermal_conductivities_W_m_K: Mapping[str, float],
    ledger_id: str = "dippr9h_liquid_mixture_conductivity",
) -> MixtureTransportLedger:
    """DIPPR9H/Vredeveld liquid-mixture thermal-conductivity ledger."""

    fractions = _validated_fraction_mapping(component_mass_fractions, "mass")
    _require_same_keys(fractions, component_thermal_conductivities_W_m_K, "conductivity")
    from chemworld.physchem.conductivity_mixtures import dippr9h_conductivity_terms

    terms = dippr9h_conductivity_terms(
        tuple(fractions.values()),
        tuple(component_thermal_conductivities_W_m_K[k] for k in fractions),
    )
    contributions: dict[str, dict[str, float]] = {}
    max_k = 0.0
    min_k = float("inf")
    for (component_id, mass_fraction), term in zip(fractions.items(), terms, strict=True):
        conductivity = component_thermal_conductivities_W_m_K[component_id]
        max_k = max(max_k, conductivity)
        min_k = min(min_k, conductivity)
        contributions[component_id] = {
            "mass_fraction": mass_fraction,
            "thermal_conductivity_W_m_K": conductivity,
            "inverse_square_contribution": term,
        }
    warnings = ["DIPPR9H assumes nonaqueous liquid mixtures and can deviate up to 20%."]
    if max_k > 2.0 * min_k:
        warnings.append("Component conductivities differ by more than 2x; DIPPR9H warning applies.")
    return MixtureTransportLedger(
        ledger_id=ledger_id,
        phase="liquid",
        property_id="mixture_thermal_conductivity",
        method_family="DIPPR9H",
        unit="W/(m*K)",
        mixture_value=sum(terms) ** -0.5,
        contributions=contributions,
        warnings=tuple(warnings),
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def binary_gas_diffusivity_fuller_report(
    *,
    temperature_K: float,
    pressure_Pa: float,
    molecular_weight_a_g_mol: float,
    molecular_weight_b_g_mol: float,
    diffusion_volume_a: float,
    diffusion_volume_b: float,
    component_a: str = "A",
    component_b: str = "B",
    relative_uncertainty: float = 0.2,
) -> TransportPropertyReport:
    """Fuller-Schettler-Giddings binary gas diffusivity estimate."""

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa <= 0:
        raise ValueError("pressure_Pa must be positive")
    if molecular_weight_a_g_mol <= 0 or molecular_weight_b_g_mol <= 0:
        raise ValueError("molecular weights must be positive")
    if diffusion_volume_a <= 0 or diffusion_volume_b <= 0:
        raise ValueError("diffusion volumes must be positive")
    pressure_atm = pressure_Pa / 101325.0
    mass_factor = sqrt(
        (molecular_weight_a_g_mol + molecular_weight_b_g_mol)
        / (2.0 * molecular_weight_a_g_mol * molecular_weight_b_g_mol)
    )
    diffusion_cm2_s = (
        1.43e-3
        * temperature_K**1.75
        * mass_factor
        / (
            pressure_atm
            * (diffusion_volume_a ** (1.0 / 3.0) + diffusion_volume_b ** (1.0 / 3.0)) ** 2.0
        )
    )
    warnings = [
        "Fuller gas diffusivity is an empirical low-pressure estimate based on diffusion volumes.",
    ]
    if pressure_atm < 0.05 or pressure_atm > 20.0:
        warnings.append("pressure outside nominal Fuller low/moderate-pressure range")
    if temperature_K < 250.0 or temperature_K > 1500.0:
        warnings.append("temperature outside broad gas-diffusivity screening range")
    return TransportPropertyReport(
        evaluation=PropertyEvaluation(
            property_id="binary_gas_diffusivity",
            correlation_id=f"fuller_diffusivity:{component_a}:{component_b}",
            equation_id="fuller_schettler_giddings",
            value=diffusion_cm2_s * 1e-4,
            unit="m^2/s",
            inputs={
                "temperature": temperature_K,
                "pressure": pressure_Pa,
                "molecular_weight_a_g_mol": molecular_weight_a_g_mol,
                "molecular_weight_b_g_mol": molecular_weight_b_g_mol,
                "diffusion_volume_a": diffusion_volume_a,
                "diffusion_volume_b": diffusion_volume_b,
            },
            warnings=tuple(warnings),
        ),
        phase="gas",
        method_family="Fuller-Schettler-Giddings",
        validity_status="estimated",
        relative_uncertainty=relative_uncertainty,
        uncertainty_note="Screening estimate; requires calibrated diffusion volumes.",
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def gas_mixture_effective_diffusivity_ledger(
    *,
    target_component: str,
    component_mole_fractions: Mapping[str, float],
    binary_diffusivities_m2_s: Mapping[str, float],
    ledger_id: str = "gas_mixture_effective_diffusivity",
) -> MixtureTransportLedger:
    """Mixture-averaged gas diffusivity for one dilute/trace component."""

    fractions = _validated_fraction_mapping(component_mole_fractions, "mole")
    if target_component not in fractions:
        raise ValueError("target_component must appear in component_mole_fractions")
    others = [component_id for component_id in fractions if component_id != target_component]
    if not others:
        raise ValueError("effective diffusivity requires at least two components")
    denominator = 0.0
    contributions: dict[str, dict[str, float]] = {}
    for component_id in others:
        if component_id not in binary_diffusivities_m2_s:
            raise ValueError(f"missing binary diffusivity for {component_id!r}")
        diffusivity = binary_diffusivities_m2_s[component_id]
        if diffusivity <= 0:
            raise ValueError(f"binary diffusivity must be positive for {component_id!r}")
        term = fractions[component_id] / diffusivity
        denominator += term
        contributions[component_id] = {
            "mole_fraction": fractions[component_id],
            "binary_diffusivity_m2_s": diffusivity,
            "resistance_term_s_m2": term,
        }
    if denominator <= 0:
        raise ValueError("effective diffusivity denominator must be positive")
    target_fraction = fractions[target_component]
    effective = (1.0 - target_fraction) / denominator
    contributions[target_component] = {
        "mole_fraction": target_fraction,
        "effective_diffusivity_m2_s": effective,
    }
    return MixtureTransportLedger(
        ledger_id=ledger_id,
        phase="gas",
        property_id="effective_gas_diffusivity",
        method_family="mixture-averaged gas diffusivity",
        unit="m^2/s",
        mixture_value=effective,
        contributions=contributions,
        warnings=(
            "Mixture-averaged diffusivity assumes binary diffusivities are "
            "available for all non-target components.",
        ),
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )


def thermal_diffusivity_report(
    *,
    thermal_conductivity_W_m_K: float,
    density_kg_m3: float,
    heat_capacity_J_kg_K: float,
    phase: str,
) -> TransportPropertyReport:
    """Thermal diffusivity alpha = k/(rho Cp)."""

    _validate_phase(phase)
    if thermal_conductivity_W_m_K <= 0:
        raise ValueError("thermal_conductivity_W_m_K must be positive")
    if density_kg_m3 <= 0:
        raise ValueError("density_kg_m3 must be positive")
    if heat_capacity_J_kg_K <= 0:
        raise ValueError("heat_capacity_J_kg_K must be positive")
    value = thermal_conductivity_W_m_K / (density_kg_m3 * heat_capacity_J_kg_K)
    return TransportPropertyReport(
        evaluation=PropertyEvaluation(
            property_id="thermal_diffusivity",
            correlation_id="thermal_diffusivity_from_k_rho_cp",
            equation_id="thermal_diffusivity_definition",
            value=value,
            unit="m^2/s",
            inputs={
                "thermal_conductivity_W_m_K": thermal_conductivity_W_m_K,
                "density_kg_m3": density_kg_m3,
                "heat_capacity_J_kg_K": heat_capacity_J_kg_K,
            },
        ),
        phase=phase,
        method_family="definition",
        validity_status="valid",
        reference_reading=_TRANSPORT_REFERENCE_READING,
    )
