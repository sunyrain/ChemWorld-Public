"""Viscosity, conductivity, diffusivity, and mixture transport helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, log, sqrt

from chemworld.physchem.property_cards import _TRANSPORT_REFERENCE_READING
from chemworld.physchem.property_equations import (
    _transport_method_family,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import (
    PropertyEvaluation,
    ValidityPolicy,
    _require_same_keys,
    _validate_phase,
    _validated_fraction_mapping,
)
from chemworld.physchem.specs import MixtureSpec, PropertyCorrelation


def liquid_transport_state(packages, amounts, temperature_K, pressure_Pa, references):
    """Declared ideal-volume, log-viscosity and DIPPR9H mixture closure.

    Caloric and pure transport values come from the common property evaluator;
    this helper selects mixing laws without adding a second property database.
    """
    from chemworld.physchem.property_packages import evaluate_property_state
    from chemworld.physchem.transport import FluidState

    report = evaluate_property_state(
        packages,
        amounts,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        phase="liquid",
        quantities=("cp", "enthalpy"),
        references=references,
    )
    ids = tuple(p.component.identifier for p in packages)
    weights = tuple(p.component.molecular_weight_g_mol for p in packages)
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
    mu = mixture_viscosity_log_rule(mixture, tuple(p.properties["viscosity"].value for p in pure))
    conductivity = liquid_mixture_thermal_conductivity_dippr9h_ledger(
        component_mass_fractions=dict(zip(ids, mixture.mass_fractions, strict=True)),
        component_thermal_conductivities_W_m_K={
            k: p.properties["thermal_conductivity"].value for k, p in zip(ids, pure, strict=True)
        },
    )
    values = {key: value.value for key, value in report.properties.items()}
    volume = sum(
        amounts[k] * mw / 1000 / p.properties["density"].value
        for k, mw, p in zip(ids, weights, pure, strict=True)
    )
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

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if molecular_weight_g_mol <= 0:
        raise ValueError("molecular_weight_g_mol must be positive")
    if molar_cv_J_mol_K <= 0:
        raise ValueError("molar_cv_J_mol_K must be positive")
    if viscosity_Pa_s <= 0:
        raise ValueError("viscosity_Pa_s must be positive")
    cv_j_kmol_k = molar_cv_J_mol_K * 1000.0
    if molecule_type == "monoatomic":
        conductivity = 2.5 * viscosity_Pa_s * cv_j_kmol_k / molecular_weight_g_mol
    elif molecule_type == "nonlinear":
        conductivity = viscosity_Pa_s / molecular_weight_g_mol * (1.15 * cv_j_kmol_k + 16903.36)
    elif molecule_type == "linear":
        if critical_temperature_K is None or critical_temperature_K <= 0:
            raise ValueError("linear DIPPR9B gas conductivity requires positive Tc")
        reduced_temperature = temperature_K / critical_temperature_K
        conductivity = (
            viscosity_Pa_s
            / molecular_weight_g_mol
            * (1.30 * cv_j_kmol_k + 14644.0 - 2928.80 / reduced_temperature)
        )
    else:
        raise ValueError("molecule_type must be monoatomic, linear, or nonlinear")
    if conductivity <= 0 or not isfinite(conductivity):
        raise ValueError("DIPPR9B gas conductivity returned nonpositive value")
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

    fractions = _validated_fraction_mapping(component_mole_fractions, "mole")
    _require_same_keys(fractions, component_viscosities_Pa_s, "viscosity")
    _require_same_keys(fractions, component_molecular_weights_g_mol, "molecular weight")
    component_ids = tuple(fractions)
    contributions: dict[str, dict[str, float]] = {}
    mixture_viscosity = 0.0
    for i in component_ids:
        yi = fractions[i]
        mui = component_viscosities_Pa_s[i]
        mwi = component_molecular_weights_g_mol[i]
        if mui <= 0:
            raise ValueError(f"viscosity must be positive for {i!r}")
        if mwi <= 0:
            raise ValueError(f"molecular weight must be positive for {i!r}")
        denominator = 0.0
        for j in component_ids:
            muj = component_viscosities_Pa_s[j]
            mwj = component_molecular_weights_g_mol[j]
            if muj <= 0:
                raise ValueError(f"viscosity must be positive for {j!r}")
            if mwj <= 0:
                raise ValueError(f"molecular weight must be positive for {j!r}")
            phi_ij = (1.0 + sqrt(mui / muj) * (mwj / mwi) ** 0.25) ** 2.0 / sqrt(
                8.0 * (1.0 + mwi / mwj)
            )
            denominator += fractions[j] * phi_ij
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
    inverse_square_sum = 0.0
    contributions: dict[str, dict[str, float]] = {}
    max_k = 0.0
    min_k = float("inf")
    for component_id, mass_fraction in fractions.items():
        conductivity = component_thermal_conductivities_W_m_K[component_id]
        if conductivity <= 0:
            raise ValueError(f"thermal conductivity must be positive for {component_id!r}")
        term = mass_fraction / (conductivity * conductivity)
        inverse_square_sum += term
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
        mixture_value=1.0 / sqrt(inverse_square_sum),
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
