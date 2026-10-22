"""Molar-volume, density, and ideal-volume-mixture helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from chemworld.physchem.property_cards import _VOLUME_REFERENCE_READING
from chemworld.physchem.property_equations import (
    _molar_volume_method_family,
    evaluate_correlation,
)
from chemworld.physchem.property_reports import (
    PropertyEvaluation,
    ValidityPolicy,
    _validate_phase,
)
from chemworld.physchem.specs import MixtureSpec, PropertyCorrelation


@dataclass(frozen=True)
class MolarVolumeReport:
    """Molar volume, optional density, and compressibility diagnostics."""

    evaluation: PropertyEvaluation
    phase: str
    method_family: str
    density_kg_m3: float | None = None
    compressibility_factor: float | None = None
    compressibility_status: str = "not_applicable"
    reference_reading: tuple[str, ...] = ()
    gas_root: dict[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_phase(self.phase)
        if self.evaluation.property_id not in {
            "liquid_molar_volume",
            "gas_molar_volume",
            "solid_molar_volume",
        }:
            raise ValueError("MolarVolumeReport requires a molar-volume property")
        molar_volume = self.evaluation.to("m^3/mol").value
        if molar_volume <= 0 or not isfinite(molar_volume):
            raise ValueError("molar volume must be positive and finite")
        if self.density_kg_m3 is not None and self.density_kg_m3 <= 0:
            raise ValueError("density_kg_m3 must be positive when provided")
        if self.compressibility_factor is not None and (
            self.compressibility_factor <= 0
            or not isfinite(self.compressibility_factor)
        ):
            raise ValueError("compressibility_factor must be positive and finite")
        if self.compressibility_status not in {
            "ideal",
            "low_correction",
            "moderate_correction",
            "large_correction",
            "not_applicable",
        }:
            raise ValueError("invalid compressibility_status")
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    @property
    def molar_volume_m3_mol(self) -> float:
        return self.evaluation.to("m^3/mol").value

    def to_dict(self) -> dict[str, object]:
        return {
            "property": self.evaluation.to_dict(),
            "phase": self.phase,
            "method_family": self.method_family,
            "molar_volume_m3_mol": self.molar_volume_m3_mol,
            "density_kg_m3": self.density_kg_m3,
            "compressibility_factor": self.compressibility_factor,
            "compressibility_status": self.compressibility_status,
            "reference_reading": list(self.reference_reading),
            "gas_root": self.gas_root,
        }


@dataclass(frozen=True)
class MixtureVolumeLedger:
    """Amgat-style mixture molar-volume ledger."""

    ledger_id: str
    phase: str
    contributions: dict[str, dict[str, float]]
    mixture_molar_volume_m3_mol: float
    mixture_density_kg_m3: float | None = None
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.ledger_id:
            raise ValueError("ledger_id cannot be empty")
        _validate_phase(self.phase)
        if not self.contributions:
            raise ValueError("MixtureVolumeLedger requires contributions")
        if self.mixture_molar_volume_m3_mol <= 0:
            raise ValueError("mixture_molar_volume_m3_mol must be positive")
        if self.mixture_density_kg_m3 is not None and self.mixture_density_kg_m3 <= 0:
            raise ValueError("mixture_density_kg_m3 must be positive when provided")
        recomputed = sum(
            entry["mole_fraction"] * entry["molar_volume_m3_mol"]
            for entry in self.contributions.values()
        )
        if abs(recomputed - self.mixture_molar_volume_m3_mol) > 1e-12:
            raise ValueError("mixture molar volume must match contributions")
        object.__setattr__(
            self,
            "contributions",
            {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
        )
        object.__setattr__(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict[str, object]:
        return {
            "ledger_id": self.ledger_id,
            "phase": self.phase,
            "contributions": {
                component_id: dict(contribution)
                for component_id, contribution in self.contributions.items()
            },
            "mixture_molar_volume_m3_mol": self.mixture_molar_volume_m3_mol,
            "mixture_density_kg_m3": self.mixture_density_kg_m3,
            "warnings": list(self.warnings),
        }


def molar_volume_to_density_kg_m3(
    molar_volume_m3_mol: float,
    molecular_weight_g_mol: float,
) -> float:
    """Convert molar volume to mass density."""

    if molar_volume_m3_mol <= 0:
        raise ValueError("molar_volume_m3_mol must be positive")
    if molecular_weight_g_mol <= 0:
        raise ValueError("molecular_weight_g_mol must be positive")
    return molecular_weight_g_mol / 1000.0 / molar_volume_m3_mol


def density_to_molar_volume_m3_mol(
    density_kg_m3: float,
    molecular_weight_g_mol: float,
) -> float:
    """Convert mass density to molar volume."""

    if density_kg_m3 <= 0:
        raise ValueError("density_kg_m3 must be positive")
    if molecular_weight_g_mol <= 0:
        raise ValueError("molecular_weight_g_mol must be positive")
    return molecular_weight_g_mol / 1000.0 / density_kg_m3


def molar_volume_report(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    phase: str,
    pressure_Pa: float | None = None,
    molecular_weight_g_mol: float | None = None,
    validity_policy: ValidityPolicy = "warn",
) -> MolarVolumeReport:
    """Evaluate a liquid or correlation-based molar-volume report."""

    _validate_phase(phase)
    if correlation.property_id not in {
        "liquid_molar_volume",
        "gas_molar_volume",
        "solid_molar_volume",
    }:
        raise ValueError("molar_volume_report requires a molar-volume property")
    if correlation.property_id != phase + "_molar_volume":
        raise ValueError("molar-volume correlation phase does not match requested phase")
    result = evaluate_correlation(
        correlation,
        temperature_K=temperature_K,
        pressure_Pa=pressure_Pa,
        validity_policy=validity_policy,
    )
    density = (
        molar_volume_to_density_kg_m3(result.to("m^3/mol").value, molecular_weight_g_mol)
        if molecular_weight_g_mol is not None
        else None
    )
    return MolarVolumeReport(
        evaluation=result,
        phase=phase,
        density_kg_m3=density,
        method_family=_molar_volume_method_family(correlation.equation_id),
        compressibility_status="not_applicable",
        reference_reading=_VOLUME_REFERENCE_READING,
    )


def ideal_gas_molar_volume_report(
    *,
    temperature_K: float,
    pressure_Pa: float,
    molecular_weight_g_mol: float | None = None,
) -> MolarVolumeReport:
    """Ideal-gas molar volume with optional density conversion."""

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa <= 0:
        raise ValueError("pressure_Pa must be positive")
    from chemworld.physchem.volume_kernels import volume_value

    molar_volume = volume_value("ideal_gas", temperature_K, P=pressure_Pa)
    density = (
        molar_volume_to_density_kg_m3(molar_volume, molecular_weight_g_mol)
        if molecular_weight_g_mol is not None
        else None
    )
    return MolarVolumeReport(
        evaluation=PropertyEvaluation(
            property_id="gas_molar_volume",
            correlation_id="ideal_gas_molar_volume",
            equation_id="ideal_gas_law",
            value=molar_volume,
            unit="m^3/mol",
            inputs={"temperature": temperature_K, "pressure": pressure_Pa},
        ),
        phase="gas",
        method_family="ideal_gas",
        density_kg_m3=density,
        compressibility_factor=1.0,
        compressibility_status="ideal",
        reference_reading=_VOLUME_REFERENCE_READING,
    )


def second_virial_coefficient_report(
    correlation: PropertyCorrelation,
    *,
    temperature_K: float,
    validity_policy: ValidityPolicy = "warn",
) -> PropertyEvaluation:
    """Evaluate a second virial coefficient in m^3/mol."""

    if correlation.property_id != "second_virial_coefficient":
        raise ValueError(
            "second_virial_coefficient_report requires second_virial_coefficient"
        )
    result = evaluate_correlation(
        correlation,
        temperature_K=temperature_K,
        validity_policy=validity_policy,
    )
    return result.to("m^3/mol")


def virial_gas_molar_volume_report(
    *,
    temperature_K: float,
    pressure_Pa: float,
    second_virial_m3_mol: float,
    molecular_weight_g_mol: float | None = None,
    warning_threshold: float = 0.05,
) -> MolarVolumeReport:
    """Second-virial gas molar volume from Z = 1 + B/Vm."""

    if temperature_K <= 0:
        raise ValueError("temperature_K must be positive")
    if pressure_Pa <= 0:
        raise ValueError("pressure_Pa must be positive")
    if warning_threshold <= 0:
        raise ValueError("warning_threshold must be positive")
    from chemworld.physchem.virial_gas import VirialRootError, density_virial_state

    try:
        state = density_virial_state(temperature_K, pressure_Pa, (second_virial_m3_mol,))
    except VirialRootError as exc:
        raise VirialRootError(
            "second virial coefficient gives no positive gas-volume root "
            "with nonsingular mechanical stability", exc.candidates,
        ) from exc
    molar_volume = state.molar_volume_m3_mol
    z_factor = state.compressibility_factor
    correction = abs(second_virial_m3_mol / molar_volume)
    if correction <= 1e-12:
        status = "ideal"
    elif correction <= warning_threshold:
        status = "low_correction"
    elif correction <= 4.0 * warning_threshold:
        status = "moderate_correction"
    else:
        status = "large_correction"
    density = (
        molar_volume_to_density_kg_m3(molar_volume, molecular_weight_g_mol)
        if molecular_weight_g_mol is not None
        else None
    )
    return MolarVolumeReport(
        evaluation=PropertyEvaluation(
            property_id="gas_molar_volume",
            correlation_id="second_virial_gas_molar_volume",
            equation_id="second_virial_volume_root",
            value=molar_volume,
            unit="m^3/mol",
            inputs={
                "temperature": temperature_K,
                "pressure": pressure_Pa,
                "second_virial_m3_mol": second_virial_m3_mol,
            },
        ),
        phase="gas",
        method_family="second_virial",
        density_kg_m3=density,
        compressibility_factor=z_factor,
        compressibility_status=status,
        reference_reading=_VOLUME_REFERENCE_READING,
        gas_root=state.to_dict(),
    )


def mixture_molar_volume_ledger(
    *,
    component_mole_fractions: Mapping[str, float],
    component_molar_volumes_m3_mol: Mapping[str, float],
    component_molecular_weights_g_mol: Mapping[str, float] | None = None,
    phase: str = "liquid",
    ledger_id: str = "mixture_molar_volume_ledger",
) -> MixtureVolumeLedger:
    """Amgat-style mole-fraction mixture molar volume ledger."""

    from chemworld.physchem.volume_mixtures import mixture_volume_value

    _validate_phase(phase)
    if not component_mole_fractions:
        raise ValueError("component_mole_fractions cannot be empty")
    total_fraction = sum(component_mole_fractions.values())
    if abs(total_fraction - 1.0) > 1e-9:
        raise ValueError("component mole fractions must sum to 1")
    contributions: dict[str, dict[str, float]] = {}
    average_mw = 0.0
    for component_id, mole_fraction in component_mole_fractions.items():
        if mole_fraction < 0:
            raise ValueError(f"negative mole fraction for {component_id!r}")
        if component_id not in component_molar_volumes_m3_mol:
            raise ValueError(f"missing molar volume for {component_id!r}")
        molar_volume = component_molar_volumes_m3_mol[component_id]
        if molar_volume <= 0:
            raise ValueError(f"molar volume must be positive for {component_id!r}")
        contribution = mole_fraction * molar_volume
        entry = {
            "mole_fraction": mole_fraction,
            "molar_volume_m3_mol": molar_volume,
            "volume_contribution_m3_mol": contribution,
        }
        if component_molecular_weights_g_mol is not None:
            if component_id not in component_molecular_weights_g_mol:
                raise ValueError(f"missing molecular weight for {component_id!r}")
            mw = component_molecular_weights_g_mol[component_id]
            if mw <= 0:
                raise ValueError(
                    f"molecular weight must be positive for {component_id!r}"
                )
            average_mw += mole_fraction * mw
            entry["molecular_weight_g_mol"] = mw
        contributions[component_id] = entry
    mixture_molar_volume = mixture_volume_value(
        "Amgat", 298.15, tuple(component_mole_fractions.values()),
        Vms=tuple(component_molar_volumes_m3_mol[key] for key in component_mole_fractions),
    )  # Pointwise rule is independent of this placeholder temperature.
    density = (
        molar_volume_to_density_kg_m3(mixture_molar_volume, average_mw)
        if component_molecular_weights_g_mol is not None
        else None
    )
    return MixtureVolumeLedger(
        ledger_id=ledger_id,
        phase=phase,
        contributions=contributions,
        mixture_molar_volume_m3_mol=mixture_molar_volume,
        mixture_density_kg_m3=density,
        warnings=(
            "Amgat ideal-volume mixing; no excess volume contribution included.",
        ),
    )


def mixture_density(
    mixture: MixtureSpec,
    component_densities_kg_m3: tuple[float, ...],
) -> PropertyEvaluation:
    """Ideal liquid-mixture density from mass-fraction specific volumes."""

    if len(component_densities_kg_m3) != len(mixture.component_ids):
        raise ValueError("Density vector must match mixture components")
    if any(rho <= 0 for rho in component_densities_kg_m3):
        raise ValueError("Component densities must be positive")
    specific_volume = sum(
        w / rho
        for w, rho in zip(
            mixture.mass_fractions,
            component_densities_kg_m3,
            strict=True,
        )
    )
    return PropertyEvaluation(
        property_id="mixture_density",
        correlation_id="ideal_specific_volume_mixing",
        equation_id="ideal_specific_volume_mixing",
        value=1.0 / specific_volume,
        unit="kg/m^3",
        inputs={"temperature": mixture.temperature_K, "pressure": mixture.pressure_Pa},
    )

