"""Phase-equilibrium utilities for ChemWorld.

The functions here provide a compact thermodynamic layer for benchmark tasks:
activity coefficients, Raoult-law K-values, isothermal flash, bubble/dew point
estimates, and a material-conserving liquid-liquid extraction stage.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import exp, isfinite, log, sqrt
from typing import Literal

from chemworld.foundation.units import convert_value
from chemworld.physchem.activity_kernels import interaction, log_activity
from chemworld.physchem.composition_math import _normalize_composition
from chemworld.physchem.equilibrium_cards import activity_model_cards
from chemworld.physchem.property_reports import ValidityPolicy
from chemworld.physchem.saturation import (
    PureSaturationReport,
    pure_saturation_pressure_report,
)
from chemworld.physchem.specs import PropertyCorrelation

ActivityModel = Literal["ideal", "margules", "wilson", "nrtl", "uniquac",
                        "regular_solution", "flory_huggins", "hansen", "unifac"]
VLESolveMode = Literal["bubble_temperature", "dew_temperature"]
FlashPhaseStatus = Literal["all_liquid", "two_phase", "all_vapor"]
AzeotropeScanStatus = Literal[
    "relative_volatility_crossing",
    "endpoint_near_crossing",
    "no_crossing",
]

AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K = (273.15, 373.15)
DAVIES_MAX_IONIC_STRENGTH_MOL_KG = 0.5


@dataclass(frozen=True)
class AqueousActivityCoefficientResult:
    """Davies activity coefficient with an explicit applicability contract."""

    ionic_strength_mol_kg: float
    charge: float
    temperature_K: float
    activity_coefficient: float
    model_id: str = "davies_aqueous_activity_v1"

    def to_dict(self) -> dict[str, float | str]:
        return {
            "model_id": self.model_id,
            "ionic_strength_mol_kg": self.ionic_strength_mol_kg,
            "charge": self.charge,
            "temperature_K": self.temperature_K,
            "activity_coefficient": self.activity_coefficient,
        }


def davies_aqueous_activity_coefficient(
    *,
    ionic_strength_mol_kg: float,
    charge: float,
    temperature_K: float = 298.15,
    debye_huckel_A: float | None = None,
    davies_b: float = 0.3,
) -> AqueousActivityCoefficientResult:
    """Return Davies gamma, with optional authored A at the stated temperature.

    Without an override, retain the existing bounded temperature approximation.
    The caller owns the validity of any supplied A/b parameterization.
    """

    if not isfinite(ionic_strength_mol_kg) or ionic_strength_mol_kg < 0.0:
        raise ValueError("ionic_strength_mol_kg must be finite and nonnegative")
    if ionic_strength_mol_kg > DAVIES_MAX_IONIC_STRENGTH_MOL_KG:
        raise ValueError("Davies activity model is outside its ionic-strength applicability domain")
    lower_temperature, upper_temperature = AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K
    if (
        not isfinite(temperature_K)
        or temperature_K < lower_temperature
        or temperature_K > upper_temperature
    ):
        raise ValueError("Davies activity model is outside its temperature applicability domain")
    if not isfinite(charge):
        raise ValueError("charge must be finite")
    if (
        (debye_huckel_A is not None and (not isfinite(debye_huckel_A) or debye_huckel_A < 0))
        or not isfinite(davies_b)
        or davies_b < 0
    ):
        raise ValueError("Davies coefficients must be finite and nonnegative")
    if charge == 0.0 or ionic_strength_mol_kg == 0.0:
        coefficient = 1.0
    else:
        sqrt_ionic_strength = sqrt(ionic_strength_mol_kg)
        # The 298 K Davies coefficient is temperature-scaled as a transparent
        # bounded approximation; this is not a Pitzer replacement.
        debye_huckel_a = (
            0.5092 * (298.15 / temperature_K) ** 1.5 if debye_huckel_A is None else debye_huckel_A
        )
        log10_gamma = (
            -debye_huckel_a
            * charge
            * charge
            * (sqrt_ionic_strength / (1.0 + sqrt_ionic_strength) - davies_b * ionic_strength_mol_kg)
        )
        coefficient = 10.0**log10_gamma
    return AqueousActivityCoefficientResult(
        ionic_strength_mol_kg=ionic_strength_mol_kg,
        charge=charge,
        temperature_K=temperature_K,
        activity_coefficient=coefficient,
    )


def weak_acid_davies_activity_ratio(
    *,
    ionic_strength_mol_kg: float,
    temperature_K: float = 298.15,
) -> float:
    """Return ``gamma_H+ gamma_A- / gamma_HA`` for a monoprotic weak acid."""

    cation = davies_aqueous_activity_coefficient(
        ionic_strength_mol_kg=ionic_strength_mol_kg,
        charge=1.0,
        temperature_K=temperature_K,
    )
    anion = davies_aqueous_activity_coefficient(
        ionic_strength_mol_kg=ionic_strength_mol_kg,
        charge=-1.0,
        temperature_K=temperature_K,
    )
    return cation.activity_coefficient * anion.activity_coefficient


@dataclass(frozen=True)
class ActivityModelSpec:
    model_id: str
    component_ids: tuple[str, ...]
    model: ActivityModel = "ideal"
    parameters: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id cannot be empty")
        if self.model not in {"ideal", "margules", "wilson", "nrtl", "uniquac",
                              "regular_solution", "flory_huggins", "hansen", "unifac"}:
            raise ValueError(f"Unsupported activity model: {self.model}")
        if not self.component_ids:
            raise ValueError("component_ids cannot be empty")
        if len(self.component_ids) != len(set(self.component_ids)):
            raise ValueError("Duplicate component ids are not allowed")
        if any(not isfinite(value) for value in self.parameters.values()):
            raise ValueError("activity-model parameters must be finite")
        if not isinstance(self.provenance, dict):
            raise ValueError("activity provenance must be a JSON object")
        object.__setattr__(self, "provenance",
                           json.loads(json.dumps(self.provenance, allow_nan=False)))
        _validate_activity_parameter_contract(self)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "component_ids": list(self.component_ids),
            "model": self.model,
            "parameters": dict(self.parameters),
            **({"provenance": json.loads(json.dumps(self.provenance))} if self.provenance else {}),
        }


@dataclass(frozen=True)
class FlashResult:
    vapor_fraction: float
    liquid_composition: dict[str, float]
    vapor_composition: dict[str, float]
    k_values: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "vapor_fraction": self.vapor_fraction,
            "liquid_composition": dict(self.liquid_composition),
            "vapor_composition": dict(self.vapor_composition),
            "k_values": dict(self.k_values),
        }


@dataclass(frozen=True)
class RachfordRiceDiagnosticReport:
    overall_composition: dict[str, float]
    k_values: dict[str, float]
    vapor_fraction: float
    liquid_composition: dict[str, float]
    vapor_composition: dict[str, float]
    objective_at_zero: float
    objective_at_one: float | None
    residual: float
    iterations: int
    phase_status: FlashPhaseStatus
    warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = (
        "reference_repos/chemicals/chemicals/rachford_rice.py: Rachford-Rice objective conventions",
        "reference_repos/thermo/thermo/flash/flash_base.py: ideal flash workflow notes",
    )

    liquid_fraction: float | None = None

    def __post_init__(self) -> None:
        if self.liquid_fraction is None:
            object.__setattr__(self, "liquid_fraction", 1-self.vapor_fraction)
        if not 0 <= self.liquid_fraction <= 1:
            raise ValueError("liquid_fraction must be in [0, 1]")
        if self.phase_status not in {"all_liquid", "two_phase", "all_vapor"}:
            raise ValueError("Unsupported Rachford-Rice phase status")
        if not 0.0 <= self.vapor_fraction <= 1.0:
            raise ValueError("vapor_fraction must be in [0, 1]")
        if self.objective_at_one is None and not any(
            value > 0 and self.k_values[key] == 0
            for key,value in self.overall_composition.items()
        ):
            raise ValueError("undefined beta=1 objective requires nonvolatile inventory")
        for field_name in ("objective_at_zero", "objective_at_one", "residual"):
            value = getattr(self,field_name)
            if value is None and field_name == "objective_at_one":
                continue
            if value is None or not isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_composition": dict(self.overall_composition),
            "k_values": dict(self.k_values),
            "vapor_fraction": self.vapor_fraction,
            "liquid_fraction": self.liquid_fraction,
            "liquid_composition": dict(self.liquid_composition),
            "vapor_composition": dict(self.vapor_composition),
            "objective_at_zero": self.objective_at_zero,
            "objective_at_one": self.objective_at_one,
            "residual": self.residual,
            "iterations": self.iterations,
            "phase_status": self.phase_status,
            "warnings": list(self.warnings),
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class VLETemperatureReport:
    solve_mode: VLESolveMode
    pressure_Pa: float
    temperature_K: float
    feed_composition: dict[str, float]
    liquid_composition: dict[str, float]
    vapor_composition: dict[str, float]
    vapor_pressures_Pa: dict[str, float]
    k_values: dict[str, float]
    residual: float
    residual_type: str
    iterations: int
    converged: bool
    bracket_temperature_K: tuple[float, float]
    saturation_reports: dict[str, PureSaturationReport]
    warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = (
        "reference_repos/thermo/thermo/flash/flash_base.py: ideal bubble/dew temperature workflow",
        "reference_repos/chemicals/chemicals/rachford_rice.py: phase split diagnostic conventions",
    )

    def __post_init__(self) -> None:
        if self.solve_mode not in {"bubble_temperature", "dew_temperature"}:
            raise ValueError("Unsupported VLE temperature solve mode")
        if self.pressure_Pa <= 0 or not isfinite(self.pressure_Pa):
            raise ValueError("pressure_Pa must be positive and finite")
        if self.temperature_K <= 0 or not isfinite(self.temperature_K):
            raise ValueError("temperature_K must be positive and finite")
        if not isfinite(self.residual):
            raise ValueError("residual must be finite")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")
        lower, upper = self.bracket_temperature_K
        if lower <= 0 or upper <= lower:
            raise ValueError("bracket_temperature_K must be positive and increasing")
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "solve_mode": self.solve_mode,
            "pressure_Pa": self.pressure_Pa,
            "temperature_K": self.temperature_K,
            "feed_composition": dict(self.feed_composition),
            "liquid_composition": dict(self.liquid_composition),
            "vapor_composition": dict(self.vapor_composition),
            "vapor_pressures_Pa": dict(self.vapor_pressures_Pa),
            "k_values": dict(self.k_values),
            "residual": self.residual,
            "residual_type": self.residual_type,
            "iterations": self.iterations,
            "converged": self.converged,
            "bracket_temperature_K": list(self.bracket_temperature_K),
            "saturation_reports": {
                component_id: report.to_dict()
                for component_id, report in self.saturation_reports.items()
            },
            "warnings": list(self.warnings),
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class GammaPhiKValueReport:
    activity_model_id: str
    pressure_Pa: float
    temperature_K: float
    liquid_composition: dict[str, float]
    vapor_pressures_Pa: dict[str, float]
    activity_coefficients: dict[str, float]
    vapor_fugacity_coefficients: dict[str, float]
    liquid_reference_fugacity_coefficients: dict[str, float]
    poynting_factors: dict[str, float]
    k_values: dict[str, float]
    relative_volatilities: dict[str, float]
    reference_component_id: str
    reference_reading: tuple[str, ...] = (
        "reference_repos/chemicals/chemicals/flash_basic.py: K_value gamma-phi equation hierarchy",
        "reference_repos/thermo/thermo/phases/phase.py: fugacity coefficient reporting conventions",
    )

    def __post_init__(self) -> None:
        if self.pressure_Pa <= 0 or not isfinite(self.pressure_Pa):
            raise ValueError("pressure_Pa must be positive and finite")
        if self.temperature_K <= 0 or not isfinite(self.temperature_K):
            raise ValueError("temperature_K must be positive and finite")
        if self.reference_component_id not in self.k_values:
            raise ValueError("reference_component_id must be present in k_values")
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "activity_model_id": self.activity_model_id,
            "pressure_Pa": self.pressure_Pa,
            "temperature_K": self.temperature_K,
            "liquid_composition": dict(self.liquid_composition),
            "vapor_pressures_Pa": dict(self.vapor_pressures_Pa),
            "activity_coefficients": dict(self.activity_coefficients),
            "vapor_fugacity_coefficients": dict(self.vapor_fugacity_coefficients),
            "liquid_reference_fugacity_coefficients": dict(
                self.liquid_reference_fugacity_coefficients
            ),
            "poynting_factors": dict(self.poynting_factors),
            "k_values": dict(self.k_values),
            "relative_volatilities": dict(self.relative_volatilities),
            "reference_component_id": self.reference_component_id,
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class AzeotropeScanPoint:
    composition: dict[str, float]
    k_values: dict[str, float]
    vapor_composition: dict[str, float]
    relative_volatility: float
    residual: float

    def __post_init__(self) -> None:
        if self.relative_volatility <= 0 or not isfinite(self.relative_volatility):
            raise ValueError("relative_volatility must be positive and finite")
        if not isfinite(self.residual):
            raise ValueError("azeotrope residual must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "composition": dict(self.composition),
            "k_values": dict(self.k_values),
            "vapor_composition": dict(self.vapor_composition),
            "relative_volatility": self.relative_volatility,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class BinaryAzeotropeDiagnosticReport:
    component_ids: tuple[str, str]
    light_component_id: str
    heavy_component_id: str
    pressure_Pa: float
    temperature_K: float
    residual_type: str
    status: AzeotropeScanStatus
    scan_points: tuple[AzeotropeScanPoint, ...]
    crossing_bracket: tuple[float, float] | None
    estimated_azeotrope_composition: dict[str, float] | None
    estimated_residual: float | None
    warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = (
        "reference_repos/chemicals/chemicals/flash_basic.py: K-value and "
        "relative-volatility equation context",
        "reference_repos/thermo/thermo/property_package.py: bubble/dew and "
        "phase-envelope workflow context",
    )

    def __post_init__(self) -> None:
        if self.status not in {
            "relative_volatility_crossing",
            "endpoint_near_crossing",
            "no_crossing",
        }:
            raise ValueError("Unsupported azeotrope scan status")
        if len(self.component_ids) != 2:
            raise ValueError("BinaryAzeotropeDiagnosticReport requires two components")
        if self.light_component_id not in self.component_ids:
            raise ValueError("light_component_id must be one of component_ids")
        if self.heavy_component_id not in self.component_ids:
            raise ValueError("heavy_component_id must be one of component_ids")
        if self.light_component_id == self.heavy_component_id:
            raise ValueError("light and heavy components must be distinct")
        if self.pressure_Pa <= 0 or not isfinite(self.pressure_Pa):
            raise ValueError("pressure_Pa must be positive and finite")
        if self.temperature_K <= 0 or not isfinite(self.temperature_K):
            raise ValueError("temperature_K must be positive and finite")
        if not self.scan_points:
            raise ValueError("scan_points cannot be empty")
        if self.crossing_bracket is not None:
            lower, upper = self.crossing_bracket
            if not 0.0 <= lower <= upper <= 1.0:
                raise ValueError("crossing_bracket must be within [0, 1]")
        if self.estimated_residual is not None and not isfinite(self.estimated_residual):
            raise ValueError("estimated_residual must be finite")
        object.__setattr__(self, "scan_points", tuple(self.scan_points))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "component_ids": list(self.component_ids),
            "light_component_id": self.light_component_id,
            "heavy_component_id": self.heavy_component_id,
            "pressure_Pa": self.pressure_Pa,
            "temperature_K": self.temperature_K,
            "residual_type": self.residual_type,
            "status": self.status,
            "scan_points": [point.to_dict() for point in self.scan_points],
            "crossing_bracket": list(self.crossing_bracket)
            if self.crossing_bracket is not None
            else None,
            "estimated_azeotrope_composition": None
            if self.estimated_azeotrope_composition is None
            else dict(self.estimated_azeotrope_composition),
            "estimated_residual": self.estimated_residual,
            "warnings": list(self.warnings),
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class UNIQUACActivityReport:
    model_id: str
    temperature_K: float
    composition: dict[str, float]
    r_parameters: dict[str, float]
    q_parameters: dict[str, float]
    tau_matrix: dict[str, dict[str, float]]
    volume_fractions: dict[str, float]
    surface_fractions: dict[str, float]
    combinatorial_log_terms: dict[str, float]
    residual_log_terms: dict[str, float]
    activity_coefficients: dict[str, float]
    coordination_number: float = 10.0
    reference_reading: tuple[str, ...] = (
        "reference_repos/thermo/thermo/uniquac.py: UNIQUAC_gammas "
        "equation contract and binary examples",
        "reference_repos/phasepy/phasepy/actmodels/uniquac.py: "
        "UNIQUAC auxiliary activity-model workflow",
    )

    def __post_init__(self) -> None:
        if self.temperature_K <= 0 or not isfinite(self.temperature_K):
            raise ValueError("temperature_K must be positive and finite")
        if self.coordination_number <= 0 or not isfinite(self.coordination_number):
            raise ValueError("coordination_number must be positive and finite")
        component_ids = set(self.composition)
        if not component_ids:
            raise ValueError("UNIQUAC report composition cannot be empty")
        for field_name in (
            "r_parameters",
            "q_parameters",
            "volume_fractions",
            "surface_fractions",
            "combinatorial_log_terms",
            "residual_log_terms",
            "activity_coefficients",
        ):
            if set(getattr(self, field_name)) != component_ids:
                raise ValueError(f"{field_name} ids must match composition ids")
        if set(self.tau_matrix) != component_ids:
            raise ValueError("tau_matrix row ids must match composition ids")
        for row_id, row in self.tau_matrix.items():
            if set(row) != component_ids:
                raise ValueError(f"tau_matrix column ids must match row {row_id!r}")
            if any(value <= 0.0 or not isfinite(value) for value in row.values()):
                raise ValueError("UNIQUAC tau values must be positive and finite")
        if any(value <= 0.0 or not isfinite(value) for value in self.r_parameters.values()):
            raise ValueError("UNIQUAC r parameters must be positive and finite")
        if any(value <= 0.0 or not isfinite(value) for value in self.q_parameters.values()):
            raise ValueError("UNIQUAC q parameters must be positive and finite")
        if any(
            value <= 0.0 or not isfinite(value) for value in self.activity_coefficients.values()
        ):
            raise ValueError("UNIQUAC activity coefficients must be positive and finite")
        object.__setattr__(self, "reference_reading", tuple(self.reference_reading))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "temperature_K": self.temperature_K,
            "composition": dict(self.composition),
            "r_parameters": dict(self.r_parameters),
            "q_parameters": dict(self.q_parameters),
            "tau_matrix": {row_id: dict(row) for row_id, row in self.tau_matrix.items()},
            "volume_fractions": dict(self.volume_fractions),
            "surface_fractions": dict(self.surface_fractions),
            "combinatorial_log_terms": dict(self.combinatorial_log_terms),
            "residual_log_terms": dict(self.residual_log_terms),
            "activity_coefficients": dict(self.activity_coefficients),
            "coordination_number": self.coordination_number,
            "reference_reading": list(self.reference_reading),
        }


@dataclass(frozen=True)
class LLEStageResult:
    organic_amounts_mol: dict[str, float]
    aqueous_amounts_mol: dict[str, float]
    recovery_to_organic: dict[str, float]
    phase_volumes_L: dict[str, float]
    material_balance_error_mol: float
    stability_diagnostic: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "organic_amounts_mol": dict(self.organic_amounts_mol),
            "aqueous_amounts_mol": dict(self.aqueous_amounts_mol),
            "recovery_to_organic": dict(self.recovery_to_organic),
            "phase_volumes_L": dict(self.phase_volumes_L),
            "material_balance_error_mol": self.material_balance_error_mol,
        }
        if self.stability_diagnostic is not None:
            result["stability_diagnostic"] = dict(self.stability_diagnostic)
        return result


@dataclass(frozen=True)
class LLEPhaseStabilityDiagnostic:
    """TPD-style liquid-liquid phase stability diagnostic.

    This report is intentionally a compact benchmark diagnostic, not a full
    thermodynamic phase-stability minimizer. It evaluates tangent-plane-like
    composition distances from the feed to partition-seeded candidate phases,
    then subtracts an explicit partition/nonideality driving-force term. The
    resulting score is auditable and deterministic: negative values indicate
    that a two-liquid split is favored by the supplied model slice.
    """

    model_id: str
    component_ids: tuple[str, ...]
    temperature_K: float
    initialization_policy: str
    phase_status: str
    feed_composition: dict[str, float]
    organic_trial_composition: dict[str, float]
    aqueous_trial_composition: dict[str, float]
    feed_activity_coefficients: dict[str, float]
    organic_activity_coefficients: dict[str, float]
    aqueous_activity_coefficients: dict[str, float]
    partition_coefficients: dict[str, float]
    organic_tpd_like: float
    aqueous_tpd_like: float
    partition_log_spread: float
    nonideality_drive: float
    minimum_tpd_like: float
    warnings: tuple[str, ...] = ()
    reference_reading: tuple[str, ...] = (
        "Michelsen tangent-plane distance stability analysis: "
        "TPD(w) = sum_i w_i [ln(w_i gamma_i(w)) - ln(z_i gamma_i(z))]",
        "phasepy.equilibrium.flash: phase-split initialization with "
        "successive substitution and Gibbs fallback",
        "thermo.flash: stability/testing stage before multiphase equilibrium solves",
    )

    def __post_init__(self) -> None:
        if self.phase_status not in {"single_liquid", "two_liquid"}:
            raise ValueError("phase_status must be single_liquid or two_liquid")
        if self.temperature_K <= 0.0 or not isfinite(self.temperature_K):
            raise ValueError("temperature_K must be positive and finite")
        if not self.initialization_policy:
            raise ValueError("initialization_policy cannot be empty")
        component_ids = set(self.component_ids)
        if not component_ids:
            raise ValueError("component_ids cannot be empty")
        for field_name in (
            "feed_composition",
            "organic_trial_composition",
            "aqueous_trial_composition",
            "feed_activity_coefficients",
            "organic_activity_coefficients",
            "aqueous_activity_coefficients",
            "partition_coefficients",
        ):
            if set(getattr(self, field_name)) != component_ids:
                raise ValueError(f"{field_name} ids must match component ids")
        for field_name in (
            "organic_tpd_like",
            "aqueous_tpd_like",
            "partition_log_spread",
            "nonideality_drive",
            "minimum_tpd_like",
        ):
            if not isfinite(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be finite")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "component_ids": list(self.component_ids),
            "temperature_K": self.temperature_K,
            "initialization_policy": self.initialization_policy,
            "phase_status": self.phase_status,
            "feed_composition": dict(self.feed_composition),
            "organic_trial_composition": dict(self.organic_trial_composition),
            "aqueous_trial_composition": dict(self.aqueous_trial_composition),
            "feed_activity_coefficients": dict(self.feed_activity_coefficients),
            "organic_activity_coefficients": dict(self.organic_activity_coefficients),
            "aqueous_activity_coefficients": dict(self.aqueous_activity_coefficients),
            "partition_coefficients": dict(self.partition_coefficients),
            "organic_tpd_like": self.organic_tpd_like,
            "aqueous_tpd_like": self.aqueous_tpd_like,
            "partition_log_spread": self.partition_log_spread,
            "nonideality_drive": self.nonideality_drive,
            "minimum_tpd_like": self.minimum_tpd_like,
            "diagnostic_kind": "empirical_partition_contact",
            "thermodynamic_stability_certified": False,
            "warnings": list(self.warnings),
            "reference_reading": list(self.reference_reading),
        }


def activity_coefficients(
    spec: ActivityModelSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
) -> dict[str, float]:
    if temperature_K <= 0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be positive and finite")
    x = _composition_vector(spec.component_ids, composition)
    try:
        values = tuple(exp(value) for value in log_activity(spec, x, temperature_K))
    except OverflowError as exc:
        raise ValueError("activity coefficient is outside numerical range") from exc
    if any(value <= 0 or not isfinite(value) for value in values):
        raise ValueError("activity coefficients must be positive and finite")
    return dict(zip(spec.component_ids, values, strict=True))


def uniquac_activity_report(
    spec: ActivityModelSpec,
    composition: Mapping[str, float],
    *,
    temperature_K: float,
) -> UNIQUACActivityReport:
    if spec.model != "uniquac":
        raise ValueError("uniquac_activity_report requires model='uniquac'")
    if temperature_K <= 0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be positive and finite")
    x_mapping = _composition_vector_mapping(spec.component_ids, composition)
    x = tuple(x_mapping[component_id] for component_id in spec.component_ids)
    r_values = tuple(
        _uniquac_structural_parameter(spec, "r", component_id)
        for component_id in spec.component_ids
    )
    q_values = tuple(
        _uniquac_structural_parameter(spec, "q", component_id)
        for component_id in spec.component_ids
    )
    tau = _uniquac_tau_matrix(spec, temperature_K)
    z_coordination = _uniquac_coordination_number(spec)

    r_sum = sum(x_i * r_i for x_i, r_i in zip(x, r_values, strict=True))
    q_sum = sum(x_i * q_i for x_i, q_i in zip(x, q_values, strict=True))
    if r_sum <= 0.0 or q_sum <= 0.0:
        raise ValueError("UNIQUAC r/q composition sums must be positive")
    phi = tuple(x_i * r_i / r_sum for x_i, r_i in zip(x, r_values, strict=True))
    theta = tuple(x_i * q_i / q_sum for x_i, q_i in zip(x, q_values, strict=True))
    l_values = tuple(
        z_coordination * 0.5 * (r_i - q_i) - (r_i - 1.0)
        for r_i, q_i in zip(r_values, q_values, strict=True)
    )
    x_l_sum = sum(x_i * l_i for x_i, l_i in zip(x, l_values, strict=True))

    combinatorial: dict[str, float] = {}
    residual: dict[str, float] = {}
    for i, component_id in enumerate(spec.component_ids):
        phi_over_x = r_values[i] / r_sum
        theta_over_phi = q_values[i] * r_sum / (r_values[i] * q_sum)
        combinatorial_log = (
            log(phi_over_x)
            + z_coordination * 0.5 * q_values[i] * log(theta_over_phi)
            + l_values[i]
            - phi_over_x * x_l_sum
        )

        theta_tau_ji = sum(theta[j] * tau[j][i] for j in range(len(spec.component_ids)))
        if theta_tau_ji <= 0.0 or not isfinite(theta_tau_ji):
            raise ValueError("UNIQUAC residual denominator must be positive")
        residual_sum = 0.0
        for j in range(len(spec.component_ids)):
            denominator = sum(theta[k] * tau[k][j] for k in range(len(spec.component_ids)))
            if denominator <= 0.0 or not isfinite(denominator):
                raise ValueError("UNIQUAC residual denominator must be positive")
            residual_sum += theta[j] * tau[i][j] / denominator
        residual_log = q_values[i] * (1.0 - log(theta_tau_ji) - residual_sum)
        log_gamma = combinatorial_log + residual_log
        if not isfinite(log_gamma):
            raise ValueError("UNIQUAC log-gamma must be finite")
        combinatorial[component_id] = combinatorial_log
        residual[component_id] = residual_log

    return UNIQUACActivityReport(
        model_id=spec.model_id,
        temperature_K=temperature_K,
        composition=x_mapping,
        r_parameters=dict(zip(spec.component_ids, r_values, strict=True)),
        q_parameters=dict(zip(spec.component_ids, q_values, strict=True)),
        tau_matrix={
            left: {right: tau[i][j] for j, right in enumerate(spec.component_ids)}
            for i, left in enumerate(spec.component_ids)
        },
        volume_fractions=dict(zip(spec.component_ids, phi, strict=True)),
        surface_fractions=dict(zip(spec.component_ids, theta, strict=True)),
        combinatorial_log_terms=combinatorial,
        residual_log_terms=residual,
        activity_coefficients=activity_coefficients(spec, composition, temperature_K=temperature_K),
        coordination_number=z_coordination,
    )


def raoult_k_values(
    activity_model: ActivityModelSpec,
    liquid_composition: Mapping[str, float],
    *,
    vapor_pressures_Pa: Mapping[str, float],
    pressure_Pa: float,
    temperature_K: float,
    vapor_fugacity_coefficients: Mapping[str, float] | None = None,
) -> dict[str, float]:
    return gamma_phi_k_value_report(
        activity_model,
        liquid_composition,
        vapor_pressures_Pa=vapor_pressures_Pa,
        pressure_Pa=pressure_Pa,
        temperature_K=temperature_K,
        vapor_fugacity_coefficients=vapor_fugacity_coefficients,
    ).k_values


def gamma_phi_k_value_report(
    activity_model: ActivityModelSpec,
    liquid_composition: Mapping[str, float],
    *,
    vapor_pressures_Pa: Mapping[str, float],
    pressure_Pa: float,
    temperature_K: float,
    vapor_fugacity_coefficients: Mapping[str, float] | None = None,
    liquid_reference_fugacity_coefficients: Mapping[str, float] | None = None,
    poynting_factors: Mapping[str, float] | None = None,
    nonvolatile_component_ids: tuple[str, ...] = (),
) -> GammaPhiKValueReport:
    """Return an auditable gamma-phi VLE K-value report.

    The compact contract follows the standard benchmark form
    `K_i = gamma_i * Psat_i * phi_l_ref_i * Poynting_i / (phi_v_i * P)`.
    Supplying no fugacity or Poynting mappings gives the modified Raoult-law
    limit.
    """

    if pressure_Pa <= 0 or not isfinite(pressure_Pa):
        raise ValueError("pressure_Pa must be positive and finite")
    if temperature_K <= 0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be positive and finite")
    x = _composition_vector_mapping(activity_model.component_ids, liquid_composition)
    _validate_vapor_pressure_values(x, vapor_pressures_Pa,
                                    nonvolatile_component_ids=nonvolatile_component_ids)
    gamma = activity_coefficients(
        activity_model,
        x,
        temperature_K=temperature_K,
    )
    phi_v = _factor_mapping(
        activity_model.component_ids,
        vapor_fugacity_coefficients,
        field_name="vapor_fugacity_coefficients",
    )
    phi_l_ref = _factor_mapping(
        activity_model.component_ids,
        liquid_reference_fugacity_coefficients,
        field_name="liquid_reference_fugacity_coefficients",
    )
    poynting = _factor_mapping(
        activity_model.component_ids,
        poynting_factors,
        field_name="poynting_factors",
    )
    k_values = {
        component_id: gamma[component_id]
        * float(vapor_pressures_Pa[component_id])
        * phi_l_ref[component_id]
        * poynting[component_id]
        / (phi_v[component_id] * pressure_Pa)
        for component_id in activity_model.component_ids
    }
    _validate_k_values(x, k_values,nonvolatile_component_ids=nonvolatile_component_ids)
    volatile = {key:value for key,value in k_values.items() if value > 0}
    if not volatile:
        raise ValueError("a VLE model requires at least one potentially volatile component")
    reference_component_id = min(volatile,key=volatile.get)
    reference_k = k_values[reference_component_id]
    relative_volatilities = {
        component_id: k_values[component_id] / reference_k
        for component_id in activity_model.component_ids
    }
    return GammaPhiKValueReport(
        activity_model_id=activity_model.model_id,
        pressure_Pa=pressure_Pa,
        temperature_K=temperature_K,
        liquid_composition=x,
        vapor_pressures_Pa={
            component_id: float(vapor_pressures_Pa[component_id])
            for component_id in activity_model.component_ids
        },
        activity_coefficients=gamma,
        vapor_fugacity_coefficients={k:v for k,v in phi_v.items()
                                     if k not in nonvolatile_component_ids},
        liquid_reference_fugacity_coefficients=phi_l_ref,
        poynting_factors=poynting,
        k_values=k_values,
        relative_volatilities=relative_volatilities,
        reference_component_id=reference_component_id,
    )


def binary_azeotrope_diagnostic_report(
    activity_model: ActivityModelSpec,
    *,
    vapor_pressures_Pa: Mapping[str, float],
    pressure_Pa: float,
    temperature_K: float,
    light_component_id: str,
    vapor_fugacity_coefficients: Mapping[str, float] | None = None,
    liquid_reference_fugacity_coefficients: Mapping[str, float] | None = None,
    poynting_factors: Mapping[str, float] | None = None,
    grid_size: int = 41,
    composition_bounds: tuple[float, float] = (1e-3, 1.0 - 1e-3),
    residual_tolerance: float = 1e-6,
) -> BinaryAzeotropeDiagnosticReport:
    """Scan binary gamma-phi relative volatility for azeotrope-like crossings.

    This diagnostic is intentionally isothermal and local. A sign change in
    `ln(K_light/K_heavy)` indicates a relative-volatility crossing that should
    be treated as azeotrope risk by flash/distillation tasks; it is not a full
    phase-stability or pressure-composition azeotrope solver.
    """

    if len(activity_model.component_ids) != 2:
        raise ValueError("binary azeotrope diagnostic requires a binary model")
    if light_component_id not in activity_model.component_ids:
        raise ValueError("light_component_id must be present in activity_model")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3")
    if residual_tolerance <= 0:
        raise ValueError("residual_tolerance must be positive")
    lower, upper = composition_bounds
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("composition_bounds must lie inside (0, 1)")
    heavy_component_id = next(
        component_id
        for component_id in activity_model.component_ids
        if component_id != light_component_id
    )
    _validate_vapor_pressure_values(
        dict.fromkeys(activity_model.component_ids, 0.5),
        vapor_pressures_Pa,
    )
    phi_v = _factor_mapping(
        activity_model.component_ids,
        vapor_fugacity_coefficients,
        field_name="vapor_fugacity_coefficients",
    )
    phi_l_ref = _factor_mapping(
        activity_model.component_ids,
        liquid_reference_fugacity_coefficients,
        field_name="liquid_reference_fugacity_coefficients",
    )
    poynting = _factor_mapping(
        activity_model.component_ids,
        poynting_factors,
        field_name="poynting_factors",
    )

    scan_points: list[AzeotropeScanPoint] = []
    for index in range(grid_size):
        fraction = lower + (upper - lower) * index / (grid_size - 1)
        composition = {
            light_component_id: fraction,
            heavy_component_id: 1.0 - fraction,
        }
        report = gamma_phi_k_value_report(
            activity_model,
            composition,
            vapor_pressures_Pa=vapor_pressures_Pa,
            pressure_Pa=pressure_Pa,
            temperature_K=temperature_K,
            vapor_fugacity_coefficients=phi_v,
            liquid_reference_fugacity_coefficients=phi_l_ref,
            poynting_factors=poynting,
        )
        relative_volatility = (
            report.k_values[light_component_id] / report.k_values[heavy_component_id]
        )
        residual = log(relative_volatility)
        vapor = _normalize_composition(
            {
                component_id: composition[component_id] * report.k_values[component_id]
                for component_id in activity_model.component_ids
            }
        )
        scan_points.append(
            AzeotropeScanPoint(
                composition=composition,
                k_values=report.k_values,
                vapor_composition=vapor,
                relative_volatility=relative_volatility,
                residual=residual,
            )
        )

    crossing = _find_azeotrope_crossing(
        scan_points,
        light_component_id=light_component_id,
        residual_tolerance=residual_tolerance,
    )
    warnings: list[str] = []
    if crossing is None:
        status: AzeotropeScanStatus = "no_crossing"
        warnings.append("No binary relative-volatility crossing found on the grid")
        crossing_bracket = None
        estimated_composition = None
        estimated_residual = None
    else:
        status, crossing_bracket, estimated_composition, estimated_residual = crossing
        if status == "endpoint_near_crossing":
            warnings.append("A grid endpoint is already within residual_tolerance")

    return BinaryAzeotropeDiagnosticReport(
        component_ids=activity_model.component_ids,
        light_component_id=light_component_id,
        heavy_component_id=heavy_component_id,
        pressure_Pa=pressure_Pa,
        temperature_K=temperature_K,
        residual_type="ln_relative_volatility",
        status=status,
        scan_points=tuple(scan_points),
        crossing_bracket=crossing_bracket,
        estimated_azeotrope_composition=estimated_composition,
        estimated_residual=estimated_residual,
        warnings=tuple(warnings),
    )


def rachford_rice_vapor_fraction(
    overall_composition: Mapping[str, float],
    k_values: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> float:
    z = _normalize_composition(overall_composition)
    _validate_k_values(z, k_values)

    def objective(beta: float) -> float:
        return sum(
            z[component_id]
            * (k_values[component_id] - 1.0)
            / (1.0 + beta * (k_values[component_id] - 1.0))
            for component_id in z
        )

    f0 = objective(0.0)
    f1 = objective(1.0)
    if f0 <= 0.0:
        return 0.0
    if f1 >= 0.0:
        return 1.0

    low = 0.0
    high = 1.0
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        value = objective(mid)
        if abs(value) < tolerance:
            return mid
        if value > 0.0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def rachford_rice_diagnostic_report(
    overall_composition: Mapping[str, float],
    k_values: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
    nonvolatile_component_ids: tuple[str, ...] = (),
) -> RachfordRiceDiagnosticReport:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    z = _normalize_composition(overall_composition)
    _validate_k_values(z,k_values,nonvolatile_component_ids=nonvolatile_component_ids)
    k = {component_id: float(k_values[component_id]) for component_id in z}

    def objective(beta: float) -> float:
        return sum(
            z[component_id] * (k[component_id] - 1.0) / (1.0 + beta * (k[component_id] - 1.0))
            for component_id in z if z[component_id] > 0
        )

    f0 = objective(0.0)
    has_nonvolatile_inventory = any(z[key] > 0 for key in nonvolatile_component_ids)
    f1 = None if has_nonvolatile_inventory else objective(1.0)
    warnings: list[str] = []
    if has_nonvolatile_inventory:
        warnings.append("beta=1 objective is undefined for retained nonvolatile inventory")
    iterations = 0
    liquid_fraction = None
    if f0 <= 0.0:
        beta = 0.0
        phase_status: FlashPhaseStatus = "all_liquid"
        warnings.append("Rachford-Rice objective indicates a single liquid phase")
    elif f1 is not None and f1 >= 0.0:
        beta = 1.0
        phase_status = "all_vapor"
        warnings.append("Rachford-Rice objective indicates a single vapor phase")
    else:
        phase_status = "two_phase"
        active = [key for key in z if z[key] > 0]
        if has_nonvolatile_inventory:
            # Solve log(L/F), retaining a positive trace liquid fraction even
            # when float(1-L/F) rounds to one. K=0 gives an exact retained
            # component, so L/F >= z_nonvolatile is a physical lower bound.
            from scipy.optimize import brentq

            def liquid_objective(log_liquid):
                fraction = exp(log_liquid)
                return sum(z[key]*(k[key]-1)/(fraction+(1-fraction)*k[key])
                           for key in active)

            lower = sum(z[key] for key in nonvolatile_component_ids)
            root, info = brentq(liquid_objective, log(lower), 0.0,
                xtol=5e-15, rtol=1e-14, maxiter=max_iterations, full_output=True)
            liquid_fraction = exp(root)
            beta = 1-liquid_fraction
            iterations = info.iterations
        else:
            liquid_fraction = None
        if not has_nonvolatile_inventory:
            analytic = None
            if len(active) == 2:
                left, right = active
                a, b = k[left] - 1, k[right] - 1
                if a * b:
                    candidate = -(z[left] * a + z[right] * b) / (a * b)
                    if 0 < candidate < 1 and abs(objective(candidate)) < tolerance:
                        analytic = candidate
            if analytic is not None:
                beta = analytic
            else:
                low, high = 0.0, 1.0
                for iteration in range(1, max_iterations + 1):
                    beta = 0.5 * (low + high)
                    value = objective(beta)
                    iterations = iteration
                    if abs(value) < tolerance:
                        break
                    if value > 0.0:
                        low = beta
                    else:
                        high = beta
                else:
                    warnings.append(
                        "Rachford-Rice bisection reached max_iterations before tolerance")
    if liquid_fraction is None:
        liquid_fraction = 1-beta
    liquid = {
        component_id: (z[component_id] / (
                        (liquid_fraction + beta * k[component_id]) if has_nonvolatile_inventory
                        else (1.0 + beta * (k[component_id] - 1.0)))
                       if z[component_id] else 0.0) for component_id in z
    }
    liquid = _normalize_composition(liquid)
    vapor = {component_id: k[component_id] * liquid[component_id] for component_id in z}
    vapor = _normalize_composition(vapor) if sum(vapor.values()) else vapor
    return RachfordRiceDiagnosticReport(
        overall_composition=z,
        k_values=k,
        vapor_fraction=beta,
        liquid_composition=liquid,
        vapor_composition=vapor,
        objective_at_zero=f0,
        objective_at_one=f1,
        residual=(sum(z[key]*(k[key]-1)/(liquid_fraction+beta*k[key])
                      for key in z if z[key] > 0) if has_nonvolatile_inventory
                  else objective(beta)),
        liquid_fraction=liquid_fraction,
        iterations=iterations,
        phase_status=phase_status,
        warnings=tuple(warnings),
    )


def flash_isothermal(
    overall_composition: Mapping[str, float],
    k_values: Mapping[str, float],
) -> FlashResult:
    z = _normalize_composition(overall_composition)
    _validate_k_values(z, k_values)
    beta = rachford_rice_vapor_fraction(z, k_values)
    liquid = {
        component_id: z[component_id] / (1.0 + beta * (k_values[component_id] - 1.0))
        for component_id in z
    }
    liquid = _normalize_composition(liquid)
    vapor = {component_id: k_values[component_id] * liquid[component_id] for component_id in z}
    vapor = _normalize_composition(vapor)
    return FlashResult(
        vapor_fraction=beta,
        liquid_composition=liquid,
        vapor_composition=vapor,
        k_values={component_id: float(k_values[component_id]) for component_id in z},
    )


def bubble_temperature_report(
    liquid_composition: Mapping[str, float],
    *,
    vapor_pressure_correlations: Mapping[str, PropertyCorrelation],
    activity_model: ActivityModelSpec,
    pressure_Pa: float,
    temperature_bounds_K: tuple[float, float] | None = None,
    validity_policy: ValidityPolicy = "raise",
    tolerance_K: float = 1e-7,
    tolerance_log_pressure: float = 1e-10,
    max_iterations: int = 100,
) -> VLETemperatureReport:
    """Solve a mixture bubble temperature with auditable K-value diagnostics."""

    x = _composition_vector_mapping(activity_model.component_ids, liquid_composition)
    _validate_vle_temperature_inputs(
        pressure_Pa=pressure_Pa,
        tolerance_K=tolerance_K,
        tolerance_log_pressure=tolerance_log_pressure,
        max_iterations=max_iterations,
    )
    _validate_vapor_pressure_correlations(
        activity_model.component_ids,
        vapor_pressure_correlations,
    )
    lower, upper = _mixture_temperature_bounds_k(
        activity_model.component_ids,
        vapor_pressure_correlations,
        temperature_bounds_K=temperature_bounds_K,
    )

    def evaluate(
        temperature_K: float,
    ) -> tuple[float, dict[str, float], dict[str, PureSaturationReport]]:
        vapor_pressures, saturation_reports = _vapor_pressures_from_correlations(
            activity_model.component_ids,
            vapor_pressure_correlations,
            temperature_K=temperature_K,
            validity_policy=validity_policy,
        )
        bubble = bubble_pressure_pa(
            x,
            vapor_pressures_Pa=vapor_pressures,
            activity_model=activity_model,
            temperature_K=temperature_K,
        )
        return log(bubble / pressure_Pa), vapor_pressures, saturation_reports

    solve = _solve_vle_temperature(
        evaluate,
        lower=lower,
        upper=upper,
        tolerance_K=tolerance_K,
        tolerance_log_pressure=tolerance_log_pressure,
        max_iterations=max_iterations,
        residual_label="bubble pressure",
    )
    residual, vapor_pressures, saturation_reports = evaluate(solve.temperature_K)
    k_values = raoult_k_values(
        activity_model,
        x,
        vapor_pressures_Pa=vapor_pressures,
        pressure_Pa=pressure_Pa,
        temperature_K=solve.temperature_K,
    )
    vapor = _normalize_composition(
        {component_id: x[component_id] * k_values[component_id] for component_id in x}
    )
    return VLETemperatureReport(
        solve_mode="bubble_temperature",
        pressure_Pa=pressure_Pa,
        temperature_K=solve.temperature_K,
        feed_composition=x,
        liquid_composition=x,
        vapor_composition=vapor,
        vapor_pressures_Pa=vapor_pressures,
        k_values=k_values,
        residual=residual,
        residual_type="log_bubble_pressure_ratio",
        iterations=solve.iterations,
        converged=solve.converged,
        bracket_temperature_K=(lower, upper),
        saturation_reports=saturation_reports,
        warnings=solve.warnings,
    )


def dew_temperature_report(
    vapor_composition: Mapping[str, float],
    *,
    vapor_pressure_correlations: Mapping[str, PropertyCorrelation],
    activity_model: ActivityModelSpec,
    pressure_Pa: float,
    temperature_bounds_K: tuple[float, float] | None = None,
    validity_policy: ValidityPolicy = "raise",
    tolerance_K: float = 1e-7,
    tolerance_log_pressure: float = 1e-10,
    max_iterations: int = 100,
    composition_iterations: int = 50,
) -> VLETemperatureReport:
    """Solve a mixture dew temperature with liquid-composition diagnostics."""

    y = _composition_vector_mapping(activity_model.component_ids, vapor_composition)
    _validate_vle_temperature_inputs(
        pressure_Pa=pressure_Pa,
        tolerance_K=tolerance_K,
        tolerance_log_pressure=tolerance_log_pressure,
        max_iterations=max_iterations,
    )
    if composition_iterations <= 0:
        raise ValueError("composition_iterations must be positive")
    _validate_vapor_pressure_correlations(
        activity_model.component_ids,
        vapor_pressure_correlations,
    )
    lower, upper = _mixture_temperature_bounds_k(
        activity_model.component_ids,
        vapor_pressure_correlations,
        temperature_bounds_K=temperature_bounds_K,
    )

    def evaluate(
        temperature_K: float,
    ) -> tuple[float, dict[str, float], dict[str, PureSaturationReport]]:
        vapor_pressures, saturation_reports = _vapor_pressures_from_correlations(
            activity_model.component_ids,
            vapor_pressure_correlations,
            temperature_K=temperature_K,
            validity_policy=validity_policy,
        )
        dew_pressure, _liquid = _dew_pressure_and_liquid_composition(
            y,
            vapor_pressures_Pa=vapor_pressures,
            activity_model=activity_model,
            temperature_K=temperature_K,
            iterations=composition_iterations,
        )
        return log(dew_pressure / pressure_Pa), vapor_pressures, saturation_reports

    solve = _solve_vle_temperature(
        evaluate,
        lower=lower,
        upper=upper,
        tolerance_K=tolerance_K,
        tolerance_log_pressure=tolerance_log_pressure,
        max_iterations=max_iterations,
        residual_label="dew pressure",
    )
    residual, vapor_pressures, saturation_reports = evaluate(solve.temperature_K)
    _dew_pressure, liquid = _dew_pressure_and_liquid_composition(
        y,
        vapor_pressures_Pa=vapor_pressures,
        activity_model=activity_model,
        temperature_K=solve.temperature_K,
        iterations=composition_iterations,
    )
    k_values = raoult_k_values(
        activity_model,
        liquid,
        vapor_pressures_Pa=vapor_pressures,
        pressure_Pa=pressure_Pa,
        temperature_K=solve.temperature_K,
    )
    vapor = _normalize_composition(
        {component_id: k_values[component_id] * liquid[component_id] for component_id in liquid}
    )
    return VLETemperatureReport(
        solve_mode="dew_temperature",
        pressure_Pa=pressure_Pa,
        temperature_K=solve.temperature_K,
        feed_composition=y,
        liquid_composition=liquid,
        vapor_composition=vapor,
        vapor_pressures_Pa=vapor_pressures,
        k_values=k_values,
        residual=residual,
        residual_type="log_dew_pressure_ratio",
        iterations=solve.iterations,
        converged=solve.converged,
        bracket_temperature_K=(lower, upper),
        saturation_reports=saturation_reports,
        warnings=solve.warnings,
    )


def bubble_pressure_pa(
    liquid_composition: Mapping[str, float],
    *,
    vapor_pressures_Pa: Mapping[str, float],
    activity_model: ActivityModelSpec,
    temperature_K: float,
) -> float:
    x = _normalize_composition(liquid_composition)
    gamma = activity_coefficients(activity_model, x, temperature_K=temperature_K)
    return sum(
        x[component_id] * gamma[component_id] * float(vapor_pressures_Pa[component_id])
        for component_id in x
    )


def dew_pressure_pa(
    vapor_composition: Mapping[str, float],
    *,
    vapor_pressures_Pa: Mapping[str, float],
    activity_model: ActivityModelSpec,
    temperature_K: float,
    iterations: int = 50,
) -> float:
    pressure, _liquid = _dew_pressure_and_liquid_composition(
        vapor_composition,
        vapor_pressures_Pa=vapor_pressures_Pa,
        activity_model=activity_model,
        temperature_K=temperature_K,
        iterations=iterations,
    )
    return pressure


@dataclass(frozen=True)
class LiquidStabilityReport:
    """Actual normalized TPD minimization, with deterministic start outcomes."""

    temperature_K: float
    feed_composition: dict[str, float]
    minimum_tpd: float
    minimizing_composition: dict[str, float]
    starts: tuple[dict[str, object], ...]
    minima: tuple[dict[str, float], ...]

    def to_dict(self):
        from dataclasses import asdict

        return {"model_id": "liquid_tpd_minimization_v1", **asdict(self)}


@dataclass(frozen=True)
class LiquidEquilibriumResult:
    temperature_K: float
    feed_amounts_mol: dict[str, float]
    phases: tuple[dict[str, float], ...]
    stability: LiquidStabilityReport
    post_split_stability: tuple[LiquidStabilityReport, ...]
    chemical_potential_error_RT: float
    material_balance_error_mol: float
    gibbs_mixing_RT_mol: float
    solver_starts: tuple[dict[str, object], ...] = ()
    mixing_chemical_potentials_RT: tuple[dict[str, float], ...] = ()

    @property
    def phase_status(self):
        if not self.phases:
            return "empty"
        return "two_liquid" if len(self.phases) == 2 else "single_liquid"

    def to_dict(self):
        from dataclasses import asdict

        return {"model_id": "strict_two_liquid_equilibrium_v1",
                "phase_status": self.phase_status,
                "chemical_potential_basis": (
                    "relative_to_common_pure_liquid_standard; divided_by_RT"),
                **asdict(self)}


def _liquid_simplex_seeds(z):
    import numpy as np

    size = len(z)
    if size == 1:
        return [np.ones(1)]
    seeds = [np.asarray(z), np.full(size, 1 / size)]
    for index in range(size):
        point = np.full(size, 1e-8)
        point[index] = 1 - (size - 1) * 1e-8
        seeds.append(point)
    if size == 2:
        seeds.extend(np.array([x, 1 - x]) for x in (.01, .05, .1, .2, .35, .5, .65, .8, .9,
                                                       .95, .99))
    else:
        seeds.extend(np.array([i, j, 8-i-j]) / 8 for i in range(1, 7)
                     for j in range(1, 8-i))
    return seeds


def _liquid_activity_state(activity_model, active_ids, composition, temperature):
    import numpy as np

    full = dict.fromkeys(activity_model.component_ids, 0.0)
    full.update(zip(active_ids, composition, strict=True))
    gamma = activity_coefficients(activity_model, full, temperature_K=temperature)
    return np.array([log(x) + log(gamma[k]) for k, x in zip(
        active_ids, composition, strict=True)])


def liquid_tpd_minimum(
    activity_model: ActivityModelSpec, composition: Mapping[str, float], *,
    temperature_K: float, max_iterations: int = 1000,
) -> LiquidStabilityReport:
    """Minimize sum w_i[ln(w_i gamma_i(w))-ln(z_i gamma_i(z))].

    Positive-inventory support is fixed; no absent component is manufactured.
    The finite deterministic multistart design is reported, not a proof of
    exhaustive global optimization for arbitrary activity functions.
    """
    import numpy as np
    from scipy.optimize import minimize

    if (not isfinite(temperature_K) or temperature_K <= 0
            or isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations <= 0
            or len(activity_model.component_ids) not in (2, 3)
            or set(composition) - set(activity_model.component_ids)):
        raise ValueError("strict liquid stability needs binary/ternary composition and positive T")
    feed = _normalize_amounts(dict.fromkeys(activity_model.component_ids, 0.0) | dict(composition))
    total = sum(feed.values())
    if not isfinite(total):
        raise ValueError("liquid composition total must be finite")
    z_full = {k: v / total for k, v in feed.items()}
    ids = tuple(k for k in activity_model.component_ids if feed[k] > 0)
    z = np.array([z_full[k] for k in ids])
    mu = _liquid_activity_state(activity_model, ids, z, temperature_K)
    def objective(w):
        point = w / w.sum()
        delta = _liquid_activity_state(activity_model, ids, point, temperature_K) - mu
        value = float(w @ delta)
        return value, delta
    outcomes, minima, best = [], [], (0.0, z)
    for index, seed in enumerate(_liquid_simplex_seeds(z)):
        if len(ids) == 1:
            value, point, success, message, iterations = 0.0, z, True, "pure support", 0
        else:
            result = minimize(
                objective, seed, jac=True, method="SLSQP", bounds=[(1e-14, 1)] * len(ids),
                constraints={"type": "eq", "fun": lambda w: w.sum() - 1,
                             "jac": lambda w: np.ones(len(ids))},
                options={"ftol": 1e-13, "maxiter": max_iterations})
            value, point = float(result.fun), result.x / result.x.sum()
            success, message, iterations = bool(result.success), str(result.message), result.nit
        outcomes.append({"start": index, "converged": success, "iterations": iterations,
                         "minimum_tpd": value, "message": message})
        full = dict.fromkeys(activity_model.component_ids, 0.0)
        full.update(zip(ids, map(float, point), strict=True))
        minima.append(full)
        if value < best[0]:
            best = value, point
    if any(not result["converged"] for result in outcomes):
        raise RuntimeError(f"liquid TPD has unresolved starts: {outcomes}")
    minimizing = dict.fromkeys(activity_model.component_ids, 0.0)
    minimizing.update(zip(ids, map(float, best[1]), strict=True))
    return LiquidStabilityReport(temperature_K, z_full, best[0], minimizing,
                                  tuple(outcomes), tuple(minima))


def equilibrate_liquids(
    activity_model: ActivityModelSpec, feed_amounts_mol: Mapping[str, float], *,
    temperature_K: float, tpd_tolerance: float = 1e-9, max_iterations: int = 1000,
) -> LiquidEquilibriumResult:
    """Constrained Gibbs minimization then actual chemical-potential refinement.

    At most two liquids. A negative post-split TPD rejects an unsupported
    additional phase. No partition coefficient or empirical drive is used.
    """
    import numpy as np
    from scipy.optimize import least_squares, minimize

    if not isfinite(tpd_tolerance) or tpd_tolerance <= 0:
        raise ValueError("liquid TPD tolerance must be positive and finite")
    feed = _normalize_amounts(dict.fromkeys(activity_model.component_ids, 0.0)
                              | dict(feed_amounts_mol))
    stability = liquid_tpd_minimum(activity_model, feed, temperature_K=temperature_K,
                                    max_iterations=max_iterations)
    ids = tuple(k for k in activity_model.component_ids if feed[k] > 0)
    n = np.array([feed[k] for k in ids])
    total, z = float(n.sum()), n / n.sum()
    def mu(point):
        return _liquid_activity_state(activity_model, ids, point, temperature_K)
    initial_g = float(total * (z @ mu(z)))
    if stability.minimum_tpd >= -tpd_tolerance:
        return LiquidEquilibriumResult(temperature_K, feed, (feed,), stability,
                                       (stability,), 0.0, 0.0, initial_g,
                                       mixing_chemical_potentials_RT=(dict(zip(
                                           ids, map(float, mu(z)), strict=True)),))
    def objective(fractions):
        first, second = z * fractions, z * (1 - fractions)
        a, b = mu(first / first.sum()), mu(second / second.sum())
        return float(first @ a + second @ b), z * (a - b)
    starts, candidates = [], []
    for index, trial in enumerate(stability.minima):
        w = np.array([trial[k] for k in ids])
        beta = min(.5, .5 * float(np.min(z / np.maximum(w, 1e-300))))
        fractions = np.clip(beta * w / z, 1e-10, 1 - 1e-10)
        result = minimize(objective, fractions, jac=True, method="L-BFGS-B",
                          bounds=[(1e-12, 1-1e-12)] * len(ids),
                          options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": max_iterations})
        starts.append({"start": index, "converged": bool(result.success),
                       "message": str(result.message), "gibbs_RT_per_mol": float(result.fun)})
        if result.success and result.fun < initial_g / total - tpd_tolerance:
            candidates.append(result)
    if not candidates:
        raise RuntimeError(f"unstable liquid could not produce a lower-G split: {starts}")
    optimum = min(candidates, key=lambda result: result.fun)
    def mismatch(fractions):
        first, second = z * fractions, z * (1 - fractions)
        return mu(first / first.sum()) - mu(second / second.sum())
    refined = least_squares(mismatch, optimum.x, bounds=(1e-12, 1 - 1e-12),
                            xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=max_iterations)
    error = float(np.max(np.abs(mismatch(refined.x))))
    first, second = n * refined.x, n * (1 - refined.x)
    if not refined.success or error > 1e-7 or np.max(abs(
        first / first.sum() - second / second.sum())) < 1e-6:
        raise RuntimeError("strict liquid split failed chemical-potential refinement")
    phases = []
    for amounts in (first, second):
        phase = dict.fromkeys(activity_model.component_ids, 0.0)
        phase.update(zip(ids, map(float, amounts), strict=True))
        phases.append(phase)
    phases.sort(key=lambda phase: tuple(phase[k] / sum(phase.values()) for k in ids))
    post = tuple(liquid_tpd_minimum(activity_model, phase, temperature_K=temperature_K,
                                    max_iterations=max_iterations) for phase in phases)
    if any(report.minimum_tpd < -1e-7 for report in post):
        raise ValueError("remaining liquid instability may require an unsupported additional phase")
    balance = max(abs(feed[k] - sum(phase[k] for phase in phases)) for k in feed)
    return LiquidEquilibriumResult(
        temperature_K, feed, tuple(phases), stability, post, error, balance,
        float(total * objective(refined.x)[0]), tuple(starts), tuple(
            dict(zip(ids, map(float, mu(np.array([p[k] / sum(p.values()) for k in ids]))),
                     strict=True)) for p in phases))


def lle_phase_stability_diagnostic(
    feed_amounts_mol: Mapping[str, float],
    *,
    partition_coefficients: Mapping[str, float],
    aqueous_volume_L: float,
    organic_volume_L: float,
    activity_model: ActivityModelSpec | None = None,
    temperature_K: float = 298.15,
    initialization_policy: str = "partition_weighted",
    stage_efficiency: float = 1.0,
    tpd_tolerance: float = 1.0e-9,
) -> LLEPhaseStabilityDiagnostic:
    """Return a deterministic TPD-style LLE phase-stability diagnostic.

    The diagnostic seeds two trial liquid phases from supplied partition
    coefficients and phase volumes, evaluates tangent-plane-like distances
    against the feed composition, and subtracts an explicit partition-drive
    term. It is designed for benchmark LLE/extraction tasks where the split
    must be explainable, mass-balanced, and replayable without exposing hidden
    runtime state.
    """

    if aqueous_volume_L <= 0.0 or organic_volume_L <= 0.0:
        raise ValueError("phase volumes must be positive")
    if temperature_K <= 0.0 or not isfinite(temperature_K):
        raise ValueError("temperature_K must be positive and finite")
    if not 0.0 <= stage_efficiency <= 1.0:
        raise ValueError("stage_efficiency must be between 0 and 1")
    if tpd_tolerance < 0.0 or not isfinite(tpd_tolerance):
        raise ValueError("tpd_tolerance must be non-negative and finite")
    feed = _normalize_amounts(feed_amounts_mol)
    component_ids = tuple(feed)
    coefficients = _partition_coefficient_mapping(component_ids, partition_coefficients)
    if activity_model is None:
        activity_model = ActivityModelSpec(
            "ideal_lle_phase_stability",
            component_ids,
            "ideal",
        )
    if set(activity_model.component_ids) != set(component_ids):
        raise ValueError("activity_model component ids must match feed components")

    z = _normalize_composition(feed)
    organic_trial = _trial_phase_composition(
        z,
        coefficients,
        organic_volume_L=organic_volume_L,
        aqueous_volume_L=aqueous_volume_L,
        phase="organic",
    )
    aqueous_trial = _trial_phase_composition(
        z,
        coefficients,
        organic_volume_L=organic_volume_L,
        aqueous_volume_L=aqueous_volume_L,
        phase="aqueous",
    )
    feed_gamma = activity_coefficients(activity_model, z, temperature_K=temperature_K)
    organic_gamma = activity_coefficients(
        activity_model,
        organic_trial,
        temperature_K=temperature_K,
    )
    aqueous_gamma = activity_coefficients(
        activity_model,
        aqueous_trial,
        temperature_K=temperature_K,
    )
    organic_tpd = _tpd_like_distance(organic_trial, organic_gamma, z, feed_gamma)
    aqueous_tpd = _tpd_like_distance(aqueous_trial, aqueous_gamma, z, feed_gamma)
    log_coefficients = {component_id: log(value) for component_id, value in coefficients.items()}
    mean_log_k = sum(z[component_id] * log_coefficients[component_id] for component_id in z)
    partition_log_spread = sqrt(
        sum(
            z[component_id] * (log_coefficients[component_id] - mean_log_k) ** 2
            for component_id in z
        )
    )
    nonideality_drive = sum(
        abs(log(feed_gamma[component_id])) * z[component_id] for component_id in z
    )
    volume_drive = 0.05 * abs(log(organic_volume_L / aqueous_volume_L))
    drive = stage_efficiency * (0.45 * partition_log_spread + 0.15 * nonideality_drive)
    drive += volume_drive
    minimum_tpd = min(organic_tpd, aqueous_tpd) - drive
    warnings = []
    if partition_log_spread < 1.0e-8 and nonideality_drive < 1.0e-8:
        warnings.append("no_partition_or_nonideality_drive")
    if stage_efficiency < 0.5:
        warnings.append("low_stage_efficiency_limits_phase_approach")
    return LLEPhaseStabilityDiagnostic(
        model_id="lle_tpd_style_phase_stability",
        component_ids=component_ids,
        temperature_K=temperature_K,
        initialization_policy=initialization_policy,
        phase_status="two_liquid" if minimum_tpd < -tpd_tolerance else "single_liquid",
        feed_composition=z,
        organic_trial_composition=organic_trial,
        aqueous_trial_composition=aqueous_trial,
        feed_activity_coefficients=feed_gamma,
        organic_activity_coefficients=organic_gamma,
        aqueous_activity_coefficients=aqueous_gamma,
        partition_coefficients=coefficients,
        organic_tpd_like=organic_tpd,
        aqueous_tpd_like=aqueous_tpd,
        partition_log_spread=partition_log_spread,
        nonideality_drive=nonideality_drive,
        minimum_tpd_like=minimum_tpd,
        warnings=tuple(warnings),
    )


def entrain_liquid_amounts(source, target, fraction):
    """Move a declared fraction of one real liquid inventory into the outlet."""
    if (set(source) != set(target) or not isfinite(fraction) or not 0 <= fraction < 1
            or any(not isfinite(v) or v < 0 for v in (*source.values(), *target.values()))):
        raise ValueError("entrainment requires matching finite inventories and a fraction in [0,1)")
    moved = {key: fraction * value for key, value in source.items()}
    return ({key: value - moved[key] for key, value in source.items()},
            {key: value + moved[key] for key, value in target.items()}, moved)


def liquid_liquid_split(
    feed_amounts_mol: Mapping[str, float],
    *,
    partition_coefficients: Mapping[str, float],
    aqueous_volume_L: float,
    organic_volume_L: float,
    stage_efficiency: float = 1.0,
    entrainment_fraction: float = 0.0,
    activity_model: ActivityModelSpec | None = None,
    temperature_K: float = 298.15,
    initialization_policy: str = "partition_weighted",
) -> LLEStageResult:
    if aqueous_volume_L <= 0 or organic_volume_L <= 0:
        raise ValueError("phase volumes must be positive")
    if not 0.0 <= stage_efficiency <= 1.0:
        raise ValueError("stage_efficiency must be between 0 and 1")
    if not 0.0 <= entrainment_fraction < 1.0:
        raise ValueError("entrainment_fraction must be in [0, 1)")
    if any(value < 0 for value in feed_amounts_mol.values()):
        raise ValueError("feed amounts cannot be negative")
    diagnostic = lle_phase_stability_diagnostic(
        feed_amounts_mol,
        partition_coefficients=partition_coefficients,
        aqueous_volume_L=aqueous_volume_L,
        organic_volume_L=organic_volume_L,
        activity_model=activity_model,
        temperature_K=temperature_K,
        initialization_policy=initialization_policy,
        stage_efficiency=stage_efficiency,
    )

    organic = {}
    aqueous = {}
    recovery = {}
    for component_id, amount in feed_amounts_mol.items():
        coefficient = float(partition_coefficients.get(component_id, 1.0))
        if coefficient < 0:
            raise ValueError("partition coefficients cannot be negative")
        ideal_organic = amount * coefficient * organic_volume_L
        ideal_organic /= coefficient * organic_volume_L + aqueous_volume_L
        organic_amount = stage_efficiency * ideal_organic
        aqueous_amount = amount - organic_amount
        organic[component_id] = max(organic_amount, 0.0)
        aqueous[component_id] = max(aqueous_amount, 0.0)
    aqueous, organic, _ = entrain_liquid_amounts(aqueous, organic, entrainment_fraction)
    recovery = {key: 0 if amount <= 0 else organic[key] / amount
                for key, amount in feed_amounts_mol.items()}
    balance_error = max(
        (
            abs(feed_amounts_mol[key] - organic.get(key, 0.0) - aqueous.get(key, 0.0))
            for key in feed_amounts_mol
        ),
        default=0.0,
    )
    return LLEStageResult(
        organic_amounts_mol=organic,
        aqueous_amounts_mol=aqueous,
        recovery_to_organic=recovery,
        phase_volumes_L={
            "aqueous": aqueous_volume_L * (1.0 - entrainment_fraction),
            "organic": organic_volume_L + aqueous_volume_L * entrainment_fraction,
        },
        material_balance_error_mol=balance_error,
        stability_diagnostic=diagnostic.to_dict(),
    )


@dataclass(frozen=True)
class _VLETemperatureSolve:
    temperature_K: float
    residual: float
    iterations: int
    converged: bool
    warnings: tuple[str, ...]


def _solve_vle_temperature(
    evaluator: Callable[
        [float],
        tuple[float, dict[str, float], dict[str, PureSaturationReport]],
    ],
    *,
    lower: float,
    upper: float,
    tolerance_K: float,
    tolerance_log_pressure: float,
    max_iterations: int,
    residual_label: str,
) -> _VLETemperatureSolve:
    f_lower = _evaluate_temperature_residual(evaluator, lower)
    f_upper = _evaluate_temperature_residual(evaluator, upper)
    if abs(f_lower) <= tolerance_log_pressure:
        return _VLETemperatureSolve(lower, f_lower, 0, True, ())
    if abs(f_upper) <= tolerance_log_pressure:
        return _VLETemperatureSolve(upper, f_upper, 0, True, ())
    if f_lower * f_upper > 0.0:
        raise ValueError(
            f"{residual_label} target is outside the temperature bracket: "
            f"residuals=({f_lower:g}, {f_upper:g})"
        )

    left = lower
    right = upper
    f_left = f_lower
    final_temperature = lower
    final_residual = f_lower
    for iteration in range(1, max_iterations + 1):
        mid = 0.5 * (left + right)
        f_mid = _evaluate_temperature_residual(evaluator, mid)
        final_temperature = mid
        final_residual = f_mid
        if abs(f_mid) <= tolerance_log_pressure or (right - left) <= tolerance_K:
            return _VLETemperatureSolve(mid, f_mid, iteration, True, ())
        if f_left * f_mid <= 0.0:
            right = mid
        else:
            left = mid
            f_left = f_mid
    return _VLETemperatureSolve(
        final_temperature,
        final_residual,
        max_iterations,
        False,
        (f"{residual_label} solve reached max_iterations before tolerance",),
    )


def _evaluate_temperature_residual(
    evaluator: Callable[
        [float],
        tuple[float, dict[str, float], dict[str, PureSaturationReport]],
    ],
    temperature_K: float,
) -> float:
    residual, _vapor_pressures, _reports = evaluator(temperature_K)
    if not isfinite(residual):
        raise ValueError("VLE temperature residual must be finite")
    return float(residual)


def _validate_vle_temperature_inputs(
    *,
    pressure_Pa: float,
    tolerance_K: float,
    tolerance_log_pressure: float,
    max_iterations: int,
) -> None:
    if pressure_Pa <= 0 or not isfinite(pressure_Pa):
        raise ValueError("pressure_Pa must be positive and finite")
    if tolerance_K <= 0 or tolerance_log_pressure <= 0:
        raise ValueError("VLE temperature tolerances must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _validate_vapor_pressure_correlations(
    component_ids: tuple[str, ...],
    correlations: Mapping[str, PropertyCorrelation],
) -> None:
    missing = sorted(set(component_ids) - set(correlations))
    extra = sorted(set(correlations) - set(component_ids))
    if missing or extra:
        raise ValueError(
            "vapor pressure correlation keys must match components: "
            f"missing={missing}, extra={extra}"
        )
    for component_id in component_ids:
        correlation = correlations[component_id]
        if correlation.property_id not in {"vapor_pressure", "sublimation_pressure"}:
            raise ValueError(
                f"correlation for {component_id!r} must be a vapor pressure "
                "or sublimation pressure correlation"
            )


def _mixture_temperature_bounds_k(
    component_ids: tuple[str, ...],
    correlations: Mapping[str, PropertyCorrelation],
    *,
    temperature_bounds_K: tuple[float, float] | None,
) -> tuple[float, float]:
    if temperature_bounds_K is not None:
        lower, upper = temperature_bounds_K
        if lower <= 0 or upper <= lower:
            raise ValueError("temperature_bounds_K must be positive and increasing")
        return float(lower), float(upper)
    lower_candidates: list[float] = []
    upper_candidates: list[float] = []
    for component_id in component_ids:
        correlation = correlations[component_id]
        bounds = correlation.validity_ranges.get("temperature")
        if bounds is None:
            lower_candidates.append(200.0)
            upper_candidates.append(650.0)
            continue
        source_unit = correlation.input_units["temperature"]
        lower_candidates.append(convert_value(bounds[0], source_unit, "K"))
        upper_candidates.append(convert_value(bounds[1], source_unit, "K"))
    lower = max(lower_candidates)
    upper = min(upper_candidates)
    if lower >= upper:
        raise ValueError("vapor-pressure correlations do not have an overlapping temperature range")
    return lower, upper


def _vapor_pressures_from_correlations(
    component_ids: tuple[str, ...],
    correlations: Mapping[str, PropertyCorrelation],
    *,
    temperature_K: float,
    validity_policy: ValidityPolicy,
) -> tuple[dict[str, float], dict[str, PureSaturationReport]]:
    vapor_pressures: dict[str, float] = {}
    reports: dict[str, PureSaturationReport] = {}
    for component_id in component_ids:
        report = pure_saturation_pressure_report(
            correlations[component_id],
            temperature_K=temperature_K,
            validity_policy=validity_policy,
        )
        vapor_pressures[component_id] = report.saturation_pressure_Pa
        reports[component_id] = report
    return vapor_pressures, reports


def _dew_pressure_and_liquid_composition(
    vapor_composition: Mapping[str, float],
    *,
    vapor_pressures_Pa: Mapping[str, float],
    activity_model: ActivityModelSpec,
    temperature_K: float,
    iterations: int,
) -> tuple[float, dict[str, float]]:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    y = _composition_vector_mapping(activity_model.component_ids, vapor_composition)
    _validate_vapor_pressure_values(y, vapor_pressures_Pa)
    pressure = 1.0 / sum(y[key] / float(vapor_pressures_Pa[key]) for key in y)
    liquid = dict(y)
    for _ in range(iterations):
        gamma = activity_coefficients(activity_model, liquid, temperature_K=temperature_K)
        denominator = sum(y[key] / (gamma[key] * float(vapor_pressures_Pa[key])) for key in y)
        if denominator <= 0.0 or not isfinite(denominator):
            raise ValueError("dew pressure denominator must be positive and finite")
        pressure = 1.0 / denominator
        liquid = _normalize_composition(
            {key: y[key] * pressure / (gamma[key] * float(vapor_pressures_Pa[key])) for key in y}
        )
    return pressure, liquid


def _validate_vapor_pressure_values(
    composition: Mapping[str, float],
    vapor_pressures_Pa: Mapping[str, float],
    *, nonvolatile_component_ids: tuple[str, ...] = (),
) -> None:
    missing = sorted(set(composition) - set(vapor_pressures_Pa))
    extra = sorted(set(vapor_pressures_Pa) - set(composition))
    if missing or extra:
        raise ValueError(
            f"vapor pressure keys must match composition: missing={missing}, extra={extra}"
        )
    declared = set(nonvolatile_component_ids)
    if declared-set(composition):
        raise ValueError("nonvolatile identities must belong to the mixture")
    if any(not isfinite(float(value)) or (value != 0 if key in declared else value <= 0)
           for key,value in vapor_pressures_Pa.items()):
        raise ValueError("vapor pressures must be finite and positive")


def _factor_mapping(
    component_ids: tuple[str, ...],
    values: Mapping[str, float] | None,
    *,
    field_name: str,
) -> dict[str, float]:
    if values is None:
        return dict.fromkeys(component_ids, 1.0)
    missing = sorted(set(component_ids) - set(values))
    extra = sorted(set(values) - set(component_ids))
    if missing or extra:
        raise ValueError(
            f"{field_name} keys must match components: missing={missing}, extra={extra}"
        )
    result = {component_id: float(values[component_id]) for component_id in component_ids}
    if any(value <= 0 or not isfinite(value) for value in result.values()):
        raise ValueError(f"{field_name} must contain finite positive values")
    return result


def _find_azeotrope_crossing(
    scan_points: list[AzeotropeScanPoint],
    *,
    light_component_id: str,
    residual_tolerance: float,
) -> (
    tuple[
        AzeotropeScanStatus,
        tuple[float, float],
        dict[str, float],
        float,
    ]
    | None
):
    for point in scan_points:
        if abs(point.residual) <= residual_tolerance:
            fraction = point.composition[light_component_id]
            return (
                "endpoint_near_crossing",
                (fraction, fraction),
                dict(point.composition),
                point.residual,
            )
    for left, right in pairwise(scan_points):
        if left.residual * right.residual > 0.0:
            continue
        left_fraction = left.composition[light_component_id]
        right_fraction = right.composition[light_component_id]
        denominator = right.residual - left.residual
        if abs(denominator) <= 1e-300:
            estimate = 0.5 * (left_fraction + right_fraction)
        else:
            estimate = (
                left_fraction - left.residual * (right_fraction - left_fraction) / denominator
            )
        estimate = min(max(estimate, left_fraction), right_fraction)
        other_component = next(
            component_id for component_id in left.composition if component_id != light_component_id
        )
        return (
            "relative_volatility_crossing",
            (left_fraction, right_fraction),
            {
                light_component_id: estimate,
                other_component: 1.0 - estimate,
            },
            0.0,
        )
    return None


def _uniquac_tau_matrix(
    spec: ActivityModelSpec,
    temperature_K: float,
) -> list[list[float]]:
    component_ids = spec.component_ids
    n = len(component_ids)
    matrix = [[1.0 for _ in range(n)] for _ in range(n)]
    for i, left in enumerate(component_ids):
        for j, right in enumerate(component_ids):
            if i == j:
                continue
            value = _uniquac_tau(spec, left, right, temperature_K)
            if value <= 0.0 or not isfinite(value):
                raise ValueError("UNIQUAC tau values must be positive and finite")
            matrix[i][j] = value
    return matrix


def _uniquac_tau(
    spec: ActivityModelSpec,
    left: str,
    right: str,
    temperature_K: float,
) -> float:
    return interaction(spec.parameters, "tau", left, right, temperature_K)


def _uniquac_structural_parameter(
    spec: ActivityModelSpec,
    prefix: str,
    component_id: str,
) -> float:
    key = f"{prefix}:{component_id}"
    if key not in spec.parameters:
        raise ValueError(f"UNIQUAC requires {prefix}:{component_id}")
    value = float(spec.parameters[key])
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"UNIQUAC {prefix} parameters must be positive and finite")
    return value


def _uniquac_coordination_number(spec: ActivityModelSpec) -> float:
    value = float(spec.parameters.get("z", 10.0))
    if value <= 0.0 or not isfinite(value):
        raise ValueError("UNIQUAC coordination number z must be positive and finite")
    return value


def _validate_activity_parameter_contract(spec: ActivityModelSpec) -> None:
    if spec.model == "unifac":
        from chemworld.physchem.unifac import validate_unifac_parameters

        validate_unifac_parameters(spec)
        return
    if spec.model in {"regular_solution", "flory_huggins", "hansen"}:
        from chemworld.physchem.regular_solution import validate_regular_parameters

        validate_regular_parameters(spec)
        return
    if spec.model == "ideal":
        return
    if spec.model == "margules":
        for i, left in enumerate(spec.component_ids):
            for right in spec.component_ids[i + 1:]:
                if not any(key in spec.parameters for key in
                           (f"A:{left}|{right}", f"A:{right}|{left}")):
                    raise ValueError(f"Margules requires an explicit A for pair {left}|{right}")
        return
    if spec.model == "uniquac":
        _uniquac_coordination_number(spec)
        for component_id in spec.component_ids:
            _uniquac_structural_parameter(spec, "r", component_id)
            _uniquac_structural_parameter(spec, "q", component_id)
    for left in spec.component_ids:
        for right in spec.component_ids:
            if left == right:
                continue
            if spec.model == "wilson":
                _validate_pair_has_any(
                    spec,
                    left,
                    right,
                    (
                        "lambda",
                        "lambda_a",
                        "lambda_b",
                        "lambda_c",
                        "lambda_d",
                        "lambda_e",
                        "lambda_f",
                    ),
                )
                direct = _directional_parameter(spec, "lambda", left, right, default=None)
                if direct is not None and direct <= 0.0:
                    raise ValueError("Wilson Lambda values must be positive")
            if spec.model == "nrtl":
                _validate_pair_has_any(
                    spec,
                    left,
                    right,
                    ("tau", "tau_a", "tau_b", "tau_e", "tau_f", "tau_g", "tau_h"),
                )
                _validate_pair_has_any(
                    spec,
                    left,
                    right,
                    ("alpha", "alpha_c", "alpha_d"),
                )
                direct_alpha = _directional_parameter(
                    spec,
                    "alpha",
                    left,
                    right,
                    default=None,
                )
                if direct_alpha is not None and direct_alpha <= 0.0:
                    raise ValueError("NRTL alpha values must be positive")
            if spec.model == "uniquac":
                _validate_pair_has_any(
                    spec,
                    left,
                    right,
                    ("tau", "tau_a", "tau_b", "tau_c", "tau_d", "tau_e", "tau_f"),
                )
                direct_tau = _directional_parameter(spec, "tau", left, right, default=None)
                if direct_tau is not None and direct_tau <= 0.0:
                    raise ValueError("UNIQUAC tau values must be positive")


def _validate_pair_has_any(
    spec: ActivityModelSpec,
    left: str,
    right: str,
    prefixes: tuple[str, ...],
) -> None:
    if not any(_has_directional_parameter(spec, prefix, left, right) for prefix in prefixes):
        allowed = ", ".join(prefixes)
        raise ValueError(f"{spec.model} requires one of {{{allowed}}} for pair {left}|{right}")


def _has_directional_parameter(
    spec: ActivityModelSpec,
    prefix: str,
    left: str,
    right: str,
) -> bool:
    return f"{prefix}:{left}|{right}" in spec.parameters


def _directional_parameter(
    spec: ActivityModelSpec,
    prefix: str,
    left: str,
    right: str,
    *,
    default: float | None,
) -> float | None:
    key = f"{prefix}:{left}|{right}"
    if key not in spec.parameters:
        return default
    return float(spec.parameters[key])


def _normalize_amounts(amounts: Mapping[str, float]) -> dict[str, float]:
    if not amounts:
        raise ValueError("feed amounts cannot be empty")
    result = {}
    for component_id, amount in amounts.items():
        if not component_id:
            raise ValueError("component ids cannot be empty")
        value = float(amount)
        if value < 0.0 or not isfinite(value):
            raise ValueError("feed amounts must be finite and non-negative")
        result[component_id] = value
    if sum(result.values()) <= 0.0:
        raise ValueError("feed amount total must be positive")
    return result


def _partition_coefficient_mapping(
    component_ids: tuple[str, ...],
    partition_coefficients: Mapping[str, float],
) -> dict[str, float]:
    result = {}
    for component_id in component_ids:
        coefficient = float(partition_coefficients.get(component_id, 1.0))
        if coefficient <= 0.0 or not isfinite(coefficient):
            raise ValueError("partition coefficients must be finite and positive")
        result[component_id] = coefficient
    return result


def _trial_phase_composition(
    feed_composition: Mapping[str, float],
    partition_coefficients: Mapping[str, float],
    *,
    organic_volume_L: float,
    aqueous_volume_L: float,
    phase: str,
) -> dict[str, float]:
    if phase not in {"organic", "aqueous"}:
        raise ValueError("phase must be organic or aqueous")
    if phase == "organic":
        weights = {
            component_id: max(
                feed_composition[component_id]
                * partition_coefficients[component_id]
                * organic_volume_L,
                1.0e-300,
            )
            for component_id in feed_composition
        }
    else:
        weights = {
            component_id: max(
                feed_composition[component_id]
                * aqueous_volume_L
                / partition_coefficients[component_id],
                1.0e-300,
            )
            for component_id in feed_composition
        }
    return _normalize_composition(weights)


def _tpd_like_distance(
    trial_composition: Mapping[str, float],
    trial_activity_coefficients: Mapping[str, float],
    feed_composition: Mapping[str, float],
    feed_activity_coefficients: Mapping[str, float],
) -> float:
    floor = 1.0e-300
    return sum(
        trial_composition[component_id]
        * (
            log(max(trial_composition[component_id], floor))
            + log(max(trial_activity_coefficients[component_id], floor))
            - log(max(feed_composition[component_id], floor))
            - log(max(feed_activity_coefficients[component_id], floor))
        )
        for component_id in feed_composition
    )


def _composition_vector(
    component_ids: tuple[str, ...],
    composition: Mapping[str, float],
) -> tuple[float, ...]:
    return tuple(
        _composition_vector_mapping(component_ids, composition)[component_id]
        for component_id in component_ids
    )


def _composition_vector_mapping(
    component_ids: tuple[str, ...],
    composition: Mapping[str, float],
) -> dict[str, float]:
    normalized = _normalize_composition(composition)
    missing = sorted(set(component_ids) - set(normalized))
    extra = sorted(set(normalized) - set(component_ids))
    if missing or extra:
        raise ValueError(f"Composition ids do not match model: missing={missing}, extra={extra}")
    return {component_id: normalized[component_id] for component_id in component_ids}


def _validate_k_values(
    composition: Mapping[str, float],
    k_values: Mapping[str, float],
    *, nonvolatile_component_ids: tuple[str, ...] = (),
) -> None:
    missing = sorted(set(composition) - set(k_values))
    extra = sorted(set(k_values) - set(composition))
    if missing or extra:
        raise ValueError(f"K-value ids do not match composition: missing={missing}, extra={extra}")
    declared = set(nonvolatile_component_ids)
    if declared-set(composition):
        raise ValueError("nonvolatile identities must belong to the mixture")
    if any(not isfinite(value) or (value != 0 if key in declared else value <= 0)
           for key,value in k_values.items()):
        raise ValueError("K-values must be finite and positive")


__all__ = [
    "AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K",
    "DAVIES_MAX_IONIC_STRENGTH_MOL_KG",
    "ActivityModel",
    "ActivityModelSpec",
    "AqueousActivityCoefficientResult",
    "AzeotropeScanPoint",
    "AzeotropeScanStatus",
    "BinaryAzeotropeDiagnosticReport",
    "FlashPhaseStatus",
    "FlashResult",
    "GammaPhiKValueReport",
    "LLEPhaseStabilityDiagnostic",
    "LLEStageResult",
    "LiquidEquilibriumResult",
    "LiquidStabilityReport",
    "RachfordRiceDiagnosticReport",
    "UNIQUACActivityReport",
    "VLESolveMode",
    "VLETemperatureReport",
    "activity_coefficients",
    "activity_model_cards",
    "binary_azeotrope_diagnostic_report",
    "bubble_pressure_pa",
    "bubble_temperature_report",
    "davies_aqueous_activity_coefficient",
    "dew_pressure_pa",
    "dew_temperature_report",
    "entrain_liquid_amounts",
    "equilibrate_liquids",
    "flash_isothermal",
    "gamma_phi_k_value_report",
    "liquid_liquid_split",
    "liquid_tpd_minimum",
    "lle_phase_stability_diagnostic",
    "rachford_rice_diagnostic_report",
    "rachford_rice_vapor_fraction",
    "raoult_k_values",
    "uniquac_activity_report",
    "weak_acid_davies_activity_ratio",
]
