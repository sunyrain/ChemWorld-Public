"""Reaction-equilibrium, electrolyte, and precipitation kernels for ChemWorld."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from itertools import pairwise
from math import exp, isfinite, log, log10
from typing import Literal

import numpy as np
from scipy.optimize import brentq, least_squares, minimize

from chemworld.physchem.reaction_network import R_J_PER_MOL_K, parse_reaction_equation

EquilibriumActivityModel = Literal["concentration"]
_ACTIVITY_FLOOR = 1e-30


@dataclass(frozen=True)
class EquilibriumReactionSpec:
    """Mass-action equilibrium reaction with van't Hoff temperature dependence."""

    reaction_id: str
    stoichiometry: dict[str, float]
    log10_k_ref: float
    reference_temperature_K: float = 298.15
    delta_h_J_mol: float = 0.0
    activity_model: EquilibriumActivityModel = "concentration"
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.reaction_id:
            raise ValueError("reaction_id cannot be empty")
        if not self.stoichiometry:
            raise ValueError("stoichiometry cannot be empty")
        if not any(value < 0.0 for value in self.stoichiometry.values()):
            raise ValueError("equilibrium reaction must contain at least one reactant")
        if not any(value > 0.0 for value in self.stoichiometry.values()):
            raise ValueError("equilibrium reaction must contain at least one product")
        if any(not isfinite(value) or value == 0.0 for value in self.stoichiometry.values()):
            raise ValueError("stoichiometric coefficients must be finite and nonzero")
        _finite(self.log10_k_ref, "log10_k_ref")
        _positive(self.reference_temperature_K, "reference_temperature_K")
        _finite(self.delta_h_J_mol, "delta_h_J_mol")
        if self.activity_model != "concentration":
            raise ValueError(f"Unsupported activity model: {self.activity_model}")

    @classmethod
    def from_equation(
        cls,
        *,
        reaction_id: str,
        equation: str,
        log10_k_ref: float,
        reference_temperature_K: float = 298.15,
        delta_h_J_mol: float = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> EquilibriumReactionSpec:
        stoichiometry, _ = parse_reaction_equation(equation)
        return cls(
            reaction_id=reaction_id,
            stoichiometry=stoichiometry,
            log10_k_ref=log10_k_ref,
            reference_temperature_K=reference_temperature_K,
            delta_h_J_mol=delta_h_J_mol,
            metadata={} if metadata is None else dict(metadata),
        )

    def equilibrium_constant(self, temperature_K: float) -> float:
        return equilibrium_constant_vant_hoff(
            log10_k_ref=self.log10_k_ref,
            temperature_K=temperature_K,
            reference_temperature_K=self.reference_temperature_K,
            delta_h_J_mol=self.delta_h_J_mol,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reaction_id": self.reaction_id,
            "stoichiometry": dict(self.stoichiometry),
            "log10_k_ref": self.log10_k_ref,
            "reference_temperature_K": self.reference_temperature_K,
            "delta_h_J_mol": self.delta_h_J_mol,
            "activity_model": self.activity_model,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EquilibriumSystemSpec:
    """A compact equilibrium problem at fixed T, P, and volume."""

    system_id: str
    reactions: tuple[EquilibriumReactionSpec, ...]
    temperature_K: float = 298.15
    pressure_Pa: float = 101_325.0
    volume_L: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id cannot be empty")
        if not self.reactions:
            raise ValueError("Equilibrium system must contain at least one reaction")
        reaction_ids = [reaction.reaction_id for reaction in self.reactions]
        if len(reaction_ids) != len(set(reaction_ids)):
            raise ValueError("Duplicate reaction_id values are not allowed")
        _positive(self.temperature_K, "temperature_K")
        _positive(self.pressure_Pa, "pressure_Pa")
        _positive(self.volume_L, "volume_L")

    @property
    def species_ids(self) -> tuple[str, ...]:
        ids = sorted({key for reaction in self.reactions for key in reaction.stoichiometry})
        return tuple(ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "reactions": [reaction.to_dict() for reaction in self.reactions],
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "volume_L": self.volume_L,
            "species_ids": list(self.species_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class EquilibriumResult:
    """Solved reaction-equilibrium state."""

    system_id: str
    initial_amounts_mol: dict[str, float]
    final_amounts_mol: dict[str, float]
    extents_mol: dict[str, float]
    equilibrium_constants: dict[str, float]
    reaction_quotients: dict[str, float]
    residuals_log: dict[str, float]
    converged: bool
    iterations: int
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id cannot be empty")
        _validate_amounts(self.initial_amounts_mol, "initial_amounts_mol")
        _validate_amounts(self.final_amounts_mol, "final_amounts_mol")
        if any(not isfinite(value) for value in self.extents_mol.values()):
            raise ValueError("extents must be finite")
        if any(
            value <= 0.0 or not isfinite(value) for value in self.equilibrium_constants.values()
        ):
            raise ValueError("equilibrium constants must be positive and finite")
        if any(value <= 0.0 or not isfinite(value) for value in self.reaction_quotients.values()):
            raise ValueError("reaction quotients must be positive and finite")
        if any(not isfinite(value) for value in self.residuals_log.values()):
            raise ValueError("residuals must be finite")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")

    def conversion(self, species_id: str) -> float:
        initial = self.initial_amounts_mol.get(species_id, 0.0)
        if initial <= 0.0:
            return 0.0
        return _clip01((initial - self.final_amounts_mol.get(species_id, 0.0)) / initial)

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "initial_amounts_mol": dict(self.initial_amounts_mol),
            "final_amounts_mol": dict(self.final_amounts_mol),
            "extents_mol": dict(self.extents_mol),
            "equilibrium_constants": dict(self.equilibrium_constants),
            "reaction_quotients": dict(self.reaction_quotients),
            "residuals_log": dict(self.residuals_log),
            "converged": self.converged,
            "iterations": self.iterations,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GibbsSpeciesSpec:
    """Species record for the compact fixed-TP Gibbs minimization solver."""

    species_id: str
    phase: str
    element_counts: dict[str, float]
    standard_gibbs_J_mol: float = 0.0
    charge: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.species_id:
            raise ValueError("species_id cannot be empty")
        if not self.phase:
            raise ValueError("phase cannot be empty")
        if not self.element_counts:
            raise ValueError("element_counts cannot be empty")
        if not any(value > 0.0 for value in self.element_counts.values()):
            raise ValueError("element_counts must contain at least one positive count")
        for element_id, count in self.element_counts.items():
            if not element_id:
                raise ValueError("element id cannot be empty")
            _nonnegative(float(count), f"element_counts[{element_id}]")
        _finite(self.standard_gibbs_J_mol, "standard_gibbs_J_mol")
        _finite(self.charge, "charge")

    def to_dict(self) -> dict[str, object]:
        return {
            "species_id": self.species_id,
            "phase": self.phase,
            "element_counts": dict(self.element_counts),
            "standard_gibbs_J_mol": self.standard_gibbs_J_mol,
            "charge": self.charge,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GibbsMinimizationSpec:
    """Fixed-TP ideal-mixture Gibbs minimization problem."""

    system_id: str
    species: tuple[GibbsSpeciesSpec, ...]
    temperature_K: float = 298.15
    pressure_Pa: float = 101_325.0
    allowed_phases: tuple[str, ...] = ()
    target_charge_eq: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id cannot be empty")
        if not self.species:
            raise ValueError("Gibbs minimization problem must contain species")
        species_ids = [item.species_id for item in self.species]
        if len(species_ids) != len(set(species_ids)):
            raise ValueError("Duplicate Gibbs species ids are not allowed")
        _positive(self.temperature_K, "temperature_K")
        _positive(self.pressure_Pa, "pressure_Pa")
        if len(self.allowed_phases) != len(set(self.allowed_phases)):
            raise ValueError("Duplicate allowed phases are not allowed")
        if self.target_charge_eq is not None:
            _finite(self.target_charge_eq, "target_charge_eq")

    @property
    def species_ids(self) -> tuple[str, ...]:
        return tuple(species.species_id for species in self.species)

    @property
    def element_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {element_id for species in self.species for element_id in species.element_counts}
            )
        )

    @property
    def phase_ids(self) -> tuple[str, ...]:
        return tuple(sorted({species.phase for species in self.species}))

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "species": [species.to_dict() for species in self.species],
            "temperature_K": self.temperature_K,
            "pressure_Pa": self.pressure_Pa,
            "allowed_phases": list(self.allowed_phases),
            "target_charge_eq": self.target_charge_eq,
            "element_ids": list(self.element_ids),
            "phase_ids": list(self.phase_ids),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GibbsMinimizationResult:
    """Result of a compact constrained Gibbs minimization."""

    system_id: str
    initial_amounts_mol: dict[str, float]
    final_amounts_mol: dict[str, float]
    total_gibbs_J: float
    element_balance_residuals_mol: dict[str, float]
    charge_balance_residual_eq: float
    phase_amounts_mol: dict[str, float]
    active_phases: tuple[str, ...]
    converged: bool
    iterations: int
    diagnostic: GibbsMinimizationDiagnostic | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id:
            raise ValueError("system_id cannot be empty")
        _validate_amounts(self.initial_amounts_mol, "initial_amounts_mol")
        _validate_amounts(self.final_amounts_mol, "final_amounts_mol")
        _finite(self.total_gibbs_J, "total_gibbs_J")
        if any(not isfinite(value) for value in self.element_balance_residuals_mol.values()):
            raise ValueError("element balance residuals must be finite")
        _finite(self.charge_balance_residual_eq, "charge_balance_residual_eq")
        _validate_amounts(self.phase_amounts_mol, "phase_amounts_mol")
        if self.iterations < 0:
            raise ValueError("iterations cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "system_id": self.system_id,
            "initial_amounts_mol": dict(self.initial_amounts_mol),
            "final_amounts_mol": dict(self.final_amounts_mol),
            "total_gibbs_J": self.total_gibbs_J,
            "element_balance_residuals_mol": dict(self.element_balance_residuals_mol),
            "charge_balance_residual_eq": self.charge_balance_residual_eq,
            "phase_amounts_mol": dict(self.phase_amounts_mol),
            "active_phases": list(self.active_phases),
            "converged": self.converged,
            "iterations": self.iterations,
            "diagnostic": None if self.diagnostic is None else self.diagnostic.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GibbsMinimizationDiagnostic:
    """Constraint and KKT-style diagnostic for the compact Gibbs solver."""

    status: Literal["ok", "warning", "failed"]
    max_element_residual_mol: float
    charge_residual_eq: float
    max_bound_violation_mol: float
    stationarity_residual_J_mol: float
    constraint_matrix_rank: int
    allowed_species_count: int
    degrees_of_freedom: int
    active_species_ids: tuple[str, ...]
    active_lower_bound_species_ids: tuple[str, ...]
    active_upper_bound_species_ids: tuple[str, ...]
    convexity_class: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonnegative(self.max_element_residual_mol, "max_element_residual_mol")
        _finite(self.charge_residual_eq, "charge_residual_eq")
        _nonnegative(self.max_bound_violation_mol, "max_bound_violation_mol")
        _nonnegative(self.stationarity_residual_J_mol, "stationarity_residual_J_mol")
        if self.constraint_matrix_rank < 0:
            raise ValueError("constraint_matrix_rank cannot be negative")
        if self.allowed_species_count < 0:
            raise ValueError("allowed_species_count cannot be negative")
        if self.degrees_of_freedom < 0:
            raise ValueError("degrees_of_freedom cannot be negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "max_element_residual_mol": self.max_element_residual_mol,
            "charge_residual_eq": self.charge_residual_eq,
            "max_bound_violation_mol": self.max_bound_violation_mol,
            "stationarity_residual_J_mol": self.stationarity_residual_J_mol,
            "constraint_matrix_rank": self.constraint_matrix_rank,
            "allowed_species_count": self.allowed_species_count,
            "degrees_of_freedom": self.degrees_of_freedom,
            "active_species_ids": list(self.active_species_ids),
            "active_lower_bound_species_ids": list(self.active_lower_bound_species_ids),
            "active_upper_bound_species_ids": list(self.active_upper_bound_species_ids),
            "convexity_class": self.convexity_class,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class AcidBaseResult:
    """Monoprotic acid/base equilibrium in water."""

    species_amounts_mol: dict[str, float]
    pH: float
    pOH: float
    hydrogen_mol_L: float
    hydroxide_mol_L: float
    acid_dissociation_fraction: float
    charge_balance_error_eq: float
    ionic_strength_mol_kg: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_amounts(self.species_amounts_mol, "species_amounts_mol")
        _finite(self.pH, "pH")
        _finite(self.pOH, "pOH")
        _positive(self.hydrogen_mol_L, "hydrogen_mol_L")
        _positive(self.hydroxide_mol_L, "hydroxide_mol_L")
        if not 0.0 <= self.acid_dissociation_fraction <= 1.0:
            raise ValueError("acid_dissociation_fraction must be between 0 and 1")
        _finite(self.charge_balance_error_eq, "charge_balance_error_eq")
        _nonnegative(self.ionic_strength_mol_kg, "ionic_strength_mol_kg")

    def to_dict(self) -> dict[str, object]:
        return {
            "species_amounts_mol": dict(self.species_amounts_mol),
            "pH": self.pH,
            "pOH": self.pOH,
            "hydrogen_mol_L": self.hydrogen_mol_L,
            "hydroxide_mol_L": self.hydroxide_mol_L,
            "acid_dissociation_fraction": self.acid_dissociation_fraction,
            "charge_balance_error_eq": self.charge_balance_error_eq,
            "ionic_strength_mol_kg": self.ionic_strength_mol_kg,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PHObservationResult:
    """Public pH-meter style observation for an aqueous acid/base state."""

    raw_signal: dict[str, object]
    processed_estimate: dict[str, float]
    uncertainty: dict[str, float]
    observed_mask: dict[str, bool]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.raw_signal:
            raise ValueError("raw_signal cannot be empty")
        if not self.processed_estimate:
            raise ValueError("processed_estimate cannot be empty")
        for key, value in self.processed_estimate.items():
            _finite(float(value), f"processed_estimate[{key}]")
        for key, value in self.uncertainty.items():
            _nonnegative(float(value), f"uncertainty[{key}]")

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_signal": dict(self.raw_signal),
            "processed_estimate": dict(self.processed_estimate),
            "uncertainty": dict(self.uncertainty),
            "observed_mask": dict(self.observed_mask),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SolubilityProductSpec:
    """Binary salt solubility-product model."""

    precipitate_id: str
    cation_id: str
    anion_id: str
    ksp: float
    cation_stoich: float = 1.0
    anion_stoich: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.precipitate_id:
            raise ValueError("precipitate_id cannot be empty")
        if not self.cation_id or not self.anion_id:
            raise ValueError("ion ids cannot be empty")
        _positive(self.ksp, "ksp")
        _positive(self.cation_stoich, "cation_stoich")
        _positive(self.anion_stoich, "anion_stoich")

    def to_dict(self) -> dict[str, object]:
        return {
            "precipitate_id": self.precipitate_id,
            "cation_id": self.cation_id,
            "anion_id": self.anion_id,
            "ksp": self.ksp,
            "cation_stoich": self.cation_stoich,
            "anion_stoich": self.anion_stoich,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrecipitationResult:
    """Precipitation or dissolution proxy result."""

    final_amounts_mol: dict[str, float]
    precipitated_mol: float
    ion_product: float
    saturation_index: float
    material_balance_error_mol: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_amounts(self.final_amounts_mol, "final_amounts_mol")
        _nonnegative(self.precipitated_mol, "precipitated_mol")
        _nonnegative(self.ion_product, "ion_product")
        _finite(self.saturation_index, "saturation_index")
        _nonnegative(self.material_balance_error_mol, "material_balance_error_mol")

    def to_dict(self) -> dict[str, object]:
        return {
            "final_amounts_mol": dict(self.final_amounts_mol),
            "precipitated_mol": self.precipitated_mol,
            "ion_product": self.ion_product,
            "saturation_index": self.saturation_index,
            "material_balance_error_mol": self.material_balance_error_mol,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PrecipitationHookResult:
    """Sequential precipitation-hook result for compact aqueous scenarios."""

    final_amounts_mol: dict[str, float]
    precipitation_events: tuple[dict[str, object], ...]
    total_precipitated_mol: float
    material_balance_error_mol: float
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_amounts(self.final_amounts_mol, "final_amounts_mol")
        _nonnegative(self.total_precipitated_mol, "total_precipitated_mol")
        _nonnegative(self.material_balance_error_mol, "material_balance_error_mol")

    @property
    def converged(self) -> bool:
        """Whether every configured precipitation hook reached a fixed point."""

        return self.metadata.get("status") == "converged"

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return explicit warnings for incomplete hook passes."""

        if self.converged:
            return ()
        return ("max_passes_reached",)

    def to_dict(self) -> dict[str, object]:
        return {
            "final_amounts_mol": dict(self.final_amounts_mol),
            "precipitation_events": [dict(event) for event in self.precipitation_events],
            "total_precipitated_mol": self.total_precipitated_mol,
            "material_balance_error_mol": self.material_balance_error_mol,
            "converged": self.converged,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ChargeBalanceResult:
    """Electroneutrality adjustment result."""

    adjusted_amounts_mol: dict[str, float]
    initial_charge_eq: float
    final_charge_eq: float
    adjustment_mol: float
    adjusted_species_id: str
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_amounts(self.adjusted_amounts_mol, "adjusted_amounts_mol")
        _finite(self.initial_charge_eq, "initial_charge_eq")
        _finite(self.final_charge_eq, "final_charge_eq")
        _finite(self.adjustment_mol, "adjustment_mol")
        if not self.adjusted_species_id:
            raise ValueError("adjusted_species_id cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "adjusted_amounts_mol": dict(self.adjusted_amounts_mol),
            "initial_charge_eq": self.initial_charge_eq,
            "final_charge_eq": self.final_charge_eq,
            "adjustment_mol": self.adjustment_mol,
            "adjusted_species_id": self.adjusted_species_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AqueousElectrolyteEquilibriumResult:
    """Coupled weak-acid, ionic-activity, and Ksp-hook result."""

    acid_base: AcidBaseResult
    precipitation: PrecipitationHookResult
    activity_coefficient_ratio: float
    iterations: int
    converged: bool
    material_balance_error_mol: float
    charge_balance_error_eq: float
    applicability: dict[str, float | str]

    def to_dict(self) -> dict[str, object]:
        return {
            "acid_base": self.acid_base.to_dict(),
            "precipitation": self.precipitation.to_dict(),
            "activity_coefficient_ratio": self.activity_coefficient_ratio,
            "iterations": self.iterations,
            "converged": self.converged,
            "material_balance_error_mol": self.material_balance_error_mol,
            "charge_balance_error_eq": self.charge_balance_error_eq,
            "applicability": dict(self.applicability),
        }


def solve_aqueous_electrolyte_equilibrium(
    *,
    acid_total_mol: float,
    volume_L: float,
    pka: float,
    temperature_K: float = 298.15,
    supporting_electrolyte_mol: float = 0.0,
    precipitating_salt_mol: float = 0.0,
    solubility_product: SolubilityProductSpec | None = None,
    max_precipitation_passes: int = 3,
    max_activity_iterations: int = 64,
    activity_tolerance: float = 1.0e-10,
) -> AqueousElectrolyteEquilibriumResult:
    """Solve a bounded aqueous electrolyte state and fail closed on invalidity.

    The strong supporting electrolyte and binary precipitating salt are both
    treated as 1:1 electrolytes.  Davies activity corrections are iterated with
    the weak-acid electroneutrality solve; optional Ksp hooks are applied first.
    """

    _nonnegative(acid_total_mol, "acid_total_mol")
    _positive(volume_L, "volume_L")
    _finite(pka, "pka")
    _positive(temperature_K, "temperature_K")
    _nonnegative(supporting_electrolyte_mol, "supporting_electrolyte_mol")
    _nonnegative(precipitating_salt_mol, "precipitating_salt_mol")
    if max_precipitation_passes <= 0:
        raise ValueError("max_precipitation_passes must be positive")
    if max_activity_iterations <= 0:
        raise ValueError("max_activity_iterations must be positive")
    _positive(activity_tolerance, "activity_tolerance")
    if solubility_product is not None and (
        solubility_product.cation_stoich != 1.0 or solubility_product.anion_stoich != 1.0
    ):
        raise ValueError("aqueous coupling currently requires a 1:1 Ksp hook")

    salt_cation_id = "Salt+" if solubility_product is None else solubility_product.cation_id
    salt_anion_id = "Salt-" if solubility_product is None else solubility_product.anion_id
    salt_amounts = {
        salt_cation_id: precipitating_salt_mol,
        salt_anion_id: precipitating_salt_mol,
    }
    precipitation = apply_precipitation_hooks(
        salt_amounts,
        () if solubility_product is None else (solubility_product,),
        volume_L=volume_L,
        max_passes=max_precipitation_passes,
    )
    if not precipitation.converged:
        raise RuntimeError("aqueous Ksp hooks did not converge")
    remaining_cation = precipitation.final_amounts_mol.get(salt_cation_id, 0.0)
    remaining_anion = precipitation.final_amounts_mol.get(salt_anion_id, 0.0)
    strong_cation = supporting_electrolyte_mol + remaining_cation
    strong_anion = supporting_electrolyte_mol + remaining_anion

    from chemworld.physchem.equilibrium import (
        AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K,
        DAVIES_MAX_IONIC_STRENGTH_MOL_KG,
        weak_acid_davies_activity_ratio,
    )

    initial_ionic_strength = (strong_cation + strong_anion) / (2.0 * volume_L)
    ionic_strength_value = initial_ionic_strength
    acid_base: AcidBaseResult | None = None
    ratio = 1.0
    converged = False
    iterations = 0
    for iteration in range(1, max_activity_iterations + 1):
        iterations = iteration
        ratio = weak_acid_davies_activity_ratio(
            ionic_strength_mol_kg=ionic_strength_value,
            temperature_K=temperature_K,
        )
        acid_base = solve_monoprotic_acid_base(
            acid_total_mol=acid_total_mol,
            volume_L=volume_L,
            pka=pka,
            temperature_K=temperature_K,
            strong_cation_mol=strong_cation,
            strong_anion_mol=strong_anion,
            activity_coefficient_ratio=ratio,
        )
        next_ionic_strength = acid_base.ionic_strength_mol_kg
        if abs(next_ionic_strength - ionic_strength_value) <= activity_tolerance:
            converged = True
            break
        ionic_strength_value = 0.5 * (ionic_strength_value + next_ionic_strength)
    if acid_base is None or not converged:
        raise RuntimeError("aqueous activity/electroneutrality iteration did not converge")

    acid_balance_error = abs(
        acid_base.species_amounts_mol.get("HA", 0.0)
        + acid_base.species_amounts_mol.get("A-", 0.0)
        - acid_total_mol
    )
    salt_balance_error = max(
        abs(remaining_cation + precipitation.total_precipitated_mol - precipitating_salt_mol),
        abs(remaining_anion + precipitation.total_precipitated_mol - precipitating_salt_mol),
    )
    material_error = max(
        acid_balance_error,
        salt_balance_error,
        precipitation.material_balance_error_mol,
    )
    charge_error = abs(acid_base.charge_balance_error_eq)
    if material_error > 1.0e-9 or charge_error > 1.0e-9:
        raise RuntimeError(
            "aqueous equilibrium ledger did not close: "
            f"material={material_error:.6g}, charge={charge_error:.6g}"
        )
    return AqueousElectrolyteEquilibriumResult(
        acid_base=acid_base,
        precipitation=precipitation,
        activity_coefficient_ratio=ratio,
        iterations=iterations,
        converged=True,
        material_balance_error_mol=material_error,
        charge_balance_error_eq=charge_error,
        applicability={
            "activity_model": "davies_aqueous_activity_v1",
            "temperature_min_K": AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K[0],
            "temperature_max_K": AQUEOUS_ACTIVITY_TEMPERATURE_RANGE_K[1],
            "ionic_strength_max_mol_kg": DAVIES_MAX_IONIC_STRENGTH_MOL_KG,
            "ionic_strength_mol_kg": acid_base.ionic_strength_mol_kg,
        },
    )


def equilibrium_constant_vant_hoff(
    *,
    log10_k_ref: float,
    temperature_K: float,
    reference_temperature_K: float = 298.15,
    delta_h_J_mol: float = 0.0,
) -> float:
    """Return K(T) from a reference K and reaction enthalpy."""

    _finite(log10_k_ref, "log10_k_ref")
    _positive(temperature_K, "temperature_K")
    _positive(reference_temperature_K, "reference_temperature_K")
    _finite(delta_h_J_mol, "delta_h_J_mol")
    ln_k_ref = log(10.0) * log10_k_ref
    ln_k = ln_k_ref - delta_h_J_mol / R_J_PER_MOL_K * (
        1.0 / temperature_K - 1.0 / reference_temperature_K
    )
    return exp(ln_k)


def reaction_extent_bounds(
    reaction: EquilibriumReactionSpec,
    initial_amounts_mol: Mapping[str, float],
) -> tuple[float, float]:
    """Return scalar extent bounds that preserve nonnegative amounts."""

    initial = _amounts(initial_amounts_mol)
    lower = -float("inf")
    upper = float("inf")
    for species_id, coefficient in reaction.stoichiometry.items():
        amount = initial.get(species_id, 0.0)
        if coefficient > 0.0:
            lower = max(lower, -amount / coefficient)
        elif coefficient < 0.0:
            upper = min(upper, amount / -coefficient)
    if lower == -float("inf"):
        lower = 0.0
    if upper == float("inf"):
        upper = max(sum(initial.values()), 1.0)
    if upper < lower:
        raise ValueError("No feasible reaction extent preserves nonnegative amounts")
    return lower, upper


def solve_reaction_extent(
    reaction: EquilibriumReactionSpec,
    initial_amounts_mol: Mapping[str, float],
    *,
    volume_L: float,
    temperature_K: float,
    tolerance: float = 1e-10,
) -> EquilibriumResult:
    """Solve one reversible reaction by scalar extent."""

    system = EquilibriumSystemSpec(
        system_id=reaction.reaction_id,
        reactions=(reaction,),
        temperature_K=temperature_K,
        volume_L=volume_L,
    )
    return solve_mass_action_equilibrium(
        system,
        initial_amounts_mol,
        tolerance=tolerance,
    )


def solve_mass_action_equilibrium(
    system: EquilibriumSystemSpec,
    initial_amounts_mol: Mapping[str, float],
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> EquilibriumResult:
    """Solve fixed-T, fixed-volume mass-action equilibrium in extent space."""

    initial = _complete_amounts(_amounts(initial_amounts_mol), system.species_ids)
    if len(system.reactions) == 1:
        return _solve_single_reaction(system, initial, tolerance=tolerance)

    extent_scale = max(sum(initial.values()), 1.0)
    lower = np.full(len(system.reactions), -extent_scale, dtype=float)
    upper = np.full(len(system.reactions), extent_scale, dtype=float)
    guess = np.clip(np.zeros(len(system.reactions)), lower, upper)

    def residual(x: np.ndarray) -> np.ndarray:
        amounts = _amounts_from_extents(system.reactions, initial, x.tolist())
        penalties = [
            min(0.0, amount) * 1e6
            for species_id, amount in amounts.items()
            if species_id in system.species_ids
        ]
        values = _log_residuals(system, _project_nonnegative(amounts))
        return np.asarray(values + penalties, dtype=float)

    solved = least_squares(
        residual,
        guess,
        bounds=(lower, upper),
        xtol=tolerance,
        ftol=tolerance,
        gtol=tolerance,
        max_nfev=max_iterations,
    )
    extents = solved.x.tolist()
    amounts = _project_nonnegative(_amounts_from_extents(system.reactions, initial, extents))
    residual_values = _log_residuals(system, amounts)
    return _build_equilibrium_result(
        system,
        initial,
        amounts,
        extents,
        converged=bool(solved.success and max(abs(v) for v in residual_values) < 1e-6),
        iterations=int(solved.nfev),
        metadata={"solver": "least_squares", "message": str(solved.message)},
    )


def solve_gibbs_minimization(
    spec: GibbsMinimizationSpec,
    initial_amounts_mol: Mapping[str, float],
    *,
    target_element_amounts_mol: Mapping[str, float] | None = None,
    target_charge_eq: float | None = None,
    tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> GibbsMinimizationResult:
    """Minimize ideal-mixture Gibbs energy subject to conservation constraints.

    This is a scoped benchmark solver for small fixed-TP problems. It enforces
    element balances, total charge, phase restrictions, and nonnegative species
    amounts, but it is not a database-backed Reaktoro or CALPHAD replacement.
    """

    _positive(tolerance, "tolerance")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    initial = _complete_amounts(_amounts(initial_amounts_mol), spec.species_ids)
    unknown = set(initial) - set(spec.species_ids)
    if unknown:
        raise ValueError(f"initial_amounts_mol contains unknown species: {sorted(unknown)}")

    allowed_phases = set(spec.allowed_phases or spec.phase_ids)
    if not allowed_phases:
        raise ValueError("At least one phase must be allowed")
    species = tuple(spec.species)
    species_index = {item.species_id: index for index, item in enumerate(species)}
    disallowed_initial = [
        item.species_id
        for item in species
        if item.phase not in allowed_phases and initial.get(item.species_id, 0.0) > tolerance
    ]
    if disallowed_initial:
        raise ValueError(
            f"phase restrictions exclude nonzero initial species: {sorted(disallowed_initial)}"
        )

    target_elements = (
        _element_totals(species, initial)
        if target_element_amounts_mol is None
        else _target_element_amounts(target_element_amounts_mol)
    )
    _check_allowed_species_cover_elements(species, allowed_phases, target_elements)

    charge_target = (
        target_charge_eq
        if target_charge_eq is not None
        else (
            spec.target_charge_eq
            if spec.target_charge_eq is not None
            else _charge_total(species, initial)
        )
    )
    _finite(charge_target, "target_charge_eq")

    x0 = np.asarray([initial[item.species_id] for item in species], dtype=float)
    bounds = [
        (0.0, _species_upper_bound(item, target_elements))
        if item.phase in allowed_phases
        else (0.0, 0.0)
        for item in species
    ]
    constraint_rows = _independent_constraint_rows(species, target_elements, charge_target)
    constraints = [
        {"type": "eq", "fun": _linear_constraint(row, target)} for _, row, target in constraint_rows
    ]

    solved = minimize(
        lambda values: _ideal_gibbs_energy(species, values, spec.temperature_K),
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
    )
    final_array = _project_array_nonnegative(solved.x)
    final = {
        item.species_id: float(final_array[species_index[item.species_id]]) for item in species
    }
    residuals = _element_residuals(species, final, target_elements)
    charge_residual = _charge_total(species, final) - charge_target
    max_element_residual = max((abs(value) for value in residuals.values()), default=0.0)
    converged = bool(
        solved.success
        and max_element_residual <= max(1e-8, 100.0 * tolerance)
        and abs(charge_residual) <= max(1e-8, 100.0 * tolerance)
    )
    phase_amounts = _phase_amounts(species, final)
    diagnostic = _gibbs_minimization_diagnostic(
        spec=spec,
        final_array=final_array,
        final_amounts_mol=final,
        target_elements=target_elements,
        charge_target=charge_target,
        allowed_phases=allowed_phases,
        bounds=bounds,
        constraint_rows=constraint_rows,
        tolerance=tolerance,
    )
    return GibbsMinimizationResult(
        system_id=spec.system_id,
        initial_amounts_mol=initial,
        final_amounts_mol=final,
        total_gibbs_J=_ideal_gibbs_energy(species, final_array, spec.temperature_K),
        element_balance_residuals_mol=residuals,
        charge_balance_residual_eq=charge_residual,
        phase_amounts_mol=phase_amounts,
        active_phases=tuple(
            sorted(phase for phase, amount in phase_amounts.items() if amount > 1e-12)
        ),
        converged=converged,
        iterations=int(getattr(solved, "nit", 0)),
        diagnostic=diagnostic,
        metadata={
            "solver": "scipy_slsqp_gibbs_minimization",
            "message": str(solved.message),
            "success": bool(solved.success),
            "temperature_K": spec.temperature_K,
            "pressure_Pa": spec.pressure_Pa,
            "allowed_phases": sorted(allowed_phases),
            "target_element_amounts_mol": dict(target_elements),
            "target_charge_eq": charge_target,
            "objective_model": "ideal_phase_mixture_plus_standard_gibbs",
            "diagnostic": diagnostic.to_dict(),
        },
    )


def diagnose_gibbs_minimization(
    spec: GibbsMinimizationSpec,
    result: GibbsMinimizationResult,
    *,
    tolerance: float = 1e-10,
) -> GibbsMinimizationDiagnostic:
    """Recompute the Gibbs-minimization diagnostic from a stored result."""

    _positive(tolerance, "tolerance")
    if spec.system_id != result.system_id:
        raise ValueError("spec and result system_id values do not match")
    species = tuple(spec.species)
    allowed_phases_raw = result.metadata.get("allowed_phases")
    if isinstance(allowed_phases_raw, (list, tuple, set)):
        allowed_phases = {str(item) for item in allowed_phases_raw}
    else:
        allowed_phases = set(spec.allowed_phases or spec.phase_ids)
    target_elements_raw = result.metadata.get("target_element_amounts_mol")
    if isinstance(target_elements_raw, Mapping):
        target_elements = _target_element_amounts(target_elements_raw)
    else:
        target_elements = _element_totals(species, result.initial_amounts_mol)
    charge_target_raw = result.metadata.get("target_charge_eq", spec.target_charge_eq)
    if charge_target_raw is None:
        charge_target = _charge_total(species, result.initial_amounts_mol)
    elif isinstance(charge_target_raw, int | float | str):
        charge_target = float(charge_target_raw)
    else:
        charge_target = _charge_total(species, result.initial_amounts_mol)
    _finite(charge_target, "target_charge_eq")
    bounds = [
        (0.0, _species_upper_bound(item, target_elements))
        if item.phase in allowed_phases
        else (0.0, 0.0)
        for item in species
    ]
    constraint_rows = _independent_constraint_rows(species, target_elements, charge_target)
    final_array = np.asarray(
        [float(result.final_amounts_mol.get(item.species_id, 0.0)) for item in species],
        dtype=float,
    )
    return _gibbs_minimization_diagnostic(
        spec=spec,
        final_array=final_array,
        final_amounts_mol=result.final_amounts_mol,
        target_elements=target_elements,
        charge_target=charge_target,
        allowed_phases=allowed_phases,
        bounds=bounds,
        constraint_rows=constraint_rows,
        tolerance=tolerance,
    )


def reaction_quotient(
    amounts_mol: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    *,
    volume_L: float,
    activity_floor: float = _ACTIVITY_FLOOR,
) -> float:
    """Return concentration-based reaction quotient."""

    log_q = reaction_quotient_log(
        amounts_mol,
        stoichiometry,
        volume_L=volume_L,
        activity_floor=activity_floor,
    )
    return _safe_exp(log_q)


def reaction_quotient_log(
    amounts_mol: Mapping[str, float],
    stoichiometry: Mapping[str, float],
    *,
    volume_L: float,
    activity_floor: float = _ACTIVITY_FLOOR,
) -> float:
    _positive(volume_L, "volume_L")
    _positive(activity_floor, "activity_floor")
    amounts = _amounts(amounts_mol)
    total = 0.0
    for species_id, coefficient in stoichiometry.items():
        concentration = max(amounts.get(species_id, 0.0) / volume_L, activity_floor)
        total += coefficient * log(concentration)
    return total


def water_ion_product(temperature_K: float = 298.15) -> float:
    """Return a compact water ion-product estimate, K_w = [H+][OH-]."""

    _positive(temperature_K, "temperature_K")
    pkw_table = (
        (273.15, 14.94),
        (298.15, 14.00),
        (323.15, 13.26),
        (348.15, 12.70),
        (373.15, 12.26),
    )
    if temperature_K <= pkw_table[0][0]:
        pkw = pkw_table[0][1]
    elif temperature_K >= pkw_table[-1][0]:
        pkw = pkw_table[-1][1]
    else:
        pkw = pkw_table[-1][1]
        for (t0, p0), (t1, p1) in pairwise(pkw_table):
            if t0 <= temperature_K <= t1:
                weight = (temperature_K - t0) / (t1 - t0)
                pkw = (1.0 - weight) * p0 + weight * p1
                break
    return 10.0 ** (-pkw)


def solve_monoprotic_acid_base(
    *,
    acid_total_mol: float,
    volume_L: float,
    pka: float,
    temperature_K: float = 298.15,
    strong_cation_mol: float = 0.0,
    strong_anion_mol: float = 0.0,
    activity_coefficient_ratio: float = 1.0,
    acid_id: str = "HA",
    conjugate_base_id: str = "A-",
    proton_id: str = "H+",
    hydroxide_id: str = "OH-",
    cation_id: str = "M+",
    anion_id: str = "X-",
) -> AcidBaseResult:
    """Solve a monoprotic weak-acid equilibrium with electroneutrality.

    ``activity_coefficient_ratio`` is ``gamma_H * gamma_A / gamma_HA``;
    the default of one recovers the concentration-activity model.
    """

    _nonnegative(acid_total_mol, "acid_total_mol")
    _positive(volume_L, "volume_L")
    _finite(pka, "pka")
    _positive(temperature_K, "temperature_K")
    _nonnegative(strong_cation_mol, "strong_cation_mol")
    _nonnegative(strong_anion_mol, "strong_anion_mol")
    _positive(activity_coefficient_ratio, "activity_coefficient_ratio")

    total_acid_conc = acid_total_mol / volume_L
    strong_cation_conc = strong_cation_mol / volume_L
    strong_anion_conc = strong_anion_mol / volume_L
    ka = 10.0 ** (-pka)
    kw = water_ion_product(temperature_K)

    def residual(hydrogen: float) -> float:
        conjugate_base = (
            total_acid_conc * ka / (ka + activity_coefficient_ratio * hydrogen)
            if total_acid_conc
            else 0.0
        )
        hydroxide = kw / hydrogen
        return hydrogen + strong_cation_conc - (hydroxide + conjugate_base + strong_anion_conc)

    low = 1e-14
    high = max(1.0, 10.0 * (total_acid_conc + strong_cation_conc + strong_anion_conc + ka))
    while residual(high) < 0.0:
        high *= 10.0
        if high > 1e6:
            raise ValueError("Could not bracket acid/base hydrogen concentration")
    hydrogen = brentq(residual, low, high, xtol=1e-14, rtol=1e-12, maxiter=200)
    hydroxide = kw / hydrogen
    conjugate_base = (
        total_acid_conc * ka / (ka + activity_coefficient_ratio * hydrogen)
        if total_acid_conc
        else 0.0
    )
    acid = max(total_acid_conc - conjugate_base, 0.0)
    pH = -log10(hydrogen)
    pOH = -log10(hydroxide)
    amounts = {
        acid_id: acid * volume_L,
        conjugate_base_id: conjugate_base * volume_L,
        proton_id: hydrogen * volume_L,
        hydroxide_id: hydroxide * volume_L,
    }
    charges = {acid_id: 0, conjugate_base_id: -1, proton_id: 1, hydroxide_id: -1}
    if strong_cation_mol > 0.0:
        amounts[cation_id] = strong_cation_mol
        charges[cation_id] = 1
    if strong_anion_mol > 0.0:
        amounts[anion_id] = strong_anion_mol
        charges[anion_id] = -1
    charge_error = net_charge_equivalents(amounts, charges)
    acid_dissociation_fraction = 0.0 if total_acid_conc <= 0.0 else conjugate_base / total_acid_conc
    ionic_strength_value = ionic_strength_from_amounts(
        amounts,
        charges,
        solvent_mass_kg=volume_L,
    )
    return AcidBaseResult(
        species_amounts_mol=amounts,
        pH=pH,
        pOH=pOH,
        hydrogen_mol_L=hydrogen,
        hydroxide_mol_L=hydroxide,
        acid_dissociation_fraction=acid_dissociation_fraction,
        charge_balance_error_eq=charge_error,
        ionic_strength_mol_kg=ionic_strength_value,
        metadata={
            "Ka": ka,
            "Kw": kw,
            "temperature_K": temperature_K,
            "volume_L": volume_L,
            "activity_coefficient_ratio": activity_coefficient_ratio,
        },
    )


def aqueous_ph_observation(
    result: AcidBaseResult,
    *,
    instrument_id: str = "ph_meter",
    noise_std_pH: float = 0.02,
    calibration_offset_pH: float = 0.0,
    resolution_pH: float = 0.01,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> PHObservationResult:
    """Generate a public pH-meter observation from an acid/base result.

    The observation intentionally exposes processed pH and uncertainty, not
    hidden species amounts. A Nernst-style millivolt signal is included as the
    raw signal so agents can reason about instrument calibration.
    """

    if not instrument_id:
        raise ValueError("instrument_id cannot be empty")
    _nonnegative(noise_std_pH, "noise_std_pH")
    _finite(calibration_offset_pH, "calibration_offset_pH")
    _nonnegative(resolution_pH, "resolution_pH")
    if rng is not None and seed is not None:
        raise ValueError("pH observation accepts either an RNG or a seed")
    rng = rng if rng is not None else np.random.default_rng(seed)
    noise = float(rng.normal(0.0, noise_std_pH)) if noise_std_pH > 0.0 else 0.0
    measured_pH = result.pH + calibration_offset_pH + noise
    if resolution_pH > 0.0:
        measured_pH = round(measured_pH / resolution_pH) * resolution_pH
    saturated = measured_pH < 0.0 or measured_pH > 14.0
    measured_pH = max(0.0, min(14.0, measured_pH))
    temperature_raw = result.metadata.get("temperature_K", 298.15)
    temperature_K = (
        float(temperature_raw) if isinstance(temperature_raw, int | float | str) else 298.15
    )
    nernst_slope_mV_pH = 1000.0 * R_J_PER_MOL_K * temperature_K * log(10.0) / 96485.33212
    electrode_mV = -nernst_slope_mV_pH * (measured_pH - 7.0)
    hydrogen = 10.0 ** (-measured_pH)
    hydrogen_key = (
        "hydrogen_activity"
        if result.metadata.get("activity_basis") == "molality"
        else "hydrogen_mol_L"
    )
    return PHObservationResult(
        raw_signal={
            "instrument_id": instrument_id,
            "signal_type": "potentiometric_ph",
            "electrode_mV": electrode_mV,
            "temperature_K": temperature_K,
            "resolution_pH": resolution_pH,
            "quality": "saturated" if saturated else "valid",
            "calibration": {"reference_pH": 7.0, "slope_mV_pH": nernst_slope_mV_pH},
        },
        processed_estimate={
            "pH": measured_pH,
            hydrogen_key: hydrogen,
            "ionic_strength_mol_kg": result.ionic_strength_mol_kg,
            "charge_balance_error_eq": result.charge_balance_error_eq,
        },
        uncertainty={
            "pH_std": noise_std_pH,
            "hydrogen_relative_std": log(10.0) * noise_std_pH,
            "calibration_offset_pH": abs(calibration_offset_pH),
        },
        observed_mask={
            "pH": True,
            hydrogen_key: True,
            "ionic_strength_mol_kg": True,
            "charge_balance_error_eq": True,
            "species_amounts_mol": False,
            "pka": False,
        },
        metadata={
            "source": "aqueous_ph_observation",
            "visibility": "public_processed_observation",
            "noise_model": "normal_pH_noise_then_resolution_rounding",
        },
    )


def precipitate_if_supersaturated(
    amounts_mol: Mapping[str, float],
    spec: SolubilityProductSpec,
    *,
    volume_L: float,
) -> PrecipitationResult:
    """Apply a binary solubility-product precipitation step."""

    amounts = _amounts(amounts_mol)
    _positive(volume_L, "volume_L")
    initial_cation = amounts.get(spec.cation_id, 0.0)
    initial_anion = amounts.get(spec.anion_id, 0.0)
    initial_product = _ion_product(
        initial_cation,
        initial_anion,
        spec=spec,
        volume_L=volume_L,
    )
    initial_saturation_index = log10(max(initial_product, _ACTIVITY_FLOOR) / spec.ksp)
    if initial_product <= spec.ksp:
        final = dict(amounts)
        final.setdefault(spec.precipitate_id, amounts.get(spec.precipitate_id, 0.0))
        return PrecipitationResult(
            final_amounts_mol=final,
            precipitated_mol=0.0,
            ion_product=initial_product,
            saturation_index=initial_saturation_index,
            material_balance_error_mol=0.0,
            metadata={"status": "undersaturated_or_saturated", "spec": spec.to_dict()},
        )

    max_precip = min(initial_cation / spec.cation_stoich, initial_anion / spec.anion_stoich)
    if max_precip <= 0.0:
        return PrecipitationResult(
            final_amounts_mol=dict(amounts),
            precipitated_mol=0.0,
            ion_product=initial_product,
            saturation_index=initial_saturation_index,
            material_balance_error_mol=0.0,
            metadata={"status": "missing_ion", "spec": spec.to_dict()},
        )

    def residual(precipitated: float) -> float:
        cation = max(initial_cation - spec.cation_stoich * precipitated, 0.0)
        anion = max(initial_anion - spec.anion_stoich * precipitated, 0.0)
        return _ion_product(cation, anion, spec=spec, volume_L=volume_L) - spec.ksp

    # The exact stoichiometric limit is a valid closed endpoint: at least one
    # limiting ion is zero and the ion product is therefore no greater than
    # Ksp.  Pulling the endpoint inward by a fixed relative epsilon can leave
    # both residuals positive for very small Ksp or strongly imbalanced ion
    # inventories, which violates Brent's sign-change requirement.
    upper = max_precip
    upper_residual = residual(upper)
    if upper_residual >= 0.0:
        precipitated = upper
        solver_status = "stoichiometric_limit"
    else:
        precipitated = brentq(
            residual,
            0.0,
            upper,
            xtol=1e-14,
            rtol=1e-12,
            maxiter=200,
        )
        solver_status = "brentq"
    final = dict(amounts)
    final[spec.cation_id] = max(initial_cation - spec.cation_stoich * precipitated, 0.0)
    final[spec.anion_id] = max(initial_anion - spec.anion_stoich * precipitated, 0.0)
    final[spec.precipitate_id] = final.get(spec.precipitate_id, 0.0) + precipitated
    final_product = _ion_product(
        final[spec.cation_id],
        final[spec.anion_id],
        spec=spec,
        volume_L=volume_L,
    )
    balance_error = abs(
        (initial_cation + initial_anion)
        - (
            final[spec.cation_id]
            + final[spec.anion_id]
            + precipitated * (spec.cation_stoich + spec.anion_stoich)
        )
    )
    return PrecipitationResult(
        final_amounts_mol=final,
        precipitated_mol=precipitated,
        ion_product=final_product,
        saturation_index=log10(max(final_product, _ACTIVITY_FLOOR) / spec.ksp),
        material_balance_error_mol=balance_error,
        metadata={
            "status": "precipitated",
            "initial_ion_product": initial_product,
            "initial_saturation_index": initial_saturation_index,
            "solver_status": solver_status,
            "spec": spec.to_dict(),
        },
    )


def apply_precipitation_hooks(
    amounts_mol: Mapping[str, float],
    specs: tuple[SolubilityProductSpec, ...] | list[SolubilityProductSpec],
    *,
    volume_L: float,
    max_passes: int = 3,
) -> PrecipitationHookResult:
    """Apply sequential solubility-product hooks to an aqueous amount ledger."""

    amounts = _amounts(amounts_mol)
    _positive(volume_L, "volume_L")
    if max_passes <= 0:
        raise ValueError("max_passes must be positive")
    events: list[dict[str, object]] = []
    total_precipitated = 0.0
    balance_error = 0.0
    specs_tuple = tuple(specs)
    for pass_index in range(max_passes):
        changed = False
        for spec in specs_tuple:
            result = precipitate_if_supersaturated(amounts, spec, volume_L=volume_L)
            amounts = result.final_amounts_mol
            balance_error += result.material_balance_error_mol
            if result.precipitated_mol > 0.0:
                changed = True
                total_precipitated += result.precipitated_mol
                events.append(
                    {
                        "pass_index": pass_index,
                        "precipitate_id": spec.precipitate_id,
                        "cation_id": spec.cation_id,
                        "anion_id": spec.anion_id,
                        "precipitated_mol": result.precipitated_mol,
                        "ion_product": result.ion_product,
                        "saturation_index": result.saturation_index,
                        "status": result.metadata.get("status", "unknown"),
                    }
                )
        if not changed:
            return PrecipitationHookResult(
                final_amounts_mol=amounts,
                precipitation_events=tuple(events),
                total_precipitated_mol=total_precipitated,
                material_balance_error_mol=balance_error,
                metadata={
                    "status": "converged",
                    "passes": pass_index + 1,
                    "spec_count": len(specs_tuple),
                },
            )
    return PrecipitationHookResult(
        final_amounts_mol=amounts,
        precipitation_events=tuple(events),
        total_precipitated_mol=total_precipitated,
        material_balance_error_mol=balance_error,
        metadata={
            "status": "max_passes_reached",
            "passes": max_passes,
            "spec_count": len(specs_tuple),
        },
    )


def net_charge_equivalents(
    amounts_mol: Mapping[str, float],
    charges: Mapping[str, int | float],
) -> float:
    """Return total charge equivalents in moles."""

    amounts = _amounts(amounts_mol)
    return sum(
        amounts.get(species_id, 0.0) * float(charges.get(species_id, 0.0)) for species_id in amounts
    )


def balance_charge_by_adjusting_ion(
    amounts_mol: Mapping[str, float],
    charges: Mapping[str, int | float],
    *,
    adjustable_species_id: str,
) -> ChargeBalanceResult:
    """Adjust one charged species to enforce electroneutrality."""

    amounts = _amounts(amounts_mol)
    if adjustable_species_id not in charges:
        raise ValueError("adjustable species must have a declared charge")
    charge = float(charges[adjustable_species_id])
    if charge == 0.0:
        raise ValueError("adjustable species must be charged")
    initial_charge = net_charge_equivalents(amounts, charges)
    adjustment = -initial_charge / charge
    new_amount = amounts.get(adjustable_species_id, 0.0) + adjustment
    if new_amount < -1e-12:
        raise ValueError("charge-balance adjustment would make amount negative")
    adjusted = dict(amounts)
    adjusted[adjustable_species_id] = max(new_amount, 0.0)
    final_charge = net_charge_equivalents(adjusted, charges)
    return ChargeBalanceResult(
        adjusted_amounts_mol=adjusted,
        initial_charge_eq=initial_charge,
        final_charge_eq=final_charge,
        adjustment_mol=adjustment,
        adjusted_species_id=adjustable_species_id,
        metadata={"method": "single_ion_adjustment"},
    )


def ionic_strength(
    molalities_mol_kg: Mapping[str, float],
    charges: Mapping[str, int | float],
) -> float:
    """Return ionic strength from molalities and charges."""

    molalities = _amounts(molalities_mol_kg)
    return 0.5 * sum(
        molality * float(charges.get(species_id, 0.0)) ** 2
        for species_id, molality in molalities.items()
    )


def ionic_strength_from_amounts(
    amounts_mol: Mapping[str, float],
    charges: Mapping[str, int | float],
    *,
    solvent_mass_kg: float,
) -> float:
    """Return ionic strength using amounts and solvent mass."""

    _positive(solvent_mass_kg, "solvent_mass_kg")
    molalities = {
        species_id: amount / solvent_mass_kg
        for species_id, amount in _amounts(amounts_mol).items()
        if float(charges.get(species_id, 0.0)) != 0.0
    }
    return ionic_strength(molalities, charges)


def solid_solubility_mole_fraction(
    *,
    temperature_K: float,
    melting_temperature_K: float,
    enthalpy_fusion_J_mol: float,
    liquid_heat_capacity_J_mol_K: float = 0.0,
    solid_heat_capacity_J_mol_K: float = 0.0,
    activity_coefficient: float = 1.0,
) -> float:
    """Shared eutectic kernel; impossible mole fractions are no longer clipped."""
    from chemworld.physchem.solubility import solid_solubility

    return solid_solubility(
        temperature_K, melting_temperature_K, enthalpy_fusion_J_mol,
        liquid_heat_capacity_J_mol_K, solid_heat_capacity_J_mol_K, activity_coefficient,
    )


def _element_totals(
    species: tuple[GibbsSpeciesSpec, ...],
    amounts_mol: Mapping[str, float],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in species:
        amount = float(amounts_mol.get(item.species_id, 0.0))
        for element_id, count in item.element_counts.items():
            totals[element_id] = totals.get(element_id, 0.0) + amount * float(count)
    return totals


def _target_element_amounts(amounts_mol: Mapping[str, float]) -> dict[str, float]:
    if not amounts_mol:
        raise ValueError("target_element_amounts_mol cannot be empty")
    targets = {key: float(value) for key, value in amounts_mol.items()}
    for element_id, amount in targets.items():
        if not element_id:
            raise ValueError("target_element_amounts_mol contains an empty element id")
        _nonnegative(amount, f"target_element_amounts_mol[{element_id}]")
    return targets


def _check_allowed_species_cover_elements(
    species: tuple[GibbsSpeciesSpec, ...],
    allowed_phases: set[str],
    target_elements: Mapping[str, float],
) -> None:
    covered = {
        element_id
        for item in species
        if item.phase in allowed_phases
        for element_id, count in item.element_counts.items()
        if count > 0.0
    }
    missing = [
        element_id
        for element_id, amount in target_elements.items()
        if amount > 0.0 and element_id not in covered
    ]
    if missing:
        raise ValueError(f"Allowed phases cannot carry target elements: {sorted(missing)}")


def _species_upper_bound(
    species: GibbsSpeciesSpec,
    target_elements: Mapping[str, float],
) -> float | None:
    bounds = [
        float(target_elements[element_id]) / float(count)
        for element_id, count in species.element_counts.items()
        if count > 0.0 and element_id in target_elements
    ]
    if not bounds:
        return None
    return max(0.0, min(bounds))


def _independent_constraint_rows(
    species: tuple[GibbsSpeciesSpec, ...],
    target_elements: Mapping[str, float],
    charge_target: float,
) -> list[tuple[str, np.ndarray, float]]:
    rows: list[tuple[str, np.ndarray, float]] = []
    for element_id, target in sorted(target_elements.items()):
        row = np.asarray(
            [float(item.element_counts.get(element_id, 0.0)) for item in species],
            dtype=float,
        )
        rows.append((f"element:{element_id}", row, float(target)))
    rows.append(
        (
            "charge",
            np.asarray([float(item.charge) for item in species], dtype=float),
            float(charge_target),
        )
    )

    selected: list[tuple[str, np.ndarray, float]] = []
    selected_rows: list[np.ndarray] = []
    current_rank = 0
    for constraint_id, row, target in rows:
        if np.allclose(row, 0.0, atol=1e-14):
            if abs(target) > 1e-12:
                raise ValueError(f"Constraint {constraint_id} has no species carrier")
            continue
        candidate = np.vstack([*selected_rows, row])
        candidate_rank = int(np.linalg.matrix_rank(candidate, tol=1e-12))
        if candidate_rank > current_rank:
            selected.append((constraint_id, row, target))
            selected_rows.append(row)
            current_rank = candidate_rank
    return selected


def _linear_constraint(row: np.ndarray, target_amount_mol: float) -> Callable[[np.ndarray], float]:
    def constraint(values: np.ndarray) -> float:
        return float(np.dot(row, values) - target_amount_mol)

    return constraint


def _gibbs_minimization_diagnostic(
    *,
    spec: GibbsMinimizationSpec,
    final_array: np.ndarray,
    final_amounts_mol: Mapping[str, float],
    target_elements: Mapping[str, float],
    charge_target: float,
    allowed_phases: set[str],
    bounds: list[tuple[float, float | None]],
    constraint_rows: list[tuple[str, np.ndarray, float]],
    tolerance: float,
) -> GibbsMinimizationDiagnostic:
    species = tuple(spec.species)
    residuals = _element_residuals(species, final_amounts_mol, target_elements)
    max_element_residual = max((abs(value) for value in residuals.values()), default=0.0)
    charge_residual = _charge_total(species, final_amounts_mol) - charge_target
    active_tol = max(1e-12, 100.0 * tolerance)

    max_bound_violation = 0.0
    active_species: list[str] = []
    active_lower: list[str] = []
    active_upper: list[str] = []
    allowed_indices: list[int] = []
    free_indices: list[int] = []
    for index, item in enumerate(species):
        amount = float(final_array[index])
        lower, upper = bounds[index]
        if item.phase in allowed_phases:
            allowed_indices.append(index)
        if amount > active_tol:
            active_species.append(item.species_id)
        if amount <= lower + active_tol:
            active_lower.append(item.species_id)
        if upper is not None and amount >= upper - active_tol:
            active_upper.append(item.species_id)
        if amount < lower:
            max_bound_violation = max(max_bound_violation, lower - amount)
        if upper is not None and amount > upper:
            max_bound_violation = max(max_bound_violation, amount - upper)
        if (
            item.phase in allowed_phases
            and amount > lower + active_tol
            and (upper is None or amount < upper - active_tol)
        ):
            free_indices.append(index)

    constraint_matrix = (
        np.vstack([row for _, row, _ in constraint_rows])
        if constraint_rows
        else np.zeros((0, len(species)))
    )
    constraint_rank = (
        int(np.linalg.matrix_rank(constraint_matrix, tol=1e-12)) if constraint_matrix.size else 0
    )
    degrees_of_freedom = max(0, len(allowed_indices) - constraint_rank)
    stationarity_residual = _gibbs_stationarity_residual(
        species,
        final_array,
        spec.temperature_K,
        constraint_matrix,
        free_indices,
    )
    notes = _gibbs_diagnostic_notes(
        species,
        allowed_phases,
        free_indices,
        degrees_of_freedom,
    )
    residual_limit = max(1e-8, 100.0 * tolerance)
    stationarity_limit = max(1e-5, 1e6 * tolerance)
    if max_element_residual <= residual_limit and abs(charge_residual) <= residual_limit:
        status: Literal["ok", "warning", "failed"] = (
            "ok"
            if max_bound_violation <= residual_limit and stationarity_residual <= stationarity_limit
            else "warning"
        )
    else:
        status = "failed"
    return GibbsMinimizationDiagnostic(
        status=status,
        max_element_residual_mol=max_element_residual,
        charge_residual_eq=charge_residual,
        max_bound_violation_mol=max_bound_violation,
        stationarity_residual_J_mol=stationarity_residual,
        constraint_matrix_rank=constraint_rank,
        allowed_species_count=len(allowed_indices),
        degrees_of_freedom=degrees_of_freedom,
        active_species_ids=tuple(sorted(active_species)),
        active_lower_bound_species_ids=tuple(sorted(active_lower)),
        active_upper_bound_species_ids=tuple(sorted(active_upper)),
        convexity_class=_gibbs_convexity_class(species, allowed_phases),
        notes=notes,
    )


def _gibbs_stationarity_residual(
    species: tuple[GibbsSpeciesSpec, ...],
    values: np.ndarray,
    temperature_K: float,
    constraint_matrix: np.ndarray,
    free_indices: list[int],
) -> float:
    if not free_indices:
        return 0.0
    chemical_potentials = _gibbs_chemical_potentials(species, values, temperature_K)
    free_mu = chemical_potentials[free_indices]
    if constraint_matrix.size:
        free_constraints = constraint_matrix[:, free_indices].T
        multipliers, *_ = np.linalg.lstsq(free_constraints, -free_mu, rcond=None)
        residual = free_constraints @ multipliers + free_mu
    else:
        residual = free_mu
    return float(np.max(np.abs(residual))) if residual.size else 0.0


def _gibbs_chemical_potentials(
    species: tuple[GibbsSpeciesSpec, ...],
    values: np.ndarray,
    temperature_K: float,
) -> np.ndarray:
    phase_totals: dict[str, float] = {}
    for index, item in enumerate(species):
        amount = max(float(values[index]), 0.0)
        phase_totals[item.phase] = phase_totals.get(item.phase, 0.0) + amount

    potentials = np.zeros(len(species), dtype=float)
    for index, item in enumerate(species):
        potential = item.standard_gibbs_J_mol
        if not _pure_condensed_phase(item.phase):
            amount = max(float(values[index]), _ACTIVITY_FLOOR)
            phase_total = max(phase_totals.get(item.phase, 0.0), _ACTIVITY_FLOOR)
            potential += (
                R_J_PER_MOL_K * temperature_K * log(max(amount / phase_total, _ACTIVITY_FLOOR))
            )
        potentials[index] = potential
    return potentials


def _gibbs_convexity_class(
    species: tuple[GibbsSpeciesSpec, ...],
    allowed_phases: set[str],
) -> str:
    allowed = [item for item in species if item.phase in allowed_phases]
    if any(_pure_condensed_phase(item.phase) for item in allowed):
        return "convex_with_linear_pure_condensed_phase_terms"
    return "strictly_convex_ideal_mixture_over_linear_constraints"


def _gibbs_diagnostic_notes(
    species: tuple[GibbsSpeciesSpec, ...],
    allowed_phases: set[str],
    free_indices: list[int],
    degrees_of_freedom: int,
) -> tuple[str, ...]:
    notes: list[str] = []
    allowed = [item for item in species if item.phase in allowed_phases]
    if any(_pure_condensed_phase(item.phase) for item in allowed):
        notes.append(
            "Pure condensed species have linear activity terms; multiple boundary "
            "optima can be degenerate."
        )
    if not free_indices:
        notes.append("All allowed species are active on a lower or upper bound.")
    if degrees_of_freedom == 0:
        notes.append("Linear conservation constraints leave no free composition degree of freedom.")
    return tuple(notes)


def _charge_total(
    species: tuple[GibbsSpeciesSpec, ...],
    amounts_mol: Mapping[str, float],
) -> float:
    return sum(item.charge * float(amounts_mol.get(item.species_id, 0.0)) for item in species)


def _ideal_gibbs_energy(
    species: tuple[GibbsSpeciesSpec, ...],
    values: np.ndarray,
    temperature_K: float,
) -> float:
    phase_totals: dict[str, float] = {}
    for index, item in enumerate(species):
        amount = max(float(values[index]), 0.0)
        phase_totals[item.phase] = phase_totals.get(item.phase, 0.0) + amount

    total = 0.0
    for index, item in enumerate(species):
        amount = max(float(values[index]), 0.0)
        if amount <= 0.0:
            continue
        total += amount * item.standard_gibbs_J_mol
        if not _pure_condensed_phase(item.phase):
            phase_total = max(phase_totals.get(item.phase, 0.0), _ACTIVITY_FLOOR)
            activity = max(amount / phase_total, _ACTIVITY_FLOOR)
            total += R_J_PER_MOL_K * temperature_K * amount * log(activity)
    return total


def _pure_condensed_phase(phase: str) -> bool:
    phase_lower = phase.lower()
    return phase_lower in {"solid", "pure_solid", "crystal"} or "solid" in phase_lower


def _element_residuals(
    species: tuple[GibbsSpeciesSpec, ...],
    amounts_mol: Mapping[str, float],
    target_elements: Mapping[str, float],
) -> dict[str, float]:
    totals = _element_totals(species, amounts_mol)
    return {
        element_id: totals.get(element_id, 0.0) - target
        for element_id, target in sorted(target_elements.items())
    }


def _phase_amounts(
    species: tuple[GibbsSpeciesSpec, ...],
    amounts_mol: Mapping[str, float],
) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in species:
        totals[item.phase] = totals.get(item.phase, 0.0) + amounts_mol.get(item.species_id, 0.0)
    return dict(sorted(totals.items()))


def _project_array_nonnegative(values: np.ndarray) -> np.ndarray:
    projected = np.asarray(values, dtype=float).copy()
    projected[np.abs(projected) < 1e-12] = 0.0
    projected[projected < 0.0] = 0.0
    return projected


def _solve_single_reaction(
    system: EquilibriumSystemSpec,
    initial: dict[str, float],
    *,
    tolerance: float,
) -> EquilibriumResult:
    reaction = system.reactions[0]
    lower, upper = reaction_extent_bounds(reaction, initial)
    span = max(upper - lower, 1.0)
    lo = lower + 1e-12 * span
    hi = upper - 1e-12 * span

    def residual(extent: float) -> float:
        amounts = _amounts_from_extents((reaction,), initial, [extent])
        return _log_residuals(system, amounts)[0]

    f_lo = residual(lo)
    f_hi = residual(hi)
    if abs(f_lo) < tolerance:
        extent = lo
        converged = True
    elif abs(f_hi) < tolerance:
        extent = hi
        converged = True
    elif f_lo * f_hi < 0.0:
        extent = float(brentq(residual, lo, hi, xtol=tolerance, rtol=tolerance, maxiter=200))
        converged = True
    else:
        candidates = (lo, hi, max(lower, min(0.0, upper)))
        extent = min(candidates, key=lambda value: abs(residual(value)))
        converged = abs(residual(extent)) < 1e-6
    amounts = _project_nonnegative(_amounts_from_extents((reaction,), initial, [extent]))
    return _build_equilibrium_result(
        system,
        initial,
        amounts,
        [extent],
        converged=converged,
        iterations=1,
        metadata={"solver": "brentq_extent" if converged else "bounded_residual_min"},
    )


def _build_equilibrium_result(
    system: EquilibriumSystemSpec,
    initial: dict[str, float],
    amounts: dict[str, float],
    extents: list[float],
    *,
    converged: bool,
    iterations: int,
    metadata: dict[str, object],
) -> EquilibriumResult:
    constants = {
        reaction.reaction_id: reaction.equilibrium_constant(system.temperature_K)
        for reaction in system.reactions
    }
    quotients = {
        reaction.reaction_id: reaction_quotient(
            amounts,
            reaction.stoichiometry,
            volume_L=system.volume_L,
        )
        for reaction in system.reactions
    }
    residual_values = _log_residuals(system, amounts)
    return EquilibriumResult(
        system_id=system.system_id,
        initial_amounts_mol=dict(initial),
        final_amounts_mol={key: amounts.get(key, 0.0) for key in sorted(amounts)},
        extents_mol={
            reaction.reaction_id: extents[index] for index, reaction in enumerate(system.reactions)
        },
        equilibrium_constants=constants,
        reaction_quotients=quotients,
        residuals_log={
            reaction.reaction_id: residual_values[index]
            for index, reaction in enumerate(system.reactions)
        },
        converged=converged,
        iterations=iterations,
        metadata={
            **metadata,
            "temperature_K": system.temperature_K,
            "pressure_Pa": system.pressure_Pa,
            "volume_L": system.volume_L,
        },
    )


def _log_residuals(system: EquilibriumSystemSpec, amounts: Mapping[str, float]) -> list[float]:
    values = []
    for reaction in system.reactions:
        log_q = reaction_quotient_log(
            amounts,
            reaction.stoichiometry,
            volume_L=system.volume_L,
        )
        log_k = log(max(reaction.equilibrium_constant(system.temperature_K), _ACTIVITY_FLOOR))
        values.append(log_q - log_k)
    return values


def _amounts_from_extents(
    reactions: tuple[EquilibriumReactionSpec, ...],
    initial: Mapping[str, float],
    extents: list[float],
) -> dict[str, float]:
    amounts = dict(initial)
    for reaction, extent in zip(reactions, extents, strict=True):
        for species_id, coefficient in reaction.stoichiometry.items():
            amounts[species_id] = amounts.get(species_id, 0.0) + coefficient * extent
    return amounts


def _complete_amounts(amounts: dict[str, float], species_ids: tuple[str, ...]) -> dict[str, float]:
    complete = dict(amounts)
    for species_id in species_ids:
        complete.setdefault(species_id, 0.0)
    return complete


def _ion_product(
    cation_mol: float,
    anion_mol: float,
    *,
    spec: SolubilityProductSpec,
    volume_L: float,
) -> float:
    cation_conc = max(cation_mol / volume_L, 0.0)
    anion_conc = max(anion_mol / volume_L, 0.0)
    return cation_conc**spec.cation_stoich * anion_conc**spec.anion_stoich


def _amounts(amounts_mol: Mapping[str, float]) -> dict[str, float]:
    amounts = {key: float(value) for key, value in amounts_mol.items()}
    _validate_amounts(amounts, "amounts_mol")
    return amounts


def _validate_amounts(amounts_mol: Mapping[str, float], name: str) -> None:
    for species_id, amount in amounts_mol.items():
        if not species_id:
            raise ValueError(f"{name} contains an empty species id")
        _nonnegative(float(amount), f"{name}[{species_id}]")


def _project_nonnegative(amounts: Mapping[str, float]) -> dict[str, float]:
    return {
        species_id: 0.0 if abs(amount) < 1e-12 else max(amount, 0.0)
        for species_id, amount in amounts.items()
    }


def _safe_exp(value: float) -> float:
    if value > 700.0:
        return exp(700.0)
    if value < -700.0:
        return exp(-700.0)
    return exp(value)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite(value: float, name: str) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _positive(value: float, name: str) -> None:
    _finite(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")


def _nonnegative(value: float, name: str) -> None:
    _finite(value, name)
    if value < 0.0:
        raise ValueError(f"{name} cannot be negative")
