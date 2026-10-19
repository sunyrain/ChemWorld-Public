"""Cooling crystallization with solubility, kinetics, and CSD metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isfinite, pi, sqrt

R_J_PER_MOL_K = 8.31446261815324
DEFAULT_MAXIMUM_COOLING_RATE_K_S = 0.25
SURFACE_SATURATION_MAX_LOADING_RATIO = 4.0


@dataclass(frozen=True)
class SolubilityCurveSpec:
    model_id: str
    reference_solubility_mol_L: float
    reference_temperature_K: float
    dissolution_enthalpy_J_mol: float
    minimum_temperature_K: float
    maximum_temperature_K: float
    provenance_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_solubility_mol_L", self.reference_solubility_mol_L),
            ("reference_temperature_K", self.reference_temperature_K),
            ("minimum_temperature_K", self.minimum_temperature_K),
            ("maximum_temperature_K", self.maximum_temperature_K),
        ):
            _positive_finite(value, name)
        if self.maximum_temperature_K <= self.minimum_temperature_K:
            raise ValueError("solubility temperature range must be increasing")
        if not (
            self.minimum_temperature_K <= self.reference_temperature_K <= self.maximum_temperature_K
        ):
            raise ValueError("reference temperature must lie inside the validity range")
        if not isfinite(self.dissolution_enthalpy_J_mol):
            raise ValueError("dissolution_enthalpy_J_mol must be finite")
        if not self.model_id or not self.provenance_id:
            raise ValueError("model_id and provenance_id cannot be empty")

    def solubility_mol_per_l(self, temperature_K: float) -> float:
        if not self.minimum_temperature_K <= temperature_K <= self.maximum_temperature_K:
            raise ValueError("temperature_K is outside the solubility-curve range")
        exponent = (
            -self.dissolution_enthalpy_J_mol
            / R_J_PER_MOL_K
            * (1.0 / temperature_K - 1.0 / self.reference_temperature_K)
        )
        value = self.reference_solubility_mol_L * exp(exponent)
        if value <= 0.0 or not isfinite(value):
            raise ValueError("solubility curve produced a nonphysical value")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "reference_solubility_mol_L": self.reference_solubility_mol_L,
            "reference_temperature_K": self.reference_temperature_K,
            "dissolution_enthalpy_J_mol": self.dissolution_enthalpy_J_mol,
            "minimum_temperature_K": self.minimum_temperature_K,
            "maximum_temperature_K": self.maximum_temperature_K,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class CrystallizationKineticsSpec:
    model_id: str
    primary_nucleation_coefficient_per_L_s: float
    primary_nucleation_exponent: float
    growth_coefficient_m_s: float
    growth_exponent: float
    crystal_density_kg_m3: float
    target_molecular_weight_kg_mol: float
    nucleus_diameter_m: float
    impurity_occlusion_mol_per_mol: float = 0.0
    supersaturation_occlusion_factor: float = 0.0
    impurity_occlusion_law_id: str = "linear_supersaturation_transfer_v1"
    impurity_surface_half_saturation_mol_L: float = 0.010
    fines_threshold_m: float = 20.0e-6
    provenance_id: str = ""

    def __post_init__(self) -> None:
        for name, value in (
            (
                "primary_nucleation_coefficient_per_L_s",
                self.primary_nucleation_coefficient_per_L_s,
            ),
            ("growth_coefficient_m_s", self.growth_coefficient_m_s),
            ("impurity_occlusion_mol_per_mol", self.impurity_occlusion_mol_per_mol),
            (
                "supersaturation_occlusion_factor",
                self.supersaturation_occlusion_factor,
            ),
        ):
            _nonnegative_finite(value, name)
        if self.impurity_occlusion_law_id not in {
            "linear_supersaturation_transfer_v1",
            "surface_saturation_occlusion_v1",
        }:
            raise ValueError("unsupported impurity_occlusion_law_id")
        for name, value in (
            ("primary_nucleation_exponent", self.primary_nucleation_exponent),
            ("growth_exponent", self.growth_exponent),
            ("crystal_density_kg_m3", self.crystal_density_kg_m3),
            ("target_molecular_weight_kg_mol", self.target_molecular_weight_kg_mol),
            ("nucleus_diameter_m", self.nucleus_diameter_m),
            (
                "impurity_surface_half_saturation_mol_L",
                self.impurity_surface_half_saturation_mol_L,
            ),
            ("fines_threshold_m", self.fines_threshold_m),
        ):
            _positive_finite(value, name)
        if not self.model_id or not self.provenance_id:
            raise ValueError("model_id and provenance_id cannot be empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "primary_nucleation_coefficient_per_L_s": (self.primary_nucleation_coefficient_per_L_s),
            "primary_nucleation_exponent": self.primary_nucleation_exponent,
            "growth_coefficient_m_s": self.growth_coefficient_m_s,
            "growth_exponent": self.growth_exponent,
            "crystal_density_kg_m3": self.crystal_density_kg_m3,
            "target_molecular_weight_kg_mol": self.target_molecular_weight_kg_mol,
            "nucleus_diameter_m": self.nucleus_diameter_m,
            "impurity_occlusion_mol_per_mol": self.impurity_occlusion_mol_per_mol,
            "supersaturation_occlusion_factor": (self.supersaturation_occlusion_factor),
            "impurity_occlusion_law_id": self.impurity_occlusion_law_id,
            "impurity_surface_half_saturation_mol_L": (
                self.impurity_surface_half_saturation_mol_L
            ),
            "fines_threshold_m": self.fines_threshold_m,
            "provenance_id": self.provenance_id,
        }


@dataclass(frozen=True)
class CrystallizationExecutionSpec:
    """Numerical and physical acceptance policy for a formal PBM execution."""

    maximum_cooling_rate_K_s: float = DEFAULT_MAXIMUM_COOLING_RATE_K_S
    minimum_target_amount_mol: float = 1.0e-10
    minimum_effective_seed_particles: float = 10.0
    minimum_crystallized_from_solution_mol: float = 1.0e-10
    max_growth_solver_iterations: int = 80
    growth_solver_tolerance_mol: float = 1.0e-12
    material_balance_tolerance_mol: float = 1.0e-10
    particle_target_balance_tolerance_mol: float = 1.0e-10
    fail_on_no_population: bool = False
    fail_on_no_transfer: bool = False
    fail_on_solver_nonconvergence: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "maximum_cooling_rate_K_s",
            "minimum_target_amount_mol",
            "minimum_effective_seed_particles",
            "minimum_crystallized_from_solution_mol",
            "growth_solver_tolerance_mol",
            "material_balance_tolerance_mol",
            "particle_target_balance_tolerance_mol",
        ):
            _positive_finite(float(getattr(self, field_name)), field_name)
        if not isinstance(self.max_growth_solver_iterations, int):
            raise ValueError("max_growth_solver_iterations must be an integer")
        if self.max_growth_solver_iterations < 1:
            raise ValueError("max_growth_solver_iterations must be positive")

    @classmethod
    def strict_runtime(cls) -> CrystallizationExecutionSpec:
        return cls(
            fail_on_no_population=True,
            fail_on_no_transfer=True,
            fail_on_solver_nonconvergence=True,
        )

    @classmethod
    def closed_loop_runtime(cls) -> CrystallizationExecutionSpec:
        """Fail on numerical invalidity while retaining valid negative experiments.

        A closed-loop learner must be able to observe that a physically legal
        cooling program produced no population or no material transfer.  Those
        outcomes are data, not runtime-domain errors.  Solver nonconvergence
        remains fail-closed.
        """

        return cls(
            fail_on_no_population=False,
            fail_on_no_transfer=False,
            fail_on_solver_nonconvergence=True,
        )

    def to_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class CrystalSizeDistribution:
    total_particle_count: float
    number_mean_diameter_m: float
    number_std_diameter_m: float
    coefficient_of_variation: float
    d10_m: float
    d50_m: float
    d90_m: float
    fines_number_fraction: float
    cohort_count: int
    number_moment_0: float
    length_moment_1_m: float
    area_moment_2_m2: float
    volume_moment_3_m3: float

    def to_dict(self) -> dict[str, object]:
        return {
            "total_particle_count": self.total_particle_count,
            "number_mean_diameter_m": self.number_mean_diameter_m,
            "number_std_diameter_m": self.number_std_diameter_m,
            "coefficient_of_variation": self.coefficient_of_variation,
            "d10_m": self.d10_m,
            "d50_m": self.d50_m,
            "d90_m": self.d90_m,
            "fines_number_fraction": self.fines_number_fraction,
            "cohort_count": self.cohort_count,
            "number_moment_0": self.number_moment_0,
            "length_moment_1_m": self.length_moment_1_m,
            "area_moment_2_m2": self.area_moment_2_m2,
            "volume_moment_3_m3": self.volume_moment_3_m3,
        }


@dataclass(frozen=True)
class CrystallizationStepReport:
    time_s: float
    temperature_K: float
    solubility_mol_L: float
    dissolved_target_concentration_mol_L: float
    supersaturation_ratio: float
    relative_supersaturation: float
    nucleation_rate_per_L_s: float
    growth_rate_m_s: float
    nucleated_particle_count: float
    target_crystallized_step_mol: float
    impurity_occluded_step_mol: float

    def to_dict(self) -> dict[str, float]:
        return {
            "time_s": self.time_s,
            "temperature_K": self.temperature_K,
            "solubility_mol_L": self.solubility_mol_L,
            "dissolved_target_concentration_mol_L": (self.dissolved_target_concentration_mol_L),
            "supersaturation_ratio": self.supersaturation_ratio,
            "relative_supersaturation": self.relative_supersaturation,
            "nucleation_rate_per_L_s": self.nucleation_rate_per_L_s,
            "growth_rate_m_s": self.growth_rate_m_s,
            "nucleated_particle_count": self.nucleated_particle_count,
            "target_crystallized_step_mol": self.target_crystallized_step_mol,
            "impurity_occluded_step_mol": self.impurity_occluded_step_mol,
        }


@dataclass(frozen=True)
class CoolingCrystallizationResult:
    model_id: str
    target_component: str
    impurity_component: str | None
    initial_temperature_K: float
    final_temperature_K: float
    duration_s: float
    feed_amounts_mol: dict[str, float]
    seed_target_mol: float
    crystals_amounts_mol: dict[str, float]
    mother_liquor_amounts_mol: dict[str, float]
    crystallized_from_solution_mol: float
    impurity_occluded_mol: float
    target_recovery: float
    crystal_purity: float
    maximum_supersaturation_ratio: float
    final_supersaturation_ratio: float
    material_balance_error_mol: float
    component_balance_errors_mol: dict[str, float]
    particle_target_balance_error_mol: float
    cooling_rate_K_s: float
    growth_solver_converged: bool
    growth_solver_iterations: int
    growth_solver_max_residual_mol: float
    crystal_size_distribution: CrystalSizeDistribution
    step_reports: tuple[CrystallizationStepReport, ...]
    warnings: tuple[str, ...]
    provenance: dict[str, str]
    population_cohorts: tuple[tuple[float, float], ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "target_component": self.target_component,
            "impurity_component": self.impurity_component,
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "duration_s": self.duration_s,
            "feed_amounts_mol": dict(self.feed_amounts_mol),
            "seed_target_mol": self.seed_target_mol,
            "crystals_amounts_mol": dict(self.crystals_amounts_mol),
            "mother_liquor_amounts_mol": dict(self.mother_liquor_amounts_mol),
            "crystallized_from_solution_mol": self.crystallized_from_solution_mol,
            "impurity_occluded_mol": self.impurity_occluded_mol,
            "target_recovery": self.target_recovery,
            "crystal_purity": self.crystal_purity,
            "maximum_supersaturation_ratio": self.maximum_supersaturation_ratio,
            "final_supersaturation_ratio": self.final_supersaturation_ratio,
            "material_balance_error_mol": self.material_balance_error_mol,
            "component_balance_errors_mol": dict(self.component_balance_errors_mol),
            "particle_target_balance_error_mol": self.particle_target_balance_error_mol,
            "cooling_rate_K_s": self.cooling_rate_K_s,
            "growth_solver_converged": self.growth_solver_converged,
            "growth_solver_iterations": self.growth_solver_iterations,
            "growth_solver_max_residual_mol": self.growth_solver_max_residual_mol,
            "crystal_size_distribution": self.crystal_size_distribution.to_dict(),
            "step_reports": [report.to_dict() for report in self.step_reports],
            "warnings": list(self.warnings),
            "provenance": dict(self.provenance),
            **(
                {"population_cohorts": [list(c) for c in self.population_cohorts]}
                if self.population_cohorts
                else {}
            ),
        }


@dataclass
class _CrystalCohort:
    particle_count: float
    diameter_m: float


@dataclass(frozen=True)
class _GrowthTransferResult:
    transferred_mol: float
    iterations: int
    residual_mol: float
    converged: bool


def cooling_crystallization(
    feed_amounts_mol: Mapping[str, float],
    *,
    target_component: str,
    impurity_component: str | None,
    solvent_volume_L: float,
    initial_temperature_K: float,
    final_temperature_K: float,
    duration_s: float,
    solubility_curve: SolubilityCurveSpec,
    kinetics: CrystallizationKineticsSpec,
    seed_mass_g: float = 0.0,
    seed_diameter_m: float = 100.0e-6,
    time_steps: int = 120,
    execution_spec: CrystallizationExecutionSpec | None = None,
    initial_cohorts: tuple[tuple[float, float], ...] | None = None,
    retain_population: bool = False,
) -> CoolingCrystallizationResult:
    """Integrate a compact size-cohort cooling-crystallization model."""

    policy = CrystallizationExecutionSpec() if execution_spec is None else execution_spec
    feed = _amounts(feed_amounts_mol)
    if target_component not in feed or feed[target_component] <= 0.0:
        raise ValueError("target_component must have a positive feed amount")
    if impurity_component is not None and impurity_component not in feed:
        raise ValueError("impurity_component must be present in feed")
    for name, value in (
        ("solvent_volume_L", solvent_volume_L),
        ("initial_temperature_K", initial_temperature_K),
        ("final_temperature_K", final_temperature_K),
        ("duration_s", duration_s),
        ("seed_diameter_m", seed_diameter_m),
    ):
        _positive_finite(value, name)
    _nonnegative_finite(seed_mass_g, "seed_mass_g")
    if final_temperature_K > initial_temperature_K:
        raise ValueError("final_temperature_K cannot exceed initial_temperature_K")
    cooling_rate = (initial_temperature_K - final_temperature_K) / duration_s
    if cooling_rate > policy.maximum_cooling_rate_K_s:
        raise ValueError("cooling ramp exceeds maximum_cooling_rate_K_s")
    if feed[target_component] < policy.minimum_target_amount_mol:
        raise ValueError("target feed is below minimum_target_amount_mol")
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    solubility_curve.solubility_mol_per_l(initial_temperature_K)
    solubility_curve.solubility_mol_per_l(final_temperature_K)

    seed_target_mol = seed_mass_g / 1000.0 / kinetics.target_molecular_weight_kg_mol
    cohorts: list[_CrystalCohort] = []
    if initial_cohorts is not None:
        for count, diameter in initial_cohorts:
            _positive_finite(count, "cohort particle count")
            _positive_finite(diameter, "cohort diameter")
            cohorts.append(_CrystalCohort(count, diameter))
        population_mol = sum(
            c.particle_count * _particle_moles(c.diameter_m, kinetics) for c in cohorts
        )
        if abs(population_mol - seed_target_mol) > 1.0e-10:
            raise ValueError("initial cohort population does not close supplied solid mass")
    elif seed_target_mol > 0.0:
        seed_particle_mol = _particle_moles(seed_diameter_m, kinetics)
        seed_particle_count = seed_target_mol / seed_particle_mol
        if seed_particle_count < policy.minimum_effective_seed_particles:
            raise ValueError("seed charge is below minimum_effective_seed_particles")
        cohorts.append(_CrystalCohort(seed_particle_count, seed_diameter_m))
    dissolved_target = feed[target_component]
    dissolved_impurity = 0.0 if impurity_component is None else feed[impurity_component]
    crystallized_from_solution = 0.0
    occluded_impurity = 0.0
    reports: list[CrystallizationStepReport] = []
    solver_converged = True
    solver_iterations = 0
    solver_max_residual = 0.0
    dt = duration_s / time_steps
    for step in range(1, time_steps + 1):
        fraction = step / time_steps
        temperature_K = initial_temperature_K + fraction * (
            final_temperature_K - initial_temperature_K
        )
        solubility = solubility_curve.solubility_mol_per_l(temperature_K)
        concentration = dissolved_target / solvent_volume_L
        supersaturation_ratio = concentration / solubility
        relative_supersaturation = max(supersaturation_ratio - 1.0, 0.0)
        nucleation_rate = (
            kinetics.primary_nucleation_coefficient_per_L_s
            * relative_supersaturation**kinetics.primary_nucleation_exponent
        )
        growth_rate = (
            kinetics.growth_coefficient_m_s * relative_supersaturation**kinetics.growth_exponent
        )
        available = max(dissolved_target - solubility * solvent_volume_L, 0.0)
        nucleus_moles = _particle_moles(kinetics.nucleus_diameter_m, kinetics)
        desired_nuclei = nucleation_rate * solvent_volume_L * dt
        nucleated_count = min(desired_nuclei, available / nucleus_moles)
        nucleated_target = nucleated_count * nucleus_moles
        if nucleated_count > 0.0:
            cohorts.append(_CrystalCohort(nucleated_count, kinetics.nucleus_diameter_m))
            dissolved_target -= nucleated_target
            crystallized_from_solution += nucleated_target
        impurity_step = _occlude_impurity(
            nucleated_target,
            relative_supersaturation=relative_supersaturation,
            dissolved_impurity=dissolved_impurity,
            solvent_volume_L=solvent_volume_L,
            kinetics=kinetics,
        )
        dissolved_impurity -= impurity_step
        occluded_impurity += impurity_step

        available = max(dissolved_target - solubility * solvent_volume_L, 0.0)
        diameter_increment = growth_rate * dt
        growth = _apply_capped_growth(
            cohorts,
            diameter_increment_m=diameter_increment,
            available_target_mol=available,
            kinetics=kinetics,
            max_iterations=policy.max_growth_solver_iterations,
            tolerance_mol=policy.growth_solver_tolerance_mol,
        )
        growth_target = growth.transferred_mol
        solver_converged = solver_converged and growth.converged
        solver_iterations = max(solver_iterations, growth.iterations)
        solver_max_residual = max(solver_max_residual, growth.residual_mol)
        dissolved_target -= growth_target
        crystallized_from_solution += growth_target
        growth_impurity = _occlude_impurity(
            growth_target,
            relative_supersaturation=relative_supersaturation,
            dissolved_impurity=dissolved_impurity,
            solvent_volume_L=solvent_volume_L,
            kinetics=kinetics,
        )
        dissolved_impurity -= growth_impurity
        occluded_impurity += growth_impurity
        reports.append(
            CrystallizationStepReport(
                time_s=step * dt,
                temperature_K=temperature_K,
                solubility_mol_L=solubility,
                dissolved_target_concentration_mol_L=(dissolved_target / solvent_volume_L),
                supersaturation_ratio=supersaturation_ratio,
                relative_supersaturation=relative_supersaturation,
                nucleation_rate_per_L_s=nucleation_rate,
                growth_rate_m_s=growth_rate,
                nucleated_particle_count=nucleated_count,
                target_crystallized_step_mol=nucleated_target + growth_target,
                impurity_occluded_step_mol=impurity_step + growth_impurity,
            )
        )

    crystals = dict.fromkeys(feed, 0.0)
    mother_liquor = dict(feed)
    crystals[target_component] = seed_target_mol + crystallized_from_solution
    mother_liquor[target_component] = dissolved_target
    if impurity_component is not None:
        crystals[impurity_component] = occluded_impurity
        mother_liquor[impurity_component] = dissolved_impurity
    target_recovery = crystallized_from_solution / feed[target_component]
    crystal_total = sum(crystals.values())
    crystal_purity = 0.0 if crystal_total <= 0.0 else crystals[target_component] / crystal_total
    csd = _crystal_size_distribution(cohorts, kinetics.fines_threshold_m)
    warnings: list[str] = []
    if not cohorts:
        warnings.append("no_crystal_population_formed")
    if crystallized_from_solution < policy.minimum_crystallized_from_solution_mol:
        warnings.append("no_meaningful_crystallization_transfer")
    if not solver_converged:
        warnings.append("growth_cap_solver_not_converged")
    final_solubility = solubility_curve.solubility_mol_per_l(final_temperature_K)
    final_supersaturation = dissolved_target / solvent_volume_L / final_solubility
    if final_supersaturation > 1.05:
        warnings.append("final_mother_liquor_remains_supersaturated")
    if csd.fines_number_fraction > 0.5:
        warnings.append("fines_dominate_number_distribution")
    augmented_feed = dict(feed)
    augmented_feed[target_component] += seed_target_mol
    component_errors = _component_balance_errors(
        augmented_feed,
        {"crystals": crystals, "mother_liquor": mother_liquor},
    )
    balance_error = max(component_errors.values(), default=0.0)
    particle_target_mol = (
        kinetics.crystal_density_kg_m3
        * pi
        / 6.0
        * csd.volume_moment_3_m3
        / kinetics.target_molecular_weight_kg_mol
    )
    particle_target_error = abs(particle_target_mol - crystals[target_component])
    if policy.fail_on_no_population and not cohorts:
        raise ValueError("no crystal population formed")
    if (
        policy.fail_on_no_transfer
        and crystallized_from_solution < policy.minimum_crystallized_from_solution_mol
    ):
        raise ValueError("no meaningful crystallization transfer")
    if policy.fail_on_solver_nonconvergence and not solver_converged:
        raise ValueError("growth cap solver did not converge")
    if balance_error > policy.material_balance_tolerance_mol:
        raise ValueError("component material balance did not close")
    if particle_target_error > policy.particle_target_balance_tolerance_mol:
        raise ValueError("particle size moments do not close the crystal target ledger")
    return CoolingCrystallizationResult(
        model_id="cooling_crystallization_population_balance_v1",
        target_component=target_component,
        impurity_component=impurity_component,
        initial_temperature_K=initial_temperature_K,
        final_temperature_K=final_temperature_K,
        duration_s=duration_s,
        feed_amounts_mol=feed,
        seed_target_mol=seed_target_mol,
        crystals_amounts_mol=crystals,
        mother_liquor_amounts_mol=mother_liquor,
        crystallized_from_solution_mol=crystallized_from_solution,
        impurity_occluded_mol=occluded_impurity,
        target_recovery=target_recovery,
        crystal_purity=crystal_purity,
        maximum_supersaturation_ratio=max(report.supersaturation_ratio for report in reports),
        final_supersaturation_ratio=final_supersaturation,
        material_balance_error_mol=balance_error,
        component_balance_errors_mol=component_errors,
        particle_target_balance_error_mol=particle_target_error,
        cooling_rate_K_s=cooling_rate,
        growth_solver_converged=solver_converged,
        growth_solver_iterations=solver_iterations,
        growth_solver_max_residual_mol=solver_max_residual,
        crystal_size_distribution=csd,
        population_cohorts=tuple((c.particle_count, c.diameter_m) for c in cohorts)
        if retain_population
        else (),
        step_reports=tuple(reports),
        warnings=tuple(warnings),
        provenance={
            "solubility_curve": solubility_curve.provenance_id,
            "kinetics": kinetics.provenance_id,
        },
    )


def _apply_capped_growth(
    cohorts: list[_CrystalCohort],
    *,
    diameter_increment_m: float,
    available_target_mol: float,
    kinetics: CrystallizationKineticsSpec,
    max_iterations: int,
    tolerance_mol: float,
) -> _GrowthTransferResult:
    if not cohorts or diameter_increment_m <= 0.0 or available_target_mol <= 0.0:
        return _GrowthTransferResult(0.0, 0, 0.0, True)
    desired = _growth_moles(cohorts, diameter_increment_m, kinetics)
    if desired <= available_target_mol:
        scale = 1.0
        transferred = desired
        iterations = 0
        residual = 0.0
        converged = True
    else:
        lower = 0.0
        upper = 1.0
        iterations = 0
        for _iteration in range(1, max_iterations + 1):
            iterations = _iteration
            middle = 0.5 * (lower + upper)
            value = _growth_moles(
                cohorts,
                middle * diameter_increment_m,
                kinetics,
            )
            if value > available_target_mol:
                upper = middle
            else:
                lower = middle
        scale = 0.5 * (lower + upper)
        transferred = _growth_moles(
            cohorts,
            scale * diameter_increment_m,
            kinetics,
        )
        residual = abs(transferred - available_target_mol)
        converged = residual <= tolerance_mol
    increment = scale * diameter_increment_m
    for cohort in cohorts:
        cohort.diameter_m += increment
    return _GrowthTransferResult(transferred, iterations, residual, converged)


def _growth_moles(
    cohorts: list[_CrystalCohort],
    diameter_increment_m: float,
    kinetics: CrystallizationKineticsSpec,
) -> float:
    return sum(
        cohort.particle_count
        * (
            _particle_moles(cohort.diameter_m + diameter_increment_m, kinetics)
            - _particle_moles(cohort.diameter_m, kinetics)
        )
        for cohort in cohorts
    )


def _particle_moles(
    diameter_m: float,
    kinetics: CrystallizationKineticsSpec,
) -> float:
    volume_m3 = pi / 6.0 * diameter_m**3
    return volume_m3 * kinetics.crystal_density_kg_m3 / kinetics.target_molecular_weight_kg_mol


def _occlude_impurity(
    target_transfer_mol: float,
    *,
    relative_supersaturation: float,
    dissolved_impurity: float,
    solvent_volume_L: float,
    kinetics: CrystallizationKineticsSpec,
) -> float:
    if kinetics.impurity_occlusion_law_id == "linear_supersaturation_transfer_v1":
        ratio = kinetics.impurity_occlusion_mol_per_mol * (
            1.0 + kinetics.supersaturation_occlusion_factor * relative_supersaturation
        )
    else:
        impurity_concentration = dissolved_impurity / max(solvent_volume_L, 1.0e-12)
        surface_coverage = impurity_concentration / (
            kinetics.impurity_surface_half_saturation_mol_L + impurity_concentration
        )
        ratio = (
            kinetics.impurity_occlusion_mol_per_mol
            * SURFACE_SATURATION_MAX_LOADING_RATIO
            * surface_coverage
            * (1.0 + kinetics.supersaturation_occlusion_factor * relative_supersaturation**2)
        )
    return min(dissolved_impurity, max(0.0, target_transfer_mol * ratio))


def _crystal_size_distribution(
    cohorts: list[_CrystalCohort],
    fines_threshold_m: float,
) -> CrystalSizeDistribution:
    total_count = sum(cohort.particle_count for cohort in cohorts)
    if total_count <= 0.0:
        return CrystalSizeDistribution(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    moment_1 = sum(cohort.particle_count * cohort.diameter_m for cohort in cohorts)
    moment_2 = sum(cohort.particle_count * cohort.diameter_m**2 for cohort in cohorts)
    moment_3 = sum(cohort.particle_count * cohort.diameter_m**3 for cohort in cohorts)
    mean = moment_1 / total_count
    variance = (
        sum(cohort.particle_count * (cohort.diameter_m - mean) ** 2 for cohort in cohorts)
        / total_count
    )
    std = sqrt(max(variance, 0.0))
    fines = (
        sum(cohort.particle_count for cohort in cohorts if cohort.diameter_m < fines_threshold_m)
        / total_count
    )
    return CrystalSizeDistribution(
        total_particle_count=total_count,
        number_mean_diameter_m=mean,
        number_std_diameter_m=std,
        coefficient_of_variation=std / mean if mean > 0.0 else 0.0,
        d10_m=_weighted_quantile(cohorts, 0.10, total_count),
        d50_m=_weighted_quantile(cohorts, 0.50, total_count),
        d90_m=_weighted_quantile(cohorts, 0.90, total_count),
        fines_number_fraction=fines,
        cohort_count=len(cohorts),
        number_moment_0=total_count,
        length_moment_1_m=moment_1,
        area_moment_2_m2=moment_2,
        volume_moment_3_m3=moment_3,
    )


def _weighted_quantile(
    cohorts: list[_CrystalCohort],
    quantile: float,
    total_count: float,
) -> float:
    threshold = quantile * total_count
    cumulative = 0.0
    for cohort in sorted(cohorts, key=lambda item: item.diameter_m):
        cumulative += cohort.particle_count
        if cumulative >= threshold:
            return cohort.diameter_m
    return max(cohort.diameter_m for cohort in cohorts)


def _amounts(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        raise ValueError("feed_amounts_mol cannot be empty")
    result = {str(key): float(value) for key, value in values.items()}
    if any(value < 0.0 or not isfinite(value) for value in result.values()):
        raise ValueError("feed_amounts_mol must contain finite nonnegative values")
    if sum(result.values()) <= 0.0:
        raise ValueError("feed_amounts_mol must contain positive total material")
    return result


def _component_balance_errors(
    feed: Mapping[str, float],
    outlets: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    return {
        key: abs(feed[key] - sum(outlet.get(key, 0.0) for outlet in outlets.values()))
        for key in feed
    }


def _positive_finite(value: float, field_name: str) -> None:
    if value <= 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be positive and finite")


def _nonnegative_finite(value: float, field_name: str) -> None:
    if value < 0.0 or not isfinite(value):
        raise ValueError(f"{field_name} must be nonnegative and finite")


__all__ = [
    "CoolingCrystallizationResult",
    "CrystalSizeDistribution",
    "CrystallizationExecutionSpec",
    "CrystallizationKineticsSpec",
    "CrystallizationStepReport",
    "SolubilityCurveSpec",
    "cooling_crystallization",
]
