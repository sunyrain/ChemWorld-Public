"""Numerical closure and time-grid convergence audits for crystallization."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite, pi
from typing import Any

from chemworld.physchem.crystallization_units import (
    CoolingCrystallizationResult,
    CrystallizationExecutionSpec,
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
    cooling_crystallization,
)
from chemworld.physchem.maturity import MaturityLevel, ModelCard, ValidationEvidence

IDAES_COMMIT = "eed7cebc3d99be616ee7ead203cecaee9f81ac01"
IDAES_BALANCE_PATH = "idaes/core/base/control_volume0d.py"


@dataclass(frozen=True)
class CrystallizationGridCase:
    """Complete, provenance-bearing input for repeatable grid refinement."""

    feed_amounts_mol: Mapping[str, float]
    target_component: str
    impurity_component: str | None
    solvent_volume_L: float
    initial_temperature_K: float
    final_temperature_K: float
    duration_s: float
    solubility_curve: SolubilityCurveSpec
    kinetics: CrystallizationKineticsSpec
    seed_mass_g: float = 0.0
    seed_diameter_m: float = 100.0e-6
    initial_cohorts: tuple[tuple[float, float], ...] | None = None
    retain_population: bool = False

    def run(
        self,
        time_steps: int,
        *,
        execution_spec: CrystallizationExecutionSpec | None = None,
    ) -> CoolingCrystallizationResult:
        """Run the existing PBM without changing its runtime implementation."""

        return cooling_crystallization(
            dict(self.feed_amounts_mol),
            target_component=self.target_component,
            impurity_component=self.impurity_component,
            solvent_volume_L=self.solvent_volume_L,
            initial_temperature_K=self.initial_temperature_K,
            final_temperature_K=self.final_temperature_K,
            duration_s=self.duration_s,
            solubility_curve=self.solubility_curve,
            kinetics=self.kinetics,
            seed_mass_g=self.seed_mass_g,
            seed_diameter_m=self.seed_diameter_m,
            time_steps=time_steps,
            execution_spec=execution_spec,
            initial_cohorts=self.initial_cohorts,
            retain_population=self.retain_population,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feed_amounts_mol": {
                key: float(value) for key, value in sorted(self.feed_amounts_mol.items())
            },
            "target_component": self.target_component,
            "impurity_component": self.impurity_component,
            "solvent_volume_L": self.solvent_volume_L,
            "initial_temperature_K": self.initial_temperature_K,
            "final_temperature_K": self.final_temperature_K,
            "duration_s": self.duration_s,
            "solubility_curve": self.solubility_curve.to_dict(),
            "kinetics": self.kinetics.to_dict(),
            "seed_mass_g": self.seed_mass_g,
            "seed_diameter_m": self.seed_diameter_m,
            **(
                {"initial_cohorts": self.initial_cohorts, "retain_population": True}
                if self.retain_population
                else {}
            ),
        }


@dataclass(frozen=True)
class CrystallizationConvergenceSpec:
    """Grid levels and acceptance thresholds for one audit."""

    base_time_steps: int = 60
    refinement_factors: tuple[int, ...] = (1, 2, 4)
    max_recovery_relative_delta: float = 0.02
    max_crystallized_relative_delta: float = 0.02
    max_d50_relative_delta: float = 0.05
    max_material_balance_error_mol: float = 1.0e-10
    max_step_ledger_error_mol: float = 1.0e-10
    max_particle_count_relative_error: float = 1.0e-12
    max_particle_target_balance_error_mol: float = 1.0e-10
    relative_floor: float = 1.0e-15

    def __post_init__(self) -> None:
        if self.base_time_steps < 2:
            raise ValueError("base_time_steps must be at least two")
        if len(self.refinement_factors) < 2:
            raise ValueError("refinement_factors must contain at least two levels")
        if self.refinement_factors[0] != 1:
            raise ValueError("refinement_factors must begin with one")
        if any(not isinstance(value, int) or value <= 0 for value in self.refinement_factors):
            raise ValueError("refinement_factors must contain positive integers")
        if tuple(sorted(set(self.refinement_factors))) != self.refinement_factors:
            raise ValueError("refinement_factors must be strictly increasing")
        for field_name in (
            "max_recovery_relative_delta",
            "max_crystallized_relative_delta",
            "max_d50_relative_delta",
            "max_material_balance_error_mol",
            "max_step_ledger_error_mol",
            "max_particle_count_relative_error",
            "max_particle_target_balance_error_mol",
            "relative_floor",
        ):
            value = float(getattr(self, field_name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_time_steps": self.base_time_steps,
            "refinement_factors": list(self.refinement_factors),
            "max_recovery_relative_delta": self.max_recovery_relative_delta,
            "max_crystallized_relative_delta": self.max_crystallized_relative_delta,
            "max_d50_relative_delta": self.max_d50_relative_delta,
            "max_material_balance_error_mol": self.max_material_balance_error_mol,
            "max_step_ledger_error_mol": self.max_step_ledger_error_mol,
            "max_particle_count_relative_error": (self.max_particle_count_relative_error),
            "max_particle_target_balance_error_mol": (self.max_particle_target_balance_error_mol),
            "relative_floor": self.relative_floor,
        }


@dataclass(frozen=True)
class CrystallizationGridPoint:
    """Auditable metrics from one time-grid resolution."""

    time_steps: int
    target_recovery: float
    crystallized_from_solution_mol: float
    d50_m: float
    total_particle_count: float
    material_balance_error_mol: float
    target_step_ledger_error_mol: float
    impurity_step_ledger_error_mol: float
    particle_count_ledger_relative_error: float
    particle_target_balance_error_mol: float
    result_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_steps": self.time_steps,
            "target_recovery": self.target_recovery,
            "crystallized_from_solution_mol": self.crystallized_from_solution_mol,
            "d50_m": self.d50_m,
            "total_particle_count": self.total_particle_count,
            "material_balance_error_mol": self.material_balance_error_mol,
            "target_step_ledger_error_mol": self.target_step_ledger_error_mol,
            "impurity_step_ledger_error_mol": self.impurity_step_ledger_error_mol,
            "particle_count_ledger_relative_error": (self.particle_count_ledger_relative_error),
            "particle_target_balance_error_mol": self.particle_target_balance_error_mol,
            "result_sha256": self.result_sha256,
        }


@dataclass(frozen=True)
class CrystallizationConvergenceReport:
    """Grid-refinement evidence without changing the audited PBM."""

    case_sha256: str
    grid_points: tuple[CrystallizationGridPoint, ...]
    recovery_relative_delta: float
    crystallized_relative_delta: float
    d50_relative_delta: float
    material_closed: bool
    step_ledger_closed: bool
    particle_count_ledger_closed: bool
    particle_size_moment_closed: bool
    grid_converged: bool
    passed: bool
    thresholds: CrystallizationConvergenceSpec
    warnings: tuple[str, ...]
    provenance: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "chemworld-crystallization-convergence-0.1",
            "case_sha256": self.case_sha256,
            "grid_points": [point.to_dict() for point in self.grid_points],
            "recovery_relative_delta": self.recovery_relative_delta,
            "crystallized_relative_delta": self.crystallized_relative_delta,
            "d50_relative_delta": self.d50_relative_delta,
            "material_closed": self.material_closed,
            "step_ledger_closed": self.step_ledger_closed,
            "particle_count_ledger_closed": self.particle_count_ledger_closed,
            "particle_size_moment_closed": self.particle_size_moment_closed,
            "grid_converged": self.grid_converged,
            "passed": self.passed,
            "thresholds": self.thresholds.to_dict(),
            "warnings": list(self.warnings),
            "provenance": list(self.provenance),
            "audit_boundary": (
                "offline numerical evidence for the existing cooling-crystallization "
                "PBM; not an experimental validation or runtime maturity promotion"
            ),
        }


def audit_crystallization_convergence(
    case: CrystallizationGridCase,
    *,
    spec: CrystallizationConvergenceSpec | None = None,
) -> CrystallizationConvergenceReport:
    """Run a declared refinement ladder and audit conservation identities."""

    policy = CrystallizationConvergenceSpec() if spec is None else spec
    results = tuple(
        case.run(policy.base_time_steps * factor) for factor in policy.refinement_factors
    )
    points = tuple(_grid_point(case, result) for result in results)
    previous, finest = points[-2:]
    recovery_delta = _relative_delta(
        previous.target_recovery,
        finest.target_recovery,
        policy.relative_floor,
    )
    crystallized_delta = _relative_delta(
        previous.crystallized_from_solution_mol,
        finest.crystallized_from_solution_mol,
        policy.relative_floor,
    )
    d50_delta = _relative_delta(previous.d50_m, finest.d50_m, policy.relative_floor)
    material_closed = all(
        point.material_balance_error_mol <= policy.max_material_balance_error_mol
        for point in points
    )
    step_ledger_closed = all(
        max(
            point.target_step_ledger_error_mol,
            point.impurity_step_ledger_error_mol,
        )
        <= policy.max_step_ledger_error_mol
        for point in points
    )
    particle_count_closed = all(
        point.particle_count_ledger_relative_error <= policy.max_particle_count_relative_error
        for point in points
    )
    particle_size_moment_closed = all(
        point.particle_target_balance_error_mol <= policy.max_particle_target_balance_error_mol
        for point in points
    )
    grid_converged = (
        recovery_delta <= policy.max_recovery_relative_delta
        and crystallized_delta <= policy.max_crystallized_relative_delta
        and d50_delta <= policy.max_d50_relative_delta
    )
    warnings: list[str] = []
    if not material_closed:
        warnings.append("component_material_balance_not_closed")
    if not step_ledger_closed:
        warnings.append("step_transfer_ledger_not_closed")
    if not particle_count_closed:
        warnings.append("particle_count_ledger_not_closed")
    if not particle_size_moment_closed:
        warnings.append("particle_size_moment_ledger_not_closed")
    if not grid_converged:
        warnings.append("time_grid_not_converged")
    if finest.total_particle_count == 0.0:
        warnings.append("finest_grid_has_no_crystal_population")
    return CrystallizationConvergenceReport(
        case_sha256=_sha256(case.to_dict()),
        grid_points=points,
        recovery_relative_delta=recovery_delta,
        crystallized_relative_delta=crystallized_delta,
        d50_relative_delta=d50_delta,
        material_closed=material_closed,
        step_ledger_closed=step_ledger_closed,
        particle_count_ledger_closed=particle_count_closed,
        particle_size_moment_closed=particle_size_moment_closed,
        grid_converged=grid_converged,
        passed=(
            material_closed
            and step_ledger_closed
            and particle_count_closed
            and particle_size_moment_closed
            and grid_converged
        ),
        thresholds=policy,
        warnings=tuple(warnings),
        provenance=(
            "ChemWorld cooling_crystallization deterministic refinement ladder",
            "analytical component, step-transfer, and no-breakage particle-count ledgers",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_BALANCE_PATH} material-balance and "
                "state-block convention boundary; not a crystallization kinetics backend"
            ),
        ),
    )


def crystallization_convergence_model_card() -> ModelCard:
    """Return the narrowly scoped model card for the audit calculation."""

    return ModelCard(
        model_id="chemworld_crystallization_convergence_audit_vnext",
        module_id="separations",
        title="Cooling-Crystallization Grid And Ledger Audit",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        summary=(
            "An offline diagnostic that reruns the existing compact cooling-"
            "crystallization PBM on a declared time-grid ladder and checks numerical "
            "convergence plus exact ledger identities."
        ),
        equations=(
            "delta_rel(x_h, x_h/2) = abs(x_h - x_h/2) / max(abs(x_h), abs(x_h/2), floor)",
            "sum(Delta n_target,step) = n_target,crystallized",
            "sum(Delta n_impurity,step) = n_impurity,occluded",
            "N_final = N_seed + sum(N_nucleated,step) when aggregation and breakage are absent",
            "n_target,solid = rho pi M3 / (6 MW)",
        ),
        assumptions=(
            "the audited cooling_crystallization implementation is deterministic",
            "the finest two declared grids are used for the convergence decision",
            "the current PBM has no aggregation or breakage particle-count source terms",
        ),
        validity_limits=(
            "applies only to the existing compact cooling-crystallization PBM inputs",
            "grid agreement is numerical evidence, not experimental kinetics validation",
            "D50 convergence alone does not validate the complete CSD or polymorph behavior",
        ),
        failure_modes=(
            "invalid case inputs fail through the audited PBM domain checks",
            "material, step-transfer, or particle-count residuals produce explicit warnings",
            "finest-grid disagreement returns a valid failed audit rather than silent acceptance",
        ),
        units={
            "component and ledger residuals": "mol",
            "particle count and relative deltas": "dimensionless",
            "D50": "m",
            "time grid": "integer integration steps",
        },
        reference_reading=(
            "Analytical conservation identities for the ChemWorld cohort PBM.",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_BALANCE_PATH} is pinned only for "
                "explicit material-balance and state-variable conventions."
            ),
        ),
        validation_evidence=(
            ValidationEvidence(
                evidence_id="crystallization-grid-ledger-audit",
                evidence_type="analytical_and_numerical_test",
                description=(
                    "Checks refinement convergence, exact material/step/particle ledgers, "
                    "non-convergence diagnostics, zero-population behavior, and determinism."
                ),
                status="implemented",
                reference_backend=f"IDAES {IDAES_COMMIT} convention boundary",
                command_or_path="tests/test_crystallization_validation.py",
                tolerance="all thresholds serialized in CrystallizationConvergenceSpec",
            ),
        ),
        model_limit_notes=(
            "Reference-validated applies to the audit identities, not to PBM parameters.",
            "The existing professional-candidate PBM maturity is neither raised nor lowered.",
            "Experimental CSD benchmarks and energy-balance coupling remain future evidence.",
        ),
        intended_use=(
            "crystallization regression gates",
            "time-step sensitivity review before vNext runtime integration",
            "machine-readable conservation evidence for WF-06",
        ),
    )


def _grid_point(
    case: CrystallizationGridCase,
    result: CoolingCrystallizationResult,
) -> CrystallizationGridPoint:
    target_step_total = sum(report.target_crystallized_step_mol for report in result.step_reports)
    impurity_step_total = sum(report.impurity_occluded_step_mol for report in result.step_reports)
    seed_target_mol = case.seed_mass_g / 1000.0 / case.kinetics.target_molecular_weight_kg_mol
    seed_particle_mol = _particle_moles(case.seed_diameter_m, case.kinetics)
    seed_count = seed_target_mol / seed_particle_mol if seed_target_mol > 0.0 else 0.0
    if case.initial_cohorts is not None:
        seed_count = sum(c[0] for c in case.initial_cohorts)
    expected_count = seed_count + sum(
        report.nucleated_particle_count for report in result.step_reports
    )
    observed_count = result.crystal_size_distribution.total_particle_count
    count_error = abs(observed_count - expected_count) / max(expected_count, 1.0)
    return CrystallizationGridPoint(
        time_steps=len(result.step_reports),
        target_recovery=result.target_recovery,
        crystallized_from_solution_mol=result.crystallized_from_solution_mol,
        d50_m=result.crystal_size_distribution.d50_m,
        total_particle_count=observed_count,
        material_balance_error_mol=result.material_balance_error_mol,
        target_step_ledger_error_mol=abs(target_step_total - result.crystallized_from_solution_mol),
        impurity_step_ledger_error_mol=abs(impurity_step_total - result.impurity_occluded_mol),
        particle_count_ledger_relative_error=count_error,
        particle_target_balance_error_mol=result.particle_target_balance_error_mol,
        result_sha256=_sha256(result.to_dict()),
    )


def _particle_moles(
    diameter_m: float,
    kinetics: CrystallizationKineticsSpec,
) -> float:
    volume_m3 = pi / 6.0 * diameter_m**3
    return volume_m3 * kinetics.crystal_density_kg_m3 / kinetics.target_molecular_weight_kg_mol


def _relative_delta(left: float, right: float, floor: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


def _sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "IDAES_BALANCE_PATH",
    "IDAES_COMMIT",
    "CrystallizationConvergenceReport",
    "CrystallizationConvergenceSpec",
    "CrystallizationGridCase",
    "CrystallizationGridPoint",
    "audit_crystallization_convergence",
    "crystallization_convergence_model_card",
]
