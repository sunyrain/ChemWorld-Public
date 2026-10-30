"""WF-50 provider and adapter proposal for crystallization audits."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise
from typing import Any

from chemworld.physchem.crystallization_units import (
    CoolingCrystallizationResult,
    CrystallizationExecutionSpec,
)
from chemworld.physchem.crystallization_validation import (
    IDAES_BALANCE_PATH,
    IDAES_COMMIT,
    CrystallizationConvergenceSpec,
    CrystallizationGridCase,
    audit_crystallization_convergence,
)
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.model_provider import ModelProviderResult

OWNED_PATHS = (
    "src/chemworld/physchem/crystallization_validation.py",
    "src/chemworld/physchem/crystallization_adapter_manifest.py",
    "tests/test_crystallization_validation.py",
)
INTEGRATION_OPERATIONS = ("cool_crystallize",)
RUNTIME_MODEL_ID = "cooling_crystallization_population_balance_v1"


def crystallization_runtime_provider_contract() -> ModelProviderContract:
    """Return the validated population-balance runtime contract."""

    return ModelProviderContract(
        model_id=RUNTIME_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.PROFESSIONAL_CANDIDATE,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.crystallization_adapter_manifest."
            "ValidatedCrystallizationRuntimeProvider"
        ),
        input_fields=("case", "time_steps", "execution_spec"),
        output_fields=("result",),
        units={
            "case": "CrystallizationGridCase with explicit K/s/L/mol inputs",
            "time_steps": "count",
            "execution_spec": "dimensioned runtime acceptance policy",
            "result": "CoolingCrystallizationResult",
        },
        validity_checks=(
            "target feed and any seed population exceed declared effective minima",
            "cooling rate remains inside the declared linear-ramp domain",
            "temperature history is monotonic and reaches the requested endpoint",
            "component and crystal-size-moment ledgers close within tolerance",
            (
                "growth-cap solver converges; population and transfer requirements follow "
                "the supplied execution policy"
            ),
        ),
        diagnostic_fields=(
            "runtime_validated",
            "temperature_history_valid",
            "material_closed",
            "liquid_composition_closed",
            "particle_size_moment_closed",
            "growth_solver_converged",
            "crystal_population_formed",
            "meaningful_transfer",
            "cooling_rate_K_s",
            "warnings",
        ),
        failure_policy=(
            "fail closed without a runtime output for invalid feed or seed, excessive "
            "cooling rate, solver non-convergence, or component/particle ledger failure; "
            "absent population or transfer fail only when required by execution_spec"
        ),
        provenance=(
            "ChemWorld cooling_crystallization population-balance cohort model v1",
            "ChemWorld van't Hoff solubility and primary nucleation/growth contracts",
            "tests/test_crystallization_units.py",
            "tests/test_crystallization_validation.py",
            "tests/test_crystallization_coupling.py",
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class ValidatedCrystallizationRuntimeProvider:
    """Fail-closed provider used by the formal cooling-crystallization runtime."""

    @property
    def model_contract(self) -> ModelProviderContract:
        return crystallization_runtime_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        violations: list[str] = []
        if not isinstance(inputs.get("case"), CrystallizationGridCase):
            violations.append("case must be a CrystallizationGridCase")
        time_steps = inputs.get("time_steps")
        if isinstance(time_steps, bool) or not isinstance(time_steps, int) or time_steps < 2:
            violations.append("time_steps must be an integer of at least two")
        if not isinstance(inputs.get("execution_spec"), CrystallizationExecutionSpec):
            violations.append("execution_spec must be a CrystallizationExecutionSpec")
        return tuple(violations)

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return _runtime_failed_result("; ".join(violations))
        case = inputs["case"]
        time_steps = inputs["time_steps"]
        execution_spec = inputs["execution_spec"]
        try:
            result = case.run(time_steps, execution_spec=execution_spec)
        except (ArithmeticError, ValueError) as error:
            return _runtime_failed_result(str(error))
        diagnostics = _runtime_diagnostics(result, execution_spec)
        failures = [
            key
            for key in (
                "temperature_history_valid",
                "material_closed",
                "liquid_composition_closed",
                "particle_size_moment_closed",
                "growth_solver_converged",
            )
            if diagnostics[key] is not True
        ]
        if (
            execution_spec.fail_on_no_population
            and diagnostics["crystal_population_formed"] is not True
        ):
            failures.append("crystal_population_formed")
        if (
            execution_spec.fail_on_no_transfer
            and diagnostics["meaningful_transfer"] is not True
        ):
            failures.append("meaningful_transfer")
        if failures:
            return _runtime_failed_result(
                "runtime crystallization acceptance failed: " + ", ".join(failures),
                diagnostics=diagnostics,
            )
        diagnostics["runtime_validated"] = True
        return ModelProviderResult(
            outputs={"result": result},
            diagnostics=diagnostics,
            warnings=result.warnings,
            provenance=self.model_contract.provenance,
        )


def crystallization_runtime_adapter_manifest() -> ModelAdapterManifest:
    """Return the integrated adapter manifest for formal crystallization."""

    return ModelAdapterManifest(
        adapter_id="foundation-crystallization-coupling",
        adapter_version="1.0",
        owner_workstream="foundation-crystallization-coupling",
        provider_contract=crystallization_runtime_provider_contract(),
        owned_paths=(
            "src/chemworld/world/crystallization.py",
            "src/chemworld/physchem/crystallization_units.py",
            "src/chemworld/physchem/crystallization_validation.py",
            "src/chemworld/physchem/crystallization_cards.py",
            "src/chemworld/physchem/crystallization_adapter_manifest.py",
            "src/chemworld/runtime/crystallization_services.py",
            "tests/test_crystallization_coupling.py",
        ),
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
    )


def crystallization_convergence_provider_contract() -> ModelProviderContract:
    """Return the WF-00-compatible diagnostic provider contract."""

    return ModelProviderContract(
        model_id="chemworld_crystallization_convergence_audit_vnext",
        module_id="separations",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.DIAGNOSTIC,
        provider_path=(
            "chemworld.physchem.crystallization_adapter_manifest.CrystallizationConvergenceProvider"
        ),
        input_fields=("case", "audit_spec"),
        output_fields=("report",),
        units={
            "case": "CrystallizationGridCase with explicit SI/L/mol units",
            "audit_spec": "grid and dimensioned residual policy",
            "report": "JSON numerical convergence and closure audit",
        },
        validity_checks=(
            "case is a complete CrystallizationGridCase",
            "audit_spec declares at least two strictly increasing grid levels",
            "all underlying PBM inputs pass cooling_crystallization domain checks",
            "all convergence and ledger thresholds are finite and positive",
        ),
        diagnostic_fields=(
            "passed",
            "grid_converged",
            "material_closed",
            "step_ledger_closed",
            "particle_count_ledger_closed",
            "particle_size_moment_closed",
            "warnings",
        ),
        failure_policy=(
            "reject invalid cases or policies with an explicit unsuccessful result; "
            "valid non-converged grids return a successful diagnostic whose passed "
            "field is false"
        ),
        provenance=(
            "ChemWorld cooling_crystallization deterministic refinement ladder",
            "analytical component, step-transfer, and particle-count ledgers",
            (
                f"IDAES {IDAES_COMMIT}:{IDAES_BALANCE_PATH} convention boundary; "
                "not a crystallization kinetics backend"
            ),
        ),
        intended_operations=INTEGRATION_OPERATIONS,
    )


class CrystallizationConvergenceProvider:
    """Diagnostic provider for the existing compact crystallization PBM."""

    @property
    def model_contract(self) -> ModelProviderContract:
        return crystallization_convergence_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        violations: list[str] = []
        if not isinstance(inputs.get("case"), CrystallizationGridCase):
            violations.append("case must be a CrystallizationGridCase")
        audit_spec = inputs.get("audit_spec")
        if audit_spec is not None and not isinstance(
            audit_spec,
            CrystallizationConvergenceSpec,
        ):
            violations.append("audit_spec must be a CrystallizationConvergenceSpec or None")
        return tuple(violations)

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        domain_violations = self.validate_domain(inputs)
        if domain_violations:
            return _failed_result("; ".join(domain_violations), self.model_contract.provenance)
        case = inputs["case"]
        audit_spec = inputs.get("audit_spec")
        try:
            report = audit_crystallization_convergence(case, spec=audit_spec)
        except ValueError as error:
            return _failed_result(str(error), self.model_contract.provenance)
        diagnostics = {
            "passed": report.passed,
            "grid_converged": report.grid_converged,
            "material_closed": report.material_closed,
            "step_ledger_closed": report.step_ledger_closed,
            "particle_count_ledger_closed": report.particle_count_ledger_closed,
            "particle_size_moment_closed": report.particle_size_moment_closed,
            "warnings": list(report.warnings),
        }
        return ModelProviderResult(
            outputs={"report": report.to_dict()},
            diagnostics=diagnostics,
            warnings=report.warnings,
            provenance=report.provenance,
        )


def crystallization_convergence_adapter_manifest() -> ModelAdapterManifest:
    """Return the claim-bound proposal for later WF-110 intake."""

    return ModelAdapterManifest(
        adapter_id="wf-50-crystallization-convergence",
        adapter_version="0.1",
        owner_workstream="wf-50-crystallization-convergence",
        provider_contract=crystallization_convergence_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=INTEGRATION_OPERATIONS,
        target_world_law="chemworld-physical-chemistry-vnext",
        status="proposal",
    )


def _failed_result(
    failure_reason: str,
    provenance: tuple[str, ...],
) -> ModelProviderResult:
    return ModelProviderResult(
        outputs={},
        diagnostics={
            "passed": False,
            "grid_converged": False,
            "material_closed": False,
            "step_ledger_closed": False,
            "particle_count_ledger_closed": False,
            "particle_size_moment_closed": False,
            "warnings": [failure_reason],
        },
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=provenance,
    )


def _runtime_diagnostics(
    result: CoolingCrystallizationResult,
    policy: CrystallizationExecutionSpec,
) -> dict[str, Any]:
    temperatures = [
        result.initial_temperature_K,
        *[report.temperature_K for report in result.step_reports],
    ]
    temperature_history_valid = (
        bool(result.step_reports)
        and all(right <= left for left, right in pairwise(temperatures))
        and abs(temperatures[-1] - result.final_temperature_K) <= 1.0e-9
    )
    liquid_closed = all(
        error <= policy.material_balance_tolerance_mol
        for error in result.component_balance_errors_mol.values()
    )
    return {
        "runtime_validated": False,
        "temperature_history_valid": temperature_history_valid,
        "material_closed": (
            result.material_balance_error_mol <= policy.material_balance_tolerance_mol
        ),
        "liquid_composition_closed": liquid_closed,
        "particle_size_moment_closed": (
            result.particle_target_balance_error_mol <= policy.particle_target_balance_tolerance_mol
        ),
        "growth_solver_converged": result.growth_solver_converged,
        "crystal_population_formed": (result.crystal_size_distribution.total_particle_count > 0.0),
        "meaningful_transfer": (
            result.crystallized_from_solution_mol >= policy.minimum_crystallized_from_solution_mol
        ),
        "cooling_rate_K_s": result.cooling_rate_K_s,
        "warnings": list(result.warnings),
    }


def _runtime_failed_result(
    failure_reason: str,
    *,
    diagnostics: Mapping[str, Any] | None = None,
) -> ModelProviderResult:
    failed_diagnostics: dict[str, Any] = {
        "runtime_validated": False,
        "temperature_history_valid": False,
        "material_closed": False,
        "liquid_composition_closed": False,
        "particle_size_moment_closed": False,
        "growth_solver_converged": False,
        "crystal_population_formed": False,
        "meaningful_transfer": False,
        "cooling_rate_K_s": None,
        "warnings": [failure_reason],
    }
    if diagnostics is not None:
        failed_diagnostics.update(dict(diagnostics))
        failed_diagnostics["runtime_validated"] = False
        failed_diagnostics["warnings"] = [
            *list(failed_diagnostics.get("warnings", ())),
            failure_reason,
        ]
    return ModelProviderResult(
        outputs={},
        diagnostics=failed_diagnostics,
        warnings=(failure_reason,),
        success=False,
        failure_reason=failure_reason,
        provenance=crystallization_runtime_provider_contract().provenance,
    )


__all__ = [
    "INTEGRATION_OPERATIONS",
    "OWNED_PATHS",
    "RUNTIME_MODEL_ID",
    "CrystallizationConvergenceProvider",
    "ValidatedCrystallizationRuntimeProvider",
    "crystallization_convergence_adapter_manifest",
    "crystallization_convergence_provider_contract",
    "crystallization_runtime_adapter_manifest",
    "crystallization_runtime_provider_contract",
]
