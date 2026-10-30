"""Auditable operation-to-model reachability contracts for the current World Law."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chemworld.physchem.concentration_adapter_manifest import (
    vacuum_concentration_provider_contract,
)
from chemworld.physchem.crystallization_adapter_manifest import (
    crystallization_runtime_provider_contract,
)
from chemworld.physchem.distillation_adapter_manifest import (
    duty_limited_distillation_provider_contract,
)
from chemworld.physchem.drying_adapter_manifest import sorbent_drying_provider_contract
from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.phase_equilibrium_adapter_manifest import (
    stability_aware_lle_provider_contract,
)
from chemworld.physchem.reaction_adapter_manifest import reaction_rate_provider_contract
from chemworld.physchem.transfer_adapter_manifest import transfer_provider_contract
from chemworld.runtime.domain_service_registry import DomainServiceRegistry
from chemworld.runtime.kernel_registry import OperationKernelRegistry
from chemworld.runtime.profiles import TaskRuntimeProfile
from chemworld.tasks import TaskSpec, get_task, list_tasks
from chemworld.world.operations import INSTRUMENTS, OPERATION_TYPES
from chemworld.world.spectroscopy_provider import (
    instrument_runtime_provider_contract,
    spectroscopy_identifiability_provider_contract,
)

MODEL_REACHABILITY_SCHEMA_VERSION = "chemworld-model-reachability-0.1"


@dataclass(frozen=True)
class OperationModelRoute:
    """One operation's service/kernel path and reachable model providers."""

    operation_type: str
    service_id: str
    kernel_id: str
    model_ids: tuple[str, ...] = ()
    instrument_model_ids: dict[str, tuple[str, ...]] = field(default_factory=dict)
    model_free_reason: str | None = None

    def __post_init__(self) -> None:
        if self.operation_type not in OPERATION_TYPES:
            raise ValueError(f"unknown operation route {self.operation_type!r}")
        if not self.service_id.strip() or not self.kernel_id.strip():
            raise ValueError("operation model route requires service_id and kernel_id")
        if not self.model_ids and not self.instrument_model_ids and not self.model_free_reason:
            raise ValueError("model-free routes require an explicit reason")
        unknown_instruments = sorted(set(self.instrument_model_ids) - set(INSTRUMENTS))
        if unknown_instruments:
            raise ValueError(f"unknown instrument routes: {unknown_instruments}")
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("an operation route cannot repeat base model ids")
        if any(
            len(model_ids) != len(set(model_ids))
            for model_ids in self.instrument_model_ids.values()
        ):
            raise ValueError("an instrument route cannot repeat model ids")

    def reachable_model_ids(self, instruments: frozenset[str]) -> frozenset[str]:
        model_ids = set(self.model_ids)
        for instrument in instruments:
            model_ids.update(self.instrument_model_ids.get(instrument, ()))
        return frozenset(model_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation_type": self.operation_type,
            "service_id": self.service_id,
            "kernel_id": self.kernel_id,
            "model_ids": list(self.model_ids),
            "instrument_model_ids": {
                instrument: list(model_ids)
                for instrument, model_ids in sorted(self.instrument_model_ids.items())
            },
            "model_free_reason": self.model_free_reason,
        }


@dataclass(frozen=True)
class ReachabilityFinding:
    check_id: str
    severity: str
    message: str
    task_id: str | None = None
    operation_type: str | None = None
    model_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "severity": self.severity,
            "message": self.message,
            "task_id": self.task_id,
            "operation_type": self.operation_type,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class ModelProviderRegistry:
    providers: tuple[ModelProviderContract, ...]

    def __post_init__(self) -> None:
        model_ids = [provider.model_id for provider in self.providers]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("duplicate model provider ids are not allowed")

    def get(self, model_id: str) -> ModelProviderContract:
        for provider in self.providers:
            if provider.model_id == model_id:
                return provider
        raise ValueError(f"unknown model provider {model_id!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            provider.model_id: provider.to_dict()
            for provider in sorted(self.providers, key=lambda item: item.model_id)
        }


@dataclass(frozen=True)
class ModelReachabilityRegistry:
    providers: ModelProviderRegistry
    routes: tuple[OperationModelRoute, ...]

    def __post_init__(self) -> None:
        operations = [route.operation_type for route in self.routes]
        if len(operations) != len(set(operations)):
            raise ValueError("duplicate operation model routes are not allowed")

    def route_for_operation(self, operation_type: str) -> OperationModelRoute:
        for route in self.routes:
            if route.operation_type == operation_type:
                return route
        raise ValueError(f"operation {operation_type!r} has no model route")

    def structural_findings(self) -> tuple[ReachabilityFinding, ...]:
        service_registry = DomainServiceRegistry.default()
        kernel_registry = OperationKernelRegistry.default()
        findings: list[ReachabilityFinding] = []
        route_operations = {route.operation_type for route in self.routes}
        for operation in sorted(set(OPERATION_TYPES) - route_operations):
            findings.append(
                ReachabilityFinding(
                    "operation_route_coverage",
                    "error",
                    "operation has no service/kernel/model route",
                    operation_type=operation,
                )
            )
        for operation in sorted(route_operations - set(OPERATION_TYPES)):
            findings.append(
                ReachabilityFinding(
                    "operation_route_unknown",
                    "error",
                    "route references an unknown operation",
                    operation_type=operation,
                )
            )
        routed_model_ids: set[str] = set()
        for route in self.routes:
            expected_service = service_registry.service_id_for_operation(route.operation_type)
            if route.service_id != expected_service:
                findings.append(
                    ReachabilityFinding(
                        "service_route_alignment",
                        "error",
                        f"expected service {expected_service!r}, got {route.service_id!r}",
                        operation_type=route.operation_type,
                    )
                )
            expected_kernel = kernel_registry.kernel_id_for_operation(route.operation_type)
            if route.kernel_id != expected_kernel:
                findings.append(
                    ReachabilityFinding(
                        "kernel_route_alignment",
                        "error",
                        f"expected kernel {expected_kernel!r}, got {route.kernel_id!r}",
                        operation_type=route.operation_type,
                    )
                )
            route_model_ids = route.reachable_model_ids(frozenset(INSTRUMENTS))
            routed_model_ids.update(route_model_ids)
            for model_id in sorted(route_model_ids):
                try:
                    provider = self.providers.get(model_id)
                except ValueError:
                    findings.append(
                        ReachabilityFinding(
                            "provider_coverage",
                            "error",
                            "route references an unregistered model provider",
                            operation_type=route.operation_type,
                            model_id=model_id,
                        )
                    )
                    continue
                if provider.role is ModelExecutionRole.REFERENCE:
                    findings.append(
                        ReachabilityFinding(
                            "reference_runtime_separation",
                            "error",
                            "reference-only provider cannot appear in a runtime route",
                            operation_type=route.operation_type,
                            model_id=model_id,
                        )
                    )
                if route.operation_type not in provider.intended_operations:
                    findings.append(
                        ReachabilityFinding(
                            "provider_operation_alignment",
                            "error",
                            "route operation is absent from provider intended_operations",
                            operation_type=route.operation_type,
                            model_id=model_id,
                        )
                    )
        for provider in self.providers.providers:
            if not _provider_symbol_exists(provider.provider_path):
                findings.append(
                    ReachabilityFinding(
                        "provider_path_resolution",
                        "error",
                        "provider_path does not resolve to an importable symbol",
                        model_id=provider.model_id,
                    )
                )
            if provider.runtime_reachable and provider.model_id not in routed_model_ids:
                findings.append(
                    ReachabilityFinding(
                        "runtime_provider_unrouted",
                        "error",
                        "runtime or diagnostic provider has no operation route",
                        model_id=provider.model_id,
                    )
                )
            if not provider.runtime_reachable and provider.model_id in routed_model_ids:
                findings.append(
                    ReachabilityFinding(
                        "reference_provider_routed",
                        "error",
                        "reference provider unexpectedly appears in a runtime route",
                        model_id=provider.model_id,
                    )
                )
        return tuple(findings)

    def reachable_model_ids(self, profile: TaskRuntimeProfile) -> frozenset[str]:
        model_ids: set[str] = set()
        for operation in profile.allowed_operations:
            route = self.route_for_operation(operation)
            model_ids.update(route.reachable_model_ids(profile.allowed_instruments))
        return frozenset(model_ids)

    def task_report(self, task: TaskSpec) -> dict[str, Any]:
        profile = TaskRuntimeProfile.from_task(task)
        reachable = self.reachable_model_ids(profile)
        declared_by_model: dict[str, set[str]] = {}
        for module in task.kernel_maturity.modules:
            for model_id in module.model_ids:
                declared_by_model.setdefault(model_id, set()).add(module.module_id)
        declared = frozenset(declared_by_model)
        declared_only = sorted(declared - reachable)
        runtime_only = sorted(reachable - declared)
        findings = [
            ReachabilityFinding(
                "declared_model_unreachable",
                "warning",
                "task maturity declares a model that no allowed operation can reach",
                task_id=task.task_id,
                model_id=model_id,
            )
            for model_id in declared_only
        ]
        findings.extend(
            ReachabilityFinding(
                "runtime_model_undeclared",
                "warning",
                "allowed runtime path reaches a model absent from task maturity",
                task_id=task.task_id,
                model_id=model_id,
            )
            for model_id in runtime_only
        )
        return {
            "task_id": task.task_id,
            "task_contract_hash": task.contract_hash,
            "runtime_profile_hash": profile.profile_hash,
            "declared_model_ids": sorted(declared),
            "reachable_model_ids": sorted(reachable),
            "declared_but_unreachable": declared_only,
            "reachable_but_undeclared": runtime_only,
            "alignment_status": "aligned" if not findings else "gaps_detected",
            "routes": [
                self.route_for_operation(operation).to_dict()
                for operation in sorted(profile.allowed_operations)
            ],
            "findings": [finding.to_dict() for finding in findings],
        }


def _provider(
    model_id: str,
    module_id: str,
    maturity: MaturityLevel,
    role: ModelExecutionRole,
    provider_path: str,
    operations: tuple[str, ...],
    *,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    units: dict[str, str],
    diagnostics: tuple[str, ...],
    provenance: tuple[str, ...],
) -> ModelProviderContract:
    return ModelProviderContract(
        model_id=model_id,
        module_id=module_id,
        maturity=maturity,
        role=role,
        provider_path=provider_path,
        input_fields=inputs,
        output_fields=outputs,
        units=units,
        validity_checks=("typed input schema", "declared operating-domain bounds"),
        diagnostic_fields=diagnostics,
        failure_policy="raise a typed failure or trigger transactional rollback",
        provenance=provenance,
        intended_operations=operations,
    )


def _provider_symbol_exists(provider_path: str) -> bool:
    parts = provider_path.split(".")
    for index in range(len(parts), 0, -1):
        try:
            target: Any = importlib.import_module(".".join(parts[:index]))
        except ModuleNotFoundError:
            continue
        for attribute in parts[index:]:
            if not hasattr(target, attribute):
                return False
            target = getattr(target, attribute)
        return True
    return False


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").strip().strip("/")


def _path_overlaps(left: str, right: str) -> bool:
    left_parts = _normalized_path(left).split("/")
    right_parts = _normalized_path(right).split("/")
    shared = min(len(left_parts), len(right_parts))
    return left_parts[:shared] == right_parts[:shared]


def audit_shared_claim_ownership(project_root: str | Path) -> dict[str, Any]:
    """Validate exact path ownership across active claims."""

    root = Path(project_root)
    claim_dir = root / "claims" / "active"
    findings: list[ReachabilityFinding] = []
    checked_claims = 0
    owned: list[tuple[str, str]] = []
    task_ids: set[str] = set()
    for path in sorted(claim_dir.glob("*.json")):
        checked_claims += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            findings.append(
                ReachabilityFinding(
                    "claim_read_error",
                    "error",
                    f"cannot read active claim {path.name}: {error}",
                )
            )
            continue
        task_id = str(payload.get("task_id", ""))
        if not task_id:
            findings.append(
                ReachabilityFinding(
                    "claim_identity_missing",
                    "error",
                    f"active claim {path.name!r} has no task_id",
                )
            )
            continue
        if task_id in task_ids:
            findings.append(
                ReachabilityFinding(
                    "duplicate_active_task_id",
                    "error",
                    f"active task_id {task_id!r} is duplicated",
                    task_id=task_id,
                )
            )
        task_ids.add(task_id)
        for owned_path in payload.get("owned_paths", ()):
            normalized = _normalized_path(str(owned_path))
            if not normalized or normalized.startswith("../") or "/../" in normalized:
                findings.append(
                    ReachabilityFinding(
                        "invalid_owned_path",
                        "error",
                        f"claim {task_id!r} has invalid owned path {normalized!r}",
                        task_id=task_id,
                    )
                )
                continue
            for other_task_id, other_path in owned:
                if _path_overlaps(normalized, other_path):
                    findings.append(
                        ReachabilityFinding(
                            "active_claim_path_overlap",
                            "error",
                            f"claim {task_id!r} path {normalized!r} overlaps "
                            f"claim {other_task_id!r} path {other_path!r}",
                            task_id=task_id,
                        )
                    )
            owned.append((task_id, normalized))
    return {
        "passed": not findings,
        "policy_version": "chemworld-exact-active-claim-ownership-0.1",
        "checked_active_claim_count": checked_claims,
        "checked_owned_path_count": len(owned),
        "findings": [finding.to_dict() for finding in findings],
    }


def default_model_provider_registry() -> ModelProviderRegistry:
    state_inputs = ("species_amounts_mol", "temperature_K", "volume_L")
    state_units = {
        "species_amounts_mol": "mol",
        "temperature_K": "K",
        "volume_L": "L",
    }
    providers = (
        _provider(
            "reaction_ode_mass_action_arrhenius_reference_slice",
            "reaction_kinetics",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.physchem.reaction_network.ReactionNetworkSpec.reaction_rates",
            ("heat", "wait", "run_flow"),
            inputs=("amounts_mol", "volume_L", "temperature_K"),
            outputs=("reaction_rates_mol_s",),
            units={
                "amounts_mol": "mol",
                "volume_L": "L",
                "temperature_K": "K",
                "reaction_rates_mol_s": "mol/s",
            },
            diagnostics=("network_id", "conservation_laws", "rate_basis"),
            provenance=(
                "chemworld.physchem.reaction_network.ReactionNetworkSpec",
                "tests/test_reaction_kinetics_reference.py",
                "tests/test_reaction_reactor_runtime_integration.py",
            ),
        ),
        _provider(
            "dynamic_batch_heat_release_jacket_sampling",
            "reactors",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.physchem.batch_reactors.DynamicBatchReactorModel.simulate",
            ("heat", "wait"),
            inputs=(*state_inputs, "duration_s", "target_temperature_K"),
            outputs=("temperature_K", "pressure_Pa", "species_amounts_mol"),
            units={
                **state_units,
                "duration_s": "s",
                "target_temperature_K": "K",
                "pressure_Pa": "Pa",
            },
            diagnostics=(
                "energy_balance_residual_J",
                "solver_diagnostic",
                "maximum_conservation_drift_mol",
            ),
            provenance=(
                "chemworld.physchem.batch_reactors.DynamicBatchReactorModel",
                "tests/test_reactor_reference.py",
                "tests/test_reaction_reactor_runtime_integration.py",
            ),
        ),
        reaction_rate_provider_contract(),
        instrument_runtime_provider_contract(),
        _provider(
            "beer_lambert_uvvis",
            "spectroscopy_instruments",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.world.spectra.uvvis_spectrum",
            ("measure",),
            inputs=("public_species_amounts", "wavelength_nm", "path_length_cm"),
            outputs=("absorbance", "spectrum"),
            units={"wavelength_nm": "nm", "path_length_cm": "cm", "absorbance": "1"},
            diagnostics=("peak_count", "spectrum_model_id"),
            provenance=("beer-lambert-public-slice",),
        ),
        _provider(
            "chromatography_retention_plate",
            "spectroscopy_instruments",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.runtime.observation_services.ChemWorldObservationKernel.observe",
            ("measure",),
            inputs=("public_species_amounts", "method", "flow_rate"),
            outputs=("retention_times", "peak_areas", "chromatogram"),
            units={"retention_times": "min", "peak_areas": "a.u.", "flow_rate": "mL/min"},
            diagnostics=("resolution", "plate_count"),
            provenance=("chromatography-retention-plate-slice",),
        ),
        spectroscopy_identifiability_provider_contract(),
        stability_aware_lle_provider_contract(),
        sorbent_drying_provider_contract(),
        vacuum_concentration_provider_contract(),
        transfer_provider_contract(),
        crystallization_runtime_provider_contract(),
        duty_limited_distillation_provider_contract(),
        _provider(
            "chemworld_geometry_resolved_pfr_v2",
            "continuous_flow",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.runtime.flow_services.ChemWorldFlowServices.run_flow",
            ("run_flow",),
            inputs=("geometry", "flow_rate", "thermal_boundary"),
            outputs=("pressure_profile", "reynolds_number", "energy_ledger"),
            units={"geometry": "m", "flow_rate": "L/s", "pressure_profile": "Pa"},
            diagnostics=("pressure_drop_Pa", "reynolds_number", "solver_diagnostic"),
            provenance=("chemworld-geometry-resolved-pfr-v2",),
        ),
        _provider(
            "nernst_butler_volmer_faradaic_v1",
            "electrochemistry",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.runtime.electrochemical_services.ChemWorldElectrochemicalServices.electrolyze",
            ("electrolyze",),
            inputs=("activities", "potential_V", "current_A", "duration_s"),
            outputs=("converted_mol", "selectivity", "electrical_work_J"),
            units={"potential_V": "V", "current_A": "A", "duration_s": "s"},
            diagnostics=("overpotential_V", "faradaic_efficiency", "ohmic_loss_J"),
            provenance=("nernst-butler-volmer-faradaic-v1",),
        ),
        _provider(
            "diffusion_layer_limiting_current_v1",
            "electrochemistry",
            MaturityLevel.PROFESSIONAL_CANDIDATE,
            ModelExecutionRole.RUNTIME,
            "chemworld.physchem.electrochem_transport.diffusion_layer_current_response",
            ("electrolyze",),
            inputs=("bulk_concentration", "diffusivity", "diffusion_length", "area"),
            outputs=("limiting_current_A", "productive_charge_C", "converted_mol"),
            units={
                "bulk_concentration": "mol/m^3",
                "diffusivity": "m^2/s",
                "diffusion_length": "m",
                "area": "m^2",
            },
            diagnostics=("charge_balance_residual_C", "material_balance_residual_mol"),
            provenance=("diffusion-layer-limiting-current-v1",),
        ),
        _provider(
            "randles_double_layer_transient_v1",
            "electrochemistry",
            MaturityLevel.PROFESSIONAL_CANDIDATE,
            ModelExecutionRole.RUNTIME,
            "chemworld.physchem.electrochem_double_layer.simulate_double_layer_potential_step",
            ("electrolyze",),
            inputs=("potential_step_V", "duration_s", "resistance_ohm", "capacitance_F"),
            outputs=("faradaic_charge_C", "capacitive_charge_C", "current_profile"),
            units={
                "potential_step_V": "V",
                "duration_s": "s",
                "resistance_ohm": "ohm",
                "capacitance_F": "F",
            },
            diagnostics=("time_constant_s", "charge_balance_residual_C"),
            provenance=("randles-double-layer-transient-v1",),
        ),
        _provider(
            "aqueous_acid_base_ph_observation",
            "equilibrium_chemistry",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.physchem.equilibrium_chemistry.solve_aqueous_electrolyte_equilibrium",
            ("measure", "electrolyze"),
            inputs=("aqueous_composition", "temperature_K", "acid_constant"),
            outputs=("pH", "acid_dissociation_fraction", "precipitation_signal"),
            units={"aqueous_composition": "mol/L", "temperature_K": "K", "pH": "1"},
            diagnostics=("charge_balance_residual", "precipitation_events"),
            provenance=("aqueous-acid-base-ph-observation",),
        ),
        _provider(
            "fixed_tp_ideal_gibbs_minimization",
            "equilibrium_chemistry",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.REFERENCE,
            "chemworld.physchem.equilibrium_chemistry.solve_gibbs_minimization",
            (),
            inputs=("species", "element_matrix", "temperature_K", "pressure_Pa"),
            outputs=("equilibrium_amounts", "total_gibbs_J"),
            units={"temperature_K": "K", "pressure_Pa": "Pa", "total_gibbs_J": "J"},
            diagnostics=("kkt_residual", "element_residual", "charge_residual"),
            provenance=("fixed-tp-ideal-gibbs-minimization",),
        ),
        _provider(
            "potentiometric_ph_public_reference",
            "spectroscopy_instruments",
            MaturityLevel.REFERENCE_VALIDATED,
            ModelExecutionRole.RUNTIME,
            "chemworld.world.spectra.ph_meter_signal",
            ("measure",),
            inputs=("pH", "instrument_noise", "measurement_seed"),
            outputs=("raw_signal", "processed_pH", "uncertainty"),
            units={"pH": "1", "instrument_noise": "pH", "uncertainty": "pH"},
            diagnostics=("instrument_id", "calibration_metadata"),
            provenance=("ph-meter-public-signal",),
        ),
    )
    return ModelProviderRegistry(providers)


def default_model_reachability_registry() -> ModelReachabilityRegistry:
    services = DomainServiceRegistry.default()
    kernels = OperationKernelRegistry.default()
    model_ids_by_operation: dict[str, tuple[str, ...]] = {
        "heat": (
            "reaction_ode_mass_action_arrhenius_reference_slice",
            "dynamic_batch_heat_release_jacket_sampling",
        ),
        "wait": (
            "reaction_ode_mass_action_arrhenius_reference_slice",
            "dynamic_batch_heat_release_jacket_sampling",
        ),
        "mix": ("chemworld_stability_aware_lle_vnext",),
        "wash": ("chemworld_stability_aware_lle_vnext",),
        "dry": ("chemworld_sorbent_drying_vnext",),
        "concentrate": ("chemworld_vacuum_concentration_vnext",),
        "transfer": ("chemworld_transfer_holdup_vnext",),
        "cool_crystallize": (
            "cooling_crystallization_population_balance_v1",
        ),
        "distill": ("chemworld_duty_limited_distillation_vnext",),
        "run_flow": (
            "reaction_ode_mass_action_arrhenius_reference_slice",
            "chemworld_geometry_resolved_pfr_v2",
        ),
        "electrolyze": (
            "nernst_butler_volmer_faradaic_v1",
            "diffusion_layer_limiting_current_v1",
            "randles_double_layer_transient_v1",
            "aqueous_acid_base_ph_observation",
        ),
        "measure": (
            "chemworld_validated_synthetic_instruments_v1",
        ),
    }
    instrument_models = {
        "hplc": ("chromatography_retention_plate",),
        "gc": ("chromatography_retention_plate",),
        "uvvis": ("beer_lambert_uvvis",),
        "ph_meter": (
            "aqueous_acid_base_ph_observation",
            "potentiometric_ph_public_reference",
        ),
        "final_assay": (),
    }
    routes: list[OperationModelRoute] = []
    for operation in OPERATION_TYPES:
        model_ids = model_ids_by_operation.get(operation, ())
        operation_instrument_models = instrument_models if operation == "measure" else {}
        routes.append(
            OperationModelRoute(
                operation_type=operation,
                service_id=services.service_id_for_operation(operation),
                kernel_id=kernels.kernel_id_for_operation(operation),
                model_ids=model_ids,
                instrument_model_ids=operation_instrument_models,
                model_free_reason=(
                    None
                    if model_ids or operation_instrument_models
                    else "typed ledger/equipment transition with no declared physical model"
                ),
            )
        )
    return ModelReachabilityRegistry(
        providers=default_model_provider_registry(),
        routes=tuple(routes),
    )


def audit_model_reachability(task_ids: tuple[str, ...] | None = None) -> dict[str, Any]:
    registry = default_model_reachability_registry()
    structural = registry.structural_findings()
    tasks = tuple(list_tasks()) if task_ids is None else tuple(get_task(item) for item in task_ids)
    task_reports = {task.task_id: registry.task_report(task) for task in tasks}
    gap_count = sum(
        len(report["declared_but_unreachable"]) + len(report["reachable_but_undeclared"])
        for report in task_reports.values()
    )
    return {
        "schema_version": MODEL_REACHABILITY_SCHEMA_VERSION,
        "world_law_id": tasks[0].world_law_id if tasks else None,
        "contract_integrity_passed": not any(
            finding.severity == "error" for finding in structural
        ),
        "declaration_alignment_status": "aligned" if gap_count == 0 else "gaps_detected",
        "declaration_gap_count": gap_count,
        "task_count": len(tasks),
        "provider_count": len(registry.providers.providers),
        "route_count": len(registry.routes),
        "providers": registry.providers.to_dict(),
        "routes": [route.to_dict() for route in registry.routes],
        "structural_findings": [finding.to_dict() for finding in structural],
        "tasks": task_reports,
    }


__all__ = [
    "MODEL_REACHABILITY_SCHEMA_VERSION",
    "ModelProviderRegistry",
    "ModelReachabilityRegistry",
    "OperationModelRoute",
    "ReachabilityFinding",
    "audit_model_reachability",
    "audit_shared_claim_ownership",
    "default_model_provider_registry",
    "default_model_reachability_registry",
]
