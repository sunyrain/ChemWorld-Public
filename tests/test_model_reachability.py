from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.runtime.kernel_contracts import (
    ContractModelProviderStub,
    ModelProviderResult,
    PhysicalModelProvider,
)
from chemworld.runtime.model_reachability import (
    ModelReachabilityRegistry,
    audit_model_reachability,
    default_model_reachability_registry,
)
from chemworld.world.operations import OPERATION_TYPES


class _ProviderFixture:
    def __init__(self) -> None:
        self._contract = ModelProviderContract(
            model_id="fixture_model",
            module_id="fixture",
            maturity=MaturityLevel.LITE,
            role=ModelExecutionRole.RUNTIME,
            provider_path="tests.test_model_reachability._ProviderFixture",
            input_fields=("x",),
            output_fields=("y",),
            units={"x": "1", "y": "1"},
            validity_checks=("x is finite",),
            diagnostic_fields=("residual",),
            failure_policy="return failed result",
            provenance=("test-fixture",),
            intended_operations=("wait",),
        )

    @property
    def model_contract(self) -> ModelProviderContract:
        return self._contract

    def validate_domain(self, inputs: dict[str, Any]) -> tuple[str, ...]:
        return () if "x" in inputs else ("x is required",)

    def evaluate(self, inputs: dict[str, Any]) -> ModelProviderResult:
        errors = self.validate_domain(inputs)
        if errors:
            return ModelProviderResult(
                outputs={},
                diagnostics={"residual": None},
                success=False,
                failure_reason="; ".join(errors),
                provenance=("test-fixture",),
            )
        return ModelProviderResult(
            outputs={"y": float(inputs["x"]) * 2.0},
            diagnostics={"residual": 0.0},
            provenance=("test-fixture",),
        )


def _evaluate(provider: PhysicalModelProvider, value: float) -> ModelProviderResult:
    return provider.evaluate({"x": value})


def test_parallel_provider_protocol_has_typed_success_and_failure_results() -> None:
    provider = _ProviderFixture()
    result = _evaluate(provider, 2.0)
    assert result.success is True
    assert result.outputs["y"] == pytest.approx(4.0)
    failure = provider.evaluate({})
    assert failure.success is False
    assert failure.failure_reason == "x is required"
    with pytest.raises(ValueError, match="failure_reason"):
        ModelProviderResult(outputs={}, diagnostics={}, success=False)


def test_contract_provider_stub_is_shape_checked_and_deterministic() -> None:
    contract = _ProviderFixture().model_contract
    provider = ContractModelProviderStub(
        model_contract=contract,
        outputs={"y": 4.25},
        diagnostics={"residual": 0.0},
    )

    first = _evaluate(provider, 1.0)
    second = _evaluate(provider, 99.0)
    assert first == second
    assert first.outputs == {"y": 4.25}
    assert first.provenance[-1] == "contract-stub"
    assert "no physical model" in first.warnings[0]

    failure = provider.evaluate({})
    assert failure.success is False
    assert failure.failure_reason == "missing required input: x"
    assert failure.diagnostics == {"residual": None}

    with pytest.raises(ValueError, match=r"missing=\['y'\]"):
        ContractModelProviderStub(
            model_contract=contract,
            outputs={},
            diagnostics={"residual": 0.0},
        )
    with pytest.raises(ValueError, match=r"unexpected=\['extra'\]"):
        ContractModelProviderStub(
            model_contract=contract,
            outputs={"y": 1.0},
            diagnostics={"residual": 0.0, "extra": 1.0},
        )


def test_default_registry_covers_every_operation_and_provider_route() -> None:
    registry = default_model_reachability_registry()
    assert {route.operation_type for route in registry.routes} == set(OPERATION_TYPES)
    assert registry.structural_findings() == ()
    assert registry.route_for_operation("heat").model_ids == (
        "reaction_ode_mass_action_arrhenius_reference_slice",
        "dynamic_batch_heat_release_jacket_sampling",
    )
    assert registry.route_for_operation("dry").model_ids == (
        "chemworld_sorbent_drying_vnext",
    )
    assert registry.route_for_operation("concentrate").model_ids == (
        "chemworld_vacuum_concentration_vnext",
    )
    assert registry.route_for_operation("transfer").model_ids == (
        "chemworld_transfer_holdup_vnext",
    )
    assert registry.route_for_operation("mix").model_ids == (
        "chemworld_stability_aware_lle_vnext",
    )
    assert registry.route_for_operation("distill").model_ids == (
        "chemworld_duty_limited_distillation_vnext",
    )
    provider_ids = {provider.model_id for provider in registry.providers.providers}
    assert not {
        "chemworld_separation_proxy",
        "activity_corrected_extraction_train_v1",
        "lle_phase_stability_diagnostic_v1",
        "vle_shortcut_distillation",
    }.intersection(provider_ids)
    measure = registry.route_for_operation("measure")
    assert "beer_lambert_uvvis" in measure.instrument_model_ids["uvvis"]
    assert registry.providers.get("fixed_tp_ideal_gibbs_minimization").role is (
        ModelExecutionRole.REFERENCE
    )


def test_structural_audit_detects_service_route_drift() -> None:
    registry = default_model_reachability_registry()
    corrupted = ModelReachabilityRegistry(
        providers=registry.providers,
        routes=tuple(
            replace(route, service_id="incorrect_service")
            if route.operation_type == "heat"
            else route
            for route in registry.routes
        ),
    )
    findings = corrupted.structural_findings()
    assert any(finding.check_id == "service_route_alignment" for finding in findings)


def test_task_report_exposes_current_maturity_declaration_gaps() -> None:
    report = audit_model_reachability()
    assert report["contract_integrity_passed"] is True
    assert report["route_count"] == len(OPERATION_TYPES)
    assert report["declaration_alignment_status"] == "aligned"

    assay = report["tasks"]["reaction-to-assay"]
    assert assay["alignment_status"] == "aligned"

    partition = report["tasks"]["partition-discovery"]
    assert partition["declared_but_unreachable"] == []
    assert partition["reachable_but_undeclared"] == []

    equilibrium = report["tasks"]["equilibrium-characterization"]
    assert equilibrium["declared_but_unreachable"] == []
    assert equilibrium["reachable_but_undeclared"] == []


def test_adapter_manifest_round_trip_is_hash_bound() -> None:
    provider = _ProviderFixture().model_contract
    manifest = ModelAdapterManifest(
        adapter_id="fixture_wait_adapter",
        adapter_version="0.1",
        owner_workstream="wf-test",
        provider_contract=provider,
        owned_paths=("src/example/provider.py",),
        integration_operations=("wait",),
        target_world_law="chemworld-physical-chemistry-vnext",
    )
    payload = manifest.to_dict()
    assert ModelAdapterManifest.from_dict(payload).to_dict() == payload
    payload["adapter_version"] = "tampered"
    with pytest.raises(ValueError, match="hash mismatch"):
        ModelAdapterManifest.from_dict(payload)
