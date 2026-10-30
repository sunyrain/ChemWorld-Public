"""Integrated WF-30 provider for bounded liquid transfer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.physchem.maturity import (
    MaturityLevel,
    ModelAdapterManifest,
    ModelExecutionRole,
    ModelProviderContract,
)
from chemworld.physchem.model_provider import ModelProviderResult
from chemworld.physchem.transfer_units import (
    IDAES_COMMIT,
    TRANSFER_MODEL_ID,
    TransferRequest,
    simulate_transfer,
)

OWNER_WORKSTREAM = "wf-30-transfer-holdup"
OWNED_PATHS = (
    "src/chemworld/physchem/transfer_units.py",
    "src/chemworld/physchem/transfer_adapter_manifest.py",
    "tests/test_transfer_units.py",
)


def transfer_provider_contract() -> ModelProviderContract:
    return ModelProviderContract(
        model_id=TRANSFER_MODEL_ID,
        module_id="separations",
        maturity=MaturityLevel.REFERENCE_VALIDATED,
        role=ModelExecutionRole.RUNTIME,
        provider_path=(
            "chemworld.physchem.transfer_adapter_manifest.TransferUnitProvider"
        ),
        input_fields=("request",),
        output_fields=("transfer_result",),
        units={"request": "TransferRequest", "transfer_result": "JSON material ledger"},
        validity_checks=(
            "request is a TransferRequest",
            "source and equipment volumes are finite and within declared capacity",
            "component amounts are finite, nonnegative, and explicitly tracked",
            "initial line and flush inventories match positive declared volumes",
            "component and volume closure remain within request tolerance",
        ),
        diagnostic_fields=(
            "material_balance_error_mol",
            "volume_balance_error_L",
            "withdrawn_source_volume_L",
            "target_delivered_volume_L",
            "final_line_volume_L",
            "source_delivery_fraction_of_withdrawn",
            "warnings",
        ),
        failure_policy=(
            "reject invalid domain inputs or return a failed result; never discard "
            "source, target, flush, or line inventory"
        ),
        provenance=(
            "WF-30 finite-volume FIFO transfer identities",
            f"IDAES {IDAES_COMMIT} component material-balance convention",
        ),
        intended_operations=("transfer",),
    )


class TransferUnitProvider:
    @property
    def model_contract(self) -> ModelProviderContract:
        return transfer_provider_contract()

    def validate_domain(self, inputs: Mapping[str, Any]) -> tuple[str, ...]:
        if not isinstance(inputs.get("request"), TransferRequest):
            return ("request must be a TransferRequest",)
        return ()

    def evaluate(self, inputs: Mapping[str, Any]) -> ModelProviderResult:
        violations = self.validate_domain(inputs)
        if violations:
            return ModelProviderResult(
                outputs={},
                diagnostics={
                    "material_balance_error_mol": None,
                    "volume_balance_error_L": None,
                    "withdrawn_source_volume_L": None,
                    "target_delivered_volume_L": None,
                    "final_line_volume_L": None,
                    "source_delivery_fraction_of_withdrawn": None,
                    "warnings": list(violations),
                },
                success=False,
                failure_reason="; ".join(violations),
                provenance=self.model_contract.provenance,
            )
        request = inputs["request"]
        result = simulate_transfer(request)
        return ModelProviderResult(
            outputs={"transfer_result": result.to_dict()},
            diagnostics={
                "material_balance_error_mol": result.material_balance_error_mol,
                "volume_balance_error_L": result.volume_balance_error_L,
                "withdrawn_source_volume_L": result.withdrawn_source_volume_L,
                "target_delivered_volume_L": result.target_delivered_volume_L,
                "final_line_volume_L": result.final_line_volume_L,
                "source_delivery_fraction_of_withdrawn": (
                    result.source_delivery_fraction_of_withdrawn
                ),
                "warnings": list(result.warnings),
            },
            warnings=result.warnings,
            provenance=result.provenance,
        )


def transfer_adapter_manifest() -> ModelAdapterManifest:
    return ModelAdapterManifest(
        adapter_id="wf-30-transfer-holdup",
        adapter_version="0.2",
        owner_workstream=OWNER_WORKSTREAM,
        provider_contract=transfer_provider_contract(),
        owned_paths=OWNED_PATHS,
        integration_operations=("transfer",),
        target_world_law="chemworld-physical-chemistry-vnext",
        status="integrated",
        replaces_model_ids=(),
    )


__all__ = [
    "OWNED_PATHS",
    "OWNER_WORKSTREAM",
    "TransferUnitProvider",
    "transfer_adapter_manifest",
    "transfer_provider_contract",
]
