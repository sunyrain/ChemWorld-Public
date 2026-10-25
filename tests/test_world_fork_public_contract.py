from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chemworld.foundation.world_fork_manifest import (
    WorldComponentInventory,
    load_world_component_inventory,
)
from chemworld.foundation.world_fork_public_contract import (
    PublicContractBundle,
    PublicContractCertificateError,
    audit_public_identity_leakage,
    build_public_contract_bundle,
    certify_public_contract_invariance,
)
from chemworld.foundation.world_fork_spec import WorldForkSpec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_component_inventory_v0.1.json"
)
FIXTURE_PATH = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_public_contract_v0.1.json"
)
REPORT_PATH = (
    REPOSITORY_ROOT
    / "workstreams"
    / "arxiv_v1"
    / "reports"
    / "work-i-world-fork-public-contract-v0.1.json"
)


def _inventory() -> WorldComponentInventory:
    return load_world_component_inventory(INVENTORY_PATH)


def _fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _spec_and_bundle() -> tuple[WorldComponentInventory, WorldForkSpec, PublicContractBundle]:
    inventory = _inventory()
    fixture = _fixture()
    assert fixture["evidence_status"] == "definition_fixture_not_execution_evidence"
    assert fixture["parent_bundle_ref"] == "public_contract_bundle"
    assert fixture["child_bundle_ref"] == "public_contract_bundle"
    spec = WorldForkSpec.from_dict(fixture["fork_spec"], inventory=inventory)
    bundle = PublicContractBundle.from_dict(fixture["public_contract_bundle"], inventory=inventory)
    return inventory, spec, bundle


def test_frozen_certificate_covers_all_public_components_and_matches_report() -> None:
    inventory, spec, bundle = _spec_and_bundle()
    certificate = certify_public_contract_invariance(
        spec=spec,
        inventory=inventory,
        parent_bundle=bundle,
        child_bundle=bundle,
    )
    report = certificate.to_dict()

    assert report["passed"] is True
    assert report["public_component_count"] == 9
    assert report["invariant_component_count"] == 9
    assert report["spec_bound_component_count"] == 9
    assert report["identity_leakage_finding_count"] == 0
    assert all(item["passed"] for item in report["component_results"])
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_bundle_is_canonical_and_bound_to_fork_component_digests() -> None:
    _inventory_value, spec, bundle = _spec_and_bundle()

    assert tuple(bundle.component_payloads) == tuple(sorted(bundle.component_payloads))
    assert bundle.component_sha256 == {
        component_id: spec.parent.component_sha256[component_id]
        for component_id in bundle.component_payloads
    }
    assert bundle.content_sha256 == (
        "77d79323b922d57909b23e6ddd36c48d2412ab7bac8fc3980dd4c307a802f957"
    )
    assert bundle.bundle_id == "chemworld-public-contract-77d79323b922d579"


def test_bundle_rejects_missing_component_and_nonfinite_payload() -> None:
    inventory, _spec, bundle = _spec_and_bundle()
    missing = dict(bundle.component_payloads)
    del missing["public_contract.task"]
    with pytest.raises(PublicContractCertificateError, match="component set mismatch"):
        build_public_contract_bundle(missing, inventory=inventory)

    nonfinite = copy.deepcopy(bundle.component_payloads)
    nonfinite["public_contract.resources"]["operation_limit"] = float("nan")
    with pytest.raises(PublicContractCertificateError, match="finite JSON data"):
        build_public_contract_bundle(nonfinite, inventory=inventory)


def test_child_contract_mutation_produces_failed_certificate() -> None:
    inventory, spec, bundle = _spec_and_bundle()
    child_payloads = copy.deepcopy(bundle.component_payloads)
    child_payloads["public_contract.actions"]["operations"].append("hidden_fork_action")
    child = build_public_contract_bundle(child_payloads, inventory=inventory)

    certificate = certify_public_contract_invariance(
        spec=spec,
        inventory=inventory,
        parent_bundle=bundle,
        child_bundle=child,
    )
    action_result = next(
        result
        for result in certificate.component_results
        if result.component_id == "public_contract.actions"
    )

    assert certificate.passed is False
    assert action_result.payloads_equal is False
    assert action_result.parent_spec_bound is True
    assert action_result.child_spec_bound is False
    assert sum(not result.passed for result in certificate.component_results) == 1


def test_identity_key_leak_is_detected_even_when_parent_and_child_match() -> None:
    inventory, spec, bundle = _spec_and_bundle()
    leaked_payloads = copy.deepcopy(bundle.component_payloads)
    leaked_payloads["public_contract.task"]["fork_id"] = spec.fork_id
    leaked = build_public_contract_bundle(leaked_payloads, inventory=inventory)

    findings = audit_public_identity_leakage(leaked, spec=spec)
    certificate = certify_public_contract_invariance(
        spec=spec,
        inventory=inventory,
        parent_bundle=leaked,
        child_bundle=leaked,
    )

    assert certificate.passed is False
    assert {finding.finding_kind for finding in findings} == {
        "forbidden_identity_key",
        "forbidden_identity_value",
    }
    assert all(finding.component_id == "public_contract.task" for finding in findings)


def test_identity_value_leak_is_detected_under_an_innocent_key() -> None:
    inventory, spec, bundle = _spec_and_bundle()
    leaked_payloads = copy.deepcopy(bundle.component_payloads)
    leaked_payloads["public_contract.observations"]["label"] = spec.child.lineage_sha256
    leaked = build_public_contract_bundle(leaked_payloads, inventory=inventory)

    findings = audit_public_identity_leakage(leaked, spec=spec)

    assert len(findings) == 1
    assert findings[0].finding_kind == "forbidden_identity_value"
    assert findings[0].path == "$.label"


def test_bundle_parser_rejects_unknown_root_field() -> None:
    inventory = _inventory()
    payload = copy.deepcopy(_fixture()["public_contract_bundle"])
    payload["bundle_id"] = "caller-supplied-identity"

    with pytest.raises(PublicContractCertificateError, match="unknown=\\['bundle_id'\\]"):
        PublicContractBundle.from_dict(payload, inventory=inventory)


def test_certificate_claim_boundary_does_not_expand_to_execution() -> None:
    inventory, spec, bundle = _spec_and_bundle()
    boundary = certify_public_contract_invariance(
        spec=spec,
        inventory=inventory,
        parent_bundle=bundle,
        child_bundle=bundle,
    ).to_dict()["claim_boundary"]

    assert boundary == {
        "public_interface_invariance": True,
        "fork_identity_non_disclosure": True,
        "physical_divergence_claim": False,
        "replay_claim": False,
        "agent_performance_claim": False,
    }
