from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chemworld.foundation.world_fork_divergence import (
    DivergenceOracleError,
    DivergenceOracleSpec,
    evaluate_divergence_oracle,
)
from chemworld.foundation.world_fork_manifest import (
    WorldComponentInventory,
    load_world_component_inventory,
)
from chemworld.foundation.world_fork_spec import WorldForkSpec, build_world_fork_spec

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_component_inventory_v0.1.json"
)
SUITE_PATH = REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_divergence_v0.1.json"
REPORT_PATH = (
    REPOSITORY_ROOT
    / "workstreams"
    / "arxiv_v1"
    / "reports"
    / "work-i-world-fork-divergence-v0.1.json"
)


def _inventory() -> WorldComponentInventory:
    return load_world_component_inventory(INVENTORY_PATH)


def _suite() -> dict[str, Any]:
    value = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert value["evidence_status"] == "definition_fixture_not_execution_evidence"
    return value


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _fork_for_oracle(
    oracle: DivergenceOracleSpec,
    inventory: WorldComponentInventory,
    *,
    world_seed: int = 0,
) -> WorldForkSpec:
    component_ids = sorted(
        component.component_id
        for component in inventory.components
        if component.layer != "identity"
    )
    parent = {
        component_id: _digest(f"f04:{oracle.intervention_class}:base:{component_id}")
        for component_id in component_ids
    }
    child = dict(parent)
    child[oracle.target_component_id] = _digest(
        f"f04:{oracle.intervention_class}:fork:{oracle.target_component_id}"
    )
    intervention_payload = (
        {
            "kind": "material_law_counterfactual",
            "material_field": "solvent",
            "public_to_baseline": [1, 0, 2, 3],
        }
        if oracle.intervention_class == "material_law_counterfactual"
        else {
            "kind": "mechanism_family",
            "mode": "constitutive_law_family",
            "severity": 1.0,
        }
    )
    return build_world_fork_spec(
        inventory=inventory,
        world_seed=world_seed,
        intervention_class=oracle.intervention_class,
        target_component_id=oracle.target_component_id,
        intervention_payload=intervention_payload,
        parent_component_sha256=parent,
        child_component_sha256=child,
    )


def _evaluations() -> list[dict[str, Any]]:
    inventory = _inventory()
    suite = _suite()
    oracle_by_id = {
        oracle.oracle_id: oracle
        for oracle in (
            DivergenceOracleSpec.from_dict(payload, inventory=inventory)
            for payload in suite["oracles"]
        )
    }
    return [
        evaluate_divergence_oracle(
            oracle=oracle_by_id[fixture["oracle_id"]],
            spec=_fork_for_oracle(
                oracle_by_id[fixture["oracle_id"]],
                inventory,
                world_seed=fixture["world_seed"],
            ),
            inventory=inventory,
            parent_checkpoints=fixture["parent_checkpoints"],
            child_checkpoints=fixture["child_checkpoints"],
        )
        for fixture in suite["qualification_fixtures"]
    ]


def test_frozen_suite_covers_both_intervention_classes_and_channels() -> None:
    inventory = _inventory()
    oracles = [
        DivergenceOracleSpec.from_dict(payload, inventory=inventory)
        for payload in _suite()["oracles"]
    ]

    assert {oracle.intervention_class for oracle in oracles} == {
        "mechanism_or_constitutive_law",
        "material_law_counterfactual",
    }
    assert all(
        {expectation.channel for expectation in oracle.expectations}
        == {"physical_state", "public_observation"}
        for oracle in oracles
    )
    assert all(oracle.oracle_id == oracle.expected_oracle_id for oracle in oracles)


def test_qualification_fixtures_pass_and_match_frozen_report() -> None:
    evaluations = _evaluations()
    report = {
        "report_version": "chemworld-work-i-divergence-suite-report-0.1",
        "evidence_status": "definition_fixture_not_execution_evidence",
        "case_count": len(evaluations),
        "physical_expectation_count": sum(
            item["physical_expectation_count"] for item in evaluations
        ),
        "physical_expectation_pass_count": sum(
            item["physical_expectation_pass_count"] for item in evaluations
        ),
        "observation_expectation_count": sum(
            item["observation_expectation_count"] for item in evaluations
        ),
        "observation_expectation_pass_count": sum(
            item["observation_expectation_pass_count"] for item in evaluations
        ),
        "cases": evaluations,
        "passed": all(item["passed"] for item in evaluations),
    }

    assert report["passed"] is True
    assert report["case_count"] == 2
    assert report["physical_expectation_pass_count"] == 2
    assert report["observation_expectation_pass_count"] == 2
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_oracle_rejects_missing_channel_and_zero_tolerances() -> None:
    inventory = _inventory()
    payload = copy.deepcopy(_suite()["oracles"][0])
    payload["expectations"] = payload["expectations"][:1]
    with pytest.raises(DivergenceOracleError, match="must include physical_state"):
        DivergenceOracleSpec.from_dict(payload, inventory=inventory)

    zero = copy.deepcopy(_suite()["oracles"][0])
    zero["expectations"][0]["minimum_absolute_delta"] = 0.0
    zero["expectations"][0]["minimum_relative_delta"] = 0.0
    with pytest.raises(DivergenceOracleError, match="at least one divergence tolerance"):
        DivergenceOracleSpec.from_dict(zero, inventory=inventory)


def test_oracle_rejects_content_id_tampering() -> None:
    inventory = _inventory()
    payload = copy.deepcopy(_suite()["oracles"][0])
    payload["oracle_id"] = "chemworld-divergence-forged"

    with pytest.raises(DivergenceOracleError, match="oracle_id does not match"):
        DivergenceOracleSpec.from_dict(payload, inventory=inventory)


def test_below_tolerance_or_wrong_direction_fails() -> None:
    inventory = _inventory()
    oracle = DivergenceOracleSpec.from_dict(_suite()["oracles"][0], inventory=inventory)
    spec = _fork_for_oracle(oracle, inventory)
    parent = {
        "post_probe": {
            "physical_state": {"reaction_extent": 0.62},
            "public_observation": {"yield": 0.60},
        }
    }
    child = {
        "post_probe": {
            "physical_state": {"reaction_extent": 0.61},
            "public_observation": {"yield": 0.72},
        }
    }

    report = evaluate_divergence_oracle(
        oracle=oracle,
        spec=spec,
        inventory=inventory,
        parent_checkpoints=parent,
        child_checkpoints=child,
    )

    assert report["passed"] is False
    physical, observation = report["expectation_results"]
    assert physical["magnitude_passed"] is False
    assert physical["direction_passed"] is True
    assert observation["magnitude_passed"] is True
    assert observation["direction_passed"] is False


@pytest.mark.parametrize(
    ("child_value", "failure_code"),
    [(None, "missing_checkpoint_or_field"), (float("inf"), "nonfinite_value")],
)
def test_missing_or_nonfinite_value_has_deterministic_failure(
    child_value: float | None,
    failure_code: str,
) -> None:
    inventory = _inventory()
    oracle = DivergenceOracleSpec.from_dict(_suite()["oracles"][0], inventory=inventory)
    spec = _fork_for_oracle(oracle, inventory)
    parent = {
        "post_probe": {
            "physical_state": {"reaction_extent": 0.62},
            "public_observation": {"yield": 0.60},
        }
    }
    child_physical = {} if child_value is None else {"reaction_extent": child_value}
    child = {
        "post_probe": {
            "physical_state": child_physical,
            "public_observation": {"yield": 0.46},
        }
    }

    report = evaluate_divergence_oracle(
        oracle=oracle,
        spec=spec,
        inventory=inventory,
        parent_checkpoints=parent,
        child_checkpoints=child,
    )

    assert report["passed"] is False
    assert report["expectation_results"][0]["failure_code"] == failure_code


def test_oracle_rejects_incompatible_fork() -> None:
    inventory = _inventory()
    mechanism = DivergenceOracleSpec.from_dict(_suite()["oracles"][0], inventory=inventory)
    material = DivergenceOracleSpec.from_dict(_suite()["oracles"][1], inventory=inventory)

    with pytest.raises(DivergenceOracleError, match="intervention class does not match"):
        evaluate_divergence_oracle(
            oracle=mechanism,
            spec=_fork_for_oracle(material, inventory),
            inventory=inventory,
            parent_checkpoints={},
            child_checkpoints={},
        )


def test_claim_boundary_excludes_runtime_replay_and_performance() -> None:
    report = _evaluations()[0]

    assert report["claim_boundary"] == {
        "expected_response_divergence": True,
        "runtime_execution_claim": False,
        "exact_replay_claim": False,
        "agent_performance_claim": False,
    }
