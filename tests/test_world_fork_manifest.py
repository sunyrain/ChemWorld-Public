from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from chemworld.foundation.world_fork_manifest import (
    WORK_I_V01_REQUIRED_COMPONENT_IDS,
    WORK_I_V01_REQUIRED_RULE_IDS,
    WORLD_COMPONENT_INVENTORY_ID,
    WORLD_COMPONENT_INVENTORY_SCHEMA_VERSION,
    WorldComponentInventory,
    WorldComponentManifestError,
    audit_world_component_inventory,
    load_world_component_inventory,
    world_component_manifest_schema,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_component_inventory_v0.1.json"
)
REPORT_PATH = (
    REPOSITORY_ROOT
    / "workstreams"
    / "arxiv_v1"
    / "reports"
    / "work-i-world-component-inventory-v0.1.json"
)


def _payload() -> dict[str, Any]:
    loaded = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_frozen_inventory_has_complete_component_partition() -> None:
    inventory = load_world_component_inventory(INVENTORY_PATH)

    assert inventory.schema_version == WORLD_COMPONENT_INVENTORY_SCHEMA_VERSION
    assert inventory.inventory_id == WORLD_COMPONENT_INVENTORY_ID
    assert set(inventory.component_by_id) == WORK_I_V01_REQUIRED_COMPONENT_IDS
    assert inventory.intervention_target_map == {
        "mechanism_or_constitutive_law": (
            "private_physics.constitutive_laws",
            "private_physics.reaction_mechanism",
        ),
        "material_law_counterfactual": ("private_physics.material_laws",),
    }

    public_components = [
        component for component in inventory.components if component.layer == "public_contract"
    ]
    assert len(public_components) == 9
    assert all(component.visibility == "public" for component in public_components)
    assert all(component.fork_policy == "invariant" for component in public_components)
    assert all(not component.allowed_intervention_classes for component in public_components)


def test_inventory_round_trip_and_digest_are_canonical() -> None:
    inventory = load_world_component_inventory(INVENTORY_PATH)
    round_tripped = WorldComponentInventory.from_dict(inventory.to_dict())

    assert round_tripped == inventory
    assert round_tripped.content_sha256 == (
        "654b710fcfb0a66232e4a3c6e14f1abb1dd6c24357e7eac995d23d11f64ee6da"
    )


def test_inventory_audit_resolves_every_implementation_anchor() -> None:
    inventory = load_world_component_inventory(INVENTORY_PATH)
    report = audit_world_component_inventory(inventory, repository_root=REPOSITORY_ROOT)

    assert report["passed"] is True
    assert report["summary"] == {
        "component_count": 17,
        "cross_component_rule_count": 5,
        "implementation_anchor_count": 32,
        "missing_implementation_anchor_count": 0,
        "layer_counts": {"identity": 2, "private_physics": 6, "public_contract": 9},
        "fork_policy_counts": {"derived": 2, "intervention_target": 3, "invariant": 12},
    }
    assert report["implementation_anchor_audit"]["missing"] == []
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_manifest_schema_freezes_root_and_component_shapes() -> None:
    schema = world_component_manifest_schema()

    assert schema["$id"] == WORLD_COMPONENT_INVENTORY_SCHEMA_VERSION
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version",
        "inventory_id",
        "status",
        "scope",
        "components",
        "cross_component_rules",
    }
    assert schema["properties"]["components"]["minItems"] == 17
    assert schema["properties"]["components"]["maxItems"] == 17
    assert schema["properties"]["components"]["items"]["additionalProperties"] is False
    assert set(schema["properties"]["components"]["items"]["properties"]) == {
        "component_id",
        "layer",
        "visibility",
        "fork_policy",
        "allowed_intervention_classes",
        "canonical_payload_sources",
        "implementation_anchors",
        "description",
    }
    assert schema["properties"]["scope"]["properties"]["agent_performance_claim_allowed"] == {
        "const": False
    }


def test_manifest_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["unexpected"] = True

    with pytest.raises(WorldComponentManifestError, match="unknown=\\['unexpected'\\]"):
        WorldComponentInventory.from_dict(payload)


def test_manifest_rejects_public_contract_mutation() -> None:
    payload = copy.deepcopy(_payload())
    action_component = next(
        component
        for component in payload["components"]
        if component["component_id"] == "public_contract.actions"
    )
    action_component["fork_policy"] = "intervention_target"
    action_component["allowed_intervention_classes"] = ["mechanism_or_constitutive_law"]

    with pytest.raises(WorldComponentManifestError, match="must be private physics"):
        WorldComponentInventory.from_dict(payload)


def test_manifest_rejects_claim_expansion_to_agent_performance() -> None:
    payload = _payload()
    payload["scope"]["agent_performance_claim_allowed"] = True

    with pytest.raises(WorldComponentManifestError, match="no agent-performance claim"):
        WorldComponentInventory.from_dict(payload)


def test_manifest_rejects_component_namespace_mismatch() -> None:
    payload = copy.deepcopy(_payload())
    action_component = next(
        component
        for component in payload["components"]
        if component["component_id"] == "public_contract.actions"
    )
    action_component["layer"] = "private_physics"
    action_component["visibility"] = "private"

    with pytest.raises(WorldComponentManifestError, match="namespace and layer disagree"):
        WorldComponentInventory.from_dict(payload)


def test_manifest_rejects_cross_component_rule_removal() -> None:
    payload = _payload()
    payload["cross_component_rules"] = payload["cross_component_rules"][:-1]

    with pytest.raises(WorldComponentManifestError, match="rule set mismatch"):
        WorldComponentInventory.from_dict(payload)
    assert {
        rule["rule_id"] for rule in _payload()["cross_component_rules"]
    } == WORK_I_V01_REQUIRED_RULE_IDS
