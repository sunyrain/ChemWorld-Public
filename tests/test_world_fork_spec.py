from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from chemworld.foundation.world_fork_manifest import (
    WorldComponentInventory,
    load_world_component_inventory,
)
from chemworld.foundation.world_fork_spec import (
    DERIVED_IDENTITY_FIELDS,
    WorldForkSpec,
    WorldForkSpecError,
    audit_world_fork_spec,
    build_world_fork_spec,
    child_lineage_sha256,
    root_lineage_sha256,
    world_snapshot_sha256,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_component_inventory_v0.1.json"
)
SPEC_PATH = REPOSITORY_ROOT / "configs" / "benchmark" / "work_i_world_fork_spec_v0.1.json"
REPORT_PATH = (
    REPOSITORY_ROOT / "workstreams" / "arxiv_v1" / "reports" / "work-i-world-fork-spec-v0.1.json"
)


def _inventory() -> WorldComponentInventory:
    return load_world_component_inventory(INVENTORY_PATH)


def _payload() -> dict[str, Any]:
    loaded = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_frozen_fork_spec_binds_lineage_inventory_and_single_diff() -> None:
    inventory = _inventory()
    spec = WorldForkSpec.from_dict(_payload(), inventory=inventory)

    assert spec.content_sha256 == (
        "084ecfdb2f8d11916474678a651f26cd3afeb6e721395636c2fa5b5b3b873f16"
    )
    assert spec.component_diff.changed_component_ids == ("private_physics.constitutive_laws",)
    assert spec.component_diff.derived_identity_fields == DERIVED_IDENTITY_FIELDS
    assert spec.child.lineage_sha256 == child_lineage_sha256(
        inventory_sha256=inventory.content_sha256,
        parent_world_sha256=spec.parent.world_sha256,
        parent_lineage_sha256=spec.parent.lineage_sha256,
        child_world_sha256=spec.child.world_sha256,
        intervention_sha256=spec.intervention_sha256,
        target_component_id=spec.target_component_id,
    )
    assert spec.fork_id.endswith(spec.child.lineage_sha256[:16])


def test_frozen_fork_audit_is_reproducible_and_claim_bounded() -> None:
    inventory = _inventory()
    spec = WorldForkSpec.from_dict(_payload(), inventory=inventory)
    report = audit_world_fork_spec(spec, inventory=inventory)

    assert report["passed"] is True
    assert report["public_contract_invariant"] is True
    assert report["public_contract_component_count"] == 9
    assert report["claim_boundary"] == {
        "single_private_physics_target": True,
        "execution_claim": False,
        "divergence_claim": False,
        "agent_performance_claim": False,
    }
    assert json.loads(REPORT_PATH.read_text(encoding="utf-8")) == report


def test_builder_supports_material_law_target_and_nested_parent_lineage() -> None:
    inventory = _inventory()
    frozen = WorldForkSpec.from_dict(_payload(), inventory=inventory)
    parent = dict(frozen.parent.component_sha256)
    child = dict(parent)
    child["private_physics.material_laws"] = _digest("nested-material-law-child")

    spec = build_world_fork_spec(
        inventory=inventory,
        world_seed=2,
        intervention_class="material_law_counterfactual",
        target_component_id="private_physics.material_laws",
        intervention_payload={
            "kind": "material_law_counterfactual",
            "material_field": "solvent",
            "public_to_baseline": [1, 0, 2, 3],
        },
        parent_component_sha256=parent,
        child_component_sha256=child,
        parent_lineage_sha256=frozen.child.lineage_sha256,
    )

    assert spec.parent.lineage_sha256 == frozen.child.lineage_sha256
    assert spec.component_diff.changed_component_ids == ("private_physics.material_laws",)
    assert spec.parent.world_sha256 == world_snapshot_sha256(
        parent,
        inventory_sha256=inventory.content_sha256,
    )
    assert spec.child.lineage_sha256 != spec.parent.lineage_sha256


def test_builder_supports_reaction_mechanism_target() -> None:
    inventory = _inventory()
    frozen = WorldForkSpec.from_dict(_payload(), inventory=inventory)
    parent = dict(frozen.parent.component_sha256)
    child = dict(parent)
    child["private_physics.reaction_mechanism"] = _digest("reaction-mechanism-child")

    spec = build_world_fork_spec(
        inventory=inventory,
        world_seed=1,
        intervention_class="mechanism_or_constitutive_law",
        target_component_id="private_physics.reaction_mechanism",
        intervention_payload={
            "kind": "mechanism_family",
            "mode": "rate_law_family",
            "severity": 1.0,
        },
        parent_component_sha256=parent,
        child_component_sha256=child,
    )

    assert spec.component_diff.changed_component_ids == ("private_physics.reaction_mechanism",)
    assert spec.parent.lineage_sha256 == root_lineage_sha256(
        inventory_sha256=inventory.content_sha256,
        world_sha256=spec.parent.world_sha256,
    )


def test_root_lineage_is_deterministic_and_world_bound() -> None:
    inventory = _inventory()
    spec = WorldForkSpec.from_dict(_payload(), inventory=inventory)

    assert spec.parent.lineage_sha256 == root_lineage_sha256(
        inventory_sha256=inventory.content_sha256,
        world_sha256=spec.parent.world_sha256,
    )
    assert (
        root_lineage_sha256(
            inventory_sha256=inventory.content_sha256,
            world_sha256=_digest("different-world"),
        )
        != spec.parent.lineage_sha256
    )


def test_builder_rejects_more_than_one_changed_component() -> None:
    inventory = _inventory()
    frozen = WorldForkSpec.from_dict(_payload(), inventory=inventory)
    parent = dict(frozen.parent.component_sha256)
    child = dict(frozen.child.component_sha256)
    child["private_physics.randomness"] = _digest("unmatched-randomness")

    with pytest.raises(WorldForkSpecError, match="change exactly its declared target"):
        build_world_fork_spec(
            inventory=inventory,
            world_seed=0,
            intervention_class="mechanism_or_constitutive_law",
            target_component_id="private_physics.constitutive_laws",
            intervention_payload={"kind": "mechanism_family"},
            parent_component_sha256=parent,
            child_component_sha256=child,
        )


def test_builder_rejects_public_contract_as_target() -> None:
    inventory = _inventory()
    frozen = WorldForkSpec.from_dict(_payload(), inventory=inventory)
    parent = dict(frozen.parent.component_sha256)
    child = dict(parent)
    child["public_contract.actions"] = _digest("changed-action-contract")

    with pytest.raises(WorldForkSpecError, match="not an intervention target"):
        build_world_fork_spec(
            inventory=inventory,
            world_seed=0,
            intervention_class="mechanism_or_constitutive_law",
            target_component_id="public_contract.actions",
            intervention_payload={"kind": "mechanism_family"},
            parent_component_sha256=parent,
            child_component_sha256=child,
        )


def test_builder_rejects_incompatible_intervention_class() -> None:
    inventory = _inventory()
    frozen = WorldForkSpec.from_dict(_payload(), inventory=inventory)

    with pytest.raises(WorldForkSpecError, match="incompatible with target"):
        build_world_fork_spec(
            inventory=inventory,
            world_seed=0,
            intervention_class="material_law_counterfactual",
            target_component_id="private_physics.constitutive_laws",
            intervention_payload={"kind": "material_law_counterfactual"},
            parent_component_sha256=frozen.parent.component_sha256,
            child_component_sha256=frozen.child.component_sha256,
        )


def test_parser_rejects_intervention_or_lineage_tampering() -> None:
    inventory = _inventory()
    intervention_tamper = copy.deepcopy(_payload())
    intervention_tamper["intervention_payload"]["severity"] = 0.5
    with pytest.raises(WorldForkSpecError, match="does not bind intervention_payload"):
        WorldForkSpec.from_dict(intervention_tamper, inventory=inventory)

    lineage_tamper = copy.deepcopy(_payload())
    lineage_tamper["child"]["lineage_sha256"] = _digest("forged-lineage")
    with pytest.raises(WorldForkSpecError, match="child lineage does not bind"):
        WorldForkSpec.from_dict(lineage_tamper, inventory=inventory)


def test_parser_rejects_missing_component_and_boolean_seed() -> None:
    inventory = _inventory()
    missing_component = copy.deepcopy(_payload())
    del missing_component["parent"]["component_sha256"]["public_contract.task"]
    with pytest.raises(WorldForkSpecError, match="component set mismatch"):
        WorldForkSpec.from_dict(missing_component, inventory=inventory)

    boolean_seed = copy.deepcopy(_payload())
    boolean_seed["world_seed"] = True
    with pytest.raises(WorldForkSpecError, match="world_seed must be"):
        WorldForkSpec.from_dict(boolean_seed, inventory=inventory)
