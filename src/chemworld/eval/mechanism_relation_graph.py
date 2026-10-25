"""Versioned diagnostic-relation graph for mechanism-adaptation protocols."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256
from chemworld.tasks import get_task

DIAGNOSTIC_RELATION_GRAPH_VERSION = "chemworld-diagnostic-relation-graph-0.1"


def build_diagnostic_relation_graph(
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the declared relation semantics with runtime task bindings."""

    task_contracts = protocol.get("task_mechanism_contracts")
    if not isinstance(task_contracts, Mapping):
        raise ValueError("protocol task_mechanism_contracts must be an object")
    entries: list[dict[str, Any]] = []
    task_contract_hashes: dict[str, str] = {}
    for task_id in sorted(str(item) for item in protocol["design"]["tasks"]):
        contract = task_contracts.get(task_id)
        if not isinstance(contract, Mapping):
            raise ValueError(f"missing mechanism contract for {task_id}")
        task_contract_hashes[task_id] = get_task(task_id).contract_hash
        relations = contract.get("diagnostic_relations")
        if not isinstance(relations, Mapping):
            raise ValueError(f"missing diagnostic relations for {task_id}")
        for candidate_id, raw_relation in sorted(relations.items()):
            if not isinstance(raw_relation, Mapping):
                raise ValueError(
                    f"diagnostic relation must be an object: {task_id}/{candidate_id}"
                )
            relation = dict(raw_relation)
            if relation.get("relation_graph_version") != DIAGNOSTIC_RELATION_GRAPH_VERSION:
                raise ValueError(
                    f"diagnostic relation graph version mismatch: {task_id}/{candidate_id}"
                )
            entries.append(
                {
                    "task_id": task_id,
                    "task_contract_hash": task_contract_hashes[task_id],
                    "candidate_id": str(candidate_id),
                    "relation_id": str(relation["relation_id"]),
                    "varied_fields": relation["varied_fields"],
                    "controlled_background": relation["controlled_background"],
                    "required_levels": relation["required_levels"],
                    "observable_channels": relation["observable_channels"],
                    "candidate_signatures": relation["candidate_signatures"],
                    "minimum_distinct_actions": int(
                        relation["minimum_distinct_actions"]
                    ),
                    "relation_declaration_sha256": canonical_json_sha256(relation),
                }
            )
    payload: dict[str, Any] = {
        "schema_version": DIAGNOSTIC_RELATION_GRAPH_VERSION,
        "graph_id": (
            "chemworld-mechanism-adaptation-diagnostic-relations-v0.3.0-rc24"
        ),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": canonical_json_sha256(protocol),
        "task_contract_hashes": task_contract_hashes,
        "relation_count": len(entries),
        "relations": entries,
        "status": "frozen_protocol_input",
    }
    payload["graph_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_diagnostic_relation_graph(
    protocol: Mapping[str, Any],
    plan: Mapping[str, Any],
    graph: Mapping[str, Any],
) -> list[str]:
    """Return fail-closed binding errors for a materialized relation graph."""

    expected = build_diagnostic_relation_graph(protocol)
    errors: list[str] = []
    for field in (
        "schema_version",
        "graph_id",
        "protocol_id",
        "protocol_sha256",
        "task_contract_hashes",
        "relation_count",
        "relations",
        "graph_sha256",
        "status",
    ):
        if graph.get(field) != expected[field]:
            errors.append(f"diagnostic relation graph {field} is stale")
    contract = plan.get("diagnostic_relation_graph")
    if not isinstance(contract, Mapping):
        errors.append("Gate A plan has no diagnostic relation graph contract")
    else:
        if contract.get("schema_version") != DIAGNOSTIC_RELATION_GRAPH_VERSION:
            errors.append("Gate A relation graph schema binding is stale")
        expected_graph_sha = contract.get("expected_graph_sha256")
        if not isinstance(expected_graph_sha, str):
            errors.append("Gate A plan must freeze expected_graph_sha256")
        elif expected_graph_sha != expected["graph_sha256"]:
            errors.append("Gate A relation graph digest binding is stale")
    return errors


__all__ = [
    "DIAGNOSTIC_RELATION_GRAPH_VERSION",
    "build_diagnostic_relation_graph",
    "validate_diagnostic_relation_graph",
]
