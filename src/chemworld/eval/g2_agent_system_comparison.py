"""Build an audited, non-leaderboard comparison of two G2 agent systems."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import (
    canonical_json_sha256,
    file_sha256,
    write_json_atomic,
)

SCHEMA_VERSION = "chemworld-g2-agent-system-comparison-0.1"
ARMS = ("opaque", "nominal")
PHYSICAL_IDENTITY_FIELDS = (
    "world_seed",
    "world_id",
    "world_family_version",
    "mechanism_hash",
    "material_family_id",
    "material_family_sha256",
    "material_instance_sha256",
    "scoring_contract_hash",
    "workflow_mode",
    "observation_noise_mode",
    "observation_noise_namespace",
    "observation_seed",
    "resource_card_sha256",
)


class G2AgentSystemComparisonError(ValueError):
    """Raised when source audits are incomplete or not comparable."""


def _load_audit(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise G2AgentSystemComparisonError(
            f"invalid {label} audit: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise G2AgentSystemComparisonError(f"{label} audit must be an object")
    unhashed = dict(payload)
    declared_hash = unhashed.pop("audit_sha256", None)
    if declared_hash != canonical_json_sha256(unhashed):
        raise G2AgentSystemComparisonError(
            f"{label} audit self-hash is invalid"
        )
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping):
        raise G2AgentSystemComparisonError(f"{label} audit has no matrix")
    gates = {
        "ten_cells": matrix.get("cell_count") == 10,
        "closed": matrix.get("all_cells_complete") is True,
        "resources": matrix.get("all_resource_ledgers_verified") is True,
        "replay": matrix.get("all_exact_replays_verified") is True,
        "provider": matrix.get("all_provider_sessions_verified") is True,
        "physical_pairs": matrix.get("all_pairs_physically_matched") is True,
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise G2AgentSystemComparisonError(
            f"{label} audit failed comparison gates: {failed}"
        )
    cells = payload.get("cells")
    if not isinstance(cells, list) or len(cells) != 10:
        raise G2AgentSystemComparisonError(
            f"{label} audit must contain ten cell rows"
        )
    return payload


def _keyed_cells(audit: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    keyed: dict[tuple[int, str], Mapping[str, Any]] = {}
    for raw_cell in audit["cells"]:
        if not isinstance(raw_cell, Mapping):
            raise G2AgentSystemComparisonError("cell row must be an object")
        key = (int(raw_cell["world_seed"]), str(raw_cell["arm"]))
        if key in keyed:
            raise G2AgentSystemComparisonError(f"duplicate cell: {key}")
        keyed[key] = raw_cell
    expected = {(seed, arm) for seed in range(5) for arm in ARMS}
    if set(keyed) != expected:
        raise G2AgentSystemComparisonError(
            "source audit does not cover seeds 0--4 in both information arms"
        )
    return keyed


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _system_profile(
    *,
    system_id: str,
    model_id: str,
    decision_transport: str,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    cells = list(_keyed_cells(audit).values())

    def arm_rows(arm: str) -> list[Mapping[str, Any]]:
        return [cell for cell in cells if cell["arm"] == arm]

    def arm_profile(arm: str) -> dict[str, Any]:
        rows = arm_rows(arm)
        return {
            "cell_count": len(rows),
            "closed_batch_count": sum(
                int(row["resource_ledger"]["closed_batches"]) for row in rows
            ),
            "final_assay_count": sum(
                int(row["resource_ledger"]["final_assays"]) for row in rows
            ),
            "discarded_batch_count": sum(
                int(row["resource_ledger"]["discarded_batches"]) for row in rows
            ),
            "operation_attempt_count": sum(
                int(row["operations"]["count"]) for row in rows
            ),
            "nonfinal_instrument_use_count": sum(
                int(row["resource_ledger"]["nonfinal_instrument_uses"])
                for row in rows
            ),
            "mean_best_final_score": _mean(
                [float(row["scores"]["best_final_score"]) for row in rows]
            ),
            "mean_operation_attempt_running_best_auc": _mean(
                [
                    float(row["scores"]["operation_attempt_running_best_auc"])
                    for row in rows
                ]
            ),
        }

    opaque = arm_profile("opaque")
    nominal = arm_profile("nominal")
    pair_rows = audit["paired_worlds"]
    return {
        "system_id": system_id,
        "model_id": model_id,
        "decision_transport": decision_transport,
        "source_audit_sha256": audit["audit_sha256"],
        "audit_gates_passed": True,
        "cell_count": len(cells),
        "closed_batch_count": sum(
            int(cell["resource_ledger"]["closed_batches"]) for cell in cells
        ),
        "final_assay_count": sum(
            int(cell["resource_ledger"]["final_assays"]) for cell in cells
        ),
        "discarded_batch_count": sum(
            int(cell["resource_ledger"]["discarded_batches"]) for cell in cells
        ),
        "operation_attempt_count": sum(
            int(cell["operations"]["count"]) for cell in cells
        ),
        "invalid_operation_count": sum(
            int(cell["operations"]["invalid_count"]) for cell in cells
        ),
        "nonfinal_instrument_use_count": sum(
            int(cell["resource_ledger"]["nonfinal_instrument_uses"])
            for cell in cells
        ),
        "assay_commitment_fraction": sum(
            int(cell["resource_ledger"]["final_assays"]) for cell in cells
        )
        / sum(int(cell["resource_ledger"]["closed_batches"]) for cell in cells),
        "provider_qualification_kinds": dict(
            sorted(
                Counter(
                    str(
                        cell["provider_sessions"].get(
                            "qualification_kind", "legacy_experiment_session"
                        )
                    )
                    for cell in cells
                ).items()
            )
        ),
        "arms": {"opaque": opaque, "nominal": nominal},
        "within_system_information_contrast": {
            "estimand": "nominal minus opaque within the same physical world",
            "n_paired_worlds": 5,
            "mean_best_final_score_delta": _mean(
                [
                    float(row["nominal_minus_opaque"]["best_final_score"])
                    for row in pair_rows
                ]
            ),
            "mean_operation_attempt_auc_delta": _mean(
                [
                    float(
                        row["nominal_minus_opaque"][
                            "operation_attempt_running_best_auc"
                        ]
                    )
                    for row in pair_rows
                ]
            ),
            "nominal_higher_best_world_count": sum(
                row["nominal_minus_opaque"]["best_final_score"] > 0
                for row in pair_rows
            ),
            "operation_attempt_delta": (
                nominal["operation_attempt_count"]
                - opaque["operation_attempt_count"]
            ),
            "final_assay_delta": (
                nominal["final_assay_count"] - opaque["final_assay_count"]
            ),
            "discarded_batch_delta": (
                nominal["discarded_batch_count"]
                - opaque["discarded_batch_count"]
            ),
            "nonfinal_instrument_use_delta": (
                nominal["nonfinal_instrument_use_count"]
                - opaque["nonfinal_instrument_use_count"]
            ),
        },
    }


def build_g2_agent_system_comparison(
    codex_audit_path: str | Path,
    deepseek_audit_path: str | Path,
) -> dict[str, Any]:
    """Compare audited complete systems while rejecting backend-causal claims."""

    codex_path = Path(codex_audit_path)
    deepseek_path = Path(deepseek_audit_path)
    codex = _load_audit(codex_path, label="Codex")
    deepseek = _load_audit(deepseek_path, label="DeepSeek")
    codex_cells = _keyed_cells(codex)
    deepseek_cells = _keyed_cells(deepseek)
    matches: list[dict[str, Any]] = []
    for key in sorted(codex_cells):
        left = codex_cells[key]
        right = deepseek_cells[key]
        mismatches = [
            field
            for field in PHYSICAL_IDENTITY_FIELDS
            if left["identity"][field] != right["identity"][field]
        ]
        matches.append(
            {
                "world_seed": key[0],
                "arm": key[1],
                "matched": not mismatches,
                "mismatched_fields": mismatches,
            }
        )
    if not all(row["matched"] for row in matches):
        raise G2AgentSystemComparisonError(
            "agent systems were not evaluated on identical physical cells"
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_audited_two_agent-system_demonstration",
        "formal_result": False,
        "comparison_unit": "complete agent system on a matched world-by-information cell",
        "claim_boundary": {
            "allowed": (
                "The same executable-world apparatus supports distinct autonomous "
                "agent systems and exposes differences in their experimental "
                "trajectories, resource allocation, assay commitment and within-system "
                "information response."
            ),
            "not_allowed": (
                "The design does not isolate a causal model-backend effect because "
                "the complete systems also differ in decision transport and scaffold."
            ),
            "leaderboard_interpretation": False,
            "descriptive_only": True,
            "n_worlds": 5,
        },
        "source_files": {
            "codex": {
                "file_sha256": file_sha256(codex_path),
                "audit_sha256": codex["audit_sha256"],
            },
            "deepseek": {
                "file_sha256": file_sha256(deepseek_path),
                "audit_sha256": deepseek["audit_sha256"],
            },
        },
        "physical_matching": {
            "matched_cell_count": len(matches),
            "all_cells_matched": True,
            "matched_fields": list(PHYSICAL_IDENTITY_FIELDS),
            "intentionally_not_matched": [
                "model_id",
                "decision_transport",
                "provider retry behavior",
                "source code hash",
                "run configuration hash",
            ],
            "cells": matches,
        },
        "systems": {
            "codex_sol_medium_mcp": _system_profile(
                system_id="codex_sol_medium_mcp",
                model_id="gpt-5.6-sol",
                decision_transport="native Codex MCP session per vessel",
                audit=codex,
            ),
            "deepseek_v4_flash_direct": _system_profile(
                system_id="deepseek_v4_flash_direct",
                model_id="deepseek-v4-flash",
                decision_transport=(
                    "direct JSON decision per primitive operation with local dynamic "
                    "schema validation"
                ),
                audit=deepseek,
            ),
        },
    }
    report["comparison_sha256"] = canonical_json_sha256(report)
    return report


def render_g2_agent_system_comparison_markdown(
    report: Mapping[str, Any],
) -> str:
    systems = report["systems"]
    lines = [
        "# G2 matched agent-system demonstration",
        "",
        "Both complete agent systems passed resource-ledger, exact-replay, provider "
        "qualification, and within-world physical-pair audits in all ten cells. The "
        "comparison is a platform and behavior-profile demonstration, not a model "
        "leaderboard or an isolated backend experiment.",
        "",
        "| Complete agent system | Closed batches | Final assays | Discards | "
        "Operations | Non-final instruments | Assay commitment |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for system in systems.values():
        lines.append(
            f"| {system['model_id']} / {system['decision_transport']} | "
            f"{system['closed_batch_count']} | {system['final_assay_count']} | "
            f"{system['discarded_batch_count']} | {system['operation_attempt_count']} | "
            f"{system['nonfinal_instrument_use_count']} | "
            f"{system['assay_commitment_fraction']:.0%} |"
        )
    lines.extend(
        [
            "",
            "## Within-system nominal-minus-opaque profiles",
            "",
            "| System | Worlds nominal higher in best | Mean Δ best | "
            "Mean Δ operation-AUC | Δ operations | Δ assays | Δ discards |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for system in systems.values():
        delta = system["within_system_information_contrast"]
        lines.append(
            f"| {system['model_id']} | "
            f"{delta['nominal_higher_best_world_count']}/5 | "
            f"{delta['mean_best_final_score_delta']:+.4f} | "
            f"{delta['mean_operation_attempt_auc_delta']:+.4f} | "
            f"{delta['operation_attempt_delta']:+d} | "
            f"{delta['final_assay_delta']:+d} | "
            f"{delta['discarded_batch_delta']:+d} |"
        )
    lines.extend(
        [
            "",
            "Physical identity matched in "
            f"{report['physical_matching']['matched_cell_count']}/10 cells across "
            "world, mechanism, material instance, scoring, noise, workflow and "
            "resource-card identities.",
            "",
            f"Interpretation boundary: {report['claim_boundary']['not_allowed']}",
            "",
            f"Comparison hash: `{report['comparison_sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_g2_agent_system_comparison(
    codex_audit_path: str | Path,
    deepseek_audit_path: str | Path,
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, Any]:
    report = build_g2_agent_system_comparison(
        codex_audit_path,
        deepseek_audit_path,
    )
    write_json_atomic(Path(json_path), report)
    markdown_file = Path(markdown_path)
    markdown_file.parent.mkdir(parents=True, exist_ok=True)
    markdown_file.write_text(
        render_g2_agent_system_comparison_markdown(report),
        encoding="utf-8",
    )
    return report


__all__ = [
    "SCHEMA_VERSION",
    "G2AgentSystemComparisonError",
    "build_g2_agent_system_comparison",
    "render_g2_agent_system_comparison_markdown",
    "write_g2_agent_system_comparison",
]
