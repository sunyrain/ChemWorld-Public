"""Provider-free materialization of the nine Work II formal runtime configs."""

from __future__ import annotations

import copy
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from chemworld.eval.provenance import canonical_json_sha256, file_sha256, write_json_atomic
from chemworld.eval.work_ii_ae_formal_cohort import FORMAL_DESIGN_VERSION
from chemworld.eval.work_ii_resource_calibration_v02 import (
    EXPECTED_PATTERN_KEYS,
    pattern_key,
    pattern_slug,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    validate_manifest as validate_w2_26_manifest,
)
from chemworld.eval.work_ii_resource_calibration_v02 import (
    validate_summary as validate_w2_26_summary,
)
from chemworld.eval.work_ii_task_resources import (
    materialize_task_resource_caps,
    resolve_task_resource_card,
)

FORMAL_RUNTIME_MANIFEST_VERSION = "chemworld-work-ii-formal-runtime-manifest-0.1"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _binding(root: Path, path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or not resolved.is_file():
        raise ValueError(f"cannot bind repository artifact: {resolved}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": file_sha256(resolved),
    }


def formal_runtime_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def build_formal_runtime_manifest(
    root: Path,
    *,
    w2_26_manifest_path: Path,
    w2_26_summary_path: Path,
    formal_design_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build nine configs from passed W2-26 cards without executing a provider."""

    root = root.resolve()
    w2_26_manifest_path = w2_26_manifest_path.resolve()
    w2_26_summary_path = w2_26_summary_path.resolve()
    formal_design_path = formal_design_path.resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(root):
        raise ValueError("formal runtime output root must be inside the repository")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite formal runtime root: {output_root}")

    w2_26_manifest = _load(w2_26_manifest_path)
    w2_26_summary = _load(w2_26_summary_path)
    formal_design = _load(formal_design_path)
    manifest_errors = validate_w2_26_manifest(root, w2_26_manifest)
    summary_errors = validate_w2_26_summary(w2_26_summary, manifest=w2_26_manifest)
    if manifest_errors:
        raise ValueError("W2-26 manifest failed: " + "; ".join(manifest_errors))
    if summary_errors:
        raise ValueError("W2-26 summary failed: " + "; ".join(summary_errors))
    if (
        w2_26_summary.get("status") != "passed"
        or w2_26_summary.get("calibration_passed") is not True
        or w2_26_summary.get("method_qualification_may_be_authorized") is not True
    ):
        raise ValueError("W2-26 summary is not passed")
    if formal_design.get("schema_version") != FORMAL_DESIGN_VERSION:
        raise ValueError("formal runtime requires the current v0.2 design")

    patterns = w2_26_manifest.get("patterns")
    patterns = patterns if isinstance(patterns, list) else []
    cards = w2_26_summary.get("resource_card_proposals")
    cards = cards if isinstance(cards, list) else []
    pattern_by_key = {pattern_key(row): row for row in patterns if isinstance(row, Mapping)}
    card_keys = [
        pattern_key(row.get("card_identity", {})) for row in cards if isinstance(row, Mapping)
    ]
    if set(pattern_by_key) != set(EXPECTED_PATTERN_KEYS) or len(pattern_by_key) != 9:
        raise ValueError("W2-26 manifest does not contain nine unique task configs")
    if set(card_keys) != set(EXPECTED_PATTERN_KEYS) or len(card_keys) != 9:
        raise ValueError("W2-26 summary does not contain nine unique task cards")

    staged: list[tuple[Path, dict[str, Any], dict[str, Any]]] = []
    rows: list[dict[str, Any]] = []
    for locus, task_id, rounds in EXPECTED_PATTERN_KEYS:
        pattern = pattern_by_key[(locus, task_id, rounds)]
        source_binding = pattern.get("campaign_config_binding")
        source_binding = source_binding if isinstance(source_binding, Mapping) else {}
        relative = source_binding.get("path")
        digest = source_binding.get("sha256")
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError(f"{locus}/{task_id}: normalized config binding is missing")
        source_path = (root / relative).resolve()
        if (
            not source_path.is_relative_to(root)
            or not source_path.is_file()
            or file_sha256(source_path) != digest
        ):
            raise ValueError(f"{locus}/{task_id}: normalized config binding is stale")
        source = _load(source_path)
        source_binding_with_canonical = dict(source_binding)
        source_binding_with_canonical["config_canonical_json_sha256"] = canonical_json_sha256(
            source
        )
        card = resolve_task_resource_card(
            w2_26_summary,
            rounds=rounds,
            locus=locus,
            task_id=task_id,
            formal_source_config=source,
            formal_source_binding=source_binding_with_canonical,
        )
        config = materialize_task_resource_caps(source, card)
        config["formal_runtime_identity"] = {
            "locus": locus,
            "task_id": task_id,
            "rounds": rounds,
            "provider_calls_executed": 0,
        }
        destination = output_root / f"{pattern_slug(pattern)}.json"
        staged.append((destination, config, card))
        rows.append(
            {
                "locus": locus,
                "task_id": task_id,
                "rounds": rounds,
                "source_campaign_config_binding": dict(source_binding),
                "formal_campaign_config_binding": {
                    "path": destination.relative_to(root).as_posix(),
                    "canonical_json_sha256": canonical_json_sha256(config),
                },
                "resource_card_sha256": card.get("card_sha256"),
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": FORMAL_RUNTIME_MANIFEST_VERSION,
        "status": "materialized_authorization_blocked",
        "formal_result": False,
        "formal_execution_allowed": False,
        "provider_calls_executed": 0,
        "w2_26_manifest_binding": _binding(root, w2_26_manifest_path),
        "w2_26_summary_binding": _binding(root, w2_26_summary_path),
        "formal_design_binding": _binding(root, formal_design_path),
        "task_configs": rows,
    }
    manifest["manifest_sha256"] = formal_runtime_manifest_sha256(manifest)
    temporary_root = output_root.with_name(f".{output_root.name}-{uuid4().hex[:8]}.tmp")
    temporary_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root.mkdir(exist_ok=False)
    try:
        for destination, config, _card in staged:
            write_json_atomic(temporary_root / destination.name, config)
        write_json_atomic(temporary_root / "manifest.json", manifest)
        temporary_root.replace(output_root)
        errors = validate_formal_runtime_manifest(
            root, manifest, manifest_path=output_root / "manifest.json"
        )
        if errors:
            shutil.rmtree(output_root)
            raise ValueError("formal runtime manifest failed: " + "; ".join(errors))
    finally:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
    return copy.deepcopy(manifest)


def validate_formal_runtime_manifest(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path | None = None,
) -> list[str]:
    """Validate the small provider-free runtime binding surface."""

    root = root.resolve()
    errors: list[str] = []
    if manifest.get("schema_version") != FORMAL_RUNTIME_MANIFEST_VERSION:
        errors.append("unexpected formal runtime manifest schema")
    if manifest.get("manifest_sha256") != formal_runtime_manifest_sha256(manifest):
        errors.append("formal runtime manifest self-hash mismatch")
    if (
        manifest.get("status") != "materialized_authorization_blocked"
        or manifest.get("formal_result") is not False
        or manifest.get("formal_execution_allowed") is not False
        or manifest.get("provider_calls_executed") != 0
    ):
        errors.append("formal runtime manifest crossed its execution boundary")
    upstream: dict[str, dict[str, Any]] = {}
    for field in (
        "w2_26_manifest_binding",
        "w2_26_summary_binding",
        "formal_design_binding",
    ):
        binding = manifest.get(field)
        binding = binding if isinstance(binding, Mapping) else {}
        relative = binding.get("path")
        digest = binding.get("sha256")
        path = (root / str(relative)).resolve()
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not path.is_relative_to(root)
            or not path.is_file()
            or file_sha256(path) != digest
        ):
            errors.append(f"formal runtime upstream binding is stale: {field}")
            continue
        upstream[field] = _load(path)
    w2_26_manifest = upstream.get("w2_26_manifest_binding")
    w2_26_summary = upstream.get("w2_26_summary_binding")
    formal_design = upstream.get("formal_design_binding")
    if w2_26_manifest is not None and validate_w2_26_manifest(root, w2_26_manifest):
        errors.append("formal runtime W2-26 manifest is invalid")
    if w2_26_manifest is not None and w2_26_summary is not None:
        if validate_w2_26_summary(w2_26_summary, manifest=w2_26_manifest):
            errors.append("formal runtime W2-26 summary is invalid")
        if (
            w2_26_summary.get("status") != "passed"
            or w2_26_summary.get("calibration_passed") is not True
            or w2_26_summary.get("method_qualification_may_be_authorized") is not True
        ):
            errors.append("formal runtime W2-26 summary is not passed")
    if formal_design is not None and formal_design.get("schema_version") != FORMAL_DESIGN_VERSION:
        errors.append("formal runtime design binding is not current v0.2")

    upstream_patterns: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    upstream_cards: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    if w2_26_manifest is not None:
        upstream_patterns = {
            pattern_key(row): row
            for row in w2_26_manifest.get("patterns", [])
            if isinstance(row, Mapping)
        }
    if w2_26_summary is not None:
        upstream_cards = {
            pattern_key(row.get("card_identity", {})): row
            for row in w2_26_summary.get("resource_card_proposals", [])
            if isinstance(row, Mapping)
        }

    rows = manifest.get("task_configs")
    rows = rows if isinstance(rows, list) else []
    keys = [pattern_key(row) for row in rows if isinstance(row, Mapping)]
    if tuple(keys) != EXPECTED_PATTERN_KEYS:
        errors.append("formal runtime manifest lacks the exact nine task configs")
        return errors
    for row in rows:
        binding = row.get("formal_campaign_config_binding")
        binding = binding if isinstance(binding, Mapping) else {}
        relative = binding.get("path")
        digest = binding.get("canonical_json_sha256")
        path = (root / str(relative)).resolve()
        if (
            not isinstance(relative, str)
            or not isinstance(digest, str)
            or not path.is_relative_to(root)
            or not path.is_file()
            or canonical_json_sha256(_load(path)) != digest
        ):
            errors.append(f"formal runtime config binding is stale: {row.get('task_id')}")
            continue
        config = _load(path)
        locus, task_id, rounds = pattern_key(row)
        upstream_pattern = upstream_patterns.get((locus, task_id, rounds), {})
        upstream_card = upstream_cards.get((locus, task_id, rounds), {})
        if row.get("source_campaign_config_binding") != upstream_pattern.get(
            "campaign_config_binding"
        ) or row.get("resource_card_sha256") != upstream_card.get("card_sha256"):
            errors.append(f"formal runtime upstream task binding differs: {locus}/{task_id}")
        identity = config.get("formal_runtime_identity")
        identity = identity if isinstance(identity, Mapping) else {}
        campaign = config.get("campaign")
        campaign = campaign if isinstance(campaign, Mapping) else {}
        resources = config.get("method_resources")
        resources = resources if isinstance(resources, Mapping) else {}
        provider = config.get("provider")
        provider = provider if isinstance(provider, Mapping) else {}
        if (
            config.get("task_id") != task_id
            or identity.get("locus") != locus
            or identity.get("task_id") != task_id
            or identity.get("rounds") != rounds
            or identity.get("provider_calls_executed") != 0
            or campaign.get("complete_experiments") != rounds
            or resources.get("complete_experiment_limit") != rounds
            or resources.get("model_call_limit") != 2
            or provider.get("accepted_turn_continuation_limit") != 1
            or provider.get("provider_process_attempt_limit") != 3
            or config.get("resource_calibration_card_binding", {}).get("card_sha256")
            != row.get("resource_card_sha256")
        ):
            errors.append(f"formal runtime config identity differs: {locus}/{task_id}")
    if manifest_path is not None and manifest_path.resolve().parent not in {
        (root / str(row["formal_campaign_config_binding"]["path"])).resolve().parent for row in rows
    }:
        errors.append("formal runtime manifest is outside its config root")
    return errors


__all__ = [
    "FORMAL_RUNTIME_MANIFEST_VERSION",
    "build_formal_runtime_manifest",
    "formal_runtime_manifest_sha256",
    "validate_formal_runtime_manifest",
]
