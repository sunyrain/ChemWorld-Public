"""Create a local paper/preprint artifact directory."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chemworld import __version__
from chemworld.data.datasets import dataset_card, export_dataset
from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.data.submission import git_commit
from chemworld.eval.baseline_report import generate_baseline_report
from chemworld.eval.provenance import build_solver_provenance_manifest
from chemworld.eval.runner import run_agent
from chemworld.eval.verify import verify_records
from chemworld.legacy.agent_registry import make_agent
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.schemas import ACTION_SCHEMA, RECIPE_SCHEMA, TRAJECTORY_SCHEMA
from chemworld.tasks import get_task, get_task_card
from chemworld.world import list_scenarios, world_law_spec


def create_paper_artifact(
    *,
    output_dir: str | Path,
    task_ids: list[str],
    agents: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    """Generate a reproducible benchmark release artifact skeleton.

    The generated directory contains machine-readable metadata, task cards,
    schema snapshots, a baseline report, a dataset example, and a reproduction
    command script. It is small enough for local smoke tests but uses the same
    structure expected by a release artifact.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": "chemworld-paper-artifact-0.1",
        "created_at": datetime.now(UTC).isoformat(),
        "chemworld_version": __version__,
        "commit_hash": git_commit(),
        "tasks": task_ids,
        "agents": agents,
        "seeds": seeds,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    task_dir = root / "tasks"
    task_dir.mkdir(exist_ok=True)
    task_cards = [get_task_card(task_id) for task_id in task_ids]
    task_contracts = [get_task(task_id).to_dict() for task_id in task_ids]
    (task_dir / "task_cards.json").write_text(
        json.dumps(task_cards, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (task_dir / "task_contracts.json").write_text(
        json.dumps(task_contracts, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (task_dir / "scenario_cards.json").write_text(
        json.dumps([scenario.to_dict() for scenario in list_scenarios()], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (task_dir / "world_law.json").write_text(
        json.dumps(world_law_spec().to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    schema_dir = root / "schemas"
    schema_dir.mkdir(exist_ok=True)
    schemas = {
        "action_schema.json": ACTION_SCHEMA,
        "recipe_schema.json": RECIPE_SCHEMA,
        "trajectory_schema.json": TRAJECTORY_SCHEMA,
    }
    for filename, schema in schemas.items():
        (schema_dir / filename).write_text(
            json.dumps(schema, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    baseline_report = generate_baseline_report(
        task_ids=task_ids,
        agents=agents,
        seeds=seeds,
        output_dir=root / "baseline_report",
    )

    trajectory_dir = root / "trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    dataset_dir = root / "dataset_examples"
    dataset_dir.mkdir(exist_ok=True)
    example_task = task_ids[0]
    example_agent = agents[0]
    example_seed = seeds[0]
    example_trajectory = trajectory_dir / f"{example_task}_{example_agent}_seed{example_seed}.jsonl"
    run_agent(
        env_id="ChemWorld",
        agent=make_agent(example_agent),
        world_split="public-test",
        budget=18,
        objective="balanced",
        seed=example_seed,
        task_id=example_task,
        output_path=example_trajectory,
    )
    records = load_jsonl(example_trajectory)
    verification = verify_records(records).to_dict()
    exported_dataset = dataset_dir / f"{example_task}_example_dataset.jsonl"
    export_dataset(example_trajectory, output=exported_dataset, format="jsonl")
    dataset_card_payload = dataset_card(exported_dataset)
    (dataset_dir / "dataset_card.json").write_text(
        json.dumps(dataset_card_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    manifest_dir = root / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    replay_manifest = _build_replay_manifest(
        trajectory_path=example_trajectory,
        records=records,
        verification=verification,
    )
    (manifest_dir / "replay_manifest.json").write_text(
        json.dumps(replay_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    claim_status = _load_current_claim_status()
    (manifest_dir / "current_status.json").write_text(
        json.dumps(claim_status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    release_checklist = _build_release_checklist(
        task_ids=task_ids,
        agents=agents,
        seeds=seeds,
        verification=verification,
        claim_status=claim_status,
    )
    (manifest_dir / "release_checklist.json").write_text(
        json.dumps(release_checklist, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    solver_provenance = build_solver_provenance_manifest(
        task_ids=task_ids,
        agents=agents,
        seeds=seeds,
    )
    (manifest_dir / "solver_provenance_manifest.json").write_text(
        json.dumps(solver_provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (root / "release_checklist.md").write_text(
        _release_checklist_markdown(release_checklist),
        encoding="utf-8",
    )
    (manifest_dir / "release_manifest.json").write_text(
        json.dumps(
            {
                **metadata,
                "task_contract_count": len(task_contracts),
                "baseline_result_count": baseline_report.result_count,
                "dataset_card_schema_version": dataset_card_payload["schema_version"],
                "replay_verified": verification["verified"],
                "solver_provenance_schema_version": solver_provenance["schema_version"],
                "required_files": _required_artifact_files(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    script_dir = root / "scripts"
    script_dir.mkdir(exist_ok=True)
    commands = [
        "# Reproduce the public baseline artifact",
        (
            "chemworld baselines report "
            f"--tasks {' '.join(task_ids)} --agents {' '.join(agents)} "
            f"--seeds {' '.join(str(seed) for seed in seeds)} "
            "--output-dir artifact/baseline_report"
        ),
        (
            "chemworld datasets export "
            f"--submission {example_trajectory.as_posix()} --format jsonl "
            "--output artifact/dataset_examples/recreated_dataset.jsonl"
        ),
        f"chemworld verify --submission {example_trajectory.as_posix()}",
    ]
    (script_dir / "reproduce_public_artifact.ps1").write_text(
        "\n".join(commands) + "\n",
        encoding="utf-8",
    )
    (root / "environment.md").write_text(
        _environment_markdown(metadata, task_contracts),
        encoding="utf-8",
    )
    (root / "limitations.md").write_text(
        _limitations_markdown(task_contracts),
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# ChemWorld Benchmark Release Artifact\n\n"
        "This directory is generated by `chemworld artifact create`. It contains "
        "task contracts, schema snapshots, baseline tables, example trajectories, "
        "dataset cards, replay manifests, release checklist files, limitations, "
        "and reproduction commands for a local preprint artifact.\n",
        encoding="utf-8",
    )

    summary = {
        **metadata,
        "path": str(root),
        "baseline_report": baseline_report.to_dict(),
        "dataset_example": str(exported_dataset),
        "dataset_card": str(dataset_dir / "dataset_card.json"),
        "example_trajectory": str(example_trajectory),
        "replay_manifest": str(manifest_dir / "replay_manifest.json"),
        "solver_provenance_manifest": str(
            manifest_dir / "solver_provenance_manifest.json"
        ),
        "release_checklist": str(root / "release_checklist.md"),
        "replay_verified": verification["verified"],
        "artifact_bundle_complete": release_checklist["artifact_bundle_complete"],
        "formal_publication_claim_allowed": release_checklist[
            "formal_publication_claim_allowed"
        ],
    }
    (root / "artifact_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def _build_replay_manifest(
    *,
    trajectory_path: Path,
    records: list[dict[str, Any]],
    verification: dict[str, Any],
) -> dict[str, Any]:
    first = records[0]
    final = records[-1]
    return {
        "schema_version": "chemworld-replay-manifest-0.1",
        "trajectory": str(trajectory_path),
        "record_count": len(records),
        "task_id": trajectory_value(first, 'benchmark_task_id', None),
        "seed": first.get("seed"),
        "agent_name": first.get("agent_metadata", {}).get("name"),
        "world_law_id": first.get("world_law_id"),
        "task_contract_hash": first.get("task_contract_hash"),
        "runtime_profile_hash": first.get("runtime_profile_hash"),
        "mechanism_id": first.get("mechanism_id"),
        "mechanism_hash": first.get("mechanism_hash"),
        "scoring_contract_hash": first.get("scoring_contract_hash"),
        "observation_contract_hash": first.get("observation_contract_hash"),
        "final_step": final.get("step"),
        "final_leaderboard_score": trajectory_value(final, 'leaderboard_score', None),
        "terminated": final.get("terminated"),
        "truncated": final.get("truncated"),
        "verification": verification,
        "verify_command": f"chemworld verify --submission {trajectory_path.as_posix()}",
    }


def _build_release_checklist(
    *,
    task_ids: list[str],
    agents: list[str],
    seeds: list[int],
    verification: dict[str, Any],
    claim_status: Mapping[str, Any],
) -> dict[str, Any]:
    items = [
        {
            "id": "task_contracts",
            "status": "included",
            "evidence": "tasks/task_contracts.json",
        },
        {
            "id": "baseline_report",
            "status": "included",
            "evidence": "baseline_report/baseline_report.json",
        },
        {
            "id": "dataset_card",
            "status": "included",
            "evidence": "dataset_examples/dataset_card.json",
        },
        {
            "id": "replay_manifest",
            "status": "verified" if verification["verified"] else "failed",
            "evidence": "manifests/replay_manifest.json",
        },
        {
            "id": "solver_provenance",
            "status": "included",
            "evidence": "manifests/solver_provenance_manifest.json",
        },
        {
            "id": "current_status",
            "status": "included",
            "evidence": "manifests/current_status.json",
        },
        {
            "id": "release_limitations",
            "status": "included",
            "evidence": "limitations.md",
        },
    ]
    artifact_bundle_complete = all(
        item["status"] in {"included", "verified"} for item in items
    )
    formal_publication_claim_allowed = bool(
        artifact_bundle_complete
        and claim_status["formal_results_present"] is True
        and claim_status["benchmark_claim_allowed"] is True
        and claim_status["publication_ready"] is True
    )
    return {
        "schema_version": "chemworld-release-checklist-0.2",
        "tasks": task_ids,
        "agents": agents,
        "seeds": seeds,
        "items": items,
        "artifact_bundle_complete": artifact_bundle_complete,
        "claim_status_source": "configs/current.json",
        "claim_status": dict(claim_status),
        "publication_ready": claim_status["publication_ready"],
        "formal_publication_claim_allowed": formal_publication_claim_allowed,
    }


def _release_checklist_markdown(checklist: dict[str, Any]) -> str:
    lines = [
        "# Release Checklist",
        "",
        "| Item | Status | Evidence |",
        "| --- | --- | --- |",
    ]
    for item in checklist["items"]:
        lines.append(f"| `{item['id']}` | {item['status']} | `{item['evidence']}` |")
    lines.extend(
        [
            "",
            f"Artifact bundle complete: `{checklist['artifact_bundle_complete']}`",
            f"Current publication status: `{checklist['claim_status']['publication_status']}`",
            f"Publication ready: `{checklist['publication_ready']}`",
            (
                "Formal publication claim allowed: "
                f"`{checklist['formal_publication_claim_allowed']}`"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _load_current_claim_status() -> dict[str, Any]:
    """Read the canonical registry and expose only claim-relevant status fields."""

    registry_path = configuration_root() / "current.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping):
        raise ValueError("configs/current.json must contain a JSON object")
    formal = registry.get("formal_evaluation")
    publication = registry.get("publication")
    if not isinstance(formal, Mapping) or not isinstance(publication, Mapping):
        raise ValueError(
            "configs/current.json must define formal_evaluation and publication states"
        )
    boolean_fields = {
        "formal_results_present": formal.get("formal_results_present"),
        "benchmark_claim_allowed": formal.get("benchmark_claim_allowed"),
        "publication_ready": publication.get("publication_ready"),
    }
    invalid = [key for key, value in boolean_fields.items() if not isinstance(value, bool)]
    if invalid:
        raise ValueError(
            "configs/current.json claim fields must be boolean: " + ", ".join(invalid)
        )
    return {
        "registry_schema_version": registry.get("schema_version"),
        "formal_evaluation_status": formal.get("status"),
        "formal_results_present": boolean_fields["formal_results_present"],
        "benchmark_claim_allowed": boolean_fields["benchmark_claim_allowed"],
        "publication_status": publication.get("status"),
        "publication_ready": boolean_fields["publication_ready"],
    }


def _limitations_markdown(task_contracts: list[dict[str, Any]]) -> str:
    lines = [
        "# Limitations",
        "",
        "ChemWorld-Bench is a virtual physical-chemistry interaction benchmark for "
        "agents, optimizers, and students. It is not a real reaction predictor, "
        "DFT or molecular-dynamics wrapper, commercial process simulator, or "
        "robot laboratory controller.",
        "",
        "Benchmark claims must include task maturity metadata. Current task "
        "boundaries are:",
        "",
        "| Task | Physics Maturity | Proxy Allowed | Boundary |",
        "| --- | --- | --- | --- |",
    ]
    for task in task_contracts:
        boundary = (
            "virtual benchmark task; do not interpret scores as real laboratory "
            "success probabilities"
        )
        lines.append(
            "| "
            f"`{task['task_id']}` | `{task['physics_maturity']}` | "
            f"`{task['proxy_allowed']}` | {boundary} |"
        )
    lines.extend(
        [
            "",
            "Known low-maturity surfaces include synthetic instruments, proxy or "
            "lite downstream separations, compact kinetic models, and benchmark "
            "safety/cost signals. Generated trajectories are virtual data and "
            "must not be presented as real experimental observations.",
            "",
        ]
    )
    return "\n".join(lines)


def _environment_markdown(
    metadata: dict[str, Any],
    task_contracts: list[dict[str, Any]],
) -> str:
    lines = [
        "# Environment Summary",
        "",
        f"- ChemWorld version: `{metadata['chemworld_version']}`",
        f"- Commit hash: `{metadata['commit_hash']}`",
        f"- Created at: `{metadata['created_at']}`",
        f"- Tasks: `{', '.join(metadata['tasks'])}`",
        f"- Agents: `{', '.join(metadata['agents'])}`",
        f"- Seeds: `{', '.join(str(seed) for seed in metadata['seeds'])}`",
        "",
        "## Task Contracts",
        "",
        "| Task | Split | Budget | Episode Mode | Contract Hash | Physics Maturity |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for task in task_contracts:
        lines.append(
            "| "
            f"`{task['task_id']}` | `{task['world_split']}` | {task['budget']} | "
            f"`{task['episode_mode']}` | `{task['contract_hash']}` | "
            f"`{task['physics_maturity']}` |"
        )
    lines.append("")
    return "\n".join(lines)


def _required_artifact_files() -> list[str]:
    return [
        "README.md",
        "environment.md",
        "tasks/task_cards.json",
        "tasks/task_contracts.json",
        "baseline_report/baseline_report.json",
        "dataset_examples/dataset_card.json",
        "manifests/replay_manifest.json",
        "manifests/current_status.json",
        "manifests/solver_provenance_manifest.json",
        "manifests/release_manifest.json",
        "limitations.md",
        "release_checklist.md",
        "scripts/reproduce_public_artifact.ps1",
    ]


__all__ = ["create_paper_artifact"]
