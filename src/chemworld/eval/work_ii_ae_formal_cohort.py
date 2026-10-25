"""Fail-closed cohort binding between A-E v0.2 qualification and formal cells."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256

FORMAL_DESIGN_VERSION = "chemworld-work-ii-formal-design-0.2"
LEGACY_FORMAL_DESIGN_VERSION = "chemworld-work-ii-formal-design-0.1"
FORMAL_DESIGN_ID = "work-ii-fixed-law-prior-formal-v0.2"
AE_CONTRACT_VERSION = "chemworld-work-ii-ae-prior-distinguishability-contract-0.2"
AE_CONTRACT_PATH = "configs/benchmark/work_ii_ae_prior_distinguishability_v0.2.json"
FORMAL_ARMS = ("opaque", "aligned_nominal", "misindexed_nominal")


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _task_worlds(value: object) -> dict[str, list[int]]:
    mapping = _mapping(value)
    result: dict[str, list[int]] = {}
    for task_id, seeds in mapping.items():
        if isinstance(task_id, str) and isinstance(seeds, list):
            result[task_id] = [int(seed) for seed in seeds]
    return result


def _flatten(task_worlds: Mapping[str, Sequence[int]]) -> list[int]:
    return [int(seed) for seeds in task_worlds.values() for seed in seeds]


def _selected_seed(namespace: str, task_id: str, index: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{task_id}:{index}".encode()).digest()
    return 100_000_000 + int.from_bytes(digest[:8], "big") % 900_000_000


def load_ae_formal_cohort(
    root: Path, design: Mapping[str, Any]
) -> tuple[dict[str, list[int]], dict[str, list[int]], list[str]]:
    """Load and validate the exact public/construction A-E cohort identity."""

    root = root.resolve()
    errors: list[str] = []
    if design.get("schema_version") != FORMAL_DESIGN_VERSION:
        errors.append("formal design is not the v0.2 cohort contract")
    if design.get("design_id") != FORMAL_DESIGN_ID:
        errors.append("formal design ID is not the v0.2 cohort contract")

    qualification = _mapping(design.get("prior_distinguishability_qualification_contract"))
    binding = _mapping(qualification.get("contract_binding"))
    relative = binding.get("path")
    if relative != AE_CONTRACT_PATH:
        errors.append("formal design does not bind the canonical A-E v0.2 contract path")
        return {}, {}, errors
    contract_path = (root / relative).resolve()
    if not contract_path.is_relative_to(root) or not contract_path.is_file():
        errors.append("formal A-E v0.2 contract binding is missing or escapes the repository")
        return {}, {}, errors
    contract = _load_object(contract_path)
    if contract.get("schema_version") != AE_CONTRACT_VERSION:
        errors.append("formal design binds an unexpected A-E contract version")
    if binding.get("canonical_sha256") != canonical_json_sha256(contract):
        errors.append("formal A-E v0.2 contract binding is stale")

    # Use the hardened scientific validator without introducing an import cycle at
    # module import time (the qualification module reuses work_ii_formal helpers).
    from chemworld.eval.work_ii_ae_prior_qualification_v02 import validate_contract

    errors.extend(f"A-E v0.2 contract: {error}" for error in validate_contract(root, contract))

    cohorts = _mapping(contract.get("cohorts"))
    expected_public = _task_worlds(
        _mapping(cohorts.get("heldout_qualification")).get("task_world_seeds")
    )
    expected_construction = _task_worlds(
        _mapping(cohorts.get("construction")).get("task_world_seeds")
    )
    world_cohort = _mapping(design.get("world_cohort"))
    public = _mapping(world_cohort.get("public_formal"))
    construction = _mapping(world_cohort.get("exposed_construction_only"))
    observed_public = _task_worlds(public.get("task_world_seeds"))
    observed_construction = _task_worlds(construction.get("task_world_seeds"))
    if observed_public != expected_public:
        errors.append("formal public worlds differ from A-E heldout qualification worlds")
    if observed_construction != expected_construction:
        errors.append("exposed construction worlds differ from A-E construction worlds")
    if (
        public.get("selection_algorithm") != "sha256_first8_modulo_namespace_v1"
        or public.get("selection_namespace")
        != _mapping(cohorts.get("heldout_qualification")).get("selection_namespace")
        or public.get("prospective_formal_participant_cohort") is not True
        or public.get("participant_formal_denominator") is not True
        or public.get("qualification_phase") != "heldout_qualification"
    ):
        errors.append("formal public cohort role or derivation changed")
    if (
        construction.get("qualification_phase") != "construction"
        or construction.get("participant_formal_denominator") is not False
        or construction.get("participant_cell_count") != 0
    ):
        errors.append("exposed construction cohort entered the participant denominator")

    namespace = str(public.get("selection_namespace", ""))
    derived = {
        task_id: [_selected_seed(namespace, task_id, index) for index in range(5)]
        for task_id in expected_public
    }
    if observed_public != derived:
        errors.append("formal public world selection is not reproducible from its namespace")
    public_flat = _flatten(observed_public)
    construction_flat = _flatten(observed_construction)
    development = _mapping(world_cohort.get("development_and_qualification"))
    development_seeds = {int(seed) for seed in development.get("world_seeds", [])}
    if len(public_flat) != 25 or len(set(public_flat)) != 25:
        errors.append("formal public cohort is not exactly 25 unique task-worlds")
    if len(construction_flat) != 25 or len(set(construction_flat)) != 25:
        errors.append("exposed construction cohort is not exactly 25 unique task-worlds")
    if set(public_flat) & set(construction_flat):
        errors.append("formal public and exposed construction cohorts overlap")
    if development_seeds & (set(public_flat) | set(construction_flat)):
        errors.append("development worlds overlap a frozen A-E cohort")
    return expected_public, expected_construction, errors


def validate_ae_public_cells(
    root: Path,
    design: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Require 75 A-E cells to be exactly public task/world/arm triplets."""

    public, construction, errors = load_ae_formal_cohort(root, design)
    expected = Counter(
        (task_id, seed, arm)
        for task_id, seeds in public.items()
        for seed in seeds
        for arm in FORMAL_ARMS
    )
    observed = Counter(
        (
            str(cell.get("task_id", "")),
            int(cell.get("world_seed", -1)),
            str(cell.get("prior_arm", "")),
        )
        for cell in cells
        if isinstance(cell, Mapping)
    )
    if observed != expected:
        errors.append("A-E participant cells differ from the exact v0.2 public cohort")
    construction_seeds = set(_flatten(construction))
    if any(
        cell.get("c2_locus") != "A_E"
        or cell.get("world_split") != "public_formal"
        or cell.get("world_seed") in construction_seeds
        for cell in cells
        if isinstance(cell, Mapping)
    ):
        errors.append("A-E participant cells include a non-public or construction identity")
    return errors


def validate_formal_ae_qualification(
    root: Path, report_path: Path, design: Mapping[str, Any]
) -> list[str]:
    """Route formal A-E evidence to the validator matching the design version."""

    if design.get("schema_version") == FORMAL_DESIGN_VERSION:
        _, _, cohort_errors = load_ae_formal_cohort(root, design)
        qualification = _mapping(
            design.get("prior_distinguishability_qualification_contract")
        )
        binding = _mapping(qualification.get("contract_binding"))
        contract_path = (root.resolve() / str(binding.get("path", ""))).resolve()
        from chemworld.eval.work_ii_ae_prior_qualification_v02 import (
            validate_formal_qualification_output,
        )

        return [
            *cohort_errors,
            *validate_formal_qualification_output(
                root.resolve(), report_path.resolve(), contract_path
            ),
        ]

    if design.get("schema_version") != LEGACY_FORMAL_DESIGN_VERSION:
        return ["unsupported formal design version for A-E qualification"]
    return [
        "legacy formal design v0.1 is historical-only and cannot authorize new A-E admission"
    ]


def qualification_tested_commit(report: Mapping[str, Any]) -> object:
    """Return the tested commit from either supported qualification schema."""

    context = _mapping(report.get("execution_context"))
    if context:
        return context.get("tested_commit")
    return _mapping(report.get("source_binding")).get("tested_commit")


__all__ = [
    "AE_CONTRACT_PATH",
    "FORMAL_DESIGN_ID",
    "FORMAL_DESIGN_VERSION",
    "LEGACY_FORMAL_DESIGN_VERSION",
    "load_ae_formal_cohort",
    "qualification_tested_commit",
    "validate_ae_public_cells",
    "validate_formal_ae_qualification",
]
