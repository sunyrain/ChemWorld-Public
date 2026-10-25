"""Provider-free Experiment 1 P benchmark asset authoring helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from math import exp, log
from pathlib import Path
from typing import Any

from chemworld.eval.provenance import canonical_json_sha256, file_sha256
from chemworld.world.phase_kernel import partition_split
from chemworld.world.scenario import DefaultScenarioGenerator, get_scenario

SCHEMA_VERSION = "chemworld-experiment-1-p-asset-authoring-1.1.0"
EXTRACTANT_IDS = ("X0", "X1", "X2", "X3")


class PAssetAuthoringError(ValueError):
    """Raised when an authoring-only P asset contract is malformed."""


def load_asset_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise PAssetAuthoringError("P asset contract schema changed")
    if value.get("status") != "development_authoring_only":
        raise PAssetAuthoringError("P asset contract is not authoring-only")
    if value.get("participant_execution_authorized") is not False:
        raise PAssetAuthoringError("Participant execution must remain disabled")
    if value.get("formal_qualification_authorized") is not False:
        raise PAssetAuthoringError("formal qualification must remain disabled")
    note = value.get("authoring_note")
    if not isinstance(note, Mapping):
        raise PAssetAuthoringError("authoring note binding missing")
    note_path = path.resolve().parents[2] / str(note.get("path", ""))
    if not note_path.is_file() or file_sha256(note_path) != note.get("sha256"):
        raise PAssetAuthoringError("authoring note binding changed")
    worlds = value.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 5:
        raise PAssetAuthoringError("exactly five P Worlds are required")
    if [row.get("world_id") for row in worlds] != [f"P-W0{i}" for i in range(1, 6)]:
        raise PAssetAuthoringError("P World identifiers changed")
    entity = value.get("entity")
    if not isinstance(entity, Mapping):
        raise PAssetAuthoringError("entity authoring contract missing")
    permutation = tuple(int(item) for item in entity.get("misindex_permutation", ()))
    if sorted(permutation) != list(range(4)) or any(
        i == value for i, value in enumerate(permutation)
    ):
        raise PAssetAuthoringError("entity misindex must be a four-item derangement")
    return value


def _contact(contract: Mapping[str, Any], **overrides: float) -> dict[str, float]:
    context = {key: float(value) for key, value in contract["contact_context"].items()}
    context.update(overrides)
    return context


def partition_truth(
    contract: Mapping[str, Any],
    *,
    extractant: int,
    product_mol: float,
    impurity_mol: float,
    coefficient_multiplier: float = 1.0,
    phase_volume_multiplier: float = 1.0,
    composition_coupling_multiplier: float = 1.0,
) -> dict[str, float]:
    result = partition_split(
        product_mol=product_mol,
        impurity_mol=impurity_mol,
        solvent=extractant,
        coefficient_multiplier=coefficient_multiplier,
        phase_volume_multiplier=phase_volume_multiplier,
        composition_coupling_multiplier=composition_coupling_multiplier,
        **_contact(contract),
    )
    product_k = float(result["partition_coefficient"])
    impurity_k = float(result["impurity_partition_coefficient"])
    organic_product = float(result["organic_product_mol"])
    aqueous_product = float(result["aqueous_product_mol"])
    organic_impurity = float(result["organic_impurity_mol"])
    aqueous_impurity = float(result["aqueous_impurity_mol"])
    return {
        "K_product": product_k,
        "K_impurity": impurity_k,
        "S_star": product_k / impurity_k,
        "product_in_organic_mol": organic_product,
        "product_in_aqueous_mol": aqueous_product,
        "impurity_in_organic_mol": organic_impurity,
        "impurity_in_aqueous_mol": aqueous_impurity,
        "product_organic_fraction": organic_product
        / max(organic_product + aqueous_product, 1.0e-12),
        "impurity_organic_fraction": organic_impurity
        / max(organic_impurity + aqueous_impurity, 1.0e-12),
        "organic_to_aqueous_product_ratio": organic_product
        / max(aqueous_product, 1.0e-12),
    }


def _world_multipliers(world_truth: Mapping[str, Any] | None) -> dict[str, float]:
    if world_truth is None:
        return {
            "coefficient_multiplier": 1.0,
            "phase_volume_multiplier": 1.0,
        }
    return {
        "coefficient_multiplier": float(world_truth["partition_coefficient_multiplier"]),
        "phase_volume_multiplier": float(world_truth["partition_phase_volume_multiplier"]),
    }


def build_extractant_dossier(
    contract: Mapping[str, Any], *, world_truth: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    anchors = contract["entity"]["feed_anchors"]
    multipliers = _world_multipliers(world_truth)
    aligned = []
    for extractant, extractant_id in enumerate(EXTRACTANT_IDS):
        aligned.append(
            {
                "extractant_id": extractant_id,
                "anchors": [
                    {
                        "anchor_id": str(anchor["anchor_id"]),
                        **partition_truth(
                            contract,
                            extractant=extractant,
                            product_mol=float(anchor["product_mol"]),
                            impurity_mol=float(anchor["impurity_mol"]),
                            **multipliers,
                        ),
                    }
                    for anchor in anchors
                ],
            }
        )
    permutation = tuple(int(item) for item in contract["entity"]["misindex_permutation"])
    misspecified = [
        {"extractant_id": EXTRACTANT_IDS[index], "anchors": aligned[source]["anchors"]}
        for index, source in enumerate(permutation)
    ]
    return {
        "schema_version": "chemworld-experiment-1-p-entity-dossier-1.1.0",
        "world_id": None if world_truth is None else str(world_truth["world_id"]),
        "private_truth_binding": multipliers,
        "aligned": aligned,
        "misspecified": misspecified,
        "permutation": list(permutation),
        "aligned_sha256": canonical_json_sha256(aligned),
        "misspecified_sha256": canonical_json_sha256(misspecified),
    }


def matched_s_star_bands(
    contract: Mapping[str, Any],
    *,
    world_truth: Mapping[str, Any] | None = None,
    signed_shift: float = 1.0,
) -> dict[str, Any]:
    spec = contract["parametric"]
    multipliers = _world_multipliers(world_truth)
    truth = partition_truth(
        contract,
        extractant=int(spec["reference_extractant"]),
        product_mol=float(spec["reference_product_mol"]),
        impurity_mol=float(spec["reference_impurity_mol"]),
        **multipliers,
    )
    center = float(truth["S_star"])
    half_width = float(spec["log_band_half_width"])
    shift = signed_shift * float(spec["misspecified_log_shift"])

    def band(value: float) -> dict[str, float]:
        return {
            "center": value,
            "lower": exp(log(value) - half_width),
            "upper": exp(log(value) + half_width),
            "log_half_width": half_width,
        }

    return {
        "world_id": None if world_truth is None else str(world_truth["world_id"]),
        "private_truth_binding": multipliers,
        "truth": truth,
        "aligned": band(center),
        "misspecified": band(exp(log(center) + shift)),
        "units": "dimensionless",
    }


def frozen_world_truths(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    generator = DefaultScenarioGenerator()
    scenario = get_scenario("reaction-to-purification")
    rows = []
    for world in contract["worlds"]:
        instance = generator.generate(
            scenario,
            int(world["world_seed"]),
            tuple(dict(item) for item in world["world_interventions"]),
        )
        payload = {
            "world_id": str(world["world_id"]),
            "world_seed": int(world["world_seed"]),
            "role": str(world["role"]),
            "private_world_id": instance.parameters.world_id,
            "partition_coefficient_multiplier": instance.parameters.domain_parameter(
                "partition_coefficient_multiplier"
            ),
            "partition_phase_volume_multiplier": instance.parameters.domain_parameter(
                "partition_phase_volume_multiplier"
            ),
            "mechanism_hash": instance.compiled_mechanism.mechanism_hash,
            "world_interventions": [dict(item) for item in world["world_interventions"]],
        }
        payload["truth_sha256"] = canonical_json_sha256(payload)
        rows.append(payload)
    return rows


def build_world_assets(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for truth in frozen_world_truths(contract):
        dossier = build_extractant_dossier(contract, world_truth=truth)
        bands = matched_s_star_bands(contract, world_truth=truth)
        row: dict[str, Any] = {
            "world_id": truth["world_id"],
            "truth": truth,
            "entity_dossier": dossier,
            "parametric_bands": bands,
        }
        row["asset_sha256"] = canonical_json_sha256(row)
        rows.append(row)
    return rows


def structural_intervention(contract: Mapping[str, Any]) -> dict[str, Any]:
    spec = contract["structural"]
    return {
        "kind": "mechanism_family",
        "mode": "constitutive_law_family",
        "severity": float(spec["severity"]),
        "constitutive_law_change": {
            "transform_id": str(spec["child_transform"]),
            "composition_coupling_multiplier_at_full_severity": float(
                spec["composition_coupling_multiplier_at_full_severity"]
            ),
        },
    }


__all__ = [
    "PAssetAuthoringError",
    "build_extractant_dossier",
    "build_world_assets",
    "frozen_world_truths",
    "load_asset_contract",
    "matched_s_star_bands",
    "partition_truth",
    "structural_intervention",
]
