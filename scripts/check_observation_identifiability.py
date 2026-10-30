"""Audit bounded public-observation identifiability without leaking evaluator truth."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

import chemworld  # noqa: F401
from chemworld.agents.task_recipes import (
    task_recipe_dimension,
    task_recipe_from_unit_vector,
)
from chemworld.data.logging import to_builtin
from chemworld.eval.observation_identifiability import (
    SPECTRUM_CONDITIONS,
    ObservationIdentifiabilityError,
    PublicSpectrumArchive,
    apply_spectrum_condition,
)
from chemworld.physchem.mechanism_library import configuration_root
from chemworld.physchem.spectroscopy import build_signal_spec, synthesize_signal
from chemworld.physchem.spectroscopy_identifiability import (
    evaluate_spectral_identifiability,
)
from chemworld.tasks import get_task
from chemworld.world.spectra import ph_meter_signal

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_PATH = (
    configuration_root() / "foundation" / "observation_identifiability_v0.5.json"
)
PROTOCOL_VERSION = "chemworld-observation-identifiability-protocol-0.1"







def load_observation_identifiability_protocol(path: Path | None = None) -> dict[str, Any]:
    resolved = DEFAULT_PROTOCOL_PATH if path is None else path
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PROTOCOL_VERSION:
        raise ObservationIdentifiabilityError("unsupported observation identifiability protocol")
    return payload




def audit_observation_identifiability(
    protocol: Mapping[str, Any], *, workspace: Path = ROOT
) -> dict[str, Any]:
    state_pair = protocol["state_pair"]
    instruments: dict[str, dict[str, Any]] = {}
    serialized_public_reports: list[str] = []
    for instrument_id in protocol["spectral_instruments"]:
        spec = build_signal_spec(
            instrument_id,
            tuple(state_pair["species"]),
            target_species=tuple(state_pair["target_species"]),
            impurity_species=tuple(state_pair["impurity_species"]),
            formulas=dict(state_pair["formulas"]),
        )
        reference = synthesize_signal(
            spec,
            dict(state_pair["reference_amounts_mol"]),
            volume_L=float(state_pair["volume_L"]),
            seed=int(state_pair["reference_seed"]),
            replicate_count=int(state_pair["replicate_count"]),
        )
        alternative = synthesize_signal(
            spec,
            dict(state_pair["alternative_amounts_mol"]),
            volume_L=float(state_pair["volume_L"]),
            seed=int(state_pair["alternative_seed"]),
            replicate_count=int(state_pair["replicate_count"]),
        )
        identifiable = evaluate_spectral_identifiability(reference, alternative)
        low_amounts = {
            species: float(state_pair["low_signal_amount_mol"])
            for species in state_pair["species"]
        }
        low_left = synthesize_signal(
            spec,
            low_amounts,
            volume_L=float(state_pair["volume_L"]),
            seed=3,
            replicate_count=int(state_pair["replicate_count"]),
        )
        low_right = synthesize_signal(
            spec,
            low_amounts,
            volume_L=float(state_pair["volume_L"]),
            seed=47,
            replicate_count=int(state_pair["replicate_count"]),
        )
        degraded = evaluate_spectral_identifiability(low_left, low_right)
        public_report = identifiable.to_dict()
        serialized_public_reports.append(json.dumps(public_report, sort_keys=True))
        instruments[instrument_id] = {
            "identifiability": public_report,
            "expected_identifiable": protocol["expected_identifiability"][instrument_id],
            "degraded_low_signal": degraded.to_dict(),
            "replicate_probe_accuracy": _nearest_centroid_accuracy(reference, alternative),
        }

    ph = protocol["ph_pair"]
    ph_reference = ph_meter_signal(
        {"pH_normalized": float(ph["reference_normalized"])},
        seed=int(ph["seed"]),
        replicate_count=int(ph["replicate_count"]),
    )
    ph_alternative = ph_meter_signal(
        {"pH_normalized": float(ph["alternative_normalized"])},
        seed=int(ph["seed"]),
        replicate_count=int(ph["replicate_count"]),
    )
    ph_low_left = ph_meter_signal(
        {"pH_normalized": float(ph["low_contrast_left"])},
        seed=int(ph["seed"]),
        replicate_count=int(ph["replicate_count"]),
    )
    ph_low_right = ph_meter_signal(
        {"pH_normalized": float(ph["low_contrast_right"])},
        seed=int(ph["seed"]),
        replicate_count=int(ph["replicate_count"]),
    )
    ph_report = {
        "reference_pH": ph_reference["pH"],
        "alternative_pH": ph_alternative["pH"],
        "between_state_delta_pH": abs(ph_reference["pH"] - ph_alternative["pH"]),
        "declared_loq_pH": ph_reference["calibration"]["loq_pH"],
        "low_contrast_delta_pH": abs(ph_low_left["pH"] - ph_low_right["pH"]),
        "state_pair_distinguishable": abs(ph_reference["pH"] - ph_alternative["pH"])
        > ph_reference["calibration"]["loq_pH"],
        "low_contrast_degraded": abs(ph_low_left["pH"] - ph_low_right["pH"])
        < ph_reference["calibration"]["loq_pH"],
    }

    packet_a, context_a = _runtime_public_packet(0.25)
    packet_b, _ = _runtime_public_packet(0.75)
    conditions = {
        condition: apply_spectrum_condition(packet_a, condition)
        for condition in protocol["spectrum_conditions"]
    }
    condition_contexts = {
        condition: {**context_a, "spectrum": packet}
        for condition, packet in conditions.items()
    }
    non_spectral_hashes = {
        condition: _canonical_sha256(
            {key: value for key, value in context.items() if key != "spectrum"}
        )
        for condition, context in condition_contexts.items()
    }
    raw_curve_hashes = {
        condition: _raw_curve_sha256(packet)
        for condition, packet in conditions.items()
        if condition != "masked"
    }

    archive_policy = protocol["archive_policy"]
    archive = PublicSpectrumArchive(
        retrieval_cost=float(archive_policy["historical_retrieval_cost"])
    )
    archive.record(
        "experiment-1:step-5:hplc",
        packet_a,
        experiment_index=1,
        measurement_step=5,
        measurement_cost=float(context_a["measurement_cost"]),
    )
    archive.record(
        "experiment-2:step-5:hplc",
        packet_b,
        experiment_index=2,
        measurement_step=5,
        measurement_cost=float(context_a["measurement_cost"]),
    )
    catalog = archive.catalog()
    retrieved = archive.retrieve("experiment-1:step-5:hplc")
    unknown_rejected = False
    try:
        archive.retrieve("missing-spectrum")
    except ObservationIdentifiabilityError:
        unknown_rejected = True
    archive_ledger = archive.ledger()
    archive_report = {
        "catalog": catalog,
        "catalog_sha256": _canonical_sha256({"catalog": catalog}),
        "catalog_contains_signal": any(
            key in json.dumps(catalog, sort_keys=True)
            for key in ("intensity", "absorbance", "replicate_signals", "peaks")
        ),
        "retrieved_packet_sha256": _canonical_sha256(retrieved),
        "expected_packet_sha256": _canonical_sha256(packet_a),
        "unknown_id_rejected": unknown_rejected,
        "ledger": archive_ledger,
    }

    forbidden_tokens = tuple(str(item) for item in protocol["forbidden_public_tokens"])
    leakage_matches = sorted(
        token
        for token in forbidden_tokens
        if any(token in payload for payload in serialized_public_reports)
    )
    controls = {
        "spectral_expectations_match": all(
            item["identifiability"]["identifiable"] is item["expected_identifiable"]
            for item in instruments.values()
        ),
        "all_instruments_have_explicit_degradation": all(
            item["degraded_low_signal"]["identifiable"] is False
            and item["degraded_low_signal"]["warnings"]
            for item in instruments.values()
        ),
        "simple_probe_finds_public_information": sum(
            item["replicate_probe_accuracy"] >= 0.75 for item in instruments.values()
        )
        >= 3,
        "ph_signal_and_degradation_are_explicit": ph_report["state_pair_distinguishable"]
        is True
        and ph_report["low_contrast_degraded"] is True,
        "public_identifiability_reports_do_not_leak_truth": not leakage_matches,
        "spectrum_conditions_are_exact": set(conditions) == set(SPECTRUM_CONDITIONS)
        and conditions["assigned"].get("assignments")
        and conditions["unassigned"].get("assignments") == []
        and conditions["masked"].get("available") is False,
        "assigned_unassigned_share_raw_curve": raw_curve_hashes["assigned"]
        == raw_curve_hashes["unassigned"],
        "non_spectral_context_is_paired": len(set(non_spectral_hashes.values())) == 1,
        "history_catalog_is_request_only": archive_report["catalog_contains_signal"] is False
        and archive_report["retrieved_packet_sha256"]
        == archive_report["expected_packet_sha256"],
        "history_failures_and_costs_are_ledgered": archive_report["unknown_id_rejected"]
        is True
        and len(archive_ledger) == 2
        and all("cost" in item and "success" in item for item in archive_ledger),
    }
    controls_ready = all(controls.values())
    source_commit, dirty = _git_provenance(workspace)
    return {
        "schema_version": "chemworld-observation-identifiability-report-0.1",
        "protocol_id": protocol["protocol_id"],
        "status": "observation_identifiability_controls_passed"
        if controls_ready
        else "observation_identifiability_controls_failed",
        "controls_ready": controls_ready,
        "source_commit": source_commit,
        "source_tree_dirty": dirty,
        "protocol_sha256": _canonical_sha256(protocol),
        "controls": controls,
        "instruments": instruments,
        "ph_meter": ph_report,
        "spectrum_conditions": {
            "condition_sha256": {
                condition: _canonical_sha256(packet) for condition, packet in conditions.items()
            },
            "raw_curve_sha256": raw_curve_hashes,
            "non_spectral_context_sha256": non_spectral_hashes,
        },
        "history_archive": archive_report,
        "leakage_matches": leakage_matches,
        "limitations": [
            "Synthetic signals are benchmark observations, not empirical sample predictions.",
            "Pairwise identifiability is not a proof of global mechanism identifiability.",
            "UV/Vis and IR deliberately expose weak-separation regimes under this state pair.",
        ],

    }


def _nearest_centroid_accuracy(reference: Any, alternative: Any) -> float:
    left = np.asarray(reference.replicate_signals, dtype=float)
    right = np.asarray(alternative.replicate_signals, dtype=float)
    left_centroid = np.mean(left[:3], axis=0)
    right_centroid = np.mean(right[:3], axis=0)
    correct = 0
    total = 0
    for expected, rows in ((0, left[3:]), (1, right[3:])):
        for row in rows:
            distances = (
                float(np.linalg.norm(row - left_centroid)),
                float(np.linalg.norm(row - right_centroid)),
            )
            correct += int(int(distances[1] < distances[0]) == expected)
            total += 1
    return correct / total


def _runtime_public_packet(profile_value: float) -> tuple[dict[str, Any], dict[str, Any]]:
    task = get_task("reaction-to-assay")
    task_payload = task.to_dict()
    vector = np.full(task_recipe_dimension(task_payload), profile_value, dtype=float)
    actions = task_recipe_from_unit_vector(task_payload, vector)["steps"]
    env = gym.make("ChemWorld", task_id=task.task_id, seed=811)
    try:
        env.reset(seed=811)
        for step, action in enumerate(actions, start=1):
            _, _, _, _, info = env.step(action)
            if action.get("operation") == "measure" and action.get("instrument") == "hplc":
                packet = info.get("raw_signal")
                if not isinstance(packet, dict):
                    raise ObservationIdentifiabilityError("runtime HPLC packet is missing")
                context = {
                    "task_id": task.task_id,
                    "step": step,
                    "cost": info.get("cost"),
                    "measurement_cost": info.get("measurement_cost"),
                    "constraint_flags": to_builtin(info.get("constraint_flags")),
                    "available_operations": list(task.allowed_operations),
                }
                return copy.deepcopy(packet), context
    finally:
        env.close()
    raise ObservationIdentifiabilityError("runtime recipe did not produce HPLC evidence")


def _raw_curve_sha256(packet: Mapping[str, Any]) -> str:
    keys = (
        "time_min",
        "wavelength_nm",
        "chemical_shift_ppm",
        "wavenumber_cm-1",
        "intensity",
        "absorbance",
        "transmittance",
        "replicate_signals",
        "raw_signal",
        "axis",
    )
    return _canonical_sha256({key: packet[key] for key in keys if key in packet})






def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_builtin(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_provenance(workspace: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


__all__ = [
    "ObservationIdentifiabilityError",
    "PublicSpectrumArchive",
    "apply_spectrum_condition",
    "audit_observation_identifiability",
    "load_observation_identifiability_protocol",
]
