"""Deterministic operation-level calibration agent for G2 tri-arm design.

This is not a leaderboard baseline.  It is a reproducible design instrument
used to compare campaign envelopes before spending external-model calls.  The
same policy runs under opaque, correct nominal, and blindly misindexed public
material dossiers; only the dossier changes between arms.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.interaction import InteractionCapabilities


@dataclass(frozen=True)
class G2CalibrationPolicy:
    """One operation-level policy shape used in resource-design sweeps."""

    policy_id: str
    diagnostic_instrument: str | None = None
    adapted_second_stage: bool = False
    reagent_mol: float = 0.020
    solvent_L: float = 0.040
    potential_V: float = 1.20
    current_mA: float = 250.0
    duration_s: float = 14_400.0
    adapted_potential_V: float = 1.40
    adapted_current_mA: float = 250.0
    adapted_duration_s: float = 7_200.0
    empirical_weight: float = 0.78
    prior_weight: float = 0.22
    exploration_bonus: float = 0.08

    @property
    def operations_per_completed_batch(self) -> int:
        return (
            6
            + int(self.diagnostic_instrument is not None)
            + 2 * int(self.adapted_second_stage)
        )


class G2TriarmCalibrationAgent(BaseAgent):
    """Choose every primitive operation while adapting material pairs by batch."""

    name = "g2_triarm_calibration"

    def __init__(self, policy: G2CalibrationPolicy) -> None:
        self.policy = policy

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        material = task_info.get("material_information", {})
        dossier = material.get("dossier") if isinstance(material, Mapping) else None
        self.material_condition = (
            str(material.get("mode", "opaque_codes"))
            if isinstance(material, Mapping)
            else "opaque_codes"
        )
        self.prior_scores = _descriptor_prior_scores(dossier)
        self._pair_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
        self._batch_index = 0
        self._current_pair: tuple[int, int] | None = None
        self._actions: list[dict[str, Any]] = []
        self._action_index = 0
        self._selected_pairs: list[tuple[int, int]] = []
        self._final_scores: list[float] = []

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if not self._actions or self._action_index >= len(self._actions):
            self._current_pair = self._select_pair()
            self._selected_pairs.append(self._current_pair)
            self._actions = self._batch_actions(self._current_pair)
            self._action_index = 0
        action = deepcopy(self._actions[self._action_index])
        self._action_index += 1
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        del observation, reward
        if not info.get("experiment_ended"):
            return
        if self._current_pair is None:
            raise RuntimeError("completed batch has no calibration material pair")
        score = info.get("leaderboard_score")
        normalized_score = (
            float(score)
            if isinstance(score, int | float) and not isinstance(score, bool)
            else 0.0
        )
        self._pair_scores[self._current_pair].append(normalized_score)
        self._final_scores.append(normalized_score)
        self._batch_index += 1
        self._current_pair = None
        self._actions = []
        self._action_index = 0

    def manifest(self) -> dict[str, Any]:
        payload = super().manifest()
        payload.update(
            {
                "role_id": "g2_triarm_resource_design_calibration",
                "policy": {
                    "policy_id": self.policy.policy_id,
                    "operations_per_completed_batch": (
                        self.policy.operations_per_completed_batch
                    ),
                    "diagnostic_instrument": self.policy.diagnostic_instrument,
                    "adapted_second_stage": self.policy.adapted_second_stage,
                },
                "material_information_condition": self.material_condition,
                "selected_pairs": [list(pair) for pair in self._selected_pairs],
                "final_scores": list(self._final_scores),
                "external_model_calls": 0,
            }
        )
        return payload

    def interaction_capabilities(self) -> InteractionCapabilities:
        return InteractionCapabilities(
            decision_scope="operation",
            consumes_intermediate_observations=True,
            consumes_spectra=False,
            adapts_within_experiment=self.policy.adapted_second_stage,
            adapts_across_experiments=True,
            emits_structured_decision_audit=False,
        )

    def _select_pair(self) -> tuple[int, int]:
        pairs = tuple((electrolyte, solvent) for electrolyte in range(4) for solvent in range(4))
        # Correct/wrong dossiers choose their declared best first. Opaque ties
        # follow a seeded cyclic order so paired worlds remain deterministic.
        offset = self.seed % len(pairs)
        cyclic_rank = {
            pair: (index - offset) % len(pairs)
            for index, pair in enumerate(pairs)
        }
        prior_order = sorted(
            pairs,
            key=lambda pair: (
                self.prior_scores[pair],
                -cyclic_rank[pair],
            ),
            reverse=True,
        )
        if self._batch_index < 2:
            return prior_order[self._batch_index]

        # The next three batches challenge the leading material prior by
        # holding its electrolyte fixed and covering the remaining solvents.
        # A six-batch campaign therefore contains two prior-led batches, three
        # falsification batches, and one evidence-led exploitation batch.
        leading_electrolyte = prior_order[0][0]
        solvent_challenge = [
            pair
            for pair in prior_order
            if pair[0] == leading_electrolyte and not self._pair_scores[pair]
        ]
        if solvent_challenge:
            return solvent_challenge[0]

        tried = [pair for pair in pairs if self._pair_scores[pair]]
        if not tried:
            return prior_order[0]
        best: tuple[int, int] | None = None
        best_value = -math.inf
        total = sum(len(values) for values in self._pair_scores.values())
        for pair in tried:
            values = self._pair_scores[pair]
            mean = sum(values) / len(values)
            prior = self.prior_scores[pair]
            bonus = self.policy.exploration_bonus * math.sqrt(
                math.log(total + 1.0) / len(values)
            )
            value = (
                self.policy.empirical_weight * mean
                + self.policy.prior_weight * prior
                + bonus
            )
            if value > best_value:
                best = pair
                best_value = value
        if best is None:
            raise RuntimeError("calibration pair selection failed")
        return best

    def _batch_actions(self, pair: tuple[int, int]) -> list[dict[str, Any]]:
        electrolyte, solvent = pair
        actions: list[dict[str, Any]] = [
            {
                "operation": "add_solvent",
                "volume_L": self.policy.solvent_L,
                "solvent": solvent,
            },
            {
                "operation": "add_reagent",
                "amount_mol": self.policy.reagent_mol,
            },
            {
                "operation": "set_potential",
                "potential_V": self.policy.potential_V,
                "current_mA": self.policy.current_mA,
                "electrolyte_profile": electrolyte,
            },
            {
                "operation": "electrolyze",
                "duration_s": self.policy.duration_s,
            },
        ]
        if self.policy.diagnostic_instrument is not None:
            actions.append(
                {
                    "operation": "measure",
                    "instrument": self.policy.diagnostic_instrument,
                }
            )
        if self.policy.adapted_second_stage:
            actions.extend(
                [
                    {
                        "operation": "set_potential",
                        "potential_V": self.policy.adapted_potential_V,
                        "current_mA": self.policy.adapted_current_mA,
                        "electrolyte_profile": electrolyte,
                    },
                    {
                        "operation": "electrolyze",
                        "duration_s": self.policy.adapted_duration_s,
                    },
                ]
            )
        actions.extend(
            [
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        )
        return actions


def _descriptor_prior_scores(dossier: Any) -> dict[tuple[int, int], float]:
    pairs = tuple((electrolyte, solvent) for electrolyte in range(4) for solvent in range(4))
    if not isinstance(dossier, Mapping):
        return dict.fromkeys(pairs, 0.0)
    choices = dossier.get("choices")
    if not isinstance(choices, Mapping):
        return dict.fromkeys(pairs, 0.0)
    electrolyte_rows = choices.get("electrolyte_profile")
    solvent_rows = choices.get("solvent")
    if not isinstance(electrolyte_rows, list) or not isinstance(solvent_rows, list):
        return dict.fromkeys(pairs, 0.0)

    electrolyte_transport = []
    electrolyte_chemistry = []
    for row in electrolyte_rows:
        properties = row["nominal_properties"]
        electrolyte_transport.append(
            (
                math.log10(float(properties["bulk_conductivity_S_m"])),
                math.log10(float(properties["diffusivity_m2_s"])),
                -math.log10(float(properties["diffusion_layer_thickness_mm"])),
            )
        )
        electrolyte_chemistry.append(
            (
                float(properties["acid_concentration_mol_L"]),
                -float(properties["acid_pKa"]),
                -float(properties["precipitation_log10_Ksp"]),
            )
        )
    solvent_transport = []
    solvent_chemistry = []
    for row in solvent_rows:
        properties = row["nominal_properties"]
        solvent_transport.append(
            (
                math.log10(float(properties["relative_conductivity"])),
                math.log10(float(properties["relative_diffusivity"])),
            )
        )
        solvent_chemistry.append(
            (
                float(properties["relative_proton_activity"]),
                -math.log10(float(properties["relative_solubility_product"])),
                float(properties["relative_diffusivity"]),
            )
        )

    electrolyte_transport_score = _normalize(np.asarray(electrolyte_transport)).mean(axis=1)
    solvent_transport_score = _normalize(np.asarray(solvent_transport)).mean(axis=1)
    electrolyte_chemistry_score = _normalize(np.asarray(electrolyte_chemistry)).mean(axis=1)
    solvent_chemistry_score = _normalize(np.asarray(solvent_chemistry)).mean(axis=1)
    raw = {
        pair: float(
            0.65
            * (
                electrolyte_transport_score[pair[0]]
                + solvent_transport_score[pair[1]]
            )
            + 0.35
            * (
                electrolyte_chemistry_score[pair[0]]
                + solvent_chemistry_score[pair[1]]
            )
        )
        for pair in pairs
    }
    minimum = min(raw.values())
    maximum = max(raw.values())
    span = maximum - minimum
    return {
        pair: (value - minimum) / span if span > 0.0 else 0.0
        for pair, value in raw.items()
    }


def _normalize(values: np.ndarray) -> np.ndarray:
    minimum = values.min(axis=0)
    span = values.max(axis=0) - minimum
    return np.divide(
        values - minimum,
        span,
        out=np.zeros_like(values),
        where=span > 0.0,
    )


__all__ = ["G2CalibrationPolicy", "G2TriarmCalibrationAgent"]
