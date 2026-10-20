"""Task-specific observation contracts for ChemWorld instruments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from chemworld.foundation import Instrument
from chemworld.world.scoring import TaskScoringContract

REACTION_DIAGNOSTIC_KEYS = (
    "yield",
    "selectivity",
    "conversion",
    "byproduct_signal",
    "degradation_warning",
)


@dataclass(frozen=True)
class TaskObservationContract:
    """Serializable mapping from task intent to instrument-visible keys."""

    success_metrics: tuple[str, ...]
    score_family: str
    required_observation_keys: tuple[str, ...]
    instrument_observable_keys: dict[str, tuple[str, ...]]
    mechanism_observable_mapping: dict[str, tuple[str, ...]]

    @classmethod
    def from_task(
        cls,
        *,
        success_metrics: tuple[str, ...],
        scoring_contract: TaskScoringContract,
        allowed_instruments: tuple[str, ...],
        instruments: Mapping[str, Instrument],
        mechanism_observable_mapping: Mapping[str, tuple[str, ...]],
    ) -> TaskObservationContract:
        required_keys = _required_observation_keys(
            success_metrics=success_metrics,
            scoring_contract=scoring_contract,
        )
        requested_instruments = tuple(
            instrument_id for instrument_id in allowed_instruments if instrument_id in instruments
        )
        if scoring_contract.score_family == "unscored":
            required_keys = tuple(
                dict.fromkeys(
                    key
                    for name in requested_instruments
                    for key in instruments[name].observable_keys
                )
            )
        instrument_keys = {
            instrument_id: tuple(
                key for key in instruments[instrument_id].observable_keys if key in required_keys
            )
            for instrument_id in requested_instruments
        }
        return cls(
            success_metrics=success_metrics,
            score_family=scoring_contract.score_family,
            required_observation_keys=tuple(required_keys),
            instrument_observable_keys=instrument_keys,
            mechanism_observable_mapping={
                role: tuple(species)
                for role, species in sorted(mechanism_observable_mapping.items())
            },
        )

    def observable_keys_for_instrument(
        self,
        instrument_id: str,
        fallback_keys: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self.instrument_observable_keys.get(instrument_id, fallback_keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success_metrics": list(self.success_metrics),
            "score_family": self.score_family,
            "required_observation_keys": list(self.required_observation_keys),
            "instrument_observable_keys": {
                instrument_id: list(keys)
                for instrument_id, keys in sorted(self.instrument_observable_keys.items())
            },
            "mechanism_observable_mapping": {
                role: list(species)
                for role, species in sorted(self.mechanism_observable_mapping.items())
            },
            "contract_hash": self.contract_hash,
        }

    @property
    def contract_hash(self) -> str:
        payload = {
            "success_metrics": list(self.success_metrics),
            "score_family": self.score_family,
            "required_observation_keys": list(self.required_observation_keys),
            "instrument_observable_keys": {
                instrument_id: list(keys)
                for instrument_id, keys in sorted(self.instrument_observable_keys.items())
            },
            "mechanism_observable_mapping": {
                role: list(species)
                for role, species in sorted(self.mechanism_observable_mapping.items())
            },
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()


def _required_observation_keys(
    *,
    success_metrics: tuple[str, ...],
    scoring_contract: TaskScoringContract,
) -> tuple[str, ...]:
    keys: list[str] = []
    if "reaction_score" in scoring_contract.component_weights:
        keys.extend(REACTION_DIAGNOSTIC_KEYS)
    for key in (*success_metrics, *scoring_contract.component_weights):
        if key in {
            "score",
            "final_assay_score",
            "trajectory_validity",
            "sample_efficiency",
            "constraint_violations",
            "mechanism_explanation",
            "failure_analysis",
            "validator_use",
            "explanation",
            "rank_confidence",
            "public_private_gap",
            "uncertainty",
            "local_model_quality",
        }:
            continue
        keys.append(key)
    if scoring_contract.score_family == "purification":
        keys.extend(("purity", "recovery", "impurity_signal", "process_mass_balance_error"))
    elif scoring_contract.score_family == "partition":
        keys.extend(
            (
                "phase_ratio",
                "product_in_organic",
                "product_in_aqueous",
                "purity",
                "recovery",
                "impurity_signal",
            )
        )
    elif scoring_contract.score_family == "crystallization":
        keys.extend(
            (
                "crystal_yield",
                "crystal_purity",
                "crystal_size",
                "crystal_csd_quality",
                "crystal_fines_fraction",
            )
        )
    elif scoring_contract.score_family == "distillation":
        keys.extend(("distillate_purity", "distillate_recovery", "solvent_loss"))
    elif scoring_contract.score_family == "continuous_flow":
        keys.append("flow_conversion")
    elif scoring_contract.score_family == "electrochemistry":
        keys.extend(
            (
                "selective_product_yield",
                "electrochemical_conversion",
                "electrochemical_selectivity",
                "faradaic_efficiency",
                "transport_efficiency",
                "ohmic_efficiency",
                "energy_efficiency",
            )
        )
    elif scoring_contract.score_family == "equilibrium_characterization":
        keys.extend(
            (
                "pH_normalized",
                "acid_dissociation_fraction",
                "precipitation_signal",
                "equilibrium_residual",
                "equilibrium_confidence",
            )
        )
    return tuple(dict.fromkeys(keys))


__all__ = ["REACTION_DIAGNOSTIC_KEYS", "TaskObservationContract"]
