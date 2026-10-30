"""Public spectrum disclosure and request-only history, independent of audit files."""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

SPECTRUM_CONDITIONS = ("assigned", "unassigned", "masked")

class ObservationIdentifiabilityError(RuntimeError):
    """Raised when a public observation or archive request violates its contract."""



class PublicSpectrumArchive:
    """Request-only archive whose catalog never contains signal arrays."""

    def __init__(self, *, retrieval_cost: float = 0.0) -> None:
        if not math.isfinite(retrieval_cost) or retrieval_cost < 0.0:
            raise ValueError("retrieval_cost must be finite and non-negative")
        self._retrieval_cost = float(retrieval_cost)
        self._packets: dict[str, dict[str, Any]] = {}
        self._catalog: list[dict[str, Any]] = []
        self._ledger: list[dict[str, Any]] = []

    def record(
        self,
        spectrum_id: str,
        packet: Mapping[str, Any],
        *,
        experiment_index: int,
        measurement_step: int,
        measurement_cost: float,
    ) -> None:
        if not spectrum_id or spectrum_id in self._packets:
            raise ObservationIdentifiabilityError("spectrum id must be non-empty and unique")
        if not math.isfinite(measurement_cost) or measurement_cost < 0.0:
            raise ObservationIdentifiabilityError(
                "measurement cost must be finite and non-negative"
            )
        stored = copy.deepcopy(dict(packet))
        self._packets[spectrum_id] = stored
        self._catalog.append(
            {
                "spectrum_id": spectrum_id,
                "instrument_id": stored.get("instrument_id"),
                "kind": stored.get("kind"),
                "experiment_index": int(experiment_index),
                "measurement_step": int(measurement_step),
                "measurement_cost": float(measurement_cost),
            }
        )

    def catalog(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._catalog)

    def retrieve(self, spectrum_id: str) -> dict[str, Any]:
        packet = self._packets.get(spectrum_id)
        success = packet is not None
        self._ledger.append(
            {
                "event": "historical_spectrum_retrieval",
                "spectrum_id": spectrum_id,
                "success": success,
                "cost": self._retrieval_cost,
            }
        )
        if packet is None:
            raise ObservationIdentifiabilityError("unknown spectrum id")
        return copy.deepcopy(packet)

    def ledger(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._ledger)



def apply_spectrum_condition(packet: Mapping[str, Any], condition: str) -> dict[str, Any]:
    """Apply assigned, unassigned, or masked disclosure to one public packet."""

    if condition not in SPECTRUM_CONDITIONS:
        raise ObservationIdentifiabilityError("unknown spectrum condition")
    if condition == "assigned":
        result = copy.deepcopy(dict(packet))
        result["spectrum_condition"] = "assigned"
        return result
    if condition == "masked":
        return {
            "schema_version": "chemworld-spectrum-condition-0.1",
            "spectrum_condition": "masked",
            "available": False,
        }
    result = copy.deepcopy(dict(packet))
    result["spectrum_condition"] = "unassigned"
    result["assignments"] = []
    result.pop("processed_estimates", None)
    result.pop("uncertainty", None)
    result.pop("calibration", None)
    for collection_name in ("peaks", "bands"):
        collection = result.get(collection_name)
        if not isinstance(collection, list):
            continue
        for index, item in enumerate(collection):
            if not isinstance(item, dict):
                continue
            item.pop("species_id", None)
            item.pop("analyte_id", None)
            item.pop("group", None)
            item.pop("metadata", None)
            if "assignment" in item:
                item["assignment"] = "unassigned"
            item.setdefault("feature_id", f"feature_{index + 1:03d}")
    return result
