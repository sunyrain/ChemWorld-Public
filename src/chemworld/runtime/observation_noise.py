"""Path-independent observation-noise coordinates for paired evaluations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any

import numpy as np

KEYED_OBSERVATION_NOISE_VERSION = "chemworld-keyed-observation-noise-0.1"


@dataclass(frozen=True)
class ObservationNoiseCoordinate:
    namespace: str
    base_observation_seed: int
    experiment_index: int
    operation_type: str
    instrument: str
    replicate_index: int

    def __post_init__(self) -> None:
        if not self.namespace.strip() or not self.operation_type.strip():
            raise ValueError("noise namespace and operation type must be non-empty")
        if (
            self.experiment_index < 0
            or self.replicate_index < 0
            or self.base_observation_seed < 0
        ):
            raise ValueError("noise seeds and indices must be non-negative")

    def to_private_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "base_observation_seed_sha256": sha256(
                str(self.base_observation_seed).encode("ascii")
            ).hexdigest(),
            "experiment_index": self.experiment_index,
            "operation_type": self.operation_type,
            "instrument": self.instrument,
            "replicate_index": self.replicate_index,
        }

    @property
    def key_sha256(self) -> str:
        return sha256(_canonical_json_bytes(self.to_private_dict())).hexdigest()


def keyed_observation_rng(
    coordinate: ObservationNoiseCoordinate,
) -> np.random.Generator:
    """Create an isolated stream for one semantic observation coordinate."""

    digest = sha256(
        b"chemworld-keyed-observation-noise-0.1\0"
        + _canonical_json_bytes(coordinate.to_private_dict())
    ).digest()
    seed = int.from_bytes(digest[:16], "big")
    return np.random.default_rng(seed)


def keyed_noise_provenance(
    coordinate: ObservationNoiseCoordinate,
) -> dict[str, Any]:
    return {
        "schema_version": KEYED_OBSERVATION_NOISE_VERSION,
        "mode": "keyed",
        "noise_key_sha256": coordinate.key_sha256,
        "coordinate": coordinate.to_audit_dict(),
        "sequential_rng_position_used": False,
    }


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "KEYED_OBSERVATION_NOISE_VERSION",
    "ObservationNoiseCoordinate",
    "keyed_noise_provenance",
    "keyed_observation_rng",
]
