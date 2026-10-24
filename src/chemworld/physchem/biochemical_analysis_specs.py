"""Read-only biochemical analysis policy and JSON result contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any


@dataclass(frozen=True)
class BiochemicalAnalysisPolicy:
    root_tolerance: float = 1e-10
    maximum_iterations: int = 60
    maximum_backtracks: int = 30
    rank_relative_tolerance: float = 1e-10
    stability_tolerance_s_inv: float = 1e-9
    amount_scale_mol: float | None = None
    time_scale_s: float = 1.0
    integer_jump_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        for key, value in asdict(self).items():
            if key in {"maximum_iterations", "maximum_backtracks"}:
                if type(value) is not int or value < 1:
                    raise ValueError("analysis iteration budgets must be positive integers")
            elif value is not None and (not isfinite(value) or value <= 0):
                raise ValueError("analysis scales and tolerances must be finite positive")
        if self.rank_relative_tolerance >= 1 or self.integer_jump_tolerance >= 0.5:
            raise ValueError("rank/jump tolerances cannot erase a physical mode or fractional jump")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BiochemicalAnalysisResult:
    payload: Mapping[str, Any]

    @property
    def success(self) -> bool:
        return self.payload["status"] == "steady_state_found"

    @property
    def steady_amounts_mol(self) -> dict[str, float] | None:
        value = self.payload.get("steady_amounts_mol")
        return None if value is None else dict(value)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.payload, allow_nan=False))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BiochemicalAnalysisResult:
        if payload.get("model") != "native-biochemical-steady-analysis-v1":
            raise ValueError("unsupported biochemical analysis result schema")
        return cls(json.loads(json.dumps(payload, allow_nan=False)))
