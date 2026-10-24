"""Read-only consumer of the existing formal-world model and unique state slot."""

from __future__ import annotations

from typing import Any

from chemworld.physchem.biochemical_analysis import analyze_biochemical_steady_state
from chemworld.physchem.biochemical_analysis_specs import (
    BiochemicalAnalysisPolicy,
    BiochemicalAnalysisResult,
)
from chemworld.runtime.sbml_models import validate_model_state


def analyze_sbml_world_state(
    state: Any,
    model: Any,
    *,
    policy: BiochemicalAnalysisPolicy | None = None,
    include_control: bool = True,
    include_lna: bool = False,
) -> BiochemicalAnalysisResult:
    """Read current formal coordinates/time; never install a hypothetical root."""
    _, values, _ = validate_model_state(state, model)
    return analyze_biochemical_steady_state(
        model.frame,
        dict(values),
        time_s=state.ledger.time_s,
        policy=policy,
        include_control=include_control,
        include_lna=include_lna,
    )
