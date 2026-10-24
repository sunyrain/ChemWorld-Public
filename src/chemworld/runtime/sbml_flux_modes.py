"""Read current original formal state without changing any native ledger."""

from typing import Any

from chemworld.physchem.elementary_flux_modes import FluxModePolicy, sbml_elementary_flux_modes
from chemworld.runtime.sbml_models import validate_model_state


def analyze_sbml_flux_modes(
    state: Any, model: Any, *, policy: FluxModePolicy | None = None
) -> dict[str, Any]:
    _, values, _ = validate_model_state(state, model)
    return sbml_elementary_flux_modes(
        model.frame, dict(values), time_s=state.ledger.time_s, policy=policy
    )


__all__ = ["analyze_sbml_flux_modes"]
