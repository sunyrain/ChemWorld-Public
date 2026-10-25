"""Read-only steady branch query on the current original SBML equipment state."""

from typing import Any

from chemworld.physchem.sbml_continuation import sbml_steady_branch
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime.sbml_models import validate_model_state


def analyze_sbml_steady_branch_world(
    state: Any, composition: Any, **options: Any
) -> dict[str, Any]:
    if composition.physics is None:
        raise ValueError("steady branch query requires the original physical composition")
    model = composition.physics.models_of("sbml").get(state.vessel_id)
    if not isinstance(model, SBMLSystemSpec):
        raise ValueError("selected vessel requires its original SBML declaration")
    _, values, _ = validate_model_state(state, model)
    result = sbml_steady_branch(model, dict(values), start_time_s=state.ledger.time_s, **options)
    result["world_query"] = {
        "vessel_id": state.vessel_id,
        "state_time_s": state.ledger.time_s,
        "world_advanced": False,
        "live_resources_charged": False,
        "queued_or_running_controls": bool(
            state.lab_clock is not None and (state.lab_clock.queued or state.lab_clock.running)
        ),
        "future_controls_predicted": False,
    }
    return result
