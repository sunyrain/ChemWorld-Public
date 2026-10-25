"""Read-only design query on the selected original formal workspace."""

from typing import Any

from chemworld.physchem.sbml_design import SBMLDesignSpec, optimize_sbml_design
from chemworld.physchem.sbml_system import SBMLSystemSpec
from chemworld.runtime.sbml_models import validate_model_state


def analyze_sbml_design_world(
    state: Any, composition: Any, spec: SBMLDesignSpec, **options: Any
) -> dict[str, Any]:
    if composition.physics is None:
        raise ValueError("design query requires the original physical composition")
    model = composition.physics.models_of("sbml").get(state.vessel_id)
    if not isinstance(model, SBMLSystemSpec):
        raise ValueError("design query requires the selected vessel's original SBML model")
    _, values, _ = validate_model_state(state, model)
    report = optimize_sbml_design(
        model, dict(values), spec, start_time_s=state.ledger.time_s, **options
    )
    report["world_query"] = {
        "vessel_id": state.vessel_id,
        "state_time_s": state.ledger.time_s,
        "world_advanced": False,
        "live_resources_charged": False,
        "queued_or_running_controls": bool(
            state.lab_clock is not None and (state.lab_clock.queued or state.lab_clock.running)
        ),
        "future_controls_predicted": False,
    }
    return report
