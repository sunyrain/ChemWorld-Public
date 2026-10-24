"""Declarative named-control targets for the existing circuit transaction path.

Controller memory is owned by the circuit state slot, never by these settings.
This adapter only validates units/capabilities and constructs the physical program.
"""

from math import isfinite

from chemworld.physchem.battery_thevenin_control import (
    TheveninControlProgram,
    thevenin_control_program,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram

NAMED_FIELDS = frozenset({
    "control_mode", "power_W", "resistance_Ohm", "current_limit_A",
    "branch_lower_A", "branch_upper_A",
})
TARGETS = {
    "current": ("current_A", "A"), "voltage": ("potential_V", "V"),
    "power": ("power_W", "W"), "resistance": ("resistance_Ohm", "Ohm"),
    "differential_power": ("power_W", "W"),
    "differential_resistance": ("resistance_Ohm", "Ohm"),
    "cccv": ("potential_V", "V"),
}


def configured_program(state, record, action):
    settings = record.settings["electrical_control"]
    mode = action.get("control_mode")
    if mode is None:
        fields = [mode for mode, key in (("current", "current_A"), ("voltage", "potential_V"))
                  if key in action]
        if len(fields) != 1 or NAMED_FIELDS & action.keys():
            raise ValueError("named circuit targets require an explicit control_mode")
        mode = fields[0]
    if mode not in settings.get("named_control_modes", ()) or mode not in TARGETS:
        raise ValueError("named control mode is not declared by this circuit")
    name, unit = TARGETS[mode]
    target_fields = {"current_A", "potential_V", "power_W", "resistance_Ohm"} & action.keys()
    if target_fields != {name}:
        raise ValueError("provide exactly the target quantity of the selected mode")
    target, duration = float(action[name]), float(action.get("ramp_duration_s", 0.))
    if not isfinite(target) or not isfinite(duration) or duration < 0:
        raise ValueError("target and nonnegative ramp duration must be finite")
    if unit in {"A", "V"}:
        low, high = settings["limits_"+unit]
        if not low <= target <= high:
            raise ValueError("target exceeds the declared source limits")
    elif unit == "Ohm" and target <= 0:
        raise ValueError("load resistance must be positive")
    old = thevenin_control_program(settings["program"])
    if duration and old.control_mode != mode:
        raise ValueError("a ramp cannot interpolate between different physical control modes")
    time = state.ledger.time_s
    points = (((time, old.value_at(time)), (time+duration, target))
              if duration else ((time, target),))
    limit = None
    if mode == "cccv":
        if "current_limit_A" not in action:
            raise ValueError("CCCV requires an explicit charging-current limit")
        current = float(action["current_limit_A"])
        low, high = settings["limits_A"]
        if not isfinite(current) or not low <= current < 0:
            raise ValueError("CCCV requires a negative charging current within source limits")
        limit = ScalarControlProgram(((time, current),), "A")
    elif "current_limit_A" in action:
        raise ValueError("current_limit_A is specific to CCCV")
    branch = None
    branch_keys = {"branch_lower_A", "branch_upper_A"} & action.keys()
    if mode in {"power", "differential_resistance"}:
        if len(branch_keys) != 2:
            raise ValueError("this mode requires both explicit current-branch endpoints")
        branch = (float(action["branch_lower_A"]), float(action["branch_upper_A"]))
        low, high = settings["limits_A"]
        if not all(isfinite(v) for v in branch) or not low <= branch[0] < branch[1] <= high:
            raise ValueError("current branch must be finite, ordered and within source limits")
    elif branch_keys:
        raise ValueError("current branch is not a parameter of this mode")
    program = TheveninControlProgram(mode, ScalarControlProgram(points, unit),
                                     current_limit=limit, current_branch_A=branch)
    if action.get("operation") != "queue_electrical_stage":
        validate_active_program(state, program)
    return program


def stage_program(payload, time_s):
    """Activate a queued constant target without copying or resetting q."""
    program = thevenin_control_program(payload)
    return TheveninControlProgram(
        program.control_mode,
        ScalarControlProgram(((time_s, program.setpoints[-1][1]),), program.unit),
        current_limit=(ScalarControlProgram(
            ((time_s, program.current_limit.setpoints[-1][1]),), "A")
            if program.current_limit is not None else None),
        current_branch_A=program.current_branch_A,
    ).to_dict()


def validate_active_program(state, program=None):
    """Check a newly active target at the actual stage-boundary physical state."""
    from chemworld.physchem.battery_thevenin import thevenin_fields
    from chemworld.runtime.thevenin import unpack_thevenin

    record, _, spec, physical, active = unpack_thevenin(state)
    program = active if program is None else program
    fields = thevenin_fields(spec, physical, program)
    low, high = record.settings["electrical_control"]["limits_A"]
    if not low <= fields["current_A"] <= high:
        raise ValueError("activated control has a current outside source limits")
