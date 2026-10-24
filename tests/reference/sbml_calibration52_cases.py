"""Fixed 12-unit coverage declared before execution in the 52B experiment note."""

CASES = (
    {"name": "trajectory_decay", "type": "trajectory", "kind": "decay"},
    {"name": "trajectory_reversible", "type": "trajectory", "kind": "reversible"},
    {"name": "trajectory_scaled", "type": "trajectory", "kind": "reversible", "scaled": True},
    {"name": "trajectory_zero", "type": "zero"},
    {"name": "fit_decay", "type": "fit", "kind": "decay"},
    {"name": "fit_reversible", "type": "fit", "kind": "reversible"},
    {"name": "fit_confounded", "type": "fit", "kind": "confounded"},
    {"name": "fit_bound", "type": "fit", "kind": "decay", "boundary": True},
    {"name": "scope_rejections", "type": "scope"},
    {"name": "budget_exclusions", "type": "budget"},
    {"name": "holdout_independence", "type": "holdout"},
    {"name": "immutable_initial_assignment", "type": "immutable"},
)
