"""Shared execution outcome vocabulary."""

OPERATION_OUTCOMES = frozenset({
    "success", "rejected", "numerical_failure", "process_failure", "stopped",
})
