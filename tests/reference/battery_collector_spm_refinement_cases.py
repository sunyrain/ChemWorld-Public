"""Fixed five-mesh follow-up for the unresolved collector-SPM responses.

Scientific inputs come unchanged from the completed supplement.  The explicit
mesh and same-end layout belong to this new plan, not to a rewritten old case.
This module constructs no model and executes no history.
"""

from copy import deepcopy

from battery_collector_spm_supplement_cases import case_data

UNITS = ("S01", "S03", "S04")
MESHES = {
    "A": {"z": 16, "r": 24},
    "B": {"z": 32, "r": 24},
    "C": {"z": 64, "r": 24},
    "D": {"z": 32, "r": 48},
    "E": {"z": 64, "r": 48},
}
EDGES = (("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"), ("B", "E"))
FINE_EDGES = (("B", "E"), ("C", "E"), ("D", "E"))
LAYOUT = {"negative_tab": "right", "positive_tab": "right"}
IMPLEMENTATIONS = ("native", "source")

EXPECTED_ROUTES = 15
EXPECTED_HISTORIES = 30
EXPECTED_STAGES = 50
EXPECTED_SAMPLES = 1550
EXPECTED_COMPARISONS = 51
EXPECTED_COMPARISON_STAGES = 85


def _validate_unit(unit):
    if unit not in UNITS:
        raise KeyError(unit)


def unit_data(unit):
    """Return the complete old case, independently owned by this caller.

    In particular, retain the old ``layouts`` and ``grids`` metadata.  Callers
    pass :data:`LAYOUT` and the selected :data:`MESHES` entry separately.
    """
    _validate_unit(unit)
    return deepcopy(case_data(unit))


def history_plan(unit):
    """Return all ten calls in mesh order, native then source for each mesh."""
    _validate_unit(unit)
    return tuple((mesh, implementation) for mesh in MESHES for implementation in IMPLEMENTATIONS)
