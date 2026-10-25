"""Fixed design and native source binding; no dynamic qualification run."""

from copy import deepcopy

import pytest
from battery_parameters45_cases import (
    CASE_IDS,
    GRIDS,
    RESPONSE_ORDER,
    case_data,
    native_spec,
    routes,
    stage_times,
    validate_case,
)


def test_six_units_and_twenty_four_independent_histories_are_explicit():
    design = routes()
    assert len(CASE_IDS) == 6
    assert len(design) == 24
    assert len({tuple(row.values()) for row in design}) == 24
    for implementation in ("native", "source"):
        assert GRIDS[implementation] == (32, 64)
        assert sum(row["implementation"] == implementation for row in design) == 12
    expected = {
        "P01": ((600, 0, "A", 31),),
        "P02": ((1800, 1, "A", 31),),
        "P03": ((5000, 5, "A", 101),),
        "P04": ((1800, 1, "A", 31), (3000, -1, "A", 101)),
        "P05": ((600, 1, "A", 31), (300, 0, "A", 31),
                (600, -.5, "A", 31), (300, 0, "A", 31)),
        "P06": ((1200, 1, "A", 31), (300, 0, "A", 31), (600, 4.05, "V", 31)),
    }
    for identity in CASE_IDS:
        data = validate_case(case_data(identity))
        assert tuple((s["duration_s"], s["target"], s["unit"], s["samples"])
                     for s in data["stages"]) == expected[identity]
        assert all(stage_times(s)[0] == 0 and stage_times(s)[-1] == s["duration_s"]
                   for s in data["stages"])
    assert case_data("P03")["stages"][-1]["expected"] == "minimum-voltage"
    assert case_data("P04")["stages"][-1]["expected"] == "maximum-voltage"


def test_native_geometry_and_nominal_basis_are_real_not_shared_with_source_mesh():
    data = case_data("P02")
    spec = native_spec(data, 32)
    assert spec.parameter_binding.validate_projection(spec)
    assert spec.negative.cells == spec.positive.cells == 32
    assert spec.negative.mesh_exponent == spec.positive.mesh_exponent == 2
    assert spec.maximum_current_A == 10
    assert spec.parameter_binding.source_nominal_capacity_Ah == 5
    assert spec.current_at_c_rate(1) == pytest.approx(5.252997061422181)
    assert spec.negative.mesh.faces_m[-1] == spec.negative.radius_m
    assert len(spec.inventory_keys) == 64
    widths = [b - a for a, b in zip(spec.negative.mesh.faces_m,
                                   spec.negative.mesh.faces_m[1:], strict=False)]
    assert max(widths) > 10 * min(widths)
    assert data["comparison_policy"]["equal_cell_counts_are_not_equal_meshes"]
    assert data["comparison_policy"]["preserve_all_original_times_states_and_event_tails"]
    assert RESPONSE_ORDER[:4] == ("voltage_V", "current_A", "charge_C", "work_J")


def test_case_reads_are_defensive_and_no_physical_source_override_is_allowed():
    data = case_data("P04")
    original = deepcopy(data)
    data["stages"][0]["duration_s"] = 1
    assert case_data("P04") == original
    for field, bad in (("source_name", "synthetic"), ("temperature_K", 310),
                       ("ce_mol_m3", 900), ("source_nominal_capacity_Ah", 5.25)):
        changed = deepcopy(original)
        changed[field] = bad
        with pytest.raises(ValueError, match="released fixed-T"):
            validate_case(changed)
    changed = deepcopy(original)
    changed["stages"][0]["target"] = 11
    with pytest.raises(ValueError, match="hardware"):
        validate_case(changed)
    with pytest.raises(ValueError, match="unknown"):
        case_data("P07")
