"""Coverage and original-input contracts; these tests run no model histories."""

from copy import deepcopy

import pytest
from battery_collector_spm_refinement_cases import (
    EDGES,
    EXPECTED_COMPARISON_STAGES,
    EXPECTED_COMPARISONS,
    EXPECTED_HISTORIES,
    EXPECTED_ROUTES,
    EXPECTED_SAMPLES,
    EXPECTED_STAGES,
    FINE_EDGES,
    IMPLEMENTATIONS,
    LAYOUT,
    MESHES,
    UNITS,
    history_plan,
    unit_data,
)
from battery_collector_spm_supplement_cases import case_data


def test_plan_contains_every_fixed_mesh_and_both_implementations():
    assert UNITS == ("S01", "S03", "S04")
    assert list(MESHES.items()) == [
        ("A", {"z": 16, "r": 24}),
        ("B", {"z": 32, "r": 24}),
        ("C", {"z": 64, "r": 24}),
        ("D", {"z": 32, "r": 48}),
        ("E", {"z": 64, "r": 48}),
    ]
    assert IMPLEMENTATIONS == ("native", "source")
    assert LAYOUT == {"negative_tab": "right", "positive_tab": "right"}
    for unit in UNITS:
        plan = history_plan(unit)
        assert isinstance(plan, tuple)
        assert plan == tuple(
            (mesh, implementation)
            for mesh in ("A", "B", "C", "D", "E")
            for implementation in ("native", "source")
        )
        assert len(set(plan)) == 10


def test_edges_separate_both_directions_and_retain_joint_comparison():
    assert EDGES == (
        ("A", "B"),
        ("B", "C"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
        ("B", "E"),
    )
    assert FINE_EDGES == (("B", "E"), ("C", "E"), ("D", "E"))
    assert set(FINE_EDGES) <= set(EDGES)
    for a, b in (("A", "B"), ("B", "C"), ("D", "E")):
        assert MESHES[b]["z"] == 2 * MESHES[a]["z"]
        assert MESHES[b]["r"] == MESHES[a]["r"]
    for a, b in (("B", "D"), ("C", "E")):
        assert MESHES[b]["z"] == MESHES[a]["z"]
        assert MESHES[b]["r"] == 2 * MESHES[a]["r"]
    assert all(MESHES["E"][axis] == 2 * MESHES["B"][axis] for axis in ("z", "r"))


def test_denominators_are_derived_from_complete_waveforms_and_all_edges():
    calls = [(unit, *call) for unit in UNITS for call in history_plan(unit)]
    paired_routes = {(unit, mesh) for unit, mesh, _implementation in calls}
    stages = [stage for unit, *_ in calls for stage in unit_data(unit)["waveform"]]
    samples = sum(
        len(unit_data(unit)["waveform"]) * unit_data(unit)["samples_per_segment"]
        for unit, *_ in calls
    )
    comparisons = [
        (unit, "cross-implementation", mesh, mesh) for unit in UNITS for mesh in MESHES
    ] + [
        (unit, implementation, a, b)
        for unit in UNITS
        for implementation in IMPLEMENTATIONS
        for a, b in EDGES
    ]
    comparison_stages = sum(len(unit_data(unit)["waveform"]) for unit, *_ in comparisons)
    assert len(paired_routes) == EXPECTED_ROUTES == 15
    assert len(calls) == EXPECTED_HISTORIES == 30
    assert len(stages) == EXPECTED_STAGES == 50
    assert samples == EXPECTED_SAMPLES == 1550
    assert len(comparisons) == len(set(comparisons)) == EXPECTED_COMPARISONS == 51
    assert comparison_stages == EXPECTED_COMPARISON_STAGES == 85
    assert sum(stage[0] for stage in stages) == 5400.0


@pytest.mark.parametrize("unit", UNITS)
def test_complete_scientific_inputs_equal_original_case_without_metadata_rewrite(unit):
    original = case_data(unit)
    actual = unit_data(unit)
    assert actual == original
    assert actual["layouts"] == ("opposite", "same_end")
    assert actual["grids"] == ("coarse", "fine")
    assert actual["samples_per_segment"] == 31
    assert actual["waveform"] == original["waveform"]


@pytest.mark.parametrize("unit", UNITS)
def test_nested_input_mutation_does_not_affect_other_calls_or_old_cases(unit):
    first, second = unit_data(unit), unit_data(unit)
    expected = deepcopy(second)
    first["negative"]["active_volume_m3"] = -1.0
    first["positive"]["diffusivity_m2_s"] = -1.0
    first["source_versions"]["pybamm"] = "changed"
    first["initial_z_fractions"]["negative"] = (0.0,)
    first["waveform"] = ((1e-4, 0.0, "A"),)
    first["layouts"] = ("opposite",)
    assert second == expected
    assert unit_data(unit) == expected
    assert case_data(unit) == expected


@pytest.mark.parametrize("unit", ("S02", "S10", "CC01", "s01", "", None, 1))
def test_unknown_units_cannot_enter_plan_or_input_lookup(unit):
    for function in (unit_data, history_plan):
        with pytest.raises(KeyError) as error:
            function(unit)
        assert error.value.args == (unit,)


@pytest.mark.parametrize("unit", ("S01", "S04"), ids=("CC", "CV"))
def test_highest_mesh_actual_released_source_and_native_state_layout(unit):
    """Discretise the actual E-scale source; never solve a time history."""
    from battery_collector_spm_cases import native_spec
    from battery_collector_spm_supplement_assess import Checks, source_state_layout
    from battery_collector_spm_supplement_source import build_reference, source_contract_snapshot

    data, grid = unit_data(unit), deepcopy(MESHES["E"])
    before = deepcopy(data)
    built = build_reference(data, grid, LAYOUT)
    actual = source_contract_snapshot(built)
    expected = source_state_layout(data, grid, uniform=False)
    cv = unit == "S04"

    assert actual["pybamm_version"] == data["source_versions"]["pybamm"] == "25.4.2"
    assert actual["uniform"] is False
    for key in ("grid", "authored_grid", "requested_grid", "actual_grid"):
        assert actual[key] == {"z": 64, "r": 48}
    assert actual["layout"] == LAYOUT
    assert actual["control"] == ("physical-CV-with-finite-current-limits" if cv else "CC")
    assert len(actual["states"]) == len(expected) == 11 + cv
    book = Checks()
    for state, canonical in zip(actual["states"], expected, strict=True):
        for key in ("name", "kind", "slices", "domains"):
            assert state[key] == canonical[key]
        for key in ("scale", "reference"):
            book.identity(state["name"] + ":" + key, [state[key]], [canonical[key]])
    assert all(check["passed"] for check in book.items), book.items

    sizes = {
        kind: sum(
            stop - start
            for state in expected
            if state["kind"] == kind
            for start, stop in state["slices"]
        )
        for kind in ("differential", "algebraic")
    }
    assert actual["rhs_size"] == sizes["differential"] == 2 * 64 * 48 + 5 == 6149
    assert actual["algebraic_size"] == sizes["algebraic"] == 4 * 64 + cv
    total = actual["rhs_size"] + actual["algebraic_size"]
    packed_indices = [
        index
        for state in actual["states"]
        for start, stop in state["slices"]
        for index in range(start, stop)
    ]
    assert packed_indices == list(range(total))
    assert built["model"].concatenated_initial_conditions.size == total
    assert built["model"].mass_matrix.entries.shape == (total, total)
    assert len(actual["z_nodes_m"]) == 64
    for electrode in ("negative", "positive"):
        assert built["mesh"][electrode + " particle"].npts == 48
    assert actual["original_events"]
    assert set(actual["original_events"]) <= {event["name"] for event in actual["events"]}
    assert actual["physical_work_no_feedback"]["no_feedback"]

    spec = native_spec(data, grid, LAYOUT)
    state = spec.initial_state()
    assert spec.collector_cells == spec.geometry.cells == 64
    assert spec.negative.cells == spec.positive.cells == 48
    assert spec.negative_tab == "right"
    assert len(spec.particle_bins) == 128
    assert len(state.amounts_mol) == len(spec.inventory_keys) == 6144
    assert set(state.amounts_mol) == set(spec.inventory_keys)
    for electrode, z, particle, keys, _identity in spec.particle_bins:
        assert keys == tuple(f"{electrode}:z:{z}:Li:{r}" for r in range(48))
        assert particle.cells == 48
        assert all(state.amounts_mol[key] > 0 for key in keys)
    assert set(state.coordinates) == {"discharged_charge_C", "delivered_work_J"}
    assert state.time_s == 0.0
    assert data == before == case_data(unit)
    assert data["samples_per_segment"] == 31
