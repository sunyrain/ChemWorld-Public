"""Source/API adapter contracts, kept separate from the fixed NT qualification."""

from copy import deepcopy

import numpy as np
import pytest
from battery_newman_tobias_cases import GRIDS, NAMES, SOURCE_F, F, case_data
from battery_newman_tobias_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 12))
def test_actual_source_build_has_only_two_mean_particle_states(index):
    sim = build_reference(case_data(index), GRIDS["reference"][0])
    sim.build()
    particle_keys = [str(k) for k in sim.model.rhs if "particle concentration" in str(k)]
    assert len(particle_keys) == 2
    assert all(k.startswith("Average ") for k in particle_keys)
    assert not any("electrolyte concentration" in str(k).lower() for k in sim.model.rhs)
    assert sim.model.options["particle"] == "uniform profile"
    assert sim.model.options["surface form"] == "false"


def test_parameter_mapping_is_net_ocv_effective_conductivity_and_explicit_temperature():
    import pybamm

    data = case_data(7)
    sim = build_reference(data, GRIDS["reference"][0])
    params = sim.parameter_values
    assert params["Ambient temperature [K]"] == data["temperature_K"]
    assert float(pybamm.constants.F.value) == SOURCE_F
    assert abs(SOURCE_F / F - 1) < 4e-11
    for index, e in enumerate(data["electrodes"]):
        region = 0 if index == 0 else 2
        name = e["name"].capitalize()
        assert params[name + " electrode conductivity [S.m-1]"] * (
            1 - data["porosities"][region]
        ) ** data["solid_bruggeman"][index] == pytest.approx(
            data["solid_conductivity_S_m"][index], rel=1e-14
        )
        phase = getattr(sim.model.param, e["name"][0]).prim
        for theta in (0.05, 0.3, 0.7, 0.95):
            result = params.evaluate(
                phase.U(pybamm.Scalar(theta), pybamm.Scalar(data["temperature_K"]))
            )
            assert result == pytest.approx(
                sum(a * theta**k for k, a in enumerate(e["polynomial"])), abs=1e-12
            )


def test_small_source_probe_retains_original_t0_and_full_node_face_arrays():
    data = case_data(2)
    data["waveform"] = ((0.2, 0.6, "A"),)
    data["samples_per_segment"] = 3
    grid = GRIDS["reference"][0]
    trace = reference_time(data, grid)
    segment = trace["segments"][0]
    assert segment["times_s"][0] == 0.0
    assert segment["end_time_s"] == 0.2
    assert np.all(np.isfinite(segment["values"]))
    for region, count in zip(("negative", "separator", "positive"), grid, strict=True):
        raw = segment["raw_profiles"][region]
        assert len(raw["nodes_m"]) == count
        assert len(raw["edges_m"]) == count + 1
        assert np.shape(raw["electrolyte_potential_V"])[1] == count
        assert np.shape(raw["electrolyte_face_current_A"])[1] == count + 1
        if region != "separator":
            assert np.shape(raw["reaction_current_A"])[1] == count
            expected = 0.6 if region == "negative" else -0.6
            assert np.sum(raw["reaction_current_A"], axis=1) == pytest.approx(expected, abs=2e-6)


def test_fixed_matrix_denominator_and_no_radial_grid_input():
    assert len(NAMES) == 12
    assert len(GRIDS["native"]) == len(GRIDS["reference"]) == 2
    assert all(len(grid) == 3 for values in GRIDS.values() for grid in values)
    assert 11 * 2 * 2 + 8 == 52


def test_native_short_adapter_and_source_field_comparison():
    from battery_newman_tobias_assess import compare, invariants, passes
    from battery_newman_tobias_traces import native_time

    data = case_data(2)
    data["waveform"] = ((0.2, 0.6, "A"),)
    data["samples_per_segment"] = 3
    native = native_time(data, GRIDS["native"][0])
    source = reference_time(data, GRIDS["reference"][0])
    assert all(passes(v) for v in invariants(data, native, "native").values())
    assert all(passes(v) for v in invariants(data, source, "reference").values())
    assert all(passes(v) for v in compare(native, source).values())


@pytest.mark.parametrize("raw", ["battery:minimum-voltage", "event: Minimum voltage [V]"])
def test_event_metadata_names_are_canonical_without_changing_raw_trace(raw):
    from battery_newman_tobias_assess import event_kind

    assert event_kind(raw) == "minimum-voltage"


@pytest.fixture
def structural_history():
    """Authored synthetic layout only: no qualification response is synthesized."""
    data = case_data(2)
    data["waveform"] = ((0.2, 0.6, "A"), (0.3, 0.0, "A"))
    data["samples_per_segment"] = 3
    initial = [0.0, 0.0, 0.0, 0.0]
    initial += [e["initial_fraction"] for e in data["electrodes"]]
    initial += [
        e["active_volume_m3"] * e["maximum_concentration_mol_m3"] * e["initial_fraction"]
        for e in data["electrodes"]
    ]
    segments = []
    for start, end in ((0.0, 0.2), (0.2, 0.5)):
        raw, sampled = {}, {}
        for name in ("negative", "separator", "positive"):
            fields = {
                "electrolyte_potential_V": [[0.0]] * 3,
                "electrolyte_face_current_A": [[0.0, 0.0]] * 3,
            }
            raw[name] = {"nodes_m": [0.5], "edges_m": [0.0, 1.0], **deepcopy(fields)}
            sampled[name] = deepcopy(fields)
            if name != "separator":
                extra = {"solid_potential_V": [[0.0]] * 3, "solid_face_current_A": [[0.0, 0.0]] * 3}
                raw[name].update(
                    extra,
                    reaction_current_A=[[0.0]] * 3,
                    interfacial_current_density_A_m2=[[0.0]] * 3,
                )
                sampled[name].update(extra, reaction_density_A=[[0.0]] * 3)
        segments.append(
            {
                "times_s": [start, (start + end) / 2, end],
                "values": [initial.copy() for _ in range(3)],
                "end_time_s": end,
                "termination": None,
                "raw_profiles": raw,
                "sampled_profiles": sampled,
            }
        )
    return data, {"segments": segments}


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_segment",
        "shortened_without_exception",
        "missing_t0",
        "nonfinite_raw",
        "wrong_initial_Li",
    ],
)
def test_history_contract_rejects_incomplete_or_corrupted_trace(structural_history, mutation):
    from battery_newman_tobias_assess import history_contract, passes

    data, trace = structural_history
    assert all(passes(v) for v in history_contract(data, trace).values())
    if mutation == "missing_segment":
        trace["segments"].pop()
    elif mutation == "shortened_without_exception":
        trace["segments"][-1]["times_s"][-1] = 0.4
        trace["segments"][-1]["end_time_s"] = 0.4
    elif mutation == "missing_t0":
        trace["segments"][0]["times_s"][0] = 0.01
    elif mutation == "nonfinite_raw":
        trace["segments"][0]["raw_profiles"]["negative"]["solid_potential_V"] = [[float("nan")]] * 3
    elif mutation == "wrong_initial_Li":
        trace["segments"][0]["values"][0][6] *= 0.9
    assert "exception" not in trace
    assert not all(passes(v) for v in history_contract(data, trace).values())


def test_missing_route_or_segment_fails_without_losing_available_checks(structural_history):
    from battery_newman_tobias_assess import assess, compare

    data, complete = structural_history
    missing = deepcopy(complete)
    missing["segments"].pop()
    compared = compare(complete, missing)
    assert compared["segment_count"] is False
    assert "segment0:voltage_V" in compared
    item = {"inputs": data, "routes": {}}
    assess(item)
    assert item["passed"] is False
    assert item["checks"]["native_coarse"]["route_completed"] is False


@pytest.mark.parametrize("implementation", ["native", "reference"])
def test_adapter_failure_retains_explicit_stage_context(monkeypatch, implementation):
    import battery_newman_tobias_pybamm as source
    import battery_newman_tobias_traces as native
    from run_battery_newman_tobias_block import capture

    def fail(*args):
        raise ValueError("intentional adapter build failure")

    if implementation == "native":
        monkeypatch.setattr(native, "native_spec", fail)
        function = native.native_time
    else:
        monkeypatch.setattr(source, "build_reference", fail)
        function = source.reference_time
    result = capture(lambda: function(case_data(2), GRIDS[implementation][0]))
    assert result["exception_type"] == "ValueError"
    assert result["partial_history"] == {
        "completed_segments": [],
        "stage": "build",
        "current_time_s": 0.0,
    }


def test_numpy_false_is_not_a_zero_error_ratio():
    from battery_newman_tobias_assess import passes

    assert not passes(np.bool_(False))
    assert passes(np.bool_(True))
    assert passes(0.0)


@pytest.mark.parametrize(
    "outcome,termination,expected",
    [
        ("success", None, True),
        ("stopped", "battery:minimum-voltage", True),
        ("event", "battery:minimum-voltage", False),
        ("stopped", None, False),
        ("success", "battery:minimum-voltage", False),
        ("failure", None, False),
        ("arbitrary", "battery:minimum-voltage", False),
        ("stopped", "arbitrary", False),
        (None, None, False),
    ],
)
def test_native_outcome_matches_actual_coupled_system_contract(outcome, termination, expected):
    from battery_newman_tobias_assess import native_outcome_valid

    assert native_outcome_valid(outcome, termination) is expected
