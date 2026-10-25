"""Small adapter preparation contracts, outside the fixed 14-unit matrix."""

import pytest
from battery_dfn_composite_cases import case_data, route_plan
from battery_dfn_composite_pybamm import build_metadata, build_reference

SMALL_GRID = {"x": (3, 3, 3), "r": 6}


def test_prospective_denominator():
    assert sum(len(route_plan(i)) for i in range(1, 15)) == 72
    assert sum(len(route_plan(i)) for i in range(1, 14)) == 60
    assert len(route_plan(14)) == 12


@pytest.mark.parametrize(
    "index,limit", ((1, None), (10, None), (5, "merged-dfn"), (6, "composite-spm"))
)
def test_actual_modular_source_build(index, limit):
    metadata = build_metadata(build_reference(case_data(index), SMALL_GRID, limit=limit))
    assert metadata["state_count"] > 0
    assert "Discharge energy [W.h]" in metadata["rhs"]
    assert "Throughput energy [W.h]" in metadata["rhs"]
    print("SOURCE_BUILD", index, limit, metadata)


@pytest.mark.parametrize(
    "index,limit", ((1, None), (10, None), (5, "merged-dfn"), (6, "composite-spm"))
)
def test_actual_source_short_original_fields(index, limit):
    from battery_dfn_composite_pybamm import source_time

    data = case_data(index)
    _duration, target, unit = data["waveform"][0]
    data["waveform"] = ((0.2, target, unit),)
    data["samples_per_segment"] = 3
    trace = source_time(data, SMALL_GRID, limit=limit)
    s = trace["segments"][0]
    assert s["times_s"][0] == 0.0
    assert s["times_s"][-1] == pytest.approx(0.2)
    assert len(s["source_state_vectors"]) == len(s["times_s"])
    assert len(s["raw_profiles"]) == (2 if limit == "merged-dfn" else 3)
    assert s["termination"] == "final time"


@pytest.mark.parametrize(
    "index,limit", ((1, None), (10, None), (5, "merged-dfn"), (6, "composite-spm"))
)
def test_native_short_original_fields(index, limit):
    from battery_dfn_composite_traces import native_time

    data = case_data(index)
    _, target, unit = data["waveform"][0]
    data["waveform"] = ((0.2, target, unit),)
    data["samples_per_segment"] = 3
    trace = native_time(data, SMALL_GRID, limit=limit)
    s = trace["segments"][0]
    assert s["times_s"][0] == 0.0
    assert s["times_s"][-1] == pytest.approx(0.2)
    assert s["outcome"] == "success"
    assert len(s["raw_profiles"]) == (2 if limit == "merged-dfn" else 3)


@pytest.fixture(
    scope="module", params=[(1, None), (10, None), (5, "merged-dfn"), (6, "composite-spm")]
)
def short_pair(request):
    from battery_dfn_composite_pybamm import source_time
    from battery_dfn_composite_traces import native_time

    index, limit = request.param
    data = case_data(index)
    _, target, unit = data["waveform"][0]
    data.update(waveform=((0.2, target, unit),), samples_per_segment=3)
    return (
        data,
        limit,
        native_time(data, SMALL_GRID, limit=limit),
        source_time(data, SMALL_GRID, limit=limit),
    )


def test_complete_native_source_raw_identities(short_pair):
    from battery_dfn_composite_assess import metrics, structural

    data, limit, native, reference = short_pair
    for kind, trace in [("native", native), ("reference", reference)]:
        assert structural(trace, data, SMALL_GRID, kind, limit)
        assert metrics(trace, data)


@pytest.mark.parametrize(
    "mutation",
    ("units", "input", "theta", "amount", "Q", "solver", "missing-state", "missing-stage", "NaN"),
)
def test_malformed_native_is_fail_closed(short_pair, mutation):
    from copy import deepcopy

    from battery_dfn_composite_assess import structural

    data, limit, trace, _ = short_pair
    trace = deepcopy(trace)
    s = trace["segments"][0]
    p = s["raw_profiles"]["host_a"]
    if mutation == "units":
        del trace["response_units"]
    elif mutation == "input":
        trace["inputs"]["conductivity_S_m"] *= 2
    elif mutation == "theta":
        p["theta"][0][0][0] += 0.01
    elif mutation == "amount":
        s["raw_amounts_mol"][0][p["inventory_keys"][0][0]] += 1e-6
    elif mutation == "Q":
        s["raw_coordinates"][0]["discharged_charge_C"] += 1
    elif mutation == "solver":
        s["solver"] = {}
    elif mutation == "missing-state":
        del s["raw_amounts_mol"]
    elif mutation == "missing-stage":
        trace["segments"] = []
    elif mutation == "NaN":
        p["q"][0][0] = float("nan")
    with pytest.raises((ValueError, KeyError, TypeError)):
        structural(trace, data, SMALL_GRID, "native", limit)


@pytest.mark.parametrize("mutation", ("scale", "raw", "short-history", "phase", "grid"))
def test_malformed_source_is_fail_closed(short_pair, mutation):
    from copy import deepcopy

    from battery_dfn_composite_assess import structural

    data, limit, _, trace = short_pair
    trace = deepcopy(trace)
    s = trace["segments"][0]
    if mutation == "scale":
        trace["source_build"]["state_layout"][0]["scale"] *= 2
    elif mutation == "raw":
        s["source_state_vectors"][0][4] += 0.01
    elif mutation == "short-history":
        s["times_s"][-1] = 0.1
        s["end_time_s"] = 0.1
    elif mutation == "phase":
        del s["raw_profiles"]["host_a"]
    elif mutation == "grid":
        trace["grid"]["r"] += 1
    with pytest.raises((ValueError, KeyError, TypeError)):
        structural(trace, data, SMALL_GRID, "reference", limit)


def test_fixed_interval_currents_sum_to_phase_total(short_pair):
    import numpy as np
    from battery_dfn_composite_assess import sampled

    data, _, native, reference = short_pair
    for trace in (native, reference):
        s = trace["segments"][0]
        obs = sampled(s, data)
        for name in s["raw_profiles"]:
            np.testing.assert_allclose(
                obs[name + ":q"].sum(axis=1), obs[name + ":qtotal"], atol=1e-14
            )


@pytest.mark.parametrize("name", tuple(r[0] for r in route_plan(14)))
def test_declared_refusal_captured(name):
    from battery_dfn_composite_traces import refusal

    result = refusal(case_data(14), name, SMALL_GRID)
    assert result["rejected"], result
    assert result["error"]["message"]
    assert result["state_before"] == result["state_after"]


def test_short_cc_rest_continuity_and_full_raw_stage_accounting():
    from battery_dfn_composite_assess import LIMITS, metrics, structural
    from battery_dfn_composite_pybamm import source_time
    from battery_dfn_composite_traces import native_time

    data = case_data(8)
    data.update(waveform=((0.1, 0.3, "A"), (0.1, 0.0, "A")), samples_per_segment=3)
    for kind, function in (("native", native_time), ("reference", source_time)):
        trace = function(data, SMALL_GRID)
        assert structural(trace, data, SMALL_GRID, kind, None)
        m = metrics(trace, data)
        for name in ("stage_particle_jump_mol", "stage_salt_jump_mol", "stage_Q_C", "stage_W_J"):
            assert m[name] <= LIMITS[name]


def test_response_comparison_keeps_original_grid_and_qualified_fields(short_pair):
    from battery_dfn_composite_assess import compare

    data, _, native, reference = short_pair
    comparison = compare(native, reference, data)
    assert comparison
    assert all("max_ratio" in row for row in comparison)
    names = {row["output"] for row in comparison}
    assert {"V", "I", "Q", "W", "host_a:raw:q", "host_a:raw:theta", "host_a:amount"} <= names


def test_lossless_failed_trace_and_partial_history_archive(tmp_path):
    import gzip
    import json

    from battery_dfn_composite_assess import finite
    from run_battery_dfn_composite_block import archive_verified, capture, compact_write

    def failure():
        error = ValueError("deliberate preparation failure")
        error.partial_history = {"completed_segments": [{"raw": float("nan")}], "stage": "extract"}
        raise error

    result = capture(failure)
    raw = tmp_path / "failure.json"
    archive = tmp_path / "failure.json.gz"
    compact_write(raw, result)
    stored = json.loads(raw.read_text())
    assert stored["exception_type"] == "ValueError"
    assert stored["partial_history"]["completed_segments"][0]["raw"] == {"nonfinite": "nan"}
    assert not finite(stored)
    assert archive_verified(raw, archive)["decompressed_byte_identity"]
    assert gzip.decompress(archive.read_bytes()) == raw.read_bytes()
    with pytest.raises(FileExistsError):
        compact_write(raw, result)


def test_exact_all_tasks_and_solver_version():
    import pybammsolvers
    from run_battery_dfn_composite_block import tasks

    assert isinstance(pybammsolvers.__version__, str)
    assert sum(len(list(tasks(i, case_data(i)))) for i in range(1, 15)) == 72


def test_unit_inputs_bound_to_original_authored_row():
    from battery_dfn_composite_assess import assess

    data = case_data(1)
    data["conductivity_S_m"] *= 2
    item = {"id": data["id"], "inputs": data, "routes": {}}
    assess(item)
    assert item["checks"]["canonical_fixed_inputs"] is False
    assert item["passed"] is False


def test_both_old_families_have_original_coarse_fine_comparisons():
    from battery_dfn_composite_assess import comparison_pairs

    for index in (5, 6):
        data = case_data(index)
        pairs = comparison_pairs(data)
        assert len(pairs) == 12
        for kind in ("native", "reference"):
            assert (kind + "_coarse_" + data["limit"], kind + "_fine_" + data["limit"]) in pairs


@pytest.mark.parametrize(
    "mutation", ("persistent-algebraic", "native-event", "source-event", "source-RHS-shape")
)
def test_canonical_raw_record_obligations(short_pair, mutation):
    from copy import deepcopy

    from battery_dfn_composite_assess import structural

    data, limit, native, source = short_pair
    kind = "native" if mutation in ("persistent-algebraic", "native-event") else "reference"
    trace = deepcopy(native if kind == "native" else source)
    s = trace["segments"][0]
    if mutation == "persistent-algebraic":
        s["raw_algebraic_values"][0] = {"fake_current": 0.0}
    elif mutation == "native-event":
        s["native_event_values_at_endpoint"].pop(next(iter(s["native_event_values_at_endpoint"])))
    elif mutation == "source-event":
        s["source_event_values_at_endpoint"].pop(next(iter(s["source_event_values_at_endpoint"])))
    else:
        s["source_initial_algebraic_residual"] += s["source_initial_rhs"]
        s["source_initial_rhs"] = []
    with pytest.raises((ValueError, KeyError)):
        structural(trace, data, SMALL_GRID, kind, limit)


@pytest.mark.parametrize("name", tuple(r[0] for r in route_plan(14)))
def test_refusal_stage_and_reason_must_match_declared_case(name):
    from copy import deepcopy

    from battery_dfn_composite_assess import declared_rejection
    from battery_dfn_composite_traces import refusal

    result = refusal(case_data(14), name, SMALL_GRID)
    assert declared_rejection(result, name)
    changed = deepcopy(result)
    changed["error"]["message"] = "unrelated missing attribute"
    assert not declared_rejection(changed, name)
    changed = deepcopy(result)
    changed["stage"] = "construction" if result["stage"] == "consumer" else "consumer"
    assert not declared_rejection(changed, name)
