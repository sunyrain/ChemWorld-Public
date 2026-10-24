"""Preparation contracts; none count toward the fixed 12-unit scientific block."""

from copy import deepcopy

import numpy as np
import pytest
from battery_partial_plating_cases import REFUSALS, case_data
from battery_partial_plating_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 12))
def test_actual_source_build_has_unscaled_dead_rate_and_real_zero_sei(index):
    simulation = build_reference(case_data(index), 8)
    simulation.build()
    model = simulation.built_model
    y = model.concatenated_initial_conditions.evaluate()
    rate = model.variables["qualification:deactivation_rate_mol_s"].evaluate(t=0.0, y=y)
    data = case_data(index)
    assert np.asarray(rate).item() == pytest.approx(
        data["gamma_s_inv"] * data["initial_plated_mol"], abs=1e-16
    )
    assert (
        np.asarray(
            model.variables["X-averaged negative total SEI thickness [m]"].evaluate(t=0.0, y=y)
        ).item()
        == 0.0
    )
    metal = [
        v
        for v in model.rhs
        if "dead lithium concentration" in v.name or "lithium plating concentration" in v.name
    ]
    assert len(metal) == 2
    assert all(model.rhs[v].shape == (1, 1) for v in metal)
    assert all(float(v.scale.evaluate()) == 1000.0 for v in metal)


def short_data(index):
    data = deepcopy(case_data(index))
    data.update(waveform=((0.2, *data["waveform"][0][1:]),), samples_per_segment=3)
    return data


@pytest.mark.parametrize("index", (1, 2, 3, 4, 8, 9))
def test_source_short_observables(index):
    from battery_partial_plating_assess import integrity
    from battery_partial_plating_traces import history_metrics

    data = short_data(index)
    trace = reference_time(data, 8)
    assert all(integrity(trace, data).values())
    segment = trace["segments"][0]
    rows = np.asarray(segment["rows"])
    assert segment["deactivation_rate_mol_s"] == pytest.approx(
        data["gamma_s_inv"] * rows[:, 5], abs=1e-14
    )
    assert segment["sei_thickness_m"] == [0.0, 0.0, 0.0]
    assert history_metrics(trace, data)["independent_kinetic_limit_ratio"] < 1.0


@pytest.mark.parametrize("index", (1, 2, 3, 4, 8, 9))
def test_native_short_observables(index):
    from battery_partial_plating_assess import integrity
    from battery_partial_plating_traces import history_metrics, native_time

    data = short_data(index)
    trace = native_time(data, 8)
    assert all(integrity(trace, data).values())
    segment = trace["segments"][0]
    rows = np.asarray(segment["rows"])
    assert segment["deactivation_rate_mol_s"] == pytest.approx(
        data["gamma_s_inv"] * rows[:, 5], abs=1e-14
    )
    metrics = history_metrics(trace, data)
    assert metrics["total_Li_error_mol"] < 1e-10
    assert metrics["independent_kinetic_limit_ratio"] < 1.0
    assert "rhs_boundary_extensions_mol" in segment["solver"]
    assert "roundoff_corrections_mol" in segment


@pytest.mark.parametrize("name", REFUSALS)
def test_each_named_refusal(name):
    from battery_partial_plating_traces import refusal

    result = refusal(case_data(12), name)
    assert result["rejected"], result
    assert result["input_state_unchanged"]
    if name == "MPM":
        assert "SPM" in result["exception"] or "partial" in result["exception"].lower()


def test_denominators_and_old_gamma_zero_route():
    from run_battery_partial_plating_block import tasks

    assert sum(len(tasks(i, case_data(i))) for i in range(1, 13)) == 62
    assert sum(len(tasks(i, case_data(i))) for i in range(1, 12)) == 48
    data = short_data(2)
    trace = reference_time(data, 8, old_reversible=True)
    assert trace["segments"][0]["deactivation_rate_mol_s"] == [0.0, 0.0, 0.0]


def test_archive_is_complete_lossless_json(tmp_path):
    import gzip
    import json

    from run_battery_partial_plating_block import archive_verified, compact_write

    raw, gz = tmp_path / "raw.json", tmp_path / "raw.json.gz"
    value = {"complete": [1.0, 2.0], "failure": {"error": "kept"}, "unicode": "锂"}
    compact_write(raw, value)
    info = archive_verified(raw, gz)
    assert info["decompressed_byte_identity"]
    assert gzip.decompress(gz.read_bytes()) == raw.read_bytes()
    assert json.loads(gzip.decompress(gz.read_bytes())) == value


def test_missing_route_and_truncated_history_cannot_qualify():
    from battery_partial_plating_assess import assess, integrity

    data = short_data(3)
    trace = reference_time(data, 8)
    assert all(integrity(trace, data).values())
    for kind in ("missing", "truncated", "no_t0", "wrong_inventory", "nan", "bad_status"):
        altered = deepcopy(trace)
        if kind == "missing":
            altered["segments"] = []
        else:
            segment = altered["segments"][0]
            if kind == "truncated":
                segment["end_time_s"] -= 0.1
            elif kind == "no_t0":
                segment["times_s"][0] = 0.01
            elif kind == "wrong_inventory":
                segment["rows"][0][12] += 0.001
            elif kind == "nan":
                segment["raw_profiles_theta"][0][0][0] = float("nan")
            else:
                segment["outcome"] = "arbitrary"
        assert not all(integrity(altered, data).values()), kind
    item = {"inputs": data, "routes": {}}
    assess(item)
    assert not item["passed"]


@pytest.mark.parametrize("implementation", ("native", "reference"))
def test_required_fields_and_exact_inventory_keys_fail_closed(implementation):
    from battery_partial_plating_assess import integrity
    from battery_partial_plating_traces import native_time

    data = short_data(3)
    trace = (native_time if implementation == "native" else reference_time)(data, 8)
    assert all(integrity(trace, data).values())
    keys = {
        "termination",
        "rows",
        "times_s",
        "raw_profiles_theta",
        "raw_r_over_R",
        "deactivation_rate_mol_s",
    }
    keys |= (
        {"outcome", "raw_amounts_mol", "solver", "roundoff_corrections_mol"}
        if (implementation == "native")
        else {"sei_thickness_m"}
    )
    for key in keys:
        altered = deepcopy(trace)
        altered["segments"][0].pop(key)
        assert not all(integrity(altered, data).values()), key
    for key in ("implementation", "radial_cells", "response_order"):
        altered = deepcopy(trace)
        altered.pop(key)
        assert not all(integrity(altered, data).values()), key
    if implementation == "native":
        for kind in ("missing", "extra", "renamed", "empty"):
            altered = deepcopy(trace)
            row = altered["segments"][0]["raw_amounts_mol"][1]
            if kind == "empty":
                row.clear()
            elif kind == "extra":
                row["invented:Li"] = 0.0
            elif kind == "renamed":
                row["invented:Li"] = row.pop("negative:Li:0")
            else:
                row.pop("negative:Li:0")
            assert not all(integrity(altered, data).values()), kind


@pytest.mark.parametrize("implementation", ("native", "reference"))
def test_stage_continuity_catches_radial_redistribution_and_charge_work_jump(implementation):
    from battery_partial_plating_assess import integrity
    from battery_partial_plating_traces import history_metrics, native_time

    data = short_data(6)
    data["waveform"] = ((0.2, -0.6, "A"), (0.2, 0.0, "A"))
    trace = (native_time if implementation == "native" else reference_time)(data, 8)
    assert all(integrity(trace, data).values())
    metrics = history_metrics(trace, data)
    for key in ("stage_radial_inventory_jump_mol", "stage_raw_inventory_jump_mol"):
        assert metrics[key] <= 1e-9
    assert metrics["stage_charge_jump_C"] <= 1e-5
    assert metrics["stage_work_jump_J"] <= 1e-4
    for kind in ("radial", "charge", "work", "raw"):
        if kind == "raw" and implementation != "native":
            continue
        altered = deepcopy(trace)
        segment = altered["segments"][1]
        if kind == "radial":
            segment["raw_profiles_theta"][0][0][0] += 0.001
            segment["raw_profiles_theta"][0][0][1] -= 0.001 / 7
            key, threshold = "stage_radial_inventory_jump_mol", 1e-9
        elif kind == "raw":
            segment["raw_amounts_mol"][0]["negative:Li:0"] += 1e-7
            segment["raw_amounts_mol"][0]["negative:Li:1"] -= 1e-7
            key, threshold = "stage_raw_inventory_jump_mol", 1e-9
        else:
            segment["rows"][0][2 if kind == "charge" else 3] += 0.01
            key = "stage_charge_jump_C" if kind == "charge" else "stage_work_jump_J"
            threshold = 1e-5 if kind == "charge" else 1e-4
        changed = history_metrics(altered, data)
        assert changed[key] > threshold, kind
        assert changed["stage_inventory_jump_mol"] <= 1e-9
