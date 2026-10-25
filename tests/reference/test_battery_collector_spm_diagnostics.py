"""Synthetic-only tests: never load an actual collector qualification archive."""

import gzip
import json
from copy import deepcopy

import numpy as np
import pytest
from battery_collector_spm_cases import GRIDS, LAYOUTS, F, case_data
from battery_collector_spm_diagnostics import _compare, common_interval_sums, diagnose, main


def _trace(kind, grid_name="coarse", case="CC01"):
    data = case_data(case)
    grid = GRIDS[grid_name]
    nz, nr, nt = grid["z"], grid["r"], data["samples_per_segment"]
    segments, start, charge, work = [], 0.0, 0.0, 0.0
    for duration, target, mode in data["waveform"]:
        times = np.linspace(start, start + duration, nt)
        current = target if mode == "A" else 0.2
        voltage = (3.8 if mode == "A" else target) + (0.02 if kind == "source" else 0.0)
        q = charge + current * (times - start)
        w = work + voltage * current * (times - start)
        values = np.column_stack((np.full(nt, voltage), np.full(nt, current), q, w))
        profiles, raw_amounts = {}, [{} for _ in times]
        for name, sign in (("negative", -1), ("positive", 1)):
            particle = data[name]
            initial = particle["active_volume_m3"] * particle["maximum_concentration_mol_m3"]
            initial *= particle["initial_fraction"]
            amounts = np.broadcast_to(
                ((initial + sign * q / F) / (nz * nr))[:, None, None],
                (nt, nz, nr),
            ).copy()
            keys = [[f"{name}:z:{z}:Li:{r}" for r in range(nr)] for z in range(nz)]
            for t in range(nt):
                for z in range(nz):
                    raw_amounts[t].update(zip(keys[z], amounts[t, z].tolist(), strict=True))
            average = amounts.sum(axis=2) / (
                particle["active_volume_m3"] * particle["maximum_concentration_mol_m3"] / nz
            )
            profiles[name] = {
                "shell_amounts_mol": (
                    amounts if kind == "native" else amounts.transpose(0, 2, 1)
                ).tolist(),
                "inventory_keys": keys,
                "reaction_current_A": np.full((nt, nz), -sign * current / nz).tolist(),
                "average_fraction": average.tolist(),
                "surface_fraction": average.tolist(),
                "z_nodes_m": ((np.arange(nz) + 0.5) * data["height_m"] / nz).tolist(),
                "z_edges_m": np.linspace(0, data["height_m"], nz + 1).tolist(),
                "r_edges_m": np.linspace(0, particle["radius_m"], nr + 1).tolist(),
                "collector_potential_V": np.full(
                    (nt, nz), 0.0 if name == "negative" else 3.8
                ).tolist(),
                "local_voltage_V": np.full((nt, nz), 3.8).tolist(),
            }
        segment = {
            "times_s": times.tolist(),
            "end_time_s": float(times[-1]),
            "outcome": "success",
            "termination": None if kind == "native" else "final time",
            "values": values.tolist(),
            "raw_profiles": profiles,
            "source_boundary_voltage_V": np.full(
                nt, 3.82 if mode == "A" else target + 0.02
            ).tolist(),
        }
        if kind == "native":
            segment.update(
                raw_amounts_mol=raw_amounts,
                raw_coordinates=[
                    {"discharged_charge_C": row[2], "delivered_work_J": row[3]}
                    for row in values.tolist()
                ],
                raw_algebraic_values=[{} for _ in times],
            )
        else:
            fields = {
                "Negative current collector potential [V]": profiles["negative"][
                    "collector_potential_V"
                ],
                "Positive current collector potential [V]": profiles["positive"][
                    "collector_potential_V"
                ],
                "Local voltage [V]": profiles["negative"]["local_voltage_V"],
                "source_boundary_voltage_V": segment["source_boundary_voltage_V"],
                "through_current_A": profiles["negative"]["reaction_current_A"],
            }
            for name in (
                "Current collector current density [A.m-2]",
                "Negative electrode current density [A.m-2]",
                "Positive electrode current density [A.m-2]",
            ):
                fields[name] = np.zeros((nt, nz)).tolist()
            segment.update(
                source_fields=fields,
                source_state_vectors=np.zeros((nt, 6)).tolist(),
                source_initial_rhs=[0.0] * 4,
                source_initial_algebraic_residual=[0.0] * 2,
            )
        segments.append(segment)
        start, charge, work = float(times[-1]), float(q[-1]), float(w[-1])
    return {
        "implementation": kind,
        "unit": case,
        "inputs": data,
        "grid": grid,
        "layout": LAYOUTS["opposite"],
        "faraday_C_mol": F,
        "segments": segments,
    }


def _report(case="CC01", grids=("coarse",)):
    routes = {}
    for grid_name in grids:
        routes[f"history_opposite_{grid_name}"] = {
            "implementation": "history",
            "unit": case,
            "grid": GRIDS[grid_name],
            "layout": LAYOUTS["opposite"],
            "native": _trace("native", grid_name, case),
            "source": _trace("source", grid_name, case),
        }
    return {
        "attempt": 2,
        "passed": 7,
        "unit_denominator": 12,
        "route_denominator": 48,
        "units": [
            {
                "id": case,
                "inputs": case_data(case),
                "passed": True,
                "failures": ["synthetic retained historical failure"],
                "routes": routes,
            }
        ],
    }


def _unit(output, name="CC01"):
    return next(unit for unit in output["units"] if unit["id"] == name)


def _route(output, name="CC01"):
    return _unit(output, name)["routes"]["history_opposite_coarse"]


def test_finite_inventory_current_and_voltage_diagnostics_do_not_reclassify_or_mutate():
    raw = _report()
    before = deepcopy(raw)
    result = diagnose(raw)
    assert raw == before
    assert result["original_summary"]["passed"] == 7
    assert result["original_unit_outcomes"][0]["failures"] == before["units"][0]["failures"]
    route = _route(result)
    assert route["issues"] == []
    for kind in ("native", "source"):
        assert route[kind]["coverage"] == "complete_sample_records"
        metrics = route[kind]["segments"][0]["metrics"]
        assert metrics["total_Li_drift_mol"]["max_abs"] < 1e-17
        assert metrics["negative_Faraday_residual_C"]["max_abs"] < 1e-12
        assert metrics["positive_Faraday_residual_C"]["max_abs"] < 1e-12
        assert metrics["negative_sum_q_minus_signed_I_A"]["max_abs"] == 0
    diff = route["same_grid_comparison"][0]["differences_left_minus_right"]
    assert diff["voltage_definition_gap_V"]["first"] == pytest.approx(-0.02)
    assert diff["source_boundary_diagnostic_V"]["max_abs"] == 0
    assert "not an error" in result["voltage_conventions"]["comparison"]
    assert "passed" not in route  # No new physical qualification decision.


def test_cv_target_work_difference_and_segment_continuity_are_descriptive():
    result = diagnose(_report("CC08"))
    route = _route(result, "CC08")
    native = route["native"]["segments"][0]["metrics"]
    source = route["source"]["segments"][0]["metrics"]
    assert native["CV_delta_W_minus_target_delta_Q_J"]["max_abs"] < 1e-12
    assert source["CV_delta_W_minus_target_delta_Q_J"]["last"] == pytest.approx(0.48)
    assert "current limiting" in source["CV_note"]
    result = diagnose(_report("CC07"))
    continuity = _route(result, "CC07")["native"]["segment_continuity"]
    assert len(continuity) == 2
    assert all(
        row["charge_jump_C"] == row["work_jump_J"] == row["time_gap_s"] == 0 for row in continuity
    )


def test_missing_expected_routes_and_units_remain_in_denominator():
    result = diagnose(_report())
    assert result["reported_routes"] == 48
    assert len(result["units"]) == 12
    assert result["coverage_counts"]["missing_routes"] == 47
    assert result["coverage_counts"]["native"]["missing"] == 39
    assert _unit(result)["coarse_fine"]["same_end_source"][0]["available"] is False


def test_failure_retains_completed_native_and_source_initialization_only():
    raw = _report()
    route = raw["units"][0]["routes"]["history_opposite_coarse"]
    raw["units"][0]["routes"]["history_opposite_coarse"] = {
        "exception": "initial voltage cutoff",
        "exception_type": "SolverError",
        "partial_history": {
            "stage": "source",
            "completed_traces": {"native": route["native"]},
            "failed_implementation_context": {
                "completed_segments": [],
                "consistent_initialization": {
                    "time_s": 0.0,
                    "state_vector": [1.0, 2.0],
                    "events": {"low voltage": [-0.3]},
                    "read_errors": {"unknown": "not present"},
                },
            },
        },
    }
    result = _route(diagnose(raw))
    assert result["exception"]["exception"] == "initial voltage cutoff"
    assert result["native"]["coverage"] == "complete_sample_records"
    assert result["source"]["coverage"] == "initialization_only"
    assert result["source"]["initialization_only_observations"]["events"] == {"low voltage": [-0.3]}
    assert result["same_grid_comparison"][0]["available"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), {"__nonfinite__": "nan"}])
def test_nonfinite_missing_and_malformed_data_are_reported_without_losing_other_routes(bad):
    raw = _report(grids=("coarse", "fine"))
    source = raw["units"][0]["routes"]["history_opposite_coarse"]["source"]["segments"][0]
    source["values"][3][0] = bad
    del source["raw_profiles"]["negative"]["shell_amounts_mol"]
    source["raw_profiles"]["positive"] = None
    result = diagnose(raw)
    json.dumps(result, allow_nan=False)
    assert result["reported_routes"] == 48
    reasons = str(_route(result)["issues"])
    assert "nonfinite" in reasons and "missing" in reasons
    fine = _unit(result)["routes"]["history_opposite_fine"]
    assert fine["source"]["coverage"] == "complete_sample_records"


def test_spatial_comparison_aggregates_whole_cells_and_does_not_interpolate_q():
    left_edges, right_edges = [0.0, 0.5, 1.0], [0.0, 0.25, 0.5, 0.75, 1.0]
    left = np.array([[3.0, 7.0], [-3.0, -7.0]])
    right = np.array([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
    edges, a, b = common_interval_sums(left, left_edges, right, right_edges)
    assert edges.tolist() == left_edges
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="endpoints"):
        common_interval_sums(left, left_edges, right, [0.0, 0.2, 0.4, 0.6, 0.8])
    output = diagnose(_report(grids=("coarse", "fine")))
    comparison = _unit(output)["coarse_fine"]["opposite_native"][0]
    assert comparison["spatial_aggregation"]["negative"]["available"] is True
    assert comparison["differences_left_minus_right"]["negative_reaction_current_A"]["max_abs"] == 0
    assert "do not establish convergence order" in output["mesh_note"]


def test_comparison_uses_shared_times_only_and_does_not_extrapolate_stopped_history():
    raw = _report()
    source = raw["units"][0]["routes"]["history_opposite_coarse"]["source"]["segments"][0]
    nt = len(source["times_s"])

    def truncate(value):
        if isinstance(value, dict):
            return {key: truncate(child) for key, child in value.items()}
        if isinstance(value, list) and len(value) == nt:
            return value[:3]
        return value

    source = truncate(source)
    source.update(outcome="stopped", termination="voltage", end_time_s=8.0)
    raw["units"][0]["routes"]["history_opposite_coarse"]["source"]["segments"][0] = source
    result = _route(diagnose(raw))
    assert result["source"]["coverage"] == "partial"
    assert result["same_grid_comparison"][0]["time_range_s"] == [0.0, 8.0]
    assert result["same_grid_comparison"][0]["exact_common_samples"] == 3


def test_duplicate_unexpected_and_refusal_records_are_not_silently_dropped():
    raw = _report()
    raw["units"].append(deepcopy(raw["units"][0]))
    raw["units"].append({"id": "new-extra", "exception": "unexpected unit"})
    raw["units"][0]["routes"]["unexpected"] = {"exception": "extra failure"}
    raw["units"].append(
        {
            "id": "CC11",
            "routes": {
                "refusal_opposite_coarse": {
                    "implementation": "refusal",
                    "rejected": False,
                    "refusals": [{"exception": "actual rejection detail"}],
                }
            },
        }
    )
    result = diagnose(raw)
    assert len(result["duplicate_units"]) == len(result["unexpected_units"]) == 1
    assert _unit(result)["unexpected_routes"]["unexpected"]["exception"] == "extra failure"
    refusal = _unit(result, "CC11")["routes"]["refusal_opposite_coarse"]
    assert refusal["coverage"] == "refusal_not_a_history"
    assert refusal["record"]["rejected"] is False


def test_exact_only_alignment_discloses_t0_only_and_full_domain_only_comparisons():
    raw = _report()
    source = raw["units"][0]["routes"]["history_opposite_coarse"]["source"]["segments"][0]
    source["times_s"][1:] = np.nextafter(source["times_s"][1:], np.inf).tolist()
    for profile in source["raw_profiles"].values():
        profile["z_edges_m"][1:-1] = np.nextafter(profile["z_edges_m"][1:-1], np.inf).tolist()
    comparison = _route(diagnose(raw))["same_grid_comparison"][0]
    assert comparison["exact_common_samples"] == 1
    assert comparison["all_samples_exactly_matched"] is False
    assert comparison["left_samples"] == comparison["right_samples"] == 31
    assert comparison["left_matched_fraction"] == pytest.approx(1 / 31)
    for aggregation in comparison["spatial_aggregation"].values():
        assert aggregation["full_domain_only"] is True
        assert aggregation["interval_count"] == 1


def test_invalid_comparison_is_reported_without_aborting_later_comparisons():
    malformed = {"times": np.array([0.0, 1.0])}  # Required values/profiles intentionally absent.
    output = _compare([malformed, None], [malformed, None], "synthetic")
    assert len(output) == 2
    assert output[0]["available"] is False and "KeyError" in output[0]["diagnostic_exception"]
    assert output[1]["available"] is False and "missing" in output[1]["reason"]


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_reads_json_or_gzip_and_never_overwrites_input_or_output(tmp_path, compressed):
    raw = json.dumps(_report()).encode()
    source = tmp_path / ("synthetic.json.gz" if compressed else "synthetic.json")
    source.write_bytes(gzip.compress(raw) if compressed else raw)
    original = source.read_bytes()
    output = tmp_path / "diagnostics.json"
    assert main([str(source), str(output)]) == 0
    assert source.read_bytes() == original
    result = json.loads(output.read_text())
    assert result["original_summary"]["passed"] == 7
    assert result["input_file"]["path"] == str(source)
    with pytest.raises(FileExistsError):
        main([str(source), str(output)])
    with pytest.raises(FileExistsError):
        main([str(source), str(source)])
