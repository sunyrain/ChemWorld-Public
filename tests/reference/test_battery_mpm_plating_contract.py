"""Preparation-only actual-source and adapter contracts; no formal routes."""

from copy import deepcopy

import numpy as np
import pytest
from battery_mpm_plating_cases import case_data
from battery_mpm_plating_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 12))
def test_actual_source_build(index):
    simulation = build_reference(case_data(index), 4, 8)
    simulation.build()
    assert simulation.built_model.len_alg >= 2
    for name in ("negative", "positive"):
        assert f"qualification:{name}:area_m2" in simulation.built_model.variables


@pytest.mark.parametrize("index", (1, 3, 4, 6, 8, 9))
def test_actual_source_short_raw_mapping(index):
    data = deepcopy(case_data(index))
    data["waveform"] = ((0.2, *data["waveform"][0][1:]),)
    data["samples_per_segment"] = 3
    trace = reference_time(data, 4, 8)
    assert trace["segments"][0]["end_time_s"] == pytest.approx(0.2)
    for name in ("negative", "positive"):
        m = trace["mapping"][name]
        local = trace["segments"][0]["local"][name]
        assert sum(m["area_weights"]) == pytest.approx(1.0)
        assert sum(m["volume_weights"]) == pytest.approx(1.0)
        e = next(e for e in data["electrodes"] if e["name"] == name)
        expected = 3 * e["active_volume_m3"] * np.sum(np.array(m["volume_weights"]) / m["radii_m"])
        assert m["active_area_m2"] == pytest.approx(expected, rel=1e-12)
        assert np.asarray(local["profiles_theta"]).shape == (3, 1 if data["single_bin"] else 4, 8)
        assert np.asarray(local["comparison_profiles_theta"]).shape == (3, 5, 5)


@pytest.mark.parametrize("index", (1, 2, 3, 4, 6, 8, 9))
def test_native_short_raw_mapping(index):
    from battery_mpm_plating_traces import history_metrics, local_metrics, native_time

    data = deepcopy(case_data(index))
    data["waveform"] = ((0.2, *data["waveform"][0][1:]),)
    data["samples_per_segment"] = 3
    trace = native_time(data, 4, 8)
    assert trace["segments"][0]["end_time_s"] == pytest.approx(0.2)
    assert trace["segments"][0]["outcome"] == "success"
    assert "roundoff_corrections_mol" in trace["segments"][0]
    assert history_metrics(trace, data)["total_Li_error_mol"] < 1e-10
    assert local_metrics(trace, data)["local_BV_limit_ratio"] < 1
    for name in ("negative", "positive"):
        local = trace["segments"][0]["local"][name]
        assert np.asarray(local["profiles_theta"]).shape == (3, 1 if data["single_bin"] else 4, 8)
    if index == 1:
        old = native_time(data, 4, 8, single_spm=True)
        assert np.asarray(trace["segments"][0]["rows"]) == pytest.approx(
            np.asarray(old["segments"][0]["rows"]), abs=1e-7
        )


def test_integrity_negative_controls_and_complete_denominator():
    from battery_mpm_plating_assess import integrity
    from run_battery_mpm_plating_block import tasks

    data = deepcopy(case_data(3))
    data["waveform"] = ((0.2, -0.6, "A"),)
    data["samples_per_segment"] = 3
    trace = reference_time(data, 4, 8)
    assert all(integrity(trace, data).values())
    assert not all(integrity({"segments": []}, data).values())
    for mutation in ("end", "initial", "no_t0", "bad_profile", "bad_outcome"):
        altered = deepcopy(trace)
        s = altered["segments"][0]
        if mutation == "end":
            s["end_time_s"] -= 0.1
        elif mutation == "initial":
            s["rows"][0][12] += 0.01
        elif mutation == "no_t0":
            s["times_s"][0] += 0.01
        elif mutation == "bad_profile":
            s["local"]["negative"]["profiles_theta"][0][0][0] = float("nan")
        else:
            s["outcome"] = "unrecognized-status"
        assert not all(integrity(altered, data).values()), mutation
    assert sum(len(tasks(i, case_data(i))) for i in range(1, 13)) == 62
    assert sum(len(tasks(i, case_data(i))) for i in range(1, 12)) == 50


def test_source_independent_particle_and_side_laws():
    from battery_mpm_plating_traces import history_metrics, local_metrics

    data = deepcopy(case_data(3))
    data["waveform"] = ((0.2, -0.6, "A"),)
    data["samples_per_segment"] = 3
    trace = reference_time(data, 4, 8)
    assert history_metrics(trace, data)["independent_kinetic_limit_ratio"] < 1
    m = local_metrics(trace, data)
    assert m["local_BV_limit_ratio"] < 1
    assert m["area_identity_relative_error"] < 1e-12


def test_lossless_full_archive(tmp_path):
    import gzip
    import hashlib

    from run_battery_mpm_plating_block import archive_result

    raw = tmp_path / "raw.json"
    archive = tmp_path / "raw.json.gz"
    content = b'{"failed":true,"profiles":[1.2345678901234567,-0.0],"exception":"retained"}\n'
    raw.write_bytes(content)
    result = archive_result(raw, archive)
    assert gzip.decompress(archive.read_bytes()) == content
    assert result["archive_identical"] is True
    assert result["uncompressed_sha256"] == hashlib.sha256(content).hexdigest()
    with pytest.raises(FileExistsError):
        archive_result(raw, archive)
