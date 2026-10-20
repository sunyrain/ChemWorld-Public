"""NATIVE-EXT-25 finite development matrix, real sampling and persistent equipment."""

import json
import subprocess
import sys
from dataclasses import replace
from math import exp, log

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from chemworld.physchem.chromatography_methods import (
    ChromatographyProgram,
    evaluate_chromatography_method,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.chromatography import (
    ANALYSIS_S,
    COST,
    SAMPLE_L,
    memory,
    model,
    programme,
    settings,
)
from chemworld.runtime.session_snapshot import SessionSnapshot
from chemworld.world.chromatography_instruments import fit_public_trace, public_trace_basis
from chemworld.world.chromatography_world import chromatography_world_spec


def session(instrument="hplc", **kwargs):
    s = ChemWorldRuntime(composition=chromatography_world_spec(instrument, **kwargs))
    s.reset(seed=12)
    return s


def compare_snapshots(left, right):
    a, b = [json.loads(json.dumps(s.snapshot().payload)) for s in (left, right)]
    # JSON restore may change summation order in balance diagnostics (about
    # 1e-17 here). All physical state, noise, readings and bookkeeping stay exact.
    ca, cb = [v["operation_record"].pop("constitution_checks") for v in (a, b)]
    assert a == b
    assert len(ca) == len(cb)
    for lcheck, rcheck in zip(
        sorted(ca, key=lambda x: x["name"]), sorted(cb, key=lambda x: x["name"]), strict=True
    ):
        assert (lcheck["name"], lcheck["passed"], lcheck["tolerance"]) == (
            rcheck["name"],
            rcheck["passed"],
            rcheck["tolerance"],
        )
        if isinstance(lcheck["value"], (int, float)):
            assert lcheck["value"] == pytest.approx(rcheck["value"], rel=0, abs=1e-14)
        else:
            assert lcheck == rcheck


def execute(s, **action):
    result = s.execute(action)
    assert result.info["transaction_status"] == "committed", result.info
    return result


def load_bottle(s, bottle):
    """Empty the real receiver, then transfer a finite aliquot, preserving the device."""
    execute(s, operation="select_vessel", vessel="reactor")
    if s._state.volume_L:
        execute(
            s,
            operation="route_material",
            connection="reactor-to-waste",
            transfer_fraction=1,
            mixing="allow",
        )
    execute(s, operation="select_vessel", vessel=bottle)
    execute(
        s,
        operation="route_material",
        connection=bottle + "-to-reactor",
        transfer_fraction=0.1,
        mixing="allow",
    )
    execute(s, operation="select_vessel", vessel="reactor")


@pytest.mark.parametrize("instrument", ["hplc", "gc"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_three_analytes_three_programmes(instrument, index):
    s = session(instrument)
    name = "programmed_" + instrument
    execute(s, operation="configure_chromatography", instrument=name, program_index=index)
    p = programme(s._state, name)
    params, analytes = model(s._state, name)
    original = ChromatographyProgram.from_dict(params["programs"][0])
    for a in analytes:
        rt = p.retention_time_s(a)
        expected = (
            evaluate_chromatography_method(
                a,
                dead_time_min=p.dead_time_min,
                temperature_K=298.15,
                mobile_phase_fraction=0.2 if instrument == "hplc" else None,
                detector_concentration=0,
            ).retention_time_min
            * 60
        )
        if index == 0:
            assert rt == pytest.approx(expected, rel=1e-10)
        else:
            assert rt < original.retention_time_s(a)
        assert ChromatographyProgram.from_dict(p.to_dict()).retention_time_s(a) == rt


def test_lss_linear_gradient_has_independent_analytic_integral_and_delay():
    s = session()
    _, analytes = model(s._state, "programmed_hplc")
    p = ChromatographyProgram(
        ScalarControlProgram(((0, 298.15),), "K"),
        ScalarControlProgram(((0, 0.2), (60, 0.8)), "dimensionless"),
    )
    for a in analytes:
        alpha = log(10) * a.hplc_log10_k_mobile_phase_slope * 0.6 / 60
        expected = (
            log((1 + a.reference_retention_factor) * exp(alpha * 3) - a.reference_retention_factor)
            / alpha
        )
        assert p.retention_time_s(a) == pytest.approx(expected, rel=1e-9)
        assert replace(p, delay_s=5).retention_time_s(a) > p.retention_time_s(a)
    # No elution is a physical outcome, not a manufactured peak at the endpoint.
    assert (
        replace(
            p, duration_s=1, mobile_phase=ScalarControlProgram(((0, 0.2),), "dimensionless")
        ).retention_time_s(analytes[-1])
        is None
    )


def test_gc_integral_grid_refinement():
    s = session("gc")
    execute(s, operation="configure_chromatography", instrument="programmed_gc", program_index=1)
    p = programme(s._state, "programmed_gc")
    _, analytes = model(s._state, "programmed_gc")
    errors = []
    for intervals in (20, 40, 80):
        relative = []
        for a in analytes:
            exact = p.retention_time_s(a)
            t = np.linspace(0, 30, intervals + 1)
            distance = cumulative_trapezoid(
                [1 / (60 * p.evaluate(a, x).retention_time_min) for x in t],
                t,
                initial=0,
            )
            numerical = np.interp(1, distance, t)
            relative.append(abs(numerical - exact) / exact)
        errors.append(max(relative))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.001


@pytest.mark.parametrize("instrument", ["hplc", "gc"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_two_standard_sample_blank_sequences_conserve_and_keep_memory(instrument, index):
    s = session(instrument, noise_sd=0)
    name = "programmed_" + instrument
    execute(s, operation="configure_chromatography", instrument=name, program_index=index)
    for previous_count, bottle in enumerate(("standard", "sample", "blank") * 2, 1):
        before_memory = memory(s._state, name).values.copy()
        load_bottle(s, bottle)
        assert memory(s._state, name).values == before_memory
        state = s._state
        injected = {k: n * SAMPLE_L / state.volume_L for k, n in state.species_amounts.items()}
        old_ledger = state.ledger
        result = execute(s, operation="measure", instrument=name)
        assert result.info["measurement_status"] == "valid"
        post = s._state
        config = settings(post, name)
        assert config["use_count"] == previous_count
        assert config["elapsed_time_s"] == previous_count * ANALYSIS_S
        assert post.ledger.sample_consumed_L - old_ledger.sample_consumed_L == pytest.approx(
            SAMPLE_L
        )
        assert post.ledger.cost - old_ledger.cost == pytest.approx(COST)
        assert post.ledger.time_s == old_ledger.time_s  # declared offline clock
        outlet = config["chromatography_acquisition"]["effluent_mol"]
        for key, amount in memory(post, name).values.items():
            assert amount + outlet[key] == pytest.approx(
                before_memory[key] + injected[key],
                rel=1e-8,
                abs=1e-14,
            )
            if bottle == "blank":
                assert injected[key] == 0
                assert outlet[key] > 0
                assert amount == pytest.approx(before_memory[key] * exp(-6), abs=1e-14)
        # The actual transaction, not only this independent calculation, checks balances.
        checks = result.info["constitution_checks"]
        assert any(c["name"] == "material_conservation" for c in checks)
        assert all(c["passed"] for c in checks)
        obs = result.observation
        for i in range(1, 4):
            assert obs.channel(f"peak_{i}_concentration").quality == "valid"
        assert obs.channel("chromatogram").available_at.time_s == previous_count * ANALYSIS_S


def test_observe_snapshot_and_restore_preserve_noise_drift_and_residue():
    s = session()
    load_bottle(s, "sample")
    execute(s, operation="measure", instrument="programmed_hplc")
    before = s.snapshot()
    for _ in range(3):
        s.observe()
        s.observation_view()
    assert s.snapshot() == before
    restored = ChemWorldRuntime.from_snapshot(before)
    left = execute(s, operation="measure", instrument="programmed_hplc")
    right = execute(restored, operation="measure", instrument="programmed_hplc")
    assert right == left
    compare_snapshots(restored, s)
    s.reset(options={"scope": "material", "vessel_ids": ["reactor"]})
    assert settings(s._state, "programmed_hplc")["use_count"] == 2
    assert (
        memory(s._state, "programmed_hplc").values
        == memory(restored._state, "programmed_hplc").values
    )
    s.reset(seed=12)
    assert all(x == 0 for x in memory(s._state, "programmed_hplc").values.values())
    assert settings(s._state, "programmed_hplc").get("use_count", 0) == 0


def test_saturation_invalidates_estimates_not_real_sampling():
    s = session(model_overrides={"saturation_AU": 0.006})
    load_bottle(s, "sample")
    volume = s._state.volume_L
    result = execute(s, operation="measure", instrument="programmed_hplc")
    assert result.observation.channel("chromatogram").quality == "saturated"
    assert result.observation.channel("peak_1_concentration").quality == "invalid"
    assert s._state.volume_L == pytest.approx(volume - SAMPLE_L)
    assert s._state.ledger.sample_consumed_L == SAMPLE_L
    assert sum(memory(s._state, "programmed_hplc").values.values()) > 0


def test_public_calibration_uses_standard_signal_not_specimen_truth():
    s = session(noise_sd=0, model_overrides={"initial_gain": 1.3})
    name = "programmed_hplc"
    for bottle, concentration in (
        ("standard_low", 0.005),
        ("standard", 0.01),
        ("standard_high", 0.02),
    ):
        load_bottle(s, bottle)
        execute(s, operation="measure", instrument=name)
        execute(
            s,
            operation="calibrate_chromatography",
            instrument=name,
            standard_1_M=concentration,
            standard_2_M=2 * concentration,
            standard_3_M=3 * concentration,
        )
    calibration = settings(s._state, name)["calibration"]
    assert all(1.28 < c["slope"] < 1.32 for c in calibration)
    # The fit input is only the observable trace and declared calibration, no state.
    obs = s.observe()
    axis = np.array(obs.channel("chromatogram").axis.values)
    signal = np.array(obs.channel("chromatogram").value)
    profile = obs.raw_signal["processing_calibration"]
    basis = public_trace_basis(axis, profile)
    # The third standard was acquired before its resulting calibration was fit.
    # Its public area is reproducible solely from the returned observation.
    areas, _, _ = fit_public_trace(axis, signal, basis, profile["resolved"], profile["response"])
    assert areas == pytest.approx([obs.channel(f"peak_{i}_area").value for i in range(1, 4)])
    _, estimates, _ = fit_public_trace(axis, signal, basis, profile["resolved"], calibration)
    assert estimates == pytest.approx((0.02, 0.04, 0.06), rel=0.002)
    result = s.execute(
        {
            "operation": "calibrate_chromatography",
            "instrument": name,
            "standard_1_M": 0.02,
            "standard_2_M": 0.04,
            "standard_3_M": 0.06,
        }
    )
    assert result.info["transaction_status"] != "committed"
    assert settings(s._state, name)["calibration"] == calibration


def test_illegal_program_does_not_reset_physical_memory():
    s = session()
    load_bottle(s, "sample")
    execute(s, operation="measure", instrument="programmed_hplc")
    before = memory(s._state, "programmed_hplc").values.copy()
    r = s.execute(
        {
            "operation": "configure_chromatography",
            "instrument": "programmed_hplc",
            "program_index": 15,
        }
    )
    assert r.info["transaction_status"] != "committed"
    assert memory(s._state, "programmed_hplc").values == before


@pytest.mark.parametrize("instrument", ["hplc", "gc"])
def test_new_process_snapshot_preserves_actual_continuation(tmp_path, instrument):
    s = session(instrument)
    name = "programmed_" + instrument
    load_bottle(s, "sample")
    execute(s, operation="measure", instrument=name)
    input_path, output_path = tmp_path / "before.json", tmp_path / "after.json"
    s.snapshot().save(input_path)
    code = """
import sys
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.session_snapshot import SessionSnapshot
s = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(sys.argv[1]))
r = s.execute({'operation': 'measure', 'instrument': sys.argv[3]})
assert r.info['transaction_status'] == 'committed'
s.snapshot().save(sys.argv[2])
"""
    subprocess.run(
        [sys.executable, "-c", code, str(input_path), str(output_path), name],
        check=True,
        timeout=45,
        capture_output=True,
        text=True,
    )
    execute(s, operation="measure", instrument=name)
    restored = ChemWorldRuntime.from_snapshot(SessionSnapshot.load(output_path))
    compare_snapshots(s, restored)


@pytest.mark.parametrize("mode", ["overlap", "uneluted"])
def test_unidentifiable_peaks_do_not_fabricate_concentrations(mode):
    spec = chromatography_world_spec()
    config = (
        spec.physics.initial_states["reactor"]
        .equipment.equipment["instrument:programmed_hplc"]
        .settings["chromatography"]
    )
    analytes = [dict(a) for a in config["analytes"]]
    for a in analytes:
        a["reference_retention_factor"] = 4 if mode == "overlap" else 100
    s = session(model_overrides={"analytes": analytes})
    load_bottle(s, "sample")
    result = execute(s, operation="measure", instrument="programmed_hplc")
    assert all(
        result.observation.channel(f"peak_{i}_concentration").quality == "invalid"
        for i in range(1, 4)
    )
    assert s._state.ledger.sample_consumed_L == SAMPLE_L
    if mode == "uneluted":
        assert all(
            value == 0
            for value in settings(s._state, "programmed_hplc")["chromatography_acquisition"][
                "effluent_mol"
            ].values()
        )


def test_failed_readout_preserves_already_executed_residue_and_cost(monkeypatch):
    from chemworld.runtime.observation_services import ChemWorldObservationKernel

    s = session()
    load_bottle(s, "sample")
    original = ChemWorldObservationKernel.observe

    def broken(self, state, action, rng):
        if action.get("instrument") == "programmed_hplc":
            raise ValueError("injected readout-domain fixture, not physical damage")
        return original(self, state, action, rng)

    monkeypatch.setattr(ChemWorldObservationKernel, "observe", broken)
    volume = s._state.volume_L
    r = s.execute({"operation": "measure", "instrument": "programmed_hplc"})
    assert r.info["measurement_status"] == "invalid"
    assert s._state.volume_L == pytest.approx(volume - SAMPLE_L)
    assert s._state.ledger.sample_consumed_L == SAMPLE_L
    assert sum(memory(s._state, "programmed_hplc").values.values()) > 0


def test_insufficient_sample_cannot_advance_instrument_history():
    s = session()
    execute(
        s,
        operation="route_material",
        connection="reactor-to-waste",
        transfer_fraction=0.95,
        mixing="allow",
    )
    before = settings(s._state, "programmed_hplc").copy()
    r = s.execute({"operation": "measure", "instrument": "programmed_hplc"})
    assert r.info["transaction_status"] != "committed"
    assert settings(s._state, "programmed_hplc") == before
    assert s._state.ledger.sample_consumed_L == 0


@pytest.mark.parametrize("instrument", ["hplc", "gc"])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_full_jsonl_history_replays(tmp_path, instrument, index):
    from chemworld.examples.programmed_chromatography import run_case

    report = run_case(instrument, index, tmp_path / "trajectory.jsonl")
    assert report["passed"]
    assert len(report["measurements"]) == 6
    assert report["maximum_balance_residual_mol"] < 1e-14
