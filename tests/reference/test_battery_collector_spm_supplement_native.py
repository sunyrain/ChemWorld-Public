"""Functional checks for supplement adapters, not the frozen scientific block."""

import json

import battery_collector_spm_supplement_native as adapter
import numpy as np
import pytest
from battery_collector_spm_cases import case_data

from chemworld.physchem.battery_collector_spm import default_collector_spm_spec
from chemworld.physchem.battery_impedance import battery_impedance
from chemworld.physchem.battery_models import BatterySPMSpec, ParticleSpec


def short_data():
    data = case_data("CC02")
    data["id"] = "functional-adapter-only"
    data["samples_per_segment"] = 3
    data["waveform"] = ((0.02, 0.03, "A"), (0.01, 0.0, "A"), (0.02, -0.02, "A"))
    return data


def test_collector_eis_explicitly_refuses_without_mutating_the_actual_state():
    spec = default_collector_spm_spec(cells=3, collector_cells=2)
    state = spec.initial_state()
    before = (dict(state.amounts_mol), dict(state.coordinates), dict(state.algebraic_values))
    with pytest.raises(ValueError, match=r"collector.*impedance.*unsupported"):
        battery_impedance(spec, [0.1], state=state)
    assert (
        dict(state.amounts_mol),
        dict(state.coordinates),
        dict(state.algebraic_values),
    ) == before


def test_existing_uniform_spm_impedance_still_uses_the_original_supported_path():
    spec = BatterySPMSpec(negative=ParticleSpec(cells=3), positive=ParticleSpec(cells=3))
    response = battery_impedance(spec, [0.1, 1.0])
    assert np.isfinite(response.impedance_ohm).all()
    assert np.asarray(response.impedance_ohm).shape == (2,)


@pytest.mark.parametrize("tab", ["left", "right"])
def test_native_history_keeps_true_port_state_faces_and_local_conservation(tab):
    data = short_data()
    trace = adapter.native_time(
        data, {"z": 2, "r": 3}, {"negative_tab": tab, "positive_tab": "right"}
    )
    assert trace["native_spec_class"] == "BatteryCollectorSPMSpec"
    assert trace["actual_grid"] == trace["requested_grid"] == trace["grid"]
    assert len(trace["segments"]) == 3
    assert trace["initial_state"]["time_s"] == 0
    assert trace["final_state"]["time_s"] == 0.05
    for index, segment in enumerate(trace["segments"]):
        assert segment["values"] == segment["physical_values"]
        assert segment["physical_tab_voltage_V"] == [row[0] for row in segment["values"]]
        assert segment["physical_work_J"] == [row[3] for row in segment["values"]]
        assert segment["controlled_port"] == "current"
        assert segment["times_s"][0] == segment["initial_state"]["time_s"]
        assert segment["raw_states"][-1] == segment["final_state"]
        assert segment["outcome"] == "success"
        assert segment["native_solver_policy"]["rtol"] == 1e-8
        assert segment["native_event_values_at_endpoint"]
        if index:
            assert segment["initial_state"] == trace["segments"][index - 1]["final_state"]
        for name in ("negative", "positive"):
            profile = segment["raw_profiles"][name]
            assert np.asarray(profile["theta"]).shape == (3, 2, 3)
            assert np.asarray(profile["open_circuit_potential_V"]).shape == (3, 2)
            np.testing.assert_allclose(
                profile["open_circuit_potential_V"],
                data[name]["ocv"][0] + data[name]["ocv"][1] * np.array(profile["surface_fraction"]),
                rtol=0,
                atol=0,
            )
        for values, field in zip(segment["values"], segment["native_fields"], strict=True):
            q = np.array(field["through_current_A"])
            np.testing.assert_allclose(
                np.diff(field["negative_collector_face_current_A"]), -q, rtol=0, atol=1e-16
            )
            np.testing.assert_allclose(
                np.diff(field["positive_collector_face_current_A"]), q, rtol=0, atol=1e-16
            )
            assert sum(q) == pytest.approx(values[1], abs=1e-15)
            np.testing.assert_allclose(field["positive_reaction_current_A"], -q, rtol=0, atol=0)
        for key in ("negative_faradaic_residual_C", "positive_faradaic_residual_C"):
            assert np.max(np.abs(segment["sample_conservation"][key])) < 1e-9
        assert np.max(np.abs(segment["sample_conservation"]["local_lithium_change_mol"])) < 1e-14
    json.dumps(trace, allow_nan=False)


def test_uniform_baseline_advances_actual_old_spm_and_has_only_its_two_particles(monkeypatch):
    def forbidden_collector(*_args, **_kwargs):
        raise AssertionError("uniform baseline must not construct or advance a collector")

    monkeypatch.setattr(adapter, "native_spec", forbidden_collector)
    monkeypatch.setattr(adapter, "collector_spm_system", forbidden_collector)
    trace = adapter.native_uniform_time(short_data(), {"z": 8, "r": 3})
    assert trace["native_spec_class"] == "BatterySPMSpec"
    assert trace["implementation"] == "native-uniform"
    assert trace["grid"] == {"z": 1, "r": 3}
    assert trace["actual_grid"] == trace["grid"]
    assert trace["requested_grid"] == {"z": 8, "r": 3}
    assert "collector_cells" not in trace["native_spec"]
    assert len(trace["initial_state"]["amounts_mol"]) == 6
    assert all(":z:" not in key for key in trace["initial_state"]["amounts_mol"])
    initial = trace["initial_state"]["amounts_mol"]
    final = trace["final_state"]["amounts_mol"]
    assert final != initial
    assert trace["final_state"]["coordinates"]["discharged_charge_C"] == pytest.approx(
        0.0002, abs=1e-12
    )
    for index, segment in enumerate(trace["segments"]):
        assert segment["values"] == segment["physical_values"]
        assert segment["source_boundary_voltage_V"] is None
        assert segment["native_event_values_at_endpoint"]
        if index:
            assert segment["initial_state"] == trace["segments"][index - 1]["final_state"]
        for name in ("negative", "positive"):
            assert np.asarray(segment["raw_profiles"][name]["theta"]).shape == (3, 1, 3)
            assert np.asarray(segment["raw_profiles"][name]["average_fraction"]).shape == (3, 1)
    json.dumps(trace, allow_nan=False)


def test_uniform_baseline_does_not_silently_average_a_nonuniform_input():
    data = short_data()
    data["initial_z_fractions"]["negative"] = [0.3, 0.4]
    with pytest.raises(ValueError, match="explicitly uniform"):
        adapter.native_uniform_time(data, {"r": 3, "z": 2})


@pytest.mark.parametrize("uniform", [False, True])
def test_original_event_endpoint_and_unexecuted_later_stages_remain_explicit(uniform):
    data = short_data()
    data["minimum_voltage_V"] = 4.0
    trace = (
        adapter.native_uniform_time(data, {"r": 3})
        if uniform
        else adapter.native_time(
            data,
            {"r": 3, "z": 2},
            {"negative_tab": "left", "positive_tab": "right"},
        )
    )
    assert len(trace["segments"]) == 1
    segment = trace["segments"][0]
    assert segment["outcome"] == "stopped"
    assert segment["termination"] and "minimum-voltage" in segment["termination"]
    assert segment["times_s"] == [0.0]
    assert segment["native_event_values_at_endpoint"][segment["termination"]] < 0
    assert segment["initial_state"] == segment["final_state"]


def test_native_finite_cv_reports_actual_selected_control_mode():
    data = short_data()
    data["waveform"] = ((0.01, 3.72, "V"),)
    trace = adapter.native_time(
        data, {"z": 2, "r": 3}, {"negative_tab": "right", "positive_tab": "right"}
    )
    segment = trace["segments"][0]
    assert segment["control"]["unit"] == "V"
    assert segment["controlled_port"] == "physical-tab"
    assert set(segment["control_modes"]) == {"constant_voltage"}
    np.testing.assert_allclose(np.array(segment["values"])[:, 0], 3.72, rtol=0, atol=1e-8)


@pytest.mark.parametrize("name", adapter.REFUSAL_NAMES)
def test_eight_refusals_call_real_api_and_preserve_full_before_after(name):
    result = adapter.run_refusal(name, short_data(), {"z": 2, "r": 3})
    assert result["rejected"]
    assert result["exception"]["type"] == result["expected_exception"] == "ValueError"
    assert result["exception"]["message"] == result["expected_reason"]
    assert result["before"] == result["after"]
    assert result["state_unchanged"]
    assert result["stage"] == ("analysis" if name in ("EIS", "SOH") else "codec")
    assert result["entrypoint"].startswith("chemworld.physchem.")
    json.dumps(result, allow_nan=False)


def test_refusal_adapter_cannot_claim_a_rejection_when_the_tested_api_accepts(monkeypatch):
    monkeypatch.setattr(adapter, "battery_spec_from_dict", lambda payload: payload)
    result = adapter.run_refusal("invalid-tab", short_data(), {"z": 2, "r": 3})
    assert not result["rejected"] and result["exception"] is None
    assert result["before"] == result["after"]


@pytest.mark.parametrize("uniform", [False, True])
def test_capture_failure_retains_all_computed_states_receipts_and_prior_stages(
    monkeypatch, uniform
):
    original = adapter._profiles
    calls = 0

    def fail_second_capture(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("synthetic profile extraction failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(adapter, "_profiles", fail_second_capture)
    with pytest.raises(RuntimeError, match="synthetic profile extraction failure") as caught:
        if uniform:
            adapter.native_uniform_time(short_data(), {"z": 2, "r": 3})
        else:
            adapter.native_time(
                short_data(),
                {"z": 2, "r": 3},
                {"negative_tab": "left", "positive_tab": "right"},
            )
    context = caught.value.partial_history
    assert context["stage"] == "capture"
    assert context["active_segment"]["index"] == 1
    assert len(context["completed_segments"]) == 1
    pending = context["pending_solution"]
    assert pending["outcome"] == "success"
    assert len(pending["raw_states"]) == 3
    assert pending["raw_states"][0] == pending["initial_state"]
    assert pending["raw_states"][-1] == pending["final_state"]
    assert pending["initial_state"] == context["completed_segments"][0]["final_state"]
    assert pending["final_state"] == context["last_integrated_state"]
    assert pending["final_state"]["time_s"] == 0.03
    assert pending["actual_duration_s"] == pytest.approx(0.01, abs=1e-17)
    assert pending["solver"] and pending["conservation_residuals"]
    assert "exchange_energy_J" in pending["final_state"]
    assert "actual_exchanges" in pending and "roundoff_corrections_mol" in pending
    assert context["exception_type"] == "RuntimeError"
    json.dumps(context, allow_nan=False)
