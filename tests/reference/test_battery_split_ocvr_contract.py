"""Short adapter contracts, explicitly outside the fixed 17/66 data block."""

from copy import deepcopy

import numpy as np
import pytest

pytest.importorskip("pybamm", reason="optional independent reference package")
from battery_split_ocvr_assess import assess, compare, history_contract, invariants, passes
from battery_split_ocvr_cases import NAMES, analysis_data, case_data
from battery_split_ocvr_oracle import cv_at, equilibrium_polynomial, response, soh
from battery_split_ocvr_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 13))
def test_actual_source_builds_three_original_states(index):
    sim = build_reference(case_data(index), "coarse")
    sim.build()
    assert sorted(str(k) for k in sim.model.rhs) == [
        "Discharge capacity [A.h]",
        "Negative particle stoichiometry",
        "Positive particle stoichiometry",
    ]
    assert not sim.model.algebraic
    assert len(sim.model.events) == 6
    assert sim.solver.method == "DOP853"
    assert not any(
        "energy" in str(k).lower() or "work" in str(k).lower() for k in sim.model.variables
    )


def test_direct_ocp_and_capacity_mapping_has_no_barrier():
    import pybamm

    data = case_data(5)
    sim = build_reference(data, "coarse")
    p = sim.parameter_values
    assert p["Negative electrode capacity [A.h]"] * 3600 == 1800
    assert p["Positive electrode capacity [A.h]"] * 3600 == 7200
    for q in (0.0, 100.0):
        expected = response(data, q, 0.0)
        for electrode, theta, ocv in (
            ("Negative", expected[3], expected[5]),
            ("Positive", expected[4], expected[6]),
        ):
            symbol = pybamm.FunctionParameter(
                electrode + " electrode OCP [V]",
                {electrode + " particle stoichiometry": pybamm.Scalar(theta)},
            )
            assert p.evaluate(symbol) == pytest.approx(ocv, abs=1e-14)


def test_fixed_denominator_and_independent_cv_targets():
    from run_battery_split_ocvr_block import unit_tasks

    assert len(NAMES) == 17
    assert sum(len(unit_tasks(i, case_data(i))) for i in range(1, 18)) == 66
    for index in (13, 14):
        data = case_data(index)
        duration, target, _ = data["waveform"][0]
        q, current, _w, switch = cv_at(data, 0.0, target, duration)
        assert abs(current) < data["maximum_current_A"]
        assert data["waveform"][1][1] > equilibrium_polynomial(data)(q)
        assert (switch == 0.0) == (index == 13)
    for name in ("base", "shifted", "nonlinear"):
        solution = soh(analysis_data("SOH", name))
        assert 0.0001 < solution["x_0"] < solution["x_100"] < 0.999
        assert 0.0001 < solution["y_100"] < solution["y_0"] < 0.999


@pytest.fixture(scope="module")
def short_cc_pair():
    from battery_split_ocvr_traces import native_time

    data = case_data(2)
    data.update(waveform=((0.2, 0.5, "A"),), observation_horizons_s=(0.2,), samples_per_segment=3)
    return data, native_time(data, "coarse"), reference_time(data, "coarse")


def test_short_native_source_preserve_true_coordinates(short_cc_pair):
    data, native, source = short_cc_pair
    for trace, impl in ((native, "native"), (source, "reference")):
        checks = invariants(data, trace, impl)
        assert all(passes(v) for v in checks.values()), checks
        assert trace["segments"][0]["times_s"][0] == 0
        assert trace["segments"][0]["values"][-1][2] == pytest.approx(0.1, abs=1e-12)
    assert "work_J" not in source["segments"][0]
    assert all(passes(v) for v in compare(native, source).values())


def test_short_cv_and_source_explicitly_no_cv():
    from battery_split_ocvr_traces import native_time

    data = case_data(13)
    data.update(
        waveform=((0.2, data["waveform"][0][1], "V"),),
        observation_horizons_s=(0.2,),
        samples_per_segment=3,
    )
    checks = invariants(data, native_time(data, "coarse"), "native")
    assert all(passes(v) for v in checks.values()), checks
    with pytest.raises(ValueError, match="prescribed current"):
        reference_time(data, "coarse")


@pytest.mark.parametrize("defect", ("missing", "short", "nonfinite", "t0", "inventory", "outcome"))
def test_integrity_rejects_incomplete_or_wrong_history(short_cc_pair, defect):
    data, native, _ = short_cc_pair
    trace = deepcopy(native)
    if defect == "missing":
        trace["segments"] = []
    elif defect == "short":
        trace["segments"][0]["times_s"][-1] = 0.15
        trace["segments"][0]["end_time_s"] = 0.15
    elif defect == "nonfinite":
        trace["segments"][0]["values"][0][0] = float("nan")
    elif defect == "t0":
        trace["segments"][0]["times_s"][0] = 0.01
    elif defect == "inventory":
        trace["segments"][0]["amounts_mol"][0][0] *= 0.9
    else:
        trace["segments"][0]["outcome"] = "anything"
    assert not all(passes(v) for v in history_contract(data, trace, "native").values())


def test_absent_routes_and_numpy_false_never_qualify():
    assert not passes(np.bool_(False))
    item = {"id": "S02", "inputs": case_data(2), "routes": {}}
    assess(item)
    assert item["passed"] is False
    assert item["checks"]["native_fine"]["route_completed"] is False


def test_short_actual_boundary_and_exact_event_aliases():
    from battery_split_ocvr_assess import event_kind
    from battery_split_ocvr_oracle import stop_time
    from battery_split_ocvr_traces import native_time

    data = case_data(9)
    data["initial_negative_fraction"] = 0.0002
    end = stop_time(data, 0.5)
    data.update(
        waveform=((1.1 * end, 0.5, "A"),),
        observation_horizons_s=(end,),
        analytic_stop_time_s=end,
        samples_per_segment=3,
    )
    for trace, impl in (
        (native_time(data, "coarse"), "native"),
        (reference_time(data, "coarse"), "reference"),
    ):
        checks = invariants(data, trace, impl)
        assert all(passes(v) for v in checks.values()), checks
        assert trace["segments"][0]["times_s"][-1] == pytest.approx(end, abs=1e-8)
    for electrode in ("negative", "positive"):
        for bound in ("minimum", "maximum"):
            assert (
                event_kind(f"event: {bound.title()} {electrode.title()} Electrode stoichiometry")
                == f"{electrode}-{bound}-fraction"
            )
    assert event_kind("anything") != "complete"
