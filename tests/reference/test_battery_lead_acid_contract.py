"""LOQS adapter development contracts, outside the fixed 13-unit data denominator."""

import numpy as np
import pytest
from battery_lead_acid_cases import NAMES, PRECISIONS, REFUSALS, SOURCE_F, case_data
from battery_lead_acid_oracle import fields
from battery_lead_acid_pybamm import build_reference, reference_time


@pytest.mark.parametrize("index", range(1, 13))
def test_actual_source_build_and_unchanged_loqs_scope(index):
    simulation = build_reference(case_data(index), "coarse")
    simulation.build()
    model = simulation.model
    assert model.options["surface form"] == "false"
    assert model.options["thermal"] == "isothermal"
    assert model.options["convection"] == "none"
    assert model.options["hydrolysis"] == "false"
    rhs = [str(key) for key in model.rhs]
    assert "X-averaged electrolyte concentration [mol.m-3]" in rhs
    assert len([key for key in rhs if "porosity" in key]) == 3
    assert not any("Pb" in key for key in rhs)
    assert not any("particle" in key for key in rhs)


def test_explicit_synthetic_volumes_and_bode_rate_mapping():
    import pybamm

    data = case_data(4)
    simulation = build_reference(data, "coarse")
    values = simulation.parameter_values
    param = simulation.model.param
    vp, vo, vs, ve, vw = data["molar_volumes_m3_mol"]
    assert vp + vo - 2 * vs == pytest.approx(2 * (vw - ve), abs=1e-19)
    assert values.evaluate(param.F) == SOURCE_F
    assert values["Initial temperature [K]"] == values["Ambient temperature [K]"] == 310.0
    assert values["Number of cells connected in series to make a battery"] == 1.0
    assert values["Number of electrodes connected in parallel to make a cell"] == 1.0
    for charge in (0, 100, 500):
        original = fields(data, charge, 0.5, SOURCE_F)
        ce = pybamm.Scalar(original["acid_concentration_mol_m3"])
        temperature = pybamm.Scalar(data["temperature_K"])
        for name, domain in (("negative", param.n), ("positive", param.p)):
            assert values.evaluate(domain.prim.U(ce, temperature)) == pytest.approx(
                original[name + "_ocp_V"], abs=1e-12
            )
            assert values.evaluate(domain.prim.j0(ce, temperature)) == pytest.approx(
                original[name + "_j0_A_m2"], abs=1e-12
            )


def test_short_actual_source_history_retains_t0_and_direct_states():
    data = case_data(2)
    data["waveform"] = ((0.2, 0.5, "A"),)
    data["observation_horizons_s"] = (0.2,)
    data["samples_per_segment"] = 3
    result = reference_time(data, "coarse")
    segment = result["segments"][0]
    assert segment["times_s"] == pytest.approx([0, 0.1, 0.2], abs=1e-14)
    assert np.shape(segment["values"]) == (3, 16)
    assert np.all(np.isfinite(segment["original_source_state_coordinates"]))
    for time, values in zip(segment["times_s"], segment["values"], strict=True):
        oracle = fields(data, 0.5 * time, 0.5, SOURCE_F)
        assert values[0] == pytest.approx(oracle["voltage_V"], abs=1e-7)
        assert values[4] == pytest.approx(oracle["acid_concentration_mol_m3"], abs=1e-4)
        assert values[7:10] == pytest.approx(oracle["porosities"], abs=1e-8)


def test_fixed_denominator_and_native_extra_event_semantics():
    assert len(NAMES) == 13 and len(PRECISIONS) == 2 and len(REFUSALS) == 8
    assert 12 * 2 * 2 + 8 == 56
    for index in range(7, 12):
        data = case_data(index)
        assert data["source_stop_semantics"].startswith("author_imposed")
        assert data["waveform"][0][0] > data["observation_horizons_s"][0]


@pytest.fixture(scope="module")
def short_cc_pair():
    from battery_lead_acid_traces import native_time

    data = case_data(2)
    data["waveform"] = ((0.2, 0.5, "A"),)
    data["observation_horizons_s"] = (0.2,)
    data["samples_per_segment"] = 3
    return data, native_time(data, "coarse"), reference_time(data, "coarse")


def test_short_native_source_observers_and_all_balances(short_cc_pair):
    from battery_lead_acid_assess import compare, invariants, passes
    from battery_lead_acid_oracle import cc_work

    data, native, source = short_cc_pair
    for trace, implementation in ((native, "native"), (source, "reference")):
        checks = invariants(data, trace, implementation)
        assert all(passes(v) for v in checks.values()), checks
        values = np.asarray(trace["segments"][0]["values"])
        assert values[0, 2:4] == pytest.approx([0, 0], abs=1e-12)
        assert values[-1, 2] == pytest.approx(0.1, abs=1e-10)
        assert values[-1, 3] == pytest.approx(cc_work(data, 0, 0.1, 0.5), abs=1e-7)
    assert all(passes(v) for v in compare(native, source).values())


def test_short_interior_cv_adapter():
    from battery_lead_acid_assess import compare, invariants, passes
    from battery_lead_acid_traces import native_time

    data = case_data(5)
    target = data["waveform"][0][1]
    data["waveform"] = ((0.2, target, "V"),)
    data["observation_horizons_s"] = (0.2,)
    data["samples_per_segment"] = 3
    native, source = native_time(data, "coarse"), reference_time(data, "coarse")
    for trace, implementation in ((native, "native"), (source, "reference")):
        checks = invariants(data, trace, implementation)
        assert all(passes(v) for v in checks.values()), checks
    assert all(passes(v) for v in compare(native, source).values())


@pytest.mark.parametrize(
    "defect", ["missing_segment", "shortened", "nonfinite", "missing_t0", "wrong_initial_amount"]
)
def test_incomplete_history_without_exception_never_passes(short_cc_pair, defect):
    from copy import deepcopy

    from battery_lead_acid_assess import history_contract, passes

    data, native, _ = short_cc_pair
    trace = deepcopy(native)
    if defect == "missing_segment":
        trace["segments"] = []
    elif defect == "shortened":
        trace["segments"][0]["times_s"][-1] = 0.15
        trace["segments"][0]["end_time_s"] = 0.15
    elif defect == "nonfinite":
        trace["segments"][0]["values"][1][0] = float("nan")
    elif defect == "missing_t0":
        trace["segments"][0]["times_s"][0] = 0.01
    else:
        trace["segments"][0]["amounts_mol"][0][0] *= 0.9
    assert not all(passes(v) for v in history_contract(data, trace, "native").values())


def test_missing_route_and_numpy_false_are_explicit_failures():
    from battery_lead_acid_assess import assess, passes

    assert not passes(np.bool_(False))
    item = {"inputs": case_data(2), "routes": {}}
    assess(item)
    assert item["passed"] is False
    assert item["checks"]["native_fine"]["route_completed"] is False
