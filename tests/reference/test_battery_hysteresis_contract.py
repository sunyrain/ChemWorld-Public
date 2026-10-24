"""Construction/coverage contracts; not substituted for the full scientific block."""

import numpy as np
import pytest
from battery_hysteresis_cases import (
    UNITS,
    analytic_h,
    branch_polynomial,
    case_data,
    native_case,
    response_atol,
    response_names,
    sweep_currents,
)
from battery_hysteresis_pybamm import (
    build_reference,
    output_names,
    scalar_outputs,
    shared_h_broadcast_errors,
)


def test_fixed_coverage_and_passive_branch_orientation():
    cases = [case_data(i, j) for i in range(1, 17) for j in range(len(UNITS[i-1]))]
    assert len(UNITS) == 16
    assert sum(d["expected"] in {"trajectory", "voltage_event"} for d in cases) == 13
    assert sum(d["expected"] == "static_sweep" for d in cases) == 3
    assert sum(d["expected"] == "domain_refusal" for d in cases) == 6
    assert 13*8+3*2+8+2+6 == 126
    for d in cases:
        assert len(response_atol(d)) == len(response_names(d)) == len(output_names(d))
        for electrode in ("negative", "positive"):
            lith, de = (branch_polynomial(d, electrode, v) for v in (True, False))
            assert de[0] > lith[0] and de[1] == lith[1]
    data = case_data(15)
    increment = ((data["negative_gap_V"]+data["positive_gap_V"])
                 *data["current_density_scale_m2_A"]/(2*data["effective_collector_area_m2"]))
    assert increment == 500.
    assert len(sweep_currents()) == 9 and sweep_currents()[4] == 0.


@pytest.mark.parametrize("index", range(1, 16))
def test_actual_released_ocp_models_build(index):
    pybamm = pytest.importorskip("pybamm")
    data = case_data(index)
    simulation = build_reference(data, 8, 4 if data["size_distribution"] == "narrow" else None,
                                 False)
    for electrode in ("negative", "positive"):
        expected = {"single": pybamm.open_circuit_potential.SingleOpenCircuitPotential,
            "current sigmoid": pybamm.open_circuit_potential.CurrentSigmoidOpenCircuitPotential,
            "Axen": pybamm.open_circuit_potential.AxenOpenCircuitPotential}[data[electrode+"_mode"]]
        actual = simulation.model.submodels[electrode+" primary open-circuit potential"]
        assert type(actual) is expected
    simulation.build()
    model = simulation.built_model
    inputs = {"native voltage target": 3.75} if index == 10 else {"Current function [A]": 0.}
    initial = np.asarray(model.concatenated_initial_conditions.evaluate(inputs=inputs)).ravel()
    assert np.isfinite(initial).all()


def test_independent_h_rest_and_asymptotic_endpoints():
    data = case_data(7)
    for electrode in ("negative", "positive"):
        for h in (-1., .2, 1.):
            assert analytic_h(data, electrode, h, 0., 100., 96485.33212) == h
            assert -1 <= analytic_h(data, electrode, h, .25, 120., 96485.33212) <= 1


@pytest.mark.parametrize("index", range(1, 16))
def test_native_declaration_preserves_explicit_h_states(index):
    data = case_data(index)
    spec = native_case(data, 8, 4 if data["size_distribution"] == "narrow" else None)
    state = spec.initial_state()
    expected = tuple("battery:"+name+":hysteresis" for name in ("negative", "positive")
                     if data[name+"_mode"] == "Axen")
    assert spec.hysteresis_keys == expected
    for key in expected:
        name = key.split(":")[1]
        assert state.coordinates[key] == data[name+"_initial_h"]


def test_all_failed_routes_remain_in_full_fixed_denominator(monkeypatch):
    import run_battery_hysteresis_block as runner

    count = 0

    def failed(function):
        nonlocal count
        count += 1
        return {"exception": "injected independent execution failure"}

    monkeypatch.setattr(runner, "capture", failed)
    units = []
    for index in range(1, 17):
        for j in range(len(UNITS[index-1])):
            data = case_data(index, j)
            item = {"inputs": data, "passed": False}
            runner.run_variant(item, data, lambda _: None, lambda: None)
            units.append(item)
    assert count == sum(len(item["routes"]) for item in units) == 126
    assert all(item["passed"] is False for item in units)


def test_history_alignment_never_interpolates_event_endpoints():
    from battery_hysteresis_assess import aligned_rows

    a = {"segments": [{"times_s": [0., 1., 1.2], "rows": [[0.], [1.], [99.]]}]}
    b = {"segments": [{"times_s": [0., 1., 1.3], "rows": [[0.], [1.], [-99.]]}]}
    left, right = aligned_rows(a, b, {})
    np.testing.assert_array_equal(left, [[0.], [1.]])
    np.testing.assert_array_equal(left, right)


def test_refusal_requires_domain_error_and_unchanged_state():
    from battery_hysteresis_assess import valid_refusal

    result = {"rejected": True, "input_state_unchanged": True,
              "exception_type": "ValueError", "exception": "OCP domain unsupported"}
    assert valid_refusal(result)
    assert not valid_refusal({**result, "input_state_unchanged": False})
    assert not valid_refusal({**result, "exception_type": "ImportError"})


@pytest.mark.parametrize("index", (13, 14))
def test_fine_released_mpm_grid_and_all_observable_shapes(index):
    pytest.importorskip("pybamm")
    data = case_data(index)
    simulation = build_reference(data, 160, 80, True)
    simulation.build()
    assert simulation.mesh["negative particle size"].npts == 80
    assert simulation.mesh["positive particle size"].npts == 80
    assert simulation.mesh["negative particle"].npts == 160
    model, inputs = simulation.built_model, {"Current function [A]": 0.}
    initial = np.asarray(model.concatenated_initial_conditions.evaluate(inputs=inputs)).ravel()

    class InitialObservation:
        def __getitem__(self, name):
            value = np.asarray(model.variables[name].evaluate(t=0., y=initial, inputs=inputs))
            return lambda times: np.repeat(value.reshape(-1, 1), len(times), axis=1)

    observation = InitialObservation()
    rows = scalar_outputs(observation, [0., 0.], data)
    assert rows.shape == (2, len(output_names(data)))
    assert np.isfinite(rows).all()
    errors = shared_h_broadcast_errors(observation, [0., 0.], data)
    assert all(error <= 1e-12 for error in errors.values())
    if index == 13:
        assert len(errors) == 4
    for name in ("negative", "positive"):
        explicit = observation[f"qualification:{name}:area-ocp"]([0.])
        published = observation[f"X-averaged {name} electrode open-circuit potential [V]"]([0.])
        np.testing.assert_array_equal(explicit, published)


def test_spatial_output_and_nonuniform_h_cannot_be_silently_averaged():
    class NonScalar:
        def __getitem__(self, name):
            return lambda times: np.zeros((2, len(times)))

    with pytest.raises(ValueError, match="not scalar"):
        scalar_outputs(NonScalar(), [0., 1.], case_data(13))

    class NonuniformH:
        def __getitem__(self, name):
            if name.endswith("shared-h"):
                return lambda times: np.zeros(len(times))
            return lambda times: np.array([np.zeros(len(times)), np.ones(len(times))])

    with pytest.raises(ValueError, match="nonuniform broadcast"):
        shared_h_broadcast_errors(NonuniformH(), [0., 1.], case_data(13))
