"""Fixed mapping/build contracts; no time integration and no scientific pass claim."""

import numpy as np
import pytest
from battery_half_cell_assess import aligned, valid_refusal
from battery_half_cell_cases import (
    GRIDS,
    REFUSALS,
    RESPONSE_ATOL,
    RESPONSES,
    UNITS,
    case_data,
    initial_metal,
    native_spec,
    net_ocv,
)
from battery_half_cell_pybamm import build_reference, output_names


def test_fixed_coverage_and_output_contract():
    assert len(UNITS) == 12 and len(REFUSALS) == 8
    assert sum(case_data(i)["expected"] != "refusals" for i in range(1, 13)) * 4 + 8 == 52
    assert GRIDS == {"native": (40, 80), "reference": (120, 200)}
    assert len(RESPONSES) == len(RESPONSE_ATOL) == len(output_names()) == 9


@pytest.mark.parametrize("index", range(1, 12))
def test_native_declaration_has_finite_metal_and_only_working_particle(index):
    data = case_data(index)
    spec = native_spec(data, 8)
    state = spec.initial_state()
    assert not hasattr(spec, "negative")
    assert spec.inventory_keys == ("metal:Li", *spec.positive.keys("positive"))
    assert len(state.amounts_mol) == 9
    assert state.amounts_mol["metal:Li"] == initial_metal(data)
    assert spec.metal.resistance_ohm == data["thickness_m"] / (
        data["conductivity_S_m"] * data["area_m2"])
    for theta in (.02, .25, .5, .85, .98):
        assert spec.positive_ocv.value(theta) == pytest.approx(net_ocv(theta), abs=1e-13)


@pytest.mark.parametrize("index", range(1, 12))
def test_actual_released_halfcell_build_and_scalar_outputs(index):
    pybamm = pytest.importorskip("pybamm")
    data = case_data(index)
    simulation = build_reference(data, 12)
    assert type(simulation.model) is pybamm.lithium_ion.SPM
    assert simulation.model.options.electrode_types == {"negative": "planar", "positive": "porous"}
    assert not any("negative particle" in str(key).lower() for key in simulation.model.rhs)
    simulation.build()
    built, solver = simulation.built_model, simulation.solver
    inputs = ({"half-cell voltage target": data["waveform"][0][1]}
              if data["waveform"][0][2] == "V" else {"Current function [A]": 0.})
    solver.set_up(built, inputs=inputs)
    state = np.asarray(solver.calculate_consistent_state(built, inputs=inputs))
    values = []
    for key in output_names():
        result = np.asarray(built.variables[key].evaluate(t=0., y=state, inputs=inputs))
        assert result.size == 1, key
        values.append(float(result.ravel()[0]))
    assert np.isfinite(values).all()
    assert values[4] == pytest.approx(initial_metal(data), abs=1e-15)
    if data["waveform"][0][2] == "A":
        assert values[0] == pytest.approx(net_ocv(data["initial_positive_fraction"]), abs=1e-11)


@pytest.mark.parametrize("index", (7, 8))
def test_fixed_source_ocv_can_be_certified_for_cv(index):
    assert native_spec(case_data(index), 40).cv_monotone_certified


def test_refusal_cannot_hide_missing_implementation():
    base = {"rejected": True, "input_state_unchanged": True,
            "exception_type": "ValueError", "exception": "half-cell analysis unsupported"}
    assert valid_refusal(base)
    for error in ("TypeError", "ImportError", "AttributeError"):
        assert not valid_refusal({**base, "exception_type": error})


def test_all_route_failures_remain_in_denominator(monkeypatch):
    import run_battery_half_cell_block as runner

    count = 0

    def failed(function):
        nonlocal count
        count += 1
        return {"exception": "injected failure"}

    monkeypatch.setattr(runner, "capture", failed)
    units = []
    for index in range(1, 13):
        unit = {"id": str(index), "inputs": case_data(index)}
        runner.run_unit(unit, lambda _: None, lambda: None)
        units.append(unit)
    assert count == sum(len(u["routes"]) for u in units) == 52
    assert not any(u["passed"] for u in units)


def test_event_endpoints_are_not_extrapolated():
    a = {"segments": [{"times_s": [0., 1., 1.2], "rows": [[0.], [1.], [10.]]}]}
    b = {"segments": [{"times_s": [0., 1., 1.3], "rows": [[0.], [1.], [-10.]]}]}
    x, y = aligned(a, b)
    np.testing.assert_array_equal(x, [[0.], [1.]])
    np.testing.assert_array_equal(x, y)
