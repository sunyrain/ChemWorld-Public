"""Build/output contracts only; no scientific time-domain qualification claim."""

import numpy as np
import pytest
from battery_composite_cases import GRIDS, REFUSALS, RESPONSES, case_data, native_spec
from battery_composite_pybamm import build_reference, output_names


def test_fixed_composite_scope_and_denominator():
    assert len(RESPONSES) == 19
    assert len(REFUSALS) == 8
    assert 11 * 4 + 2 + len(REFUSALS) == 54
    assert GRIDS == {"native": (40, 80), "reference": (120, 200)}


@pytest.mark.parametrize("index", range(1, 12))
def test_native_input_mapping_is_three_independent_materials(index):
    data = case_data(index)
    spec = native_spec(data, 8)
    state = spec.initial_state()
    assert len(spec.particle_bins) == 3
    assert len(spec.inventory_keys) == 24
    assert not hasattr(spec, "negative")
    for declared, phase, population in zip(data["phases"], spec.phases,
                                           spec.particle_bins, strict=True):
        electrode, _, particle, keys, identity = population
        assert identity == f"{electrode}:phase:{declared['phase_id']}"
        assert phase.phase_id == declared["phase_id"]
        assert sum(state.amounts_mol[key] for key in keys) == pytest.approx(
            declared["initial_fraction"] * declared["active_volume_m3"]
            * declared["maximum_concentration_mol_m3"], abs=1e-16)
        assert particle.diffusivity_m2_s == declared["diffusivity_m2_s"]
        assert phase.ocv.value(.2) == pytest.approx(sum(
            coefficient * .2**i for i, coefficient in enumerate(declared["polynomial"])),
            abs=1e-13)


@pytest.mark.parametrize("index,single", [(i, False) for i in range(1, 12)] + [(1, True)])
def test_actual_source_three_material_build_and_output(index, single):
    pybamm = pytest.importorskip("pybamm")
    data = case_data(index)
    simulation = build_reference(data, 12, single=single)
    assert type(simulation.model) is pybamm.lithium_ion.SPM
    assert simulation.model.options["surface form"] == "algebraic"
    assert simulation.model.options["particle phases"] == (("1", "1") if single else ("2", "1"))
    simulation.build()
    model, solver = simulation.built_model, simulation.solver
    inputs = ({"composite voltage target": data["waveform"][0][1]}
              if data["waveform"][0][2] == "V" else
              {"Current function [A]": data["waveform"][0][1]})
    solver.set_up(model, inputs=inputs)
    state = np.asarray(solver.calculate_consistent_state(model, inputs=inputs))
    values = []
    for key in output_names(data, single):
        array = np.asarray(model.variables[key].evaluate(t=0., y=state, inputs=inputs))
        assert array.size == 1, key
        values.append(float(array.ravel()[0]))
    assert len(values) == (14 if single else 19)
    assert np.isfinite(values).all()
    assert abs(sum(values[4::5])) < 1e-7
    if not single and data.get("internal_exchange"):
        assert abs(values[4]) > 1e-4 and abs(values[9]) > 1e-4
        assert values[4] * values[9] < 0


def test_all_route_failures_retained(monkeypatch):
    import run_battery_composite_block as runner

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
    assert count == sum(len(u["routes"]) for u in units) == 54
    assert not any(u["passed"] for u in units)


def test_identical_material_collapse_preserves_phase_total_inventory():
    from battery_composite_assess import collapse_identical

    row = [3.7, .6, 60., 220., .4, .5, .51, .006, .02,
           .2, .5, .51, .003, .02, -.6, .4, .39, .007, -.03]
    trace = {"segments": [{"rows": [row], "profiles_theta": [[.5]*9, [.5]*9, [.4]*9]}]}
    collapsed = collapse_identical(trace)
    result = collapsed["segments"][0]["rows"][0]
    assert len(result) == 14
    np.testing.assert_allclose(result[:4], row[:4])
    assert result[4] == pytest.approx(.6)
    assert result[7] == pytest.approx(.009)
    assert trace["segments"][0]["rows"][0] == row


def test_missing_interfaces_not_counted_as_domain_refusal():
    from battery_composite_assess import valid_refusal

    base = {"rejected": True, "input_state_unchanged": True,
            "exception_type": "ValueError", "exception": "unsupported composite thermal family"}
    assert valid_refusal(base)
    for error in ("TypeError", "ImportError", "AttributeError"):
        assert not valid_refusal({**base, "exception_type": error})


def test_event_endpoints_not_extrapolated():
    from battery_composite_assess import aligned

    a = {"segments": [{"times_s": [0., 1., 1.2], "rows": [[0.], [1.], [8.]]}]}
    b = {"segments": [{"times_s": [0., 1., 1.3], "rows": [[0.], [1.], [-8.]]}]}
    left, right = aligned(a, b)
    np.testing.assert_array_equal(left, [[0.], [1.]])
    np.testing.assert_array_equal(left, right)
