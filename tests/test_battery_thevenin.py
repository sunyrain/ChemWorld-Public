"""Actual common-kernel ECM trajectories; not the independent PyBaMM qualification."""

import json
from copy import deepcopy
from dataclasses import replace
from math import exp

import numpy as np
import pytest
from scipy.linalg import expm

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.scientific_inputs import synthetic_source
from chemworld.physchem.battery_thevenin import (
    AMBIENT_LOSS,
    CELL_T,
    CHARGE,
    CONSTANT_UNITS,
    JIG_T,
    MODEL_HEAT,
    SOC,
    THROUGHPUT_E,
    THROUGHPUT_Q,
    WORK,
    TheveninSpec,
    rc_key,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from test_tabulated_functions import function


def test_lazy_public_facade_exposes_the_actual_kernel():
    from chemworld import physchem
    from chemworld.physchem.tabulated_functions import TabulatedScalar

    assert physchem.TheveninSpec is TheveninSpec
    assert physchem.thevenin_system is thevenin_system
    assert physchem.thevenin_fields is thevenin_fields
    assert physchem.TabulatedScalar is TabulatedScalar


def fixture(elements=1, *, entropy=0., r0=.1, **overrides):
    values = {"capacity_C": 100., "maximum_current_A": 10., "minimum_voltage_V": .1,
              "maximum_voltage_V": 10., "initial_soc": .7, "minimum_soc": 0., "maximum_soc": 1.,
              "initial_cell_temperature_K": 298.15, "initial_jig_temperature_K": 298.15,
              "minimum_temperature_K": 250., "maximum_temperature_K": 400.,
              "cell_heat_capacity_J_K": 100., "jig_heat_capacity_J_K": 500.,
              "cell_jig_conductance_W_K": .5, "jig_ambient_conductance_W_K": .2,
              "reference_frame_temperature_K": 298.15, "housing_volume_L": .01}
    values.update({f"initial_rc_{i}_V": 0. for i in range(elements)})
    values.update(overrides)
    units = {**CONSTANT_UNITS, **{f"initial_rc_{i}_V": "V" for i in range(elements)}}
    source = synthetic_source()
    constants = ScientificTable("thevenin-fixture", "parameters",
        tuple(QuantityColumn(key, units[key], "parameter", (-1e9, 1e9),
                             "Self-authored native-unit fixture") for key in values), (source,),
        (ScientificRow("cell", source.provenance.source_id, "model", "constants", "parameters",
                       {key: InputValue(value, units[key], None, "Uncertainty not supplied")
                        for key, value in values.items()}),))
    return TheveninSpec(
        constants, "cell", function({"soc": (0., .5, 1.)}, lambda soc: 3+soc, output_unit="V"),
        function({}, lambda: r0),
        tuple((function({}, lambda i=i: .2*(i+1)),
               function({}, lambda i=i: 5./(i+1), output_unit="F")) for i in range(elements)),
        function({}, lambda: entropy, output_unit="V/K"),
        ScalarControlProgram(((0., 298.15),), "K"))


@pytest.mark.parametrize("elements", (0, 1, 2, 4))
@pytest.mark.parametrize("current", (-.5, .5))
def test_nonzero_rc_charge_work_and_body_heat_use_one_shared_advance(elements, current):
    spec = fixture(elements)
    state = spec.initial_state()
    program = ScalarControlProgram(((0., current),), "A")
    result = thevenin_system(spec, program).advance(state, 3., evaluation_times_s=[0., 1., 3.])
    assert result.outcome == "success" and result.actual_duration_s == 3.
    final = result.final_state
    assert final.amounts_mol == {} and final.algebraic_values == {}
    assert final.coordinates[SOC] == pytest.approx(.7-current*3/100, abs=1e-10)
    assert final.coordinates[CHARGE] == pytest.approx(current*3, abs=1e-8)
    for i in range(elements):
        expected = current*.2*(i+1)*(1-exp(-3))
        assert final.coordinates[rc_key(i)] == pytest.approx(expected, rel=5e-7, abs=1e-9)
    expected_work = current*3.7*3-current**2*9/200-current**2*.1*3
    expected_work -= sum(current**2*.2*(i+1)*(3-(1-exp(-3))) for i in range(elements))
    assert final.coordinates[WORK] == pytest.approx(expected_work, rel=5e-7, abs=1e-8)
    assert final.coordinates[THROUGHPUT_Q] == pytest.approx(abs(current)*3, abs=1e-8)
    assert final.coordinates[THROUGHPUT_E] == pytest.approx(abs(expected_work), rel=5e-7, abs=1e-8)
    assert final.coordinates[CELL_T] > state.coordinates[CELL_T]
    assert final.temperature_K == state.temperature_K  # Formal frame, not cell body.
    assert result.energy_balance_residual_J is None
    assert result.conservation_residuals["circuit_charge_C"] == pytest.approx(0., abs=1e-8)
    assert result.conservation_residuals["empirical_node_heat_J"] == pytest.approx(0., abs=1e-6)
    assert state == spec.initial_state()


@pytest.mark.parametrize("entropy", (0., 1e-4, -1e-4))
def test_two_body_temperatures_match_an_independent_linear_matrix_exponential(entropy):
    spec = fixture(2, entropy=entropy, initial_cell_temperature_K=310.,
                   initial_jig_temperature_K=295.)
    current, duration = .5, 10.
    state = spec.initial_state()
    result = thevenin_system(spec, ScalarControlProgram(((0., current),), "A")).advance(
        state, duration)
    # Independent constant-coefficient RC + two-body equations, shifted by Tamb.
    matrix = np.zeros((5, 5))
    for i in (0, 1):
        resistance, capacitance = .2*(i+1), 5./(i+1)
        matrix[i, i], matrix[i, -1] = -1/(resistance*capacitance), current/capacitance
        matrix[2, i] = current/100
    matrix[2, 2], matrix[2, 3] = -(.5+current*entropy)/100, .5/100
    matrix[2, -1] = (current**2*.1-current*298.15*entropy)/100
    matrix[3, 2], matrix[3, 3] = .5/500, -(.5+.2)/500
    expected = expm(matrix*duration) @ np.array([0., 0., 310.-298.15, 295.-298.15, 1.])
    actual = result.final_state.coordinates
    assert actual[CELL_T] == pytest.approx(expected[2]+298.15, abs=2e-6)
    assert actual[JIG_T] == pytest.approx(expected[3]+298.15, abs=2e-6)
    assert actual[MODEL_HEAT] != 0 and actual[AMBIENT_LOSS] != 0


@pytest.mark.parametrize("target,limit,expected_current", ((3.5, 10., None),
                                                         (3.3, .1, .1), (4., .1, -.1)))
def test_cv_uses_the_same_terminal_equation_and_physical_source_limits(
    target, limit, expected_current
):
    spec = fixture(maximum_current_A=limit)
    initial = spec.initial_state()
    program = ScalarControlProgram(((0., target),), "V")
    result = thevenin_system(spec, program).advance(initial, .5)
    fields = thevenin_fields(spec, result.final_state, program)
    assert abs(fields["current_A"]) <= limit+1e-9
    if expected_current is None:
        assert fields["voltage_V"] == pytest.approx(target, abs=2e-8)
        assert result.final_state.coordinates[WORK] == pytest.approx(
            target*result.final_state.coordinates[CHARGE], abs=1e-7)
    else:
        assert fields["current_A"] == pytest.approx(expected_current, abs=1e-8)
        assert abs(fields["voltage_V"]-target) > .05
    assert initial == spec.initial_state()


@pytest.mark.parametrize("minimum_soc", (0., .2))
def test_first_soc_stop_keeps_actual_time_and_unclipped_state(minimum_soc):
    spec = fixture(0, capacity_C=.05, minimum_soc=minimum_soc)
    initial = spec.initial_state()
    result = thevenin_system(spec, ScalarControlProgram(((0., .5),), "A")).advance(initial, 1.)
    assert result.outcome == "stopped"
    assert result.stopping_event == "thevenin:minimum-soc"
    assert result.actual_duration_s == pytest.approx((.7-minimum_soc)*.05/.5, abs=1e-8)
    assert result.final_state.coordinates[SOC] == pytest.approx(minimum_soc, abs=1e-9)
    assert initial.coordinates[SOC] == .7


def test_source_table_roundtrip_and_zero_time_readout_preserve_state():
    spec = fixture(2)
    restored = TheveninSpec.from_dict(json.loads(json.dumps(spec.to_dict(), sort_keys=True)))
    assert restored.to_dict() == spec.to_dict()
    state = restored.initial_state()
    fields = thevenin_fields(restored, state, ScalarControlProgram(((0., .5),), "A"))
    assert fields["voltage_V"] == pytest.approx(3.65)
    assert state == restored.initial_state()
    with pytest.raises(TypeError):
        restored.native_parameters["capacity_C"] = 1.


def test_constant_source_units_and_uncertainty_are_not_discarded():
    original = fixture(0)
    payload = original.to_dict()
    rows = payload["parameters"]["rows"]
    for key, source_unit in (("capacity_C", "Ah"), ("initial_cell_temperature_K", "degC")):
        for column in payload["parameters"]["columns"]:
            if column["quantity_id"] == key:
                column["unit"] = source_unit
        cell = rows[0]["values"][key]
        converted = InputValue(cell["value"], cell["unit"], .02,
                               "Declared source standard uncertainty").to(source_unit)
        rows[0]["values"][key] = converted.to_dict()
    restored = TheveninSpec.from_dict(payload)
    assert restored.native_parameters == pytest.approx(original.native_parameters)
    saved = restored.parameters.rows[0].values
    assert saved["capacity_C"].unit == "Ah"
    assert saved["capacity_C"].uncertainty == pytest.approx(.02/3600)
    assert saved["initial_cell_temperature_K"].value == 25.
    assert saved["initial_cell_temperature_K"].uncertainty == pytest.approx(.02)


def test_actual_current_temperature_soc_dependent_laws_feed_back_on_the_same_state():
    grids = {"soc": (0., 1.), "temperature_K": (250., 400.), "current_A": (-10., 10.)}
    units = {"soc": "dimensionless", "temperature_K": "K", "current_A": "A"}
    r0 = function(grids, lambda soc, temperature, current:
                  .1+.02*soc+.0001*(temperature-298.15)+.001*current, input_units=units)
    resistance = function(grids, lambda soc, temperature, current:
                          .2*(1+.1*soc+.001*(temperature-298.15)+.001*current), input_units=units)
    capacitance = function(grids, lambda soc, temperature, current:
                           5*(1+.05*soc+.0001*(temperature-298.15)),
                           input_units=units, output_unit="F")
    spec = replace(fixture(), series_resistance=r0, rc_elements=((resistance, capacitance),))
    program = ScalarControlProgram(((0., .5),), "A")
    initial = spec.initial_state()
    result = thevenin_system(spec, program).advance(initial, 2.)
    fields = thevenin_fields(spec, result.final_state, program)
    actual = result.final_state.coordinates
    expected = .1+.02*actual[SOC]+.0001*(actual[CELL_T]-298.15)+.001*.5
    assert fields["r0_ohm"] == pytest.approx(expected, abs=1e-13)
    assert fields["r0_ohm"] != pytest.approx(.1145, abs=1e-7)
    assert fields["resistance_ohm"][0] != .2
    assert fields["capacitance_F"][0] != 5.
    assert initial == spec.initial_state()


def test_current_dependent_parameters_and_cv_invertibility_are_distinct():
    spec = replace(fixture(0), series_resistance=function(
        {"current_A": (-10., 0., 10.)}, lambda current: 2-.15*current,
        input_units={"current_A": "A"}))
    fields = thevenin_fields(spec, spec.initial_state(), ScalarControlProgram(((0., .5),), "A"))
    assert fields["r0_ohm"] == pytest.approx(1.925)
    with pytest.raises(ValueError, match="monotone"):
        thevenin_system(spec, ScalarControlProgram(((0., 3.5),), "V"))
    ideal = fixture(0, r0=0.)
    fields = thevenin_fields(ideal, ideal.initial_state(),
                            ScalarControlProgram(((0., .5),), "A"))
    assert fields["voltage_V"] == pytest.approx(3.7)
    with pytest.raises(ValueError, match="monotone"):
        thevenin_system(ideal, ScalarControlProgram(((0., 3.5),), "V"))


@pytest.mark.parametrize("defect", ("family", "missing_constant", "bad_unit", "negative_capacity",
                                   "ambient", "negative_rc", "unknown_field"))
def test_incompatible_parameters_or_models_fail_explicitly(defect):
    payload = deepcopy(fixture().to_dict())
    if defect == "family":
        payload["model_family"] = "not-a-lithium-SPM-alias"
    elif defect == "missing_constant":
        payload["parameters"]["columns"].pop()
    elif defect == "bad_unit":
        payload["parameters"]["rows"][0]["values"]["capacity_C"]["unit"] = "V"
    elif defect == "negative_capacity":
        payload["parameters"]["rows"][0]["values"]["capacity_C"]["value"] = -1.
    elif defect == "ambient":
        payload["ambient_program"]["unit"] = "degC"
    elif defect == "negative_rc":
        value = payload["rc_elements"][0]["resistance"]["table"]["rows"][0]["values"]["response"]
        value["value"] = -1
    else:
        payload["unmodeled_diffusion"] = True
    with pytest.raises(ValueError):
        TheveninSpec.from_dict(payload)
