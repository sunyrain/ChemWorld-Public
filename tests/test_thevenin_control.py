"""Named controller state, branch certification and actual common-clock equations."""

import json
from dataclasses import replace
from math import exp, log, sqrt

import numpy as np
import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.scientific_inputs import synthetic_source
from chemworld.physchem.battery_thevenin import (
    CHARGE,
    SOC,
    WORK,
    TheveninSpec,
    thevenin_fields,
    thevenin_system,
)
from chemworld.physchem.battery_thevenin_control import (
    CONTROL_UNITS,
    CONTROLLER_CURRENT,
    TheveninControlProgram,
    TheveninControlSpec,
    thevenin_control_program,
)
from chemworld.physchem.reactor_shared import ScalarControlProgram
from chemworld.physchem.residual_system import ResidualDomainError
from chemworld.physchem.thevenin_frequency import thevenin_impedance
from test_battery_thevenin import fixture
from test_tabulated_functions import function


def controller(**overrides):
    values = {
        "initial_current_A": 0.1, "current_scale_A": 1.0, "power_scale_W": 1.0,
        "resistance_scale_Ohm": 1.0, "voltage_scale_V": 1.0,
        "power_response_time_s": 100.0, "resistance_response_time_s": 100.0,
        "voltage_response_time_s": 1.0, "antiwindup_time_s": 100 / 3600,
    }
    values.update(overrides)
    source = synthetic_source()
    table = ScientificTable(
        "controller-fixture",
        "parameters",
        tuple(
            QuantityColumn(k, unit, "parameter", (-1e9, 1e9), "Explicit functional fixture")
            for k, unit in CONTROL_UNITS.items()
        ),
        (source,),
        (
            ScientificRow(
                "controller",
                source.provenance.source_id,
                "model",
                "controller",
                "parameters",
                {
                    k: InputValue(v, CONTROL_UNITS[k], None, "Uncertainty not supplied")
                    for k, v in values.items()
                },
            ),
        ),
    )
    return TheveninControlSpec(table, "controller")


def scalar(value, unit):
    return ScalarControlProgram(((0.0, value),), unit)


def program(mode, target, branch=None, limit=None):
    unit = {
        "current": "A",
        "voltage": "V",
        "power": "W",
        "resistance": "Ohm",
        "differential_power": "W",
        "differential_resistance": "Ohm",
        "cccv": "V",
    }[mode]
    return TheveninControlProgram(
        mode, scalar(target, unit), scalar(limit, "A") if limit is not None else None, branch
    )


def constant_spec(*, initial_current=0.1, r0=0.1, **kwargs):
    return replace(
        fixture(0, r0=r0, **kwargs),
        ocv=function({}, lambda: 3.6, output_unit="V"),
        control=controller(initial_current_A=initial_current),
    )


def test_old_payload_remains_exact_and_controller_state_has_one_declared_owner():
    old = fixture()
    assert "control" not in old.to_dict()
    assert TheveninSpec.from_dict(old.to_dict()).to_dict() == old.to_dict()
    spec = replace(old, control=controller(initial_current_A=-0.3))
    restored = TheveninSpec.from_dict(json.loads(json.dumps(spec.to_dict(), allow_nan=False)))
    assert restored.to_dict() == spec.to_dict()
    assert restored.initial_state().coordinates[CONTROLLER_CURRENT] == -0.3
    assert [c.name for c in spec.coordinate_declarations].count(CONTROLLER_CURRENT) == 1
    legacy = scalar(0.2, "A")
    assert thevenin_control_program(legacy).control_mode == "current"
    assert thevenin_control_program(legacy.to_dict()).target == legacy
    explicit = program("cccv", 3.8, limit=-0.2)
    assert thevenin_control_program(json.loads(json.dumps(explicit.to_dict()))) == explicit


@pytest.mark.parametrize("defect", ("units", "tau", "branch", "singularity", "cccv", "unknown"))
def test_invalid_control_contracts_are_rejected_without_guessing(defect):
    with pytest.raises(ValueError):
        if defect == "units":
            TheveninControlProgram("power", scalar(1.0, "A"), current_branch_A=(0.0, 1.0))
        elif defect == "tau":
            controller(power_response_time_s=0.0)
        elif defect == "branch":
            program("power", 1.0)
        elif defect == "singularity":
            program("differential_resistance", 1.0, (-1.0, 1.0))
        elif defect == "cccv":
            program("cccv", 4.0, limit=0.1)
        else:
            TheveninControlProgram.from_dict({**program("current", 0.1).to_dict(), "typo": 1})


@pytest.mark.parametrize("target", (1.0, -1.0))
def test_signed_power_and_charge_work_are_the_same_port_solution(target):
    spec = constant_spec()
    control = program("power", target, (-2.0, 2.0))
    result = thevenin_system(spec, control).advance(spec.initial_state(), 2.0)
    fields = thevenin_fields(spec, result.final_state, control)
    expected = (3.6 - sqrt(3.6**2 - 4 * 0.1 * target)) / (2 * 0.1)
    assert result.outcome == "success"
    assert fields["current_A"] == pytest.approx(expected, abs=1e-10)
    assert fields["current_A"] * fields["voltage_V"] == pytest.approx(target, abs=1e-10)
    assert result.final_state.coordinates[WORK] == pytest.approx(2 * target, abs=1e-9)
    assert result.final_state.coordinates[CHARGE] == pytest.approx(2 * expected, abs=1e-9)


def test_two_power_roots_require_explicit_distinct_branches_and_fold_is_not_a_solution():
    spec = constant_spec(r0=1.0, maximum_current_A=4.0)
    roots = []
    for branch in ((0.0, 1.4), (2.1, 3.5)):
        control = program("power", 3.0, branch)
        fields = thevenin_fields(spec, spec.initial_state(), control)
        roots.append(fields["current_A"])
        assert fields["current_A"] * fields["voltage_V"] == pytest.approx(3.0, abs=1e-10)
    np.testing.assert_allclose(roots, ((3.6 - sqrt(0.96)) / 2, (3.6 + sqrt(0.96)) / 2), atol=1e-10)
    for target, branch, message in (
        (3.0, (0.0, 4.0), "multiple"),
        (3.4, (0.0, 1.4), "no_solution"),
        (3.24, (0.0, 4.0), "unresolved|singular|multiple"),
    ):
        with pytest.raises(ResidualDomainError, match=message):
            thevenin_fields(spec, spec.initial_state(), program("power", target, branch))


def test_current_dependent_ir_monotonicity_does_not_hide_two_power_roots():
    law = function(
        {"current_A": (-3.0, 0.0, 3.0)},
        lambda i: 0.1 + 0.4 * abs(i),
        input_units={"current_A": "A"},
    )
    spec = replace(constant_spec(maximum_current_A=3.0), series_resistance=law)
    assert law.positive_coordinate_product_derivative("current_A")
    with pytest.raises(ResidualDomainError, match="multiple"):
        thevenin_fields(spec, spec.initial_state(), program("power", 3.7, (0.0, 3.0)))
    expected = sorted(
        r.real for r in np.roots((0.4, 0.1, -3.6, 3.7)) if abs(r.imag) < 1e-10 and r > 0
    )
    actual = [
        thevenin_fields(spec, spec.initial_state(), program("power", 3.7, branch))["current_A"]
        for branch in ((0.0, 1.6), (1.7, 3.0))
    ]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_zero_series_resistance_keeps_power_and_load_well_defined_but_not_cv():
    spec = constant_spec(r0=0.0)
    for control, current in (
        (program("power", 0.72, (0.0, 2.0)), 0.2),
        (program("resistance", 18.0), 0.2),
    ):
        fields = thevenin_fields(spec, spec.initial_state(), control)
        assert fields["current_A"] == pytest.approx(current, abs=1e-11)
    with pytest.raises(ValueError, match="nonsingular"):
        thevenin_system(spec, scalar(3.6, "V"))


def test_dynamic_power_is_an_actual_slow_current_state_and_cc_freezes_without_reset():
    spec = constant_spec(r0=0.0, initial_current=0.05)
    control = program("differential_power", 0.72)
    result = thevenin_system(spec, control).advance(spec.initial_state(), 3.0)
    expected = 0.2 + (0.05 - 0.2) * exp(-0.036 * 3)
    fields = thevenin_fields(spec, result.final_state, control)
    assert fields["current_A"] == pytest.approx(expected, abs=1e-8)
    assert fields["controller_current_A"] == fields["current_A"]
    assert fields["current_A"] != pytest.approx(0.2)
    exact_charge = 0.2 * 3 + (0.05 - 0.2) * (1 - exp(-0.036 * 3)) / 0.036
    assert result.final_state.coordinates[CHARGE] == pytest.approx(exact_charge, abs=3e-8)
    parked = thevenin_system(spec, scalar(0.0, "A")).advance(result.final_state, 1.0).final_state
    assert (
        parked.coordinates[CONTROLLER_CURRENT] == result.final_state.coordinates[CONTROLLER_CURRENT]
    )
    resumed = thevenin_system(spec, control).advance(parked, 2.0).final_state
    assert resumed.coordinates[CONTROLLER_CURRENT] == pytest.approx(
        0.2 + (0.05 - 0.2) * exp(-0.036 * 5), abs=2e-8
    )


def test_upstream_differential_resistance_instability_stops_before_zero_singularity():
    spec = constant_spec(r0=0.0, initial_current=0.19)
    control = program("differential_resistance", 18.0, (0.02, 0.4))
    short = thevenin_system(spec, control).advance(spec.initial_state(), 0.1)
    assert short.final_state.coordinates[CONTROLLER_CURRENT] < 0.19  # Away from I*=.2, as upstream.
    stopped = thevenin_system(spec, control).advance(spec.initial_state(), 20.0)
    assert stopped.outcome == "stopped"
    assert stopped.stopping_event == "thevenin:minimum-control-branch"
    assert 0 < stopped.actual_duration_s < 20.0
    assert stopped.final_state.coordinates[CONTROLLER_CURRENT] == pytest.approx(0.02, abs=1e-10)


def test_cccv_retains_internal_windup_and_exact_source_initial_condition():
    spec = constant_spec(r0=0.0, initial_current=-0.2)
    control = program("cccv", 3.7, limit=-0.2)
    initial = spec.initial_state()
    result = thevenin_system(spec, control).advance(initial, 0.2)
    expected = -0.2 - 0.1 * (100 / 3600) * (1 - exp(-0.2 / (100 / 3600)))
    fields = thevenin_fields(spec, result.final_state, control)
    assert fields["controller_current_A"] == pytest.approx(expected, abs=1e-8)
    assert fields["controller_current_A"] < -0.2
    assert fields["current_A"] == pytest.approx(-0.2, abs=1e-12)
    assert fields["voltage_V"] == pytest.approx(3.6)
    assert result.final_state.coordinates[SOC] == pytest.approx(0.7004, abs=1e-10)
    assert initial.coordinates[CONTROLLER_CURRENT] == -0.2


def test_dynamic_current_cap_stops_at_partial_time_and_does_not_clip_state():
    spec = constant_spec(r0=0.0, initial_current=0.1, maximum_current_A=0.2)
    control = program("differential_power", 3.6)
    result = thevenin_system(spec, control).advance(spec.initial_state(), 10.0)
    expected_time = -log((1.0 - 0.2) / (1.0 - 0.1)) / 0.036
    assert result.outcome == "stopped"
    assert result.stopping_event == "thevenin:maximum-source-current"
    assert result.actual_duration_s == pytest.approx(expected_time, abs=3e-7)
    assert result.final_state.coordinates[CONTROLLER_CURRENT] == pytest.approx(0.2, abs=1e-12)
    assert result.final_state.coordinates[WORK] == pytest.approx(
        3.6 * result.final_state.coordinates[CHARGE], abs=1e-9
    )


def test_installed_frozen_controller_has_no_false_rest_impedance_mode():
    spec = replace(fixture(2), control=controller(initial_current_A=-0.7))
    state = spec.initial_state()
    result = thevenin_impedance(spec, (0.01, 1.0), state=state)
    plain = thevenin_impedance(replace(spec, control=None), (0.01, 1.0))
    assert result.success
    np.testing.assert_allclose(result.impedance_ohm, plain.impedance_ohm, atol=1e-11)
    assert state.coordinates[CONTROLLER_CURRENT] == -0.7


def test_dynamic_mode_requires_installed_coordinate_and_branch_state_is_validated():
    with pytest.raises(ValueError, match="installed"):
        thevenin_system(fixture(), program("differential_power", 1.0))
    spec = constant_spec(initial_current=0.0)
    with pytest.raises(ValueError, match="outside its declared"):
        thevenin_system(spec, program("differential_resistance", 18.0, (0.01, 0.4))).advance(
            spec.initial_state(), 0.0
        )
    spec = constant_spec(initial_current=.3, maximum_current_A=.2)
    with pytest.raises(ValueError, match="actual current exceeds"):
        thevenin_system(spec, program("differential_power", 1.)).advance(spec.initial_state(), 0.)
