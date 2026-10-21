"""Formal configuration is immutable data; the native entity owns every live value."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from chemworld.foundation import WorldState
from chemworld.physchem.sbml_system import SBMLObservable, SBMLSystemSpec, parameter_observable
from test_sbml import model


def spec(**kwargs):
    return SBMLSystemSpec(model(), maximum_step_s=.02, **kwargs)


def test_declaration_roundtrip_readout_units_and_no_live_clock_or_coordinate_values():
    original = spec(observables=(SBMLObservable("amount", "A"),
        SBMLObservable("concentration", "B"), SBMLObservable("volume", "c2")))
    payload = json.loads(json.dumps(original.to_dict(), allow_nan=False))
    assert SBMLSystemSpec.from_dict(payload).to_dict() == payload
    assert original.readout(original.frame.initial_values, 0.) == {
        "amount_A": .01, "concentration_B": 0., "volume_c2": .02}
    assert [o.unit for o in original.observables] == ["mol", "mol/L", "L"]
    assert "coordinates" not in payload and "time_s" not in payload
    with pytest.raises(FrozenInstanceError):
        original.maximum_events = 1


@pytest.mark.parametrize("mutation", ["step", "unknown_observable", "duplicate", "scale", "budget"])
def test_invalid_execution_profile_is_refused_before_world_registration(mutation):
    payload = spec().to_dict()
    if mutation == "step":
        payload["maximum_step_s"] = None
    elif mutation == "unknown_observable":
        payload["observables"] = [{"kind": "amount", "source_id": "unknown"}]
    elif mutation == "duplicate":
        payload["observables"] = [{"kind": "amount", "source_id": "A"}]*2
    elif mutation == "scale":
        payload["coordinate_scales"] = {"N:A": -1.}
    else:
        payload["maximum_pending_events"] = 0
    with pytest.raises(ValueError):
        SBMLSystemSpec.from_dict(payload)


def test_formal_workspace_has_explicitly_absent_heat_capacity_and_roundtrips():
    state = WorldState({}, 0., 298.15, 101325., "formal", "model_workspace",
                       liquid_heat_capacity_J_L_K=None)
    assert WorldState.from_dict(state.to_dict()).to_dict() == state.to_dict()
    assert state.liquid_heat_capacity_J_L_K is None
    with pytest.raises(ValueError, match="Liquid heat capacity"):
        replace(state, phase="liquid")
    for updates in ({"volume_L": .01}, {"species_amounts": {"fake": 1.}},
                    {"liquid_heat_capacity_J_L_K": 4180.}):
        with pytest.raises(ValueError, match="formal workspaces"):
            replace(state, **updates)


def test_explicit_controls_and_solver_use_v2_while_uncontrolled_bdf_keeps_v1():
    assert spec().to_dict()["format"] == "chemworld_sbml_system_v1"
    controlled = spec(controlled_parameters=("k", "p"), solver_method="LSODA")
    payload = json.loads(json.dumps(controlled.to_dict()))
    assert payload["format"] == "chemworld_sbml_system_v2"
    restored = SBMLSystemSpec.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.continuous_system().policy.method == "LSODA"
    assert restored.frame.initial_values["P:k"] == .2


@pytest.mark.parametrize("kwargs", [{"controlled_parameters": ("q",)},
    {"controlled_parameters": ("k", "k")}, {"controlled_parameters": "k"},
    {"controlled_parameters": ("unknown",)}, {"solver_method": "CVODE"},
    {"solver_method": "LSODA", "numerical_form": "residual"}])
def test_control_and_solver_policy_refuses_undeclared_basis_or_algorithm_substitution(kwargs):
    with pytest.raises(ValueError):
        spec(**kwargs)


def test_parameter_readout_dimensions_and_closed_codec_remain_explicit():
    observable = parameter_observable(model(), "k")
    assert observable.unit == "Hz"
    original = spec(observables=(observable, SBMLObservable("reaction_rate", "J")))
    assert original.to_dict()["format"] == "chemworld_sbml_system_v2"
    assert SBMLSystemSpec.from_dict(original.to_dict()).to_dict() == original.to_dict()
    with pytest.raises(ValueError, match="source dimensions"):
        spec(observables=(SBMLObservable("parameter", "k", "mol"),))
    with pytest.raises(ValueError, match="source units"):
        parameter_observable(model(), "unknown")
