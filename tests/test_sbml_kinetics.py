"""Native source bases and concentration-rate chain rules, without another solver."""

import math
from dataclasses import replace

import pytest

from chemworld.physchem.caloric_math import FirstDerivative
from chemworld.physchem.coupled_system import (
    CoupledState,
    CoupledSystem,
    HeatFlux,
    KineticOnlyThermodynamics,
)
from chemworld.physchem.sbml_kinetics import SBMLKinetics, sbml_continuous_system
from chemworld.physchem.stoichiometric_rates import stoichiometric_amount_rates
from reference.run_sbml_input_block import cases
from test_sbml import MODEL, model


def rate_rule(text, variable, value):
    return text.replace('</listOfRules>', f'''<rateRule variable="{variable}"><math
        xmlns="http://www.w3.org/1998/Math/MathML"><cn>{value}</cn></math></rateRule>
        </listOfRules>''')


def concentration_rule(text):
    text = text.replace('hasOnlySubstanceUnits="false" boundaryCondition="false"',
                        'hasOnlySubstanceUnits="false" boundaryCondition="true"', 1)
    return rate_rule(text, "A", .1)


def test_source_compartments_amounts_local_parameters_and_extent_units_share_one_basis():
    frame = SBMLKinetics(model())
    assert frame.initial_values == pytest.approx({
        "N:A": .01, "N:B": 0, "V:c1": .01, "V:c2": .02, "P:p": 3})
    scope = frame.context(frame.initial_values, 0)
    assert scope["A"] == 1 and scope["k"] == .2 and scope["v"] == .2
    assert scope["J"] == .003
    rates = frame.rates(frame.initial_values, 0)
    assert rates.coordinate_rates == pytest.approx({
        "N:A": -.003, "N:B": .003, "V:c1": 0, "V:c2": 0, "P:p": .1})
    assert rates.reaction_extents_mol_s == {"J": .003}
    assert frame.quantities(frame.initial_values, 0) == ({"A": .01, "B": 0},
                                                      {"c1": .01, "c2": .02})


def test_mmol_ml_minute_input_converts_only_at_the_boundary():
    text = list(cases())[5][3]
    frame = SBMLKinetics(model(text))
    assert frame.initial_values["N:A"] == pytest.approx(1e-5)
    assert frame.initial_values["V:c1"] == pytest.approx(1e-5)
    scope = frame.context(frame.initial_values, 60)
    assert scope["$time"] == 1 and scope["A"] == 1
    rates = frame.rates(frame.initial_values, 60)
    assert rates.coordinate_rates["N:A"] == pytest.approx(-5e-8)
    assert rates.coordinate_rates["N:B"] == pytest.approx(5e-8)
    assert rates.coordinate_rates["P:p"] == pytest.approx(.1/60)
    assert rates.reaction_extents_mol_s["J"] == pytest.approx(5e-8)


def test_reaction_amount_rate_and_volume_rate_produce_dilution_in_the_source_view():
    frame = SBMLKinetics(model(rate_rule(MODEL, "c1", .002)))
    rates = frame.rates(frame.initial_values, 0).coordinate_rates
    assert rates["N:A"] == -.003 and rates["V:c1"] == .002
    jets = {key: FirstDerivative(value, rates[key]) for key, value in frame.initial_values.items()}
    view = frame.context(jets, FirstDerivative(0, 1))
    assert view["A"].slope == pytest.approx(-.5)  # -.003/.01 - .01*.002/.01**2


@pytest.mark.parametrize("index", [0, 5])
def test_source_concentration_rate_rule_restores_the_volume_chain_term(index):
    text = concentration_rule(rate_rule(list(cases())[index][3], "c1", .002))
    frame = SBMLKinetics(model(text))
    rates = frame.rates(frame.initial_values, 0).coordinate_rates
    scale = 1. if index == 0 else .001/60
    assert rates["N:A"] == pytest.approx(.003*scale)
    assert rates["N:B"] == pytest.approx(.003*scale)
    jets = {key: FirstDerivative(value, rates[key]) for key, value in frame.initial_values.items()}
    view = frame.context(jets, FirstDerivative(0, 1))
    assert view["A"].slope == pytest.approx(.1 if index == 0 else .1/60)


def test_smooth_assignment_volume_is_derived_and_its_time_derivative_enters_the_same_rule():
    text = concentration_rule(MODEL).replace('</listOfRules>', '''<assignmentRule variable="c1">
        <math xmlns="http://www.w3.org/1998/Math/MathML"><apply><plus/><cn>.01</cn><apply>
        <times/><cn>.002</cn><csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">
        time</csymbol></apply></apply></math></assignmentRule></listOfRules>''')
    frame = SBMLKinetics(model(text))
    assert "V:c1" not in frame.initial_values
    assert frame.rates(frame.initial_values, 0).coordinate_rates["N:A"] == pytest.approx(.003)
    shifted = SBMLKinetics(model(text), initial_model_time=10)
    assert shifted.initial_values["N:A"] == pytest.approx(.03)
    assert shifted.context(shifted.initial_values, 0)["$time"] == 10
    assert shifted.rates(shifted.initial_values, 0).coordinate_rates["N:A"] == pytest.approx(.005)


def test_initial_volume_assignment_and_constant_amount_source_view():
    frame = SBMLKinetics(model(list(cases())[11][3]))
    assert frame.initial_values["V:c1"] == .015 and frame.initial_values["N:A"] == .015
    text = rate_rule(list(cases())[3][3], "c1", .002)
    frame = SBMLKinetics(model(text))
    values = dict(frame.initial_values)
    values["V:c1"] *= 2
    assert frame.context(values, 1)["A"] == .5
    assert frame.rates(values, 1).coordinate_rates["N:A"] == 0


def test_missing_volume_constraint_and_invalid_constant_reactant_are_not_silently_repaired():
    with pytest.raises(ValueError, match="uninitialized"):
        SBMLKinetics(model(MODEL.replace(' size="0.01"', '')))
    invalid = list(cases())[3][3].replace(
        'hasOnlySubstanceUnits="false" boundaryCondition="true"',
        'hasOnlySubstanceUnits="false" boundaryCondition="false"')
    with pytest.raises(ValueError, match="constant non-boundary"):
        SBMLKinetics(model(invalid))
    frame = SBMLKinetics(model())
    values = {**frame.initial_values, "P:p": -1}
    with pytest.raises(ValueError, match="constraint"):
        frame.validate_values(values, 0)


def test_common_stoichiometric_sum_preserves_jets_and_the_historical_product_grouping():
    rates = stoichiometric_amount_rates(("A", "B"), (("r", {"A": -2., "B": 1.}),),
                                        {"r": FirstDerivative(3., 1.)}, rate_scale=.1)
    assert (rates["A"].value, rates["A"].slope) == pytest.approx((-.6, -.2))
    assert (rates["B"].value, rates["B"].slope) == pytest.approx((.3, .1))


@pytest.mark.parametrize("form", ["ode", "residual"])
def test_shared_continuous_segment_before_first_event_has_no_fabricated_energy(form):
    frame = SBMLKinetics(model())
    system = sbml_continuous_system(frame, numerical_form=form)
    initial = CoupledState({}, 298.15, .03, coordinates=frame.initial_values)
    result = system.advance(initial, .1)
    expected_a = .01*math.exp(-.3*.1)
    assert result.actual_duration_s == pytest.approx(.1)
    assert result.final_state.coordinates["N:A"] == pytest.approx(expected_a, abs=2e-10)
    assert result.final_state.coordinates["N:B"] == pytest.approx(.01-expected_a, abs=2e-10)
    assert result.final_state.coordinates["P:p"] == pytest.approx(3.01)
    assert result.energy_balance_residual_J is None
    assert result.solver_diagnostic["energy_scope"] == "not_declared_formal_kinetics"
    assert result.final_state.amounts_mol == {}
    assert result.final_state.temperature_K == initial.temperature_K
    with pytest.raises(ValueError, match="no declared heat capacity"):
        system.thermodynamics.heat_capacity_j_k(initial)


def test_formal_segment_cannot_declare_a_thermostat_heat_or_chemical_port():
    with pytest.raises(ValueError, match="fixed external temperature"):
        CoupledSystem(KineticOnlyThermodynamics())
    frame = SBMLKinetics(model())
    initial = CoupledState({}, 298.15, .03, coordinates=frame.initial_values)
    system = replace(sbml_continuous_system(frame), heat_flux=lambda _: HeatFlux(heater_W=1.))
    with pytest.raises(ValueError, match="undeclared heat"):
        system.advance(initial, .1)
