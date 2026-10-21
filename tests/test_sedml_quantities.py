"""Explicit source selectors consume public native channels without truth access."""

import pytest

from chemworld.data.sedml_declarations import records
from chemworld.data.sedml_quantities import model_quantity
from reference.run_sbml_trajectory_block import cases
from test_sbml import model
from test_sedml_declarations import SEDML, document


def variable(attributes):
    text = SEDML.replace('symbol="urn:sedml:symbol:time"', attributes)
    generator = records(document(text).declaration, "listOfDataGenerators")[0]
    return records(generator, "listOfVariables")[0]


def target(group, kind, identity, symbol=""):
    return variable(f'''target="/s:sbml/s:model/s:{group}/s:{kind}[@id='{identity}']" {symbol}''')


def test_source_time_origin_and_minute_basis_do_not_change_the_native_clock():
    parsed = model(list(cases())[1][3])
    quantity = model_quantity(parsed, variable('symbol="KISAO:0000832"'))
    assert quantity.observable is None and quantity.source_unit == "minute"
    assert quantity.value({}, time_s=90., initial_model_time=5.) == 6.5


@pytest.mark.parametrize("index,amount,concentration,volume,rate", [
    (0, .015, 1., .015, .003), (1, 15., 1., 15., 3.),
])
def test_amount_concentration_volume_and_rate_use_each_source_unit_basis(
        index, amount, concentration, volume, rate):
    parsed = model(list(cases())[index][3])
    channels = {"amount_A": .015, "concentration_A": 1., "volume_c1": .015,
                "reaction_rate_J": .003/(60. if index else 1.)}
    selections = [
        (target("listOfSpecies", "species", "A", 'symbol="KISAO:0000836"'), amount),
        (target("listOfSpecies", "species", "A", 'symbol="KISAO:0000838"'), concentration),
        (target("listOfCompartments", "compartment", "c1"), volume),
        (target("listOfReactions", "reaction", "J"), rate),
    ]
    for selected, expected in selections:
        quantity = model_quantity(parsed, selected)
        assert quantity.value(channels, time_s=0.) == pytest.approx(expected)


def test_parameter_si_observation_is_converted_back_to_original_source_units_once():
    parsed = model(list(cases())[1][3])
    quantity = model_quantity(parsed, target("listOfParameters", "parameter", "k"))
    assert quantity.observable.unit == "Hz"
    assert quantity.value({"parameter_k": .2/60.}, time_s=0.) == pytest.approx(.2)


@pytest.mark.parametrize("attributes", [
    'symbol="urn:unknown"',
    '''target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']" symbol="KISAO:0000832"''',
    '''target="/s:sbml/s:model/s:listOfParameters/s:parameter[@id='k']" symbol="KISAO:0000836"''',
    '''target="/s:sbml/s:model/s:listOfSpecies/s:species[@id='A']/@initialConcentration"''',
])
def test_unsupported_or_ambiguous_output_quantity_is_not_silently_reinterpreted(attributes):
    with pytest.raises(ValueError):
        model_quantity(model(), variable(attributes))
