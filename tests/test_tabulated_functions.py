"""Sourced interpolation and analytic chain derivatives, no battery qualification."""

import json
from copy import deepcopy
from dataclasses import replace
from itertools import product
from math import prod

import pytest

from chemworld.data.scientific_inputs import (
    InputValue,
    QuantityColumn,
    ScientificRow,
    ScientificTable,
)
from chemworld.examples.scientific_inputs import synthetic_source
from chemworld.foundation.dimensions import dimension_definition
from chemworld.foundation.units import convert_value
from chemworld.physchem.tabulated_functions import TabulatedScalar


def function(grids, equation, *, output_unit="Ohm", input_units=None):
    names = tuple(grids)
    units = input_units or dict.fromkeys(names, "dimensionless")
    source = synthetic_source()
    columns = tuple(QuantityColumn(name, units[name], "parameter", (-1e6, 1e6), "declared axis")
                    for name in names)
    columns += (QuantityColumn("response", output_unit, "parameter", (-1e9, 1e9),
                               "self-authored interpolation fixture"),)
    rows = []
    for index, point in enumerate(product(*grids.values())):
        values = {name: InputValue(value, units[name], None, "Input uncertainty not supplied")
                  for name, value in zip(names, point, strict=True)}
        values["response"] = InputValue(equation(*point), output_unit, .02,
                                        "Synthetic standard uncertainty, not a prediction interval")
        rows.append(ScientificRow(str(index), source.provenance.source_id, "function", str(index),
                                  "parameters", values))
    table = ScientificTable("parameter-function", "parameters", columns, (source,), tuple(rows))
    return TabulatedScalar(table, tuple((name, name, units[name]) for name in names),
                           "response", output_unit)


@pytest.mark.parametrize("dimensions", (0, 1, 2, 3))
def test_full_cartesian_values_derivatives_and_vertex_bounds(dimensions):
    grids = dict.fromkeys(("z", "T", "I")[:dimensions], (-2., 0., 3.))

    def expression(*p):
        return 2 + sum((i+1)*x for i, x in enumerate(p)) + (prod(p) if len(p) > 1 else 0)

    law = function(grids, expression)
    point = dict(zip(grids, [.3, .7, -.4], strict=False))
    assert law.value(point) == pytest.approx(expression(*point.values()), abs=1e-13)
    gradient = law.gradient(point)
    for index, key in enumerate(grids):
        expected = index+1 + (prod(x for k, x in point.items() if k != key)
                              if dimensions > 1 else 0)
        assert gradient[key] == pytest.approx(expected, abs=1e-13)
    assert law.value_range[0] <= law.value(point) <= law.value_range[1]
    assert len(law.row_ids) == 3**dimensions
    with pytest.raises(TypeError):
        gradient["extra"] = 1


def test_import_preserves_uncertainty_sources_and_temperature_offset():
    original = function({"T": (10., 30.), "I": (-2., 2.)}, lambda T, current: 1+T/100+current/20,
                        input_units={"T": "degC", "I": "A"}, output_unit="kOhm")
    law = replace(original, axes=(("temperature_K", "T", "K"), ("current_A", "I", "A")),
                  output_unit="Ohm")
    point = {"temperature_K": 293.15, "current_A": .5}
    assert law.value(point) == pytest.approx(1225)
    assert law.gradient(point) == pytest.approx({"temperature_K": 10., "current_A": 50.})
    saved = json.dumps(law.to_dict(), sort_keys=True)
    restored = TabulatedScalar.from_json(saved)
    assert restored.to_dict() == law.to_dict()
    assert restored.value(point) == law.value(point)
    assert restored.table.rows[0].values["response"].uncertainty == .02
    assert restored.table.rows[0].values["response"].unit == "kOhm"
    assert restored.table.sources == original.table.sources
    assert law.domain["temperature_K"] == (283.15, 303.15)


def test_public_domain_and_trial_extension_do_not_clip_inputs():
    law = function({"z": (0., 1.)}, lambda z: 2+3*z)
    with pytest.raises(ValueError, match="outside"):
        law.value({"z": 2})
    assert law.value({"z": 2}, trial=True) == 8
    assert law.gradient({"z": 2}, trial=True)["z"] == 3
    assert law.value({"z": 1}) == 5
    with pytest.raises(ValueError, match="boundary"):
        law.gradient({"z": 1})


def test_nonsmooth_knot_is_rejected_even_for_trial_linearization():
    law = function({"I": (-1., 0., 1.)}, lambda current: abs(current)+1)
    assert law.value({"I": 0}) == 1
    for trial in (False, True):
        with pytest.raises(ValueError, match="unique derivative"):
            law.gradient({"I": 0}, trial=trial)
    smooth = function({"I": (-1., 0., 1.)}, lambda current: 2*current+3)
    assert smooth.gradient({"I": 0})["I"] == 2


def test_small_real_kink_is_not_smoothed_into_a_unique_derivative():
    law = function({"I": (-1., 0., 1.)}, lambda current: 1 + 2**-50 * abs(current))
    with pytest.raises(ValueError, match="unique derivative"):
        law.gradient({"I": 0})


@pytest.mark.parametrize("point", ({}, {"z": True}, {"z": float("nan")}, {"z": "0.5"},
                                    {"z": .5, "hidden": 1}))
def test_bad_function_arguments_are_not_coerced_or_ignored(point):
    law = function({"z": (0., 1.)}, lambda z: z)
    with pytest.raises(ValueError):
        law.value(point)


@pytest.mark.parametrize("defect", ("missing", "duplicate", "units", "axis_repeat", "same_output",
                                   "unknown_row", "duplicate_row", "unknown_version", "extra"))
def test_ambiguous_or_malformed_tables_are_rejected(defect):
    law = function({"z": (0., 1.), "I": (-1., 1.)}, lambda z, current: z+current)
    payload = deepcopy(law.to_dict())
    if defect == "missing":
        payload["row_ids"].pop()
    elif defect == "duplicate":
        payload["table"]["rows"][1]["values"]["I"] = payload["table"]["rows"][0]["values"]["I"]
    elif defect == "units":
        payload["axes"][0][2] = "K"
    elif defect == "axis_repeat":
        payload["axes"][1][0] = "z"
    elif defect == "same_output":
        payload["output_column"] = "z"
    elif defect == "unknown_row":
        payload["row_ids"][0] = "missing"
    elif defect == "duplicate_row":
        payload["row_ids"].append(payload["row_ids"][0])
    elif defect == "unknown_version":
        payload["schema_version"] = "other"
    else:
        payload["python_callback"] = "do_not_execute"
    with pytest.raises(ValueError):
        TabulatedScalar.from_dict(payload)


def test_source_rows_are_explicitly_selected_without_priority_guessing():
    law = function({"z": (0., 1., 2.)}, lambda z: z*z)
    selected = replace(law, row_ids=("0", "2"))
    assert selected.value({"z": 1}) == 2
    assert law.value({"z": 1}) == 1
    assert selected.row_ids == ("0", "2")


@pytest.mark.parametrize("field,value", (("axes", "z"), ("axes", ["abc"]),
                                        ("row_ids", "012"), ("row_ids", [None])))
def test_sequence_bindings_are_not_coerced_from_strings(field, value):
    payload = function({"z": (0., 1.)}, lambda z: z).to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        TabulatedScalar.from_dict(payload)


def test_trial_extension_requires_an_explicit_boolean():
    law = function({"z": (0., 1.)}, lambda z: z)
    with pytest.raises(ValueError, match="Boolean"):
        law.value({"z": 2}, trial="false")


@pytest.mark.parametrize("coefficient,expected", ((.5, True), (-1.5, False)))
def test_current_drop_monotonicity_is_certified_on_all_cell_sides(coefficient, expected):
    law = function({"T": (0., 1.), "I": (-1., 0., 1.)},
                   lambda T, current: 2+T+coefficient*current)
    assert law.positive_coordinate_product_derivative("I") is expected
    assert law.positive_coordinate_product_derivative("absent_axis")


def test_zero_resistance_has_no_invertible_current_drop():
    law = function({}, lambda: 0.)
    assert not law.positive_coordinate_product_derivative("I")


def test_entropic_coefficients_use_temperature_intervals_not_absolute_offsets():
    assert convert_value(1., "mV/K", "V/K") == .001
    value = InputValue(2., "mV/K", .1, "Declared source uncertainty").to("V/K")
    assert value.value == .002 and value.uncertainty == pytest.approx(.0001)
    assert dimension_definition("electric_potential_per_temperature").vector == (
        dimension_definition("electric_potential").vector
        / dimension_definition("temperature").vector)
