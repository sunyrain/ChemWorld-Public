"""Safe expression contracts and independent elementary analytic identities."""

import math
from dataclasses import FrozenInstanceError

import pytest

from chemworld.physchem.analytic_functions import smooth_function
from chemworld.physchem.caloric_math import derivative_values
from chemworld.physchem.mathml import MATHML_NS, SBML_AVOGADRO, SBML_NS, MathExpression


def expression(body):
    return MathExpression(f'<math xmlns="{MATHML_NS}" xmlns:sbml="{SBML_NS}">{body}</math>')


def test_piecewise_short_circuit_preserves_the_active_chain_derivatives():
    model = expression("""<piecewise><piece><apply><power/><ci>x</ci><cn>3</cn></apply>
        <apply><gt/><ci>x</ci><cn>0</cn></apply></piece>
        <otherwise><apply><ln/><cn>-1</cn></apply></otherwise></piecewise>""")
    assert model.symbols == ("x",)
    assert derivative_values(lambda x: model.evaluate({"x": x}), 2, order=3) == (8, 12, 12, 6)
    with pytest.raises(ValueError):
        model.evaluate({"x": -1})
    with pytest.raises(FrozenInstanceError):
        model.xml = "changed"


@pytest.mark.parametrize(("name", "body", "expected"), [
    ("plus", "", 0), ("times", "", 1), ("and", "", True), ("or", "", False),
    ("xor", "", False), ("times", "<cn>3</cn>", 3),
    ("eq", "<cn>3</cn><cn>3</cn><cn>3</cn>", True),
    ("gt", "<cn>3</cn><cn>2</cn><cn>1</cn>", True),
    ("xor", "<true/><true/><true/>", True),
    ("implies", "<false/><apply><ln/><cn>-1</cn></apply>", True),
    ("log", "<logbase><cn>2</cn></logbase><cn>32</cn>", 5),
    ("root", "<degree><cn>3</cn></degree><cn>-8</cn>", -2),
    ("log", "<cn>1000</cn>", 3), ("root", "<cn>9</cn>", 3),
    ("factorial", "<cn>5</cn>", 120), ("min", "<cn>5</cn>", 5),
])
def test_scalar_operations_and_zero_one_many_argument_semantics(name, body, expected):
    result = expression(f"<apply><{name}/>{body}</apply>").evaluate({})
    assert result == pytest.approx(expected)


def test_original_literal_units_special_numbers_and_sbml_avogadro_are_distinct_from_si():
    rational = expression('<cn type="rational" sbml:units="mole">3<sep/>4</cn>')
    assert rational.number_units == ("mole",) and rational.evaluate({}) == .75
    assert expression('<cn type="e-notation">1.2<sep/>-3</cn>').evaluate({}) == .0012
    special = expression('<csymbol definitionURL="http://www.sbml.org/sbml/symbols/avogadro" '
                         'encoding="text">avogadro</csymbol>')
    assert special.evaluate({}) == SBML_AVOGADRO == 6.02214179e23
    assert SBML_AVOGADRO != 6.02214076e23
    time = expression('<csymbol definitionURL="http://www.sbml.org/sbml/symbols/time">t</csymbol>')
    assert time.evaluate({"$time": 12, "time": 99}) == 12


def test_lexical_functions_do_not_capture_caller_scope_and_recursion_fails():
    f = expression('<lambda><bvar><ci>x</ci></bvar><apply><times/><ci>x</ci><ci>x</ci>'
                   '</apply></lambda>')
    call = expression('<apply><ci>f</ci><ci>A</ci></apply>')
    assert call.function_names == ("f",)
    assert call.evaluate({"A": 3, "x": 70}, functions={"f": f}) == 9
    with pytest.raises(ValueError, match="lexical"):
        expression('<lambda><bvar><ci>x</ci></bvar><apply><plus/><ci>x</ci><ci>global</ci>'
                   '</apply></lambda>')
    recursive = expression('<lambda><bvar><ci>x</ci></bvar><apply><ci>f</ci><ci>x</ci>'
                           '</apply></lambda>')
    with pytest.raises(ValueError, match="recursive"):
        call.evaluate({"A": 3}, functions={"f": recursive})
    with pytest.raises(ValueError, match="argument count"):
        expression('<apply><ci>f</ci></apply>').evaluate({}, functions={"f": f})


@pytest.mark.parametrize("body", [
    '<apply><__import__/ ><cn>2</cn></apply>', '<cn>NaN</cn>',
    '<cn type="rational">1<sep/>0</cn>', '<cn type="e-notation">1<sep/>999999</cn>',
    '<apply><divide/><cn>1</cn></apply>', '<apply><plus bogus="1"/><cn>2</cn></apply>',
    '<apply><selector/><ci>x</ci></apply>', '<ci>A.B</ci>',
    '<cn bad="1">2</cn>', '<apply><plus/><cn>2</cn></apply>ignored',
    '<apply><csymbol definitionURL="http://www.sbml.org/sbml/symbols/delay"/>'
    '<ci>x</ci><cn>1</cn></apply>', '<lambda><bvar><ci>x</ci></bvar><bvar><ci>x</ci>'
    '</bvar><ci>x</ci></lambda>',
])
def test_unsupported_malformed_or_nonfinite_literal_rejects_whole_expression(body):
    with pytest.raises(ValueError):
        expression(body)


def test_xml_bombs_namespace_depth_and_active_nonfinite_outputs_fail():
    with pytest.raises(ValueError, match="DTD"):
        MathExpression('<!DOCTYPE math [<!ENTITY x "payload">]><math>&x;</math>')
    with pytest.raises(ValueError, match="namespace"):
        MathExpression('<math><cn>1</cn></math>')
    with pytest.raises(ValueError, match="depth"):
        expression('<apply><minus/>'*70+'<cn>1</cn>'+'</apply>'*70)
    for body in ('<infinity/>', '<notanumber/>', '<apply><divide/><cn>1</cn><cn>0</cn></apply>'):
        with pytest.raises(ValueError):
            expression(body).evaluate({})


def test_xml_node_limit_is_applied_during_parsing():
    from chemworld.data.xml_inputs import bounded_xml

    with pytest.raises(ValueError, match="node/depth bounds"):
        bounded_xml("<root>"+"<child/>"*10+"</root>", max_nodes=5)


def test_shared_chain_functions_through_third_order_without_numerical_differences():
    x = .4
    assert derivative_values(lambda t: smooth_function("sin", t), x, order=3) == pytest.approx(
        (math.sin(x), math.cos(x), -math.sin(x), -math.cos(x)))
    # asin(sin(x)) is x throughout this open principal interval.
    assert derivative_values(lambda t: smooth_function("arcsin", smooth_function("sin", t)),
                             x, order=3) == pytest.approx((x, 1, 0, 0), abs=1e-14)
    model = expression('<apply><power/><ci>x</ci><ci>x</ci></apply>')
    value, first, second = derivative_values(lambda x: model.evaluate({"x": x}), 2, order=2)
    assert (value, first, second) == pytest.approx(
        (4, 4*(1+math.log(2)), 4*((1+math.log(2))**2+.5)))


def test_principal_arccot_has_a_branch_cut_at_zero_not_the_continuous_atan2_convention():
    assert smooth_function("arccot", -.7) == math.atan(1/-.7)
    assert smooth_function("arccot", 0.) == math.pi/2
    with pytest.raises(ValueError, match="branch cut"):
        derivative_values(lambda x: smooth_function("arccot", x), 0., order=1)


@pytest.mark.parametrize("body", [
    '<cn>1_000</cn>', '<cn>\uff11\uff12</cn>', '<cn type="e-notation">1_0<sep/>2</cn>',
    '<cn type="rational">1<sep>ignored</sep>2</cn>',
    '<piecewise><piece>ignored<cn>1</cn><true/></piece></piecewise>',
    '<apply><root/><degree>ignored<cn>2</cn></degree><cn>4</cn></apply>',
    '<lambda><bvar>ignored<ci>x</ci></bvar><ci>x</ci></lambda>',
])
def test_python_numeric_extensions_and_ignored_structural_text_are_not_mathml(body):
    with pytest.raises(ValueError):
        expression(body)


@pytest.mark.parametrize("body", [
    '<apply><floor/><ci>x</ci></apply>', '<apply><ceiling/><ci>x</ci></apply>',
    '<apply><abs/><ci>x</ci></apply>', '<apply><min/><ci>x</ci><cn>0</cn></apply>',
    '<apply><max/><ci>x</ci><cn>0</cn></apply>',
])
def test_no_analytic_derivative_is_invented_at_a_nonsmooth_cut(body):
    compiled = expression(body)
    with pytest.raises(ValueError):
        derivative_values(lambda x: compiled.evaluate({"x": x}), 0., order=2)


def test_nonrecursive_exponential_function_expansion_has_a_finite_evaluation_budget():
    functions = {"f0": expression('<lambda><bvar><ci>x</ci></bvar><ci>x</ci></lambda>')}
    for index in range(1, 19):
        call = f'<apply><ci>f{index-1}</ci><ci>x</ci></apply>'
        functions[f"f{index}"] = expression('<lambda><bvar><ci>x</ci></bvar><apply><plus/>'
                                            +call+call+'</apply></lambda>')
    compiled = expression('<apply><ci>f18</ci><cn>1</cn></apply>')
    with pytest.raises(ValueError, match="evaluation budget"):
        compiled.evaluate({}, functions=functions)
