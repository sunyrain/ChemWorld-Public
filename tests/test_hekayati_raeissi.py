"""Derived critical properties retain native input identity and model choice."""

import json

import pytest

from chemworld.data.constants import ConstantDatabase, ConstantFieldNotFoundError
from chemworld.data.scientific_inputs import InputValue
from chemworld.physchem import hekayati_raeissi_critical
from chemworld.physchem.critical_properties import (
    CriticalEstimate,
    eos_component_from_critical_inputs,
    estimate_critical_property,
)
from chemworld.physchem.eos import CubicEOSSpec, evaluate_cubic_eos
from chemworld.physchem.specs import ComponentSpec

COMPONENT = ComponentSpec("toluene", "C7H8", cas_number="108-88-3")
METHOD = "Hekayati_Raeissi"


def supplied(value, unit):
    return InputValue(value, unit, None, "Explicit input, uncertainty not established")


def test_published_example_and_distinct_volume_models():
    values = hekayati_raeissi_critical(MW=92.13842, V_sat=.00010686, Pc=4108000.)
    assert values == pytest.approx((599.7965819136947, 4108000., .000314909150453723),
                                  rel=3e-13)
    explicit = hekayati_raeissi_critical(MW=92.13842, Vc=values[2], Pc=4108000.)
    assert explicit[0] != pytest.approx(values[0], rel=1e-4)
    both = hekayati_raeissi_critical(MW=92.13842, Vc=values[2],
                                    V_sat=.0002, Pc=4108000.)
    assert both == explicit
    assert hekayati_raeissi_critical(MW=92., Vc=.0003, Tc=600., Pc=4e6) == (600., 4e6, .0003)


def test_multiple_outputs_require_selection_and_known_values_are_not_estimates():
    params = {"MW": 92.13842, "V_sat": .00010686, "Pc": 4108000.}
    with pytest.raises(ValueError, match="select output_property"):
        estimate_critical_property(COMPONENT, method=METHOD, parameters=params)
    for target in ("Tc", "Vc"):
        report = estimate_critical_property(COMPONENT, method=METHOD, parameters=params,
                                            output_property=target)
        assert report.property_id == target
        assert report.inputs["estimated_properties"] == ["Tc", "Vc"]
        assert report.inputs["model_basis"] == "V_sat_at_298.15_K"
        assert report.inputs["saturation_temperature_K"] == 298.15
        assert "uncertainty not established" in report.input_value().uncertainty_note
    for target in ("Pc", "MW", "unknown"):
        with pytest.raises(ValueError, match="missing"):
            estimate_critical_property(COMPONENT, method=METHOD, parameters=params,
                                        output_property=target)
    with pytest.raises(ValueError, match="select output_property"):
        estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "Tc": 600., "Vc": .0003})
    one = estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "Tc": 600.})
    assert one.property_id == "Vc"


def test_source_bearing_units_preserved_through_json_and_native_eos(tmp_path):
    db = ConstantDatabase(tmp_path)
    source = db.source("HEOS")
    path = tmp_path / source.relative_path
    path.parent.mkdir(parents=True)
    path.write_text("CAS\tname\tMW\tPc\tomega\n108-88-3\tToluene\t92.13842\t4108000\t.2\n",
                    encoding="utf-8")
    mw, pc, omega = (db.value(COMPONENT.cas_number, key, method="HEOS")
                     for key in ("MW", "Pc", "omega"))
    params = {"MW": mw, "Pc": pc, "V_sat": supplied(.10686, "L/mol"),
              "saturation_temperature_K": supplied(25., "degC")}
    report = estimate_critical_property(COMPONENT, method=METHOD, parameters=params,
                                        output_property="Tc")
    recovered = CriticalEstimate.from_dict(json.loads(json.dumps(report.to_dict())))
    assert recovered == report
    assert report.inputs["MW"]["source"]["source_id"] == "chemicals-constants:HEOS"
    assert report.inputs["V_sat"]["unit"] == "L/mol"
    assert report.inputs["saturation_temperature_K"]["unit"] == "degC"
    assert report.quantity().value == pytest.approx(599.7965819136947)
    spec = eos_component_from_critical_inputs(COMPONENT, Tc=recovered, Pc=pc, omega=omega)
    state = evaluate_cubic_eos(CubicEOSSpec("estimated", "peng_robinson", (spec,)),
                               {COMPONENT.identifier: 1}, temperature_K=700, pressure_Pa=1e6)
    assert state.compressibility_factor > 0 and state.molar_volume_m3_mol > 0
    with pytest.raises(ConstantFieldNotFoundError):
        db.value(COMPONENT.cas_number, "Tc", method="HEOS")
    with pytest.raises(ValueError, match="identity"):
        estimate_critical_property(ComponentSpec("water", "H2O", cas_number="7732-18-5"),
                                    method=METHOD, parameters=params, output_property="Tc")
    with pytest.raises(ValueError, match="field"):
        estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "MW": pc}, output_property="Tc")


def test_provided_vc_choice_and_missing_or_wrong_inputs_are_explicit():
    params = {"MW": supplied(.09213842, "kg/mol"), "Pc": supplied(41.08, "bar"),
              "V_sat": supplied(.10686, "L/mol"), "Vc": supplied(.316, "L/mol")}
    report = estimate_critical_property(COMPONENT, method=METHOD, parameters=params)
    assert report.inputs["model_basis"] == "supplied_Vc"
    assert report.inputs["unused_inputs"] == ["V_sat", "saturation_temperature_K"]
    assert report.inputs["supplied_Tc_Pc_policy"] == "preserve_without_reconciliation"
    mass = InputValue(.09213842, "kg/mol", .00001, "Explicit uncertainty")
    assert mass.to("g/mol").uncertainty == .01
    assert mass.to("g/mol").to("kg/mol") == mass
    with pytest.raises(ValueError, match="missing"):
        estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "MW": InputValue(
                                        None, "g/mol", None, "Unavailable", "missing")})
    with pytest.raises(ValueError):
        estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "MW": supplied(92., "K")})
    with pytest.raises(ValueError, match=r"298\.15"):
        estimate_critical_property(COMPONENT, method=METHOD,
                                    parameters={**params, "saturation_temperature_K": 300.})
    with pytest.raises(ValueError, match="not Pc"):
        estimate_critical_property(COMPONENT, method="Tb_Tc_relationship",
                                    parameters={"Tb": 300, "fit": "Perry8E"}, output_property="Pc")


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), 0, -1, "92", 1e308])
def test_molecular_weight_rejects_invalid_numeric_domain(value):
    with pytest.raises(ValueError):
        hekayati_raeissi_critical(MW=value, V_sat=.00010686, Tc=600.)
