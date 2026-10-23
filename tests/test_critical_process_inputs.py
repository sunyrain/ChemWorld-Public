import json
from dataclasses import replace

import pytest

from chemworld.data.constants import ConstantDatabase
from chemworld.data.scientific_inputs import InputSource, InputValue
from chemworld.data.volume import VolumeDatabase
from chemworld.physchem import critical_compressibility, critical_correlation_input
from chemworld.physchem.critical_properties import estimate_critical_property
from chemworld.physchem.data_conflicts import DataSourceProvenance
from chemworld.physchem.property_inputs import PropertyInputCard
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec
from chemworld.physchem.transport_devices import LiquidTransportProperties
from chemworld.world.transport_world import synthetic_transport_properties


def source():
    return InputSource(
        DataSourceProvenance(
            "derived-test", 0, "Explicit synthetic derivation", "1", "synthetic test"
        ),
        "synthetic",
    )


def derived(component):
    tc = estimate_critical_property(
        component, method="Tb_Tc_relationship", parameters={"Tb": 360.0, "fit": "Perry8E"}
    )
    vc = estimate_critical_property(component, method="Mersmann_Kind", parameters={})
    pc = estimate_critical_property(component, method="Ihmels", parameters={"Tc": tc, "Vc": vc})
    return {
        "Tc": tc,
        "Pc": pc,
        "Vc": vc,
        "omega": InputValue(0.2, "dimensionless", None, "Explicit synthetic omega"),
    }


@pytest.mark.parametrize(
    "family,method,keys",
    [
        ("volume", "COSTALD", ("Tc", "Vc", "omega")),
        ("virial", "Abbott", ("Tc", "Pc", "omega")),
    ],
)
def test_derived_parameters_reach_original_card_with_nested_sources(family, method, keys):
    component = ComponentSpec("decane", "C10H22")
    inputs = derived(component)
    card = critical_correlation_input(
        component,
        family=family,
        method=method,
        parameters={k: inputs[k] for k in keys},
        source=source(),
        temperature_range_K=(300, 500),
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    curve = restored.package.correlations[0]
    assert curve.metadata["parameter_sources"]["Tc"]["method"] == "Tb_Tc_relationship"
    assert card.reference is None and restored.to_dict() == card.to_dict()
    assert restored.package.evaluate(curve.property_id, temperature_K=400).value != 0
    with pytest.raises(ValueError, match="identity"):
        critical_correlation_input(
            ComponentSpec("other", "C10H22"),
            family=family,
            method=method,
            parameters={k: inputs[k] for k in keys},
            source=source(),
            temperature_range_K=(300, 500),
        )


def test_zc_definition_is_derived_and_not_a_source_constant():
    component = ComponentSpec("decane", "C10H22")
    inputs = derived(component)
    zc = estimate_critical_property(
        component, method="Zc_definition", parameters={k: inputs[k] for k in ("Tc", "Pc", "Vc")}
    )
    assert zc.value == critical_compressibility(*(inputs[k].value for k in ("Tc", "Pc", "Vc")))
    assert zc.unit == "dimensionless" and zc.kind == "pure_estimate"
    card = critical_correlation_input(
        component,
        family="volume",
        method="Rackett",
        parameters={"Tc": inputs["Tc"], "Pc": inputs["Pc"], "Zc": zc},
        source=source(),
        temperature_range_K=(300, 500),
    )
    assert card.package.correlations[0].metadata["parameter_sources"]["Zc"] == zc.to_dict()


def test_xml_critical_aliases_preserve_units_identity_and_original_fields(tmp_path):
    path = tmp_path / "Misc/ChemSep8.32.xml"
    path.parent.mkdir()
    path.write_text(
        """<compounds><compound><CompoundID value="synthetic ethanol"/>
        <CAS value="64-17-5"/><CriticalTemperature units="K" value="600"/>
        <CriticalPressure units="Pa" value="4000000"/>
        <CriticalVolume units="m3/kmol" value="0.3"/>
        <CriticalCompressibility units="_" value="0.25"/>
        <AcentricityFactor units="_" value="0.2"/></compound></compounds>""",
        encoding="utf-8",
    )
    record = ConstantDatabase(tmp_path).record("64-17-5", method="CHEMSEP_XML")
    component = ComponentSpec("ethanol", "C2H6O", cas_number="64-17-5")
    values = {key: record.value(key) for key in ("Tc", "Pc", "Vc", "Zc", "omega")}
    estimate = estimate_critical_property(
        component, method="Zc_definition",
        parameters={key: values[key] for key in ("Tc", "Pc", "Vc")},
    )
    assert estimate.value == pytest.approx(4e6 * 0.0003 / (8.31446261815324 * 600))
    card = critical_correlation_input(
        component, family="volume", method="COSTALD", source=source(),
        parameters={key: values[key] for key in ("Tc", "Vc", "omega")},
        temperature_range_K=(300, 400),
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    curve = restored.package.correlations[0]
    assert curve.coefficients["Vc"] == pytest.approx(0.0003)
    assert curve.metadata["parameter_sources"]["Vc"]["field_id"] == "CriticalVolume"
    assert restored.to_dict() == card.to_dict()
    with pytest.raises(ValueError, match="field"):
        estimate_critical_property(
            component, method="Zc_definition",
            parameters={"Tc": values["Tc"], "Pc": values["Pc"], "Vc": values["Zc"]},
        )
    with pytest.raises(ValueError, match="identity"):
        estimate_critical_property(
            ComponentSpec("water", "H2O", cas_number="7732-18-5"),
            method="Zc_definition", parameters={key: values[key] for key in ("Tc", "Pc", "Vc")},
        )


def test_virial_residual_domain_is_independent_of_ideal_datum():
    from tests.test_virial_consumers import thermal_model

    from chemworld.physchem import VirialGasThermodynamics, virial_gas_batch_system
    from chemworld.physchem.coupled_system import CoupledState, HeatFlux
    from chemworld.physchem.solver_backend import ODESolverPolicy

    component = ComponentSpec("gas", "C7H8", default_phase="gas")
    card = critical_correlation_input(
        component,
        family="virial",
        method="Abbott",
        source=source(),
        parameters={"Tc": 600, "Pc": 4e6, "omega": 0.2},
        temperature_range_K=(850, 950),
    )
    ideal = thermal_model()
    ideal = replace(
        ideal,
        heat_capacity=replace(ideal.heat_capacity, validity_ranges={"temperature": (200, 1000)}),
    )
    model = VirialGasThermodynamics(card.package, ideal)
    state = CoupledState({"gas": 2}, 900, 20)
    result = virial_gas_batch_system(
        model,
        heat_flux=lambda _: HeatFlux(heater_W=5),
        policy=ODESolverPolicy("declared-virial", method="DOP853", rtol=1e-10, atol=1e-12),
    ).advance(state, 60)
    assert 900 < result.final_state.temperature_K < 910
    assert abs(result.energy_balance_residual_J) < 1e-4
    with pytest.raises(ValueError, match="outside validity"):
        model.validate_state(replace(state, temperature_K=800))


def test_volume_source_supplement_retains_critical_estimate(tmp_path):
    db = VolumeDatabase(tmp_path)
    path = tmp_path / db.source("COSTALD_PARAMETERS").relative_path
    path.parent.mkdir(parents=True)
    path.write_text(
        "CAS\tChemical\tomega_SRK\tVchar\tZ_RA\n124-18-5\tDecane\t.2\t.0003\t.27\n",
        encoding="utf-8",
    )
    component = ComponentSpec("decane", "C10H22", cas_number="124-18-5")
    tc = derived(component)["Tc"]
    candidate = db.volumes(component.cas_number, method="COSTALD_PARAMETERS")[0]
    card = db.property_input(
        component.cas_number,
        component,
        method="COSTALD_PARAMETERS",
        curve_id=candidate.curve_id,
        parameters={"Tc": tc},
        temperature_bounds_K=(300, 500),
    )
    restored = PropertyInputCard.from_json(json.dumps(card.to_dict()))
    assert restored.package.correlations[0].metadata["parameter_sources"]["Tc"] == tc.to_dict()
    assert restored.package.correlations[0].coefficients["Vc"] == 0.0003


def mixing(method, pressure=1e6):
    base = synthetic_transport_properties()
    parameters = (
        {
            k: {"Tcs": 600 + i * 50, "Pcs": 4e6 + i * 1e6, "Zrs": 0.25 + i * 0.02}
            for i, k in enumerate(base.component_ids)
        }
        if method == "Rackett_mixture"
        else {
            k: {"Tcs": 600 + i * 50, "Vcs": 0.0002 + i * 0.0001, "omegas": 0.2 + i * 0.1}
            for i, k in enumerate(base.component_ids)
        }
    )
    return replace(
        base,
        pressure_Pa=pressure,
        caloric_references={
            k: CaloricReference(phase="liquid", pressure_Pa=pressure) for k in base.component_ids
        },
        volume_mixing_method=method,
        volume_mixing_parameters=parameters,
        volume_pressure_parameters={"Psat": 1e5} if method.endswith("compressed") else {},
        volume_mixing_note="Explicit synthetic mixing inputs; additive calorics",
    )


@pytest.mark.parametrize(
    "method", ["Rackett_mixture", "COSTALD_mixture", "COSTALD_mixture_compressed"]
)
def test_runtime_volume_uses_current_composition_and_survives_json(method):
    properties = mixing(method)
    restored = LiquidTransportProperties.from_dict(json.loads(json.dumps(properties.to_dict())))
    for amounts in ({"water": 1.0, "ethanol": 0.0}, {"water": 0.2, "ethanol": 0.8}):
        assert restored.evaluate(amounts, 300) == properties.evaluate(amounts, 300)
    assert restored.evaluate({"water": 1.0, "ethanol": 0.0}, 300)[2] != pytest.approx(
        restored.evaluate({"water": 0.2, "ethanol": 0.8}, 300)[2]
    )
    assert restored.evaluate({"water": 0.2, "ethanol": 0.8}, 310)[2] != pytest.approx(
        restored.evaluate({"water": 0.2, "ethanol": 0.8}, 300)[2]
    )
    with pytest.raises(ValueError, match="temperature"):
        restored.evaluate({"water": 0.2, "ethanol": 0.8}, 400)


def test_compressed_volume_responds_to_pressure_and_declarations_reject_ambiguity():
    a, b = (mixing("COSTALD_mixture_compressed", p) for p in (1e6, 1e7))
    assert (
        b.evaluate({"water": 0.2, "ethanol": 0.8}, 300)[2]
        < a.evaluate({"water": 0.2, "ethanol": 0.8}, 300)[2]
    )
    for kwargs in (
        {"volume_mixing_note": ""},
        {"volume_mixing_parameters": {}},
        {"volume_pressure_parameters": {"P": 1e7}},
        {"volume_mixing_method": "Amgat"},
    ):
        with pytest.raises(ValueError):
            replace(a, **kwargs)
    with pytest.raises(TypeError):
        a.volume_mixing_parameters["water"]["Tcs"] = 700


def test_bridge_rejects_missing_source_units_and_cross_family_options():
    component = ComponentSpec("test", "C7H8")
    p = {"Tc": 600, "Pc": 4e6, "omega": 0.2}
    for changes in (
        {"source": None},
        {"family": "unknown"},
        {"output": "density"},
        {"pressure_range_Pa": (1e5, 1e6)},
        {"parameters": {**p, "Tc": InputValue(600, "Pa", None, "Wrong units")}},
    ):
        with pytest.raises((TypeError, ValueError)):
            critical_correlation_input(
                component,
                **{
                    **{
                        "family": "virial",
                        "method": "Abbott",
                        "parameters": p,
                        "source": source(),
                        "temperature_range_K": (300, 500),
                    },
                    **changes,
                },
            )
