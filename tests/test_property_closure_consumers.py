"""Declared energy pairs, reference changes and pure closure consumers."""

import json
from dataclasses import replace

import pytest

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.enthalpy import caloric_temperature
from chemworld.physchem.property_packages import ComponentPropertyPackage, evaluate_property_state
from chemworld.physchem.property_reports import CaloricReference
from chemworld.physchem.specs import ComponentSpec, PropertyCorrelation
from chemworld.physchem.transport_devices import FiniteTransportSpec, LiquidTransportProperties
from chemworld.runtime import ChemWorldRuntime
from chemworld.world.transport_world import synthetic_transport_properties, transport_world_spec


def cv_package(phase):
    return ComponentPropertyPackage(
        ComponentSpec("A", "H2O"),
        (PropertyCorrelation(
            "declared-Cv", "cp_polynomial", {"a": 30, "b": 0.1}, {"temperature": "K"},
            "J/(mol*K)", property_id="isochoric_heat_capacity",
            validity_ranges={"temperature": (280, 400)}, metadata={"phase": phase},
            source_note="Synthetic variable-Cv closure fixture",
        ),),
    )


@pytest.mark.parametrize("phase", ("liquid", "solid"))
def test_condensed_cv_u_request_needs_only_its_own_declared_data(phase):
    package = cv_package(phase)
    reference = CaloricReference(
        phase=phase, energy_basis="internal_energy", temperature_K=300, energy_J_mol=-500
    )
    arguments = {
        "pressure_Pa": 101325, "phase": phase, "reference": reference,
        "quantities": ("internal_energy", "cv", "dU_dT_V"),
    }
    report = package.evaluate_state(temperature_K=350, **arguments)
    expected = -500 + 30 * 50 + 0.05 * (350**2 - 300**2)
    assert report.properties["internal_energy"].value == pytest.approx(expected, abs=1e-10)
    assert report.properties["cv"].value == 65
    assert report.properties["dU_dT_V"].value == 65
    high = package.evaluate_state(temperature_K=350.001, **arguments)
    low = package.evaluate_state(temperature_K=349.999, **arguments)
    delta = (
        high.properties["internal_energy"].value - low.properties["internal_energy"].value
    ) / 0.002
    assert delta == pytest.approx(65, rel=1e-9)
    for quantity in ("cp", "enthalpy", "entropy"):
        with pytest.raises(ValueError, match="additional closure"):
            package.evaluate_state(
                temperature_K=350, **{**arguments, "quantities": (quantity,)}
            )
    with pytest.raises(ValueError):
        package.evaluate_state(temperature_K=450, **arguments)


def test_mixed_energy_bases_do_not_get_averaged_as_the_same_quantity():
    package = cv_package("liquid")
    other = replace(package, component=ComponentSpec("B", "C2H6O"))
    with pytest.raises(ValueError, match="cannot mix H and U"):
        evaluate_property_state(
            (package, other), {"A": 1, "B": 1}, temperature_K=310, pressure_Pa=101325,
            phase="liquid", quantities=("internal_energy",),
            references={"A": CaloricReference(phase="liquid", energy_basis="internal_energy")},
        )


@pytest.mark.parametrize("temperature", (280.0, 310.0, 400.0))
def test_caloric_inverse_uses_the_same_variable_cv_reference_and_keeps_inputs(temperature):
    package = cv_package("liquid")
    reference = CaloricReference(phase="liquid", energy_basis="internal_energy")
    before = package.to_dict()

    def energy(t):
        return 2 * package.evaluate_state(
            temperature_K=t, pressure_Pa=101325, phase="liquid", reference=reference,
            quantities=("internal_energy",),
        ).properties["internal_energy"].value

    result = caloric_temperature(
        energy, energy(temperature), (280, 400), energy_basis="internal_energy"
    )
    assert result == pytest.approx(temperature, abs=1e-9)
    assert package.to_dict() == before
    with pytest.raises(ValueError, match="outside"):
        caloric_temperature(energy, energy(400) + 1, (280, 400), energy_basis="internal_energy")


@pytest.mark.parametrize("value", (None, float("nan"), float("inf")))
def test_unavailable_calorics_cannot_produce_a_temperature(value):
    with pytest.raises(ValueError, match="unavailable or nonfinite"):
        caloric_temperature(lambda _: value, 0, (280, 400), energy_basis="enthalpy")


def test_vle_actual_phase_inventory_uses_one_latent_heat_and_reference():
    properties = synthetic_vle_properties()
    liquid = {"light": 0.3, "heavy": 0.1}
    vapor = {"light": 0.05, "heavy": 0.2}
    before = liquid.copy(), vapor.copy(), properties.to_dict()
    hl, hv = properties.inventory_enthalpies(350, 1e5, liquid, vapor)
    assert hl == pytest.approx(
        0.3 * (80 * (350 - 298.15) - 30000)
        + 0.1 * (100 * (350 - 298.15) - 40000), abs=1e-9
    )
    assert hv == pytest.approx(
        (0.05 * 80 + 0.2 * 100) * (350 - 298.15), abs=1e-9
    )
    shifted = replace(properties, gas_references={
        key: replace(ref, energy_J_mol=ref.energy_J_mol + 700)
        for key, ref in properties.gas_references.items()
    })
    other = shifted.inventory_enthalpies(350, 1e5, liquid, vapor)
    assert other[0] - hl == pytest.approx(700 * sum(liquid.values()), abs=1e-9)
    assert other[1] - hv == pytest.approx(700 * sum(vapor.values()), abs=1e-9)
    assert (liquid, vapor, properties.to_dict()) == before


def test_transport_roundtrip_keeps_explicit_fluid_and_apparatus_datums():
    properties = synthetic_transport_properties()
    with pytest.raises(ValueError, match="matching liquid H references"):
        replace(properties, pressure_Pa=2e5)
    references = {
        key: CaloricReference(
            phase="liquid", pressure_Pa=2e5, temperature_K=300, energy_J_mol=-500
        ) for key in properties.component_ids
    }
    properties = replace(properties, pressure_Pa=2e5, caloric_references=references)
    spec = FiniteTransportSpec(properties, support_reference_temperature_K=290)
    restored = FiniteTransportSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    amounts = {"water": 0.8, "ethanol": 0.2}
    expected = (0.8 * 75 + 0.2 * 110) * 15 - 500 + 5 * 25
    assert restored.enthalpy(0, amounts, 315) == pytest.approx(expected, abs=1e-9)
    assert restored.at_ph(0, amounts, expected) == pytest.approx(315, abs=1e-9)
    with pytest.raises(ValueError, match="temperature domain"):
        properties.enthalpy(dict.fromkeys(amounts, 0), 100)
    legacy = synthetic_transport_properties().to_dict()
    legacy.pop("caloric_references")
    assert LiquidTransportProperties.from_dict(legacy).references == (
        synthetic_transport_properties().references
    )


def test_reference_shift_keeps_actual_pump_temperature_and_resource_response():
    properties = synthetic_transport_properties()
    shifted = replace(properties, caloric_references={
        key: replace(ref, energy_J_mol=500) for key, ref in properties.references.items()
    })
    results = []
    for current in (properties, shifted):
        session = ChemWorldRuntime(composition=transport_world_spec(FiniteTransportSpec(current)))
        try:
            session.reset(seed=24)
            for action in (
                {"operation": "set_flow_rate", "flow_rate_mL_min": 20},
                {"operation": "run_flow", "duration_s": 1},
            ):
                result = session.execute(action)
                assert result.info["transaction_status"] == "committed"
            results.append((session._state.temperature_K, session._state.ledger.to_dict()))
        finally:
            session.close()
    assert results[0][0] == pytest.approx(results[1][0], abs=1e-9)
    assert results[0][1] == pytest.approx(results[1][1], abs=1e-9)


def test_nonthermal_battery_has_no_energy_certificate():
    from chemworld.physchem.battery_models import BatterySPMSpec, advance_battery
    from chemworld.physchem.reactor_shared import ScalarControlProgram

    spec = BatterySPMSpec()
    result = advance_battery(spec, spec.initial_state(), ScalarControlProgram(((0, 0.01),), "A"), 1)
    assert result.energy_balance_residual_J is None


@pytest.mark.parametrize("domain", ("ssa", "aqueous"))
def test_isothermal_domains_do_not_claim_unmodelled_energy_closure(domain):
    from chemworld.world.aqueous_world import aqueous_world_spec
    from chemworld.world.biochemical_worlds import biochemical_world

    world = aqueous_world_spec() if domain == "aqueous" else biochemical_world("reversible", seed=7)
    session = ChemWorldRuntime(composition=world)
    try:
        session.reset(seed=7)
        result = session.execute({"operation": "wait", "duration_s": 1})
        assert result.info["transaction_status"] == "committed"
        diagnostic = result.info["outcome_diagnostics"]
        assert diagnostic.get("energy_balance_residual_J") is None
        if domain == "aqueous":
            assert "caloric_closure" in diagnostic["not_covered"]
        else:
            assert "not modelled" in diagnostic["energy_scope"]
    finally:
        session.close()
