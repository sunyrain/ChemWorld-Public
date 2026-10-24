"""State caloric identities, request boundaries and real consumer integration."""

from dataclasses import replace
from math import log

import numpy as np
import pytest
from scipy.optimize import brentq

from chemworld.physchem.coupled_system import CoupledState
from chemworld.physchem.curated_properties import curated_property_package
from chemworld.physchem.enthalpy import caloric_polynomial_change, sensible_enthalpy_change
from chemworld.physchem.property_packages import (
    PropertyPackageThermodynamics,
    SpeciesCaloricModel,
    evaluate_caloric_correlation,
    evaluate_property_state,
)
from chemworld.physchem.property_reports import R_J_PER_MOL_K, CaloricReference
from chemworld.physchem.specs import PropertyCorrelation


def polynomial(*, basis="enthalpy", coefficients=None, unit="J/(mol*K)"):
    return PropertyCorrelation(
        correlation_id="declared-synthetic-caloric", equation_id="cp_polynomial",
        property_id="isochoric_heat_capacity" if basis == "internal_energy"
        else "ideal_gas_heat_capacity",
        coefficients={"a": 30, "b": .01} if coefficients is None else coefficients,
        input_units={"temperature": "K"}, output_unit=unit,
        validity_ranges={"temperature": (250, 800)}, source_note="Synthetic integration fixture",
    )


def test_caloric_pair_derivatives_and_entropy_share_an_explicit_reference():
    correlation = polynomial()
    reference = CaloricReference(phase="gas", temperature_K=300, energy_J_mol=100,
                                 entropy_J_mol_K=200, source_note="Synthetic reference")
    def at(temperature, pressure=2e5):
        return evaluate_caloric_correlation(
            correlation, reference=reference, temperature_K=temperature, pressure_Pa=pressure,
            quantities=("cp", "cv", "enthalpy", "internal_energy", "entropy",
                        "chemical_potential", "dH_dT_P", "dU_dT_V"))
    values = at(350)
    expected_h = 100 + 30 * 50 + .005 * (350 ** 2 - 300 ** 2)
    expected_s = 200 + 30 * log(350 / 300) + .01 * 50 - R_J_PER_MOL_K * log(2e5 / 101325)
    assert values["enthalpy"].value == pytest.approx(expected_h)
    assert values["entropy"].value == pytest.approx(expected_s)
    assert values["enthalpy"].value - values["internal_energy"].value == pytest.approx(
        R_J_PER_MOL_K * 350)
    epsilon = 1e-3
    for energy, capacity in (("enthalpy", "cp"), ("internal_energy", "cv")):
        derivative = (at(350 + epsilon)[energy].value - at(350 - epsilon)[energy].value) / (
            2 * epsilon)
        assert derivative == pytest.approx(values[capacity].value, rel=1e-8)
    derivative = (at(350 + epsilon)["chemical_potential"].value
                  - at(350 - epsilon)["chemical_potential"].value) / (2 * epsilon)
    assert derivative == pytest.approx(-expected_s, rel=1e-8)
    assert values["dH_dT_P"].value == values["cp"].value
    assert values["dU_dT_V"].value == values["cv"].value


def test_phase_free_cv_inventory_does_not_invent_pv_or_an_entropy_datum():
    correlation = polynomial(basis="internal_energy", coefficients={"a": 0})
    reference = CaloricReference(energy_basis="internal_energy", temperature_K=350,
                                 energy_J_mol=-100, source_note="Mathematical zero-Cv inventory")
    values = evaluate_caloric_correlation(correlation, reference=reference, temperature_K=400,
                                          quantities=("cv", "internal_energy"))
    assert values["cv"].value == 0 and values["internal_energy"].value == -100
    for quantity in ("cp", "enthalpy", "chemical_potential"):
        with pytest.raises(ValueError):
            evaluate_caloric_correlation(correlation, reference=reference, temperature_K=400,
                                          quantities=(quantity,))
    with pytest.raises(ValueError, match="Cv integrates"):
        sensible_enthalpy_change(correlation, initial_temperature_K=350, final_temperature_K=400)


def test_caloric_units_and_reference_basis_are_explicit():
    reference = CaloricReference(phase="gas")
    original = polynomial()
    scaled = polynomial(coefficients={"a": .03, "b": .00001}, unit="kJ/(mol*K)")
    def evaluate(correlation):
        return evaluate_caloric_correlation(correlation, reference=reference, temperature_K=500,
                                             quantities=("cp", "enthalpy", "internal_energy"))
    for quantity, value in evaluate(original).items():
        assert evaluate(scaled)[quantity].value == pytest.approx(value.value, rel=1e-12)
    with pytest.raises(ValueError, match="bases"):
        evaluate_caloric_correlation(original, reference=replace(reference,
                                      energy_basis="internal_energy"), temperature_K=400,
                                      quantities=("cp",))
    with pytest.raises(ValueError, match="entropy datum"):
        evaluate_caloric_correlation(original, reference=reference, temperature_K=400,
                                      quantities=("chemical_potential",))


def test_caloric_interval_checks_real_polynomial_extrema_not_only_samples():
    # All five former quarter-interval samples are positive, but Cp(351)<0.
    correlation = polynomial(coefficients={"a": 351 ** 2 - .01, "b": -702, "c": 1})
    with pytest.raises(ValueError, match="positive"):
        caloric_polynomial_change(correlation, initial_temperature_K=300, final_temperature_K=400)


@pytest.mark.parametrize("policy", ["warn", "ignore"])
def test_permissive_caloric_evaluation_retains_invalid_domain_metadata(policy):
    values = evaluate_caloric_correlation(polynomial(), reference=CaloricReference(phase="gas"),
                                          temperature_K=900, quantities=("cp",),
                                          validity_policy=policy)
    assert not values["cp"].valid and values["cp"].warnings


def test_each_requested_property_retains_its_own_domain_and_requested_order():
    package = curated_property_package("water")
    reference = CaloricReference(phase="gas", temperature_K=40)
    report = package.evaluate_state(temperature_K=350, pressure_Pa=1000, phase="gas",
                                     quantities=("cp", "enthalpy", "density"),
                                     reference=reference, validity_policy="warn")
    assert list(report.properties) == ["cp", "enthalpy", "density"]
    assert report.properties["cp"].valid and report.properties["density"].valid
    assert not report.properties["enthalpy"].valid and not report.valid
    for value in report.properties.values():
        assert value.quantity().to(value.unit).value == value.value


def test_state_requests_reuse_component_aliases_and_do_not_require_unrequested_data():
    package = curated_property_package("water")
    report = evaluate_property_state(
        (package,), {"dihydrogen_oxide": 1, "7732-18-5": 2}, temperature_K=400,
        pressure_Pa=1000, phase="gas", quantities=("cp", "density"))
    assert dict(report.component_amounts_mol) == {"water": 3}
    assert set(report.properties) == {"cp", "density"}
    assert report.valid
    for phase, quantity in (("liquid", "cp"), ("gas", "viscosity"),
                             ("gas", "thermal_conductivity"), ("gas", "chemical_potential")):
        with pytest.raises(ValueError):
            package.evaluate_state(temperature_K=400, pressure_Pa=1000, phase=phase,
                                     quantities=(quantity,))


def test_equipment_uses_variable_caloric_data_and_independent_integrated_energy():
    from chemworld.physchem.heat_transfer_units import (
        HeatTransferEquipmentSpec,
        equipment_heat_transfer,
    )

    correlation = polynomial(basis="internal_energy", coefficients={"a": 10, "b": .05})
    caloric = PropertyPackageThermodynamics({"A": SpeciesCaloricModel(
        correlation, CaloricReference(temperature_K=350, energy_basis="internal_energy"))},
        support_heat_capacity_J_K=5)
    state = CoupledState({"A": 2}, 350, 1)
    equipment = HeatTransferEquipmentSpec("property-heater", surface_type="jacket", area_m2=1,
                                         clean_overall_u_W_m2_K=10,
                                         provenance_id="synthetic-variable-caloric-fixture")
    result = equipment_heat_transfer(equipment, process_temperature_K=350,
                                     utility_temperature_K=400, duration_s=10,
                                     caloric_model=caloric, caloric_state=state)
    # Integral of C_total(T)/(UA*(Tutility-T)), with C_total=25+0.1T.
    expected = brentq(lambda t: (65 * log(50 / (400 - t)) - .1 * (t - 350)) / 10 - 10,
                      350, 399.999)
    assert result.final_temperature_K == pytest.approx(expected, abs=2e-5)
    delta = result.final_temperature_K - 350
    independent_energy = 25 * delta + .05 * (result.final_temperature_K ** 2 - 350 ** 2)
    assert result.sensible_energy_J == pytest.approx(independent_energy, abs=1e-8)
    assert result.heat_energy_J == pytest.approx(independent_energy, rel=1e-6, abs=1e-8)
    assert abs(result.energy_balance_residual_J) / max(1, abs(independent_energy)) < 1e-6
    assert result.model_id == "equipment_caloric_heat_transfer_v1"
    with pytest.raises(ValueError, match="one energy basis"):
        equipment_heat_transfer(equipment, process_temperature_K=350, utility_temperature_K=400,
                                 duration_s=10, process_heat_capacity_J_K=60,
                                 caloric_model=caloric, caloric_state=state)


def variable_native_spec():
    from chemworld.examples.state_property_research import variable_caloric_world

    return variable_caloric_world()


def test_common_solver_rejects_an_accepted_state_outside_caloric_domain():
    from chemworld.physchem.coupled_system import CoupledSystem, HeatFlux

    caloric = PropertyPackageThermodynamics({"A": SpeciesCaloricModel(
        polynomial(basis="internal_energy", coefficients={"a": 30}),
        CaloricReference(temperature_K=350, energy_basis="internal_energy"))})
    system = CoupledSystem(caloric, heat_flux=lambda state: HeatFlux(heater_W=100))
    initial = CoupledState({"A": 1}, 790, 1)
    with pytest.raises(RuntimeError, match="outside the declared model"):
        system.advance(initial, 10)
    assert initial.temperature_K == 790


def test_native_compilation_rejects_feed_outside_its_target_caloric_domain():
    from chemworld.world.composition import WorldCompositionError, compile_world_composition

    spec = variable_native_spec()
    states = dict(spec.physics.initial_states)
    source = spec.physics.feed_sources["CO-feed"]
    states[source] = states[source].replace(temperature_K=900)
    spec = replace(spec, physics=replace(spec.physics, initial_states=states))
    with pytest.raises(WorldCompositionError, match="caloric data domain"):
        compile_world_composition(spec)


def test_native_reactor_and_heater_share_current_temperature_and_composition(monkeypatch):
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.surface_world import surface_world_spec

    spec = variable_native_spec()
    calls = []
    original = PropertyPackageThermodynamics.heat_capacity_j_k

    def capacity(self, state):
        calls.append((state.temperature_K, dict(state.amounts_mol)))
        return original(self, state)

    monkeypatch.setattr(PropertyPackageThermodynamics, "heat_capacity_j_k", capacity)
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=12)
        baseline = surface_world_spec(include_surface=False, include_bulk=True)
        constants = baseline.physics.models_of('thermal')["reactor"].thermodynamics

        def energy(state, support=constants.support_heat_capacity_J_K):
            t = state.temperature_K
            return support * (t - 350) + sum(
                amount * (constants.species[key].u_ref_J_mol
                          + constants.species[key].cv_J_mol_K
                          * (.5 * (t - 350) + (t ** 2 - 350 ** 2) / 1400))
                for key, amount in state.species_amounts.items())

        before = energy(session._state)
        receipt_h = 0
        for component, species, amount in (("CO-feed", "CO", 6e-5), ("O2-feed", "O2", 3e-5)):
            # Same authored U offset, independently integrated variable Cv and gas PV term.
            datum = constants.species[species]
            source = session._state.inactive_vessels[spec.physics.feed_sources[component]]
            source_temperature = source.temperature_K
            receipt_h += amount * (datum.u_ref_J_mol + datum.cv_J_mol_K * (
                .5 * (source_temperature - 350) + (source_temperature ** 2 - 350 ** 2) / 1400)
                + R_J_PER_MOL_K * source_temperature)
            result = session.execute({"operation": "add_component", "component": component,
                                      "amount_mol": amount})
            assert result.info["transaction_status"] == "committed"
        assert energy(session._state) - before == pytest.approx(receipt_h, abs=1e-8)
        before, ledger = energy(session._state), session._state.ledger
        result = session.execute({"operation": "heat", "target_temperature_K": 380,
                                  "duration_s": 20, "stirring_speed_rpm": 600})
        assert result.info["transaction_status"] == "committed"
        after_ledger = session._state.ledger
        incoming = (after_ledger.energy_jacket_J - ledger.energy_jacket_J
                    - after_ledger.heat_loss_J + ledger.heat_loss_J)
        assert energy(session._state) - before == pytest.approx(incoming, abs=1e-6)
        assert len({round(t, 4) for t, _ in calls}) > 5
        assert any(values.get("CO2", 0) > 0 for _, values in calls)
        assert session._state.species_amounts["CO2"] > 0
        receiver_support = baseline.physics.models_of('thermal')[
            "receiver"].thermodynamics.support_heat_capacity_J_K
        total_before = energy(session._state) + energy(
            session._state.inactive_vessels["receiver"], receiver_support)
        routed = session.execute({"operation": "route_material",
                                  "connection": "reactor-to-receiver",
                                  "transfer_fraction": .2, "mixing": 1})
        assert routed.info["transaction_status"] == "committed"
        total_after = energy(session._state) + energy(
            session._state.inactive_vessels["receiver"], receiver_support)
        assert total_after == pytest.approx(total_before, abs=1e-7)
    finally:
        session.close()


def test_variable_caloric_world_serializes_restores_and_replays_without_reparameterizing():
    import json

    from chemworld.runtime import ChemWorldRuntime

    spec = variable_native_spec()
    restored_spec = type(spec).from_dict(json.loads(json.dumps(spec.to_dict())))
    session = ChemWorldRuntime(composition=restored_spec)
    session.reset(seed=12)
    session.execute({"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5})
    continued = ChemWorldRuntime.from_snapshot(json.loads(json.dumps(session.snapshot().to_dict())))
    try:
        result = session.execute({"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600})
        other = continued.execute({"operation": "wait", "duration_s": 2, "stirring_speed_rpm": 600})
        np.testing.assert_allclose(session._state.temperature_K, continued._state.temperature_K,
                                   rtol=1e-12)
        assert result.observation.to_dict() == other.observation.to_dict()
        assert result.info["material_balance"] == other.info["material_balance"]
    finally:
        session.close()
        continued.close()


def test_native_heater_and_equipment_consume_the_same_caloric_state():
    from chemworld.physchem.heat_transfer_units import (
        HeatTransferEquipmentSpec,
        equipment_heat_transfer,
    )
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.coupled_thermal import local_thermodynamics

    spec = variable_native_spec()
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=12)
        session.execute({"operation": "add_component", "component": "CO-feed", "amount_mol": 6e-5})
        model = spec.physics.models_of('thermal')["reactor"]
        thermo, state = local_thermodynamics(session._state, model)
        device = next(value.settings["thermal_device"]
                      for value in session._state.equipment.equipment.values()
                      if value.equipment_type == "thermal_device"
                      and value.attached_vessel_id == "reactor")
        ua = device["conductance_W_K"] * device["calibration_factor"]
        target = state.temperature_K + 10
        assert 0 < ua * (target - state.temperature_K) < device["maximum_power_W"]
        # This two-port linear heat balance is exactly equivalent to one UA
        # connected to the weighted utility temperature; caloric data stay identical.
        total_ua = ua + model.loss_conductance_W_K
        utility = (ua * target + model.loss_conductance_W_K
                   * model.environment_temperature_K) / total_ua
        equipment = HeatTransferEquipmentSpec(
            "same-caloric-heater", "jacket", area_m2=1, clean_overall_u_W_m2_K=total_ua,
            provenance_id="same-state-linear-two-port-fixture")
        predicted = equipment_heat_transfer(
            equipment, process_temperature_K=state.temperature_K, utility_temperature_K=utility,
            duration_s=2, caloric_state=state, caloric_model=thermo)
        native = session.execute({"operation": "heat", "target_temperature_K": target,
                                   "duration_s": 2, "stirring_speed_rpm": 600})
        assert native.info["transaction_status"] == "committed"
        assert session._state.temperature_K == pytest.approx(
            predicted.final_temperature_K, abs=1e-6)
    finally:
        session.close()
