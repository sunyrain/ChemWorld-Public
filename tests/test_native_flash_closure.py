"""Fixed VLE inputs and independent closure/entry checks."""

import json
from dataclasses import replace
from itertools import product
from math import exp

import pytest

from chemworld.examples.native_flash_research import synthetic_vle_properties
from chemworld.physchem.equilibrium import ActivityModelSpec
from chemworld.physchem.flash_units import VLEPropertyPackage, ph_flash, tp_flash

GRID = tuple(product((.2, .5, .8), (5e4, 1e5, 2e5), (-35000, -15000, 12000)))


@pytest.mark.parametrize("temperature,phase_name", [(300, "all_liquid"), (450, "all_vapor")])
def test_single_phase_queries_preserve_each_exact_input_inventory(temperature, phase_name):
    model = synthetic_vle_properties()
    feed = {"light": 0.005068212543753439, "heavy": 0.001072317628199126}
    original = dict(feed)
    for _ in range(10):
        result = tp_flash(model, feed, temperature_K=temperature, pressure_Pa=1e5)
        assert result.phase_status == phase_name
        occupied = (result.liquid_amounts_mol if phase_name == "all_liquid"
                    else result.vapor_amounts_mol)
        empty = (result.vapor_amounts_mol if phase_name == "all_liquid"
                 else result.liquid_amounts_mol)
        assert occupied == original
        assert empty == dict.fromkeys(original, 0.0)
        recovered = ph_flash(model, occupied, pressure_Pa=1e5,
                             enthalpy_J=result.total_enthalpy_j)
        feed = (recovered.liquid_amounts_mol if phase_name == "all_liquid"
                else recovered.vapor_amounts_mol)
        assert feed == original


@pytest.mark.parametrize("fraction,pressure,enthalpy", GRID)
def test_fixed_ph_grid_and_tp_round_trip(fraction, pressure, enthalpy):
    model = synthetic_vle_properties()
    feed = {"light": fraction, "heavy": 1 - fraction}
    result = ph_flash(model, feed, pressure_Pa=pressure, enthalpy_J=enthalpy)
    assert result.converged and result.heat_duty_J == 0
    for key, amount in feed.items():
        assert result.liquid_amounts_mol[key] + result.vapor_amounts_mol[key] == pytest.approx(
            amount, abs=1e-8)
    # Independent polynomial and latent expression, not the reported residual.
    independent_h = sum(
        result.vapor_amounts_mol[key] * cp * (result.temperature_K - 298.15)
        + result.liquid_amounts_mol[key] * (cp * (result.temperature_K - 298.15) - latent)
        for key, cp, latent in (("light", 80, 30000), ("heavy", 100, 40000)))
    assert independent_h == pytest.approx(enthalpy, rel=1e-6, abs=1e-6)
    fixed = tp_flash(model, feed, temperature_K=result.temperature_K, pressure_Pa=pressure,
                     feed_enthalpy_J=enthalpy)
    assert fixed.total_enthalpy_j == pytest.approx(enthalpy, rel=1e-6)
    assert fixed.heat_duty_J == pytest.approx(0, abs=1e-4)
    restored = ph_flash(model, feed, pressure_Pa=pressure, enthalpy_J=fixed.total_enthalpy_j)
    assert restored.temperature_K == pytest.approx(result.temperature_K, abs=1e-6)


def test_flash_domain_failures_nonconvergence_and_no_fabricated_tp_heat():
    model = synthetic_vle_properties()
    feed = {"light": .5, "heavy": .5}
    result = tp_flash(model, feed, temperature_K=350, pressure_Pa=1e5)
    assert result.heat_duty_J is None and result.energy_balance_error_J is None
    with pytest.raises(ValueError, match="unreachable"):
        ph_flash(model, feed, pressure_Pa=1e5, enthalpy_J=1e9)
    with pytest.raises(ValueError, match="domain"):
        tp_flash(model, feed, temperature_K=600, pressure_Pa=1e5)
    with pytest.raises(RuntimeError, match="did not converge"):
        nonideal = replace(model, activity_model=ActivityModelSpec(
            "convex-liquid", model.component_ids, "margules",
            {"A:light|heavy": 1.2, "A:heavy|light": .8}))
        tp_flash(nonideal, feed, temperature_K=350, pressure_Pa=1e5, max_iterations=1)
    model = replace(model, activity_model=ActivityModelSpec(
        "unstable-liquid", model.component_ids, "margules",
        {"A:light|heavy": 3, "A:heavy|light": 3}))
    with pytest.raises(ValueError, match="two liquid phases"):
        tp_flash(model, feed, temperature_K=350, pressure_Pa=1e5)


def test_pure_saturation_plateau_and_zero_component_survive_ph():
    model = synthetic_vle_properties()
    for beta in (0, 1e-7, .5, 1 - 1e-7, 1):
        target = 80 * (330 - 298.15) - 30000 * (1 - beta)
        result = ph_flash(model, {"light": 1, "heavy": 0}, pressure_Pa=1e5, enthalpy_J=target)
        assert result.temperature_K == pytest.approx(330, abs=1e-7)
        assert result.vapor_fraction == pytest.approx(beta, abs=1e-8)
        assert result.total_enthalpy_j == pytest.approx(target, abs=1e-5)
        assert result.vapor_amounts_mol["heavy"] == 0


def test_flash_serialization_keeps_same_datums_and_property_parameters():
    model = synthetic_vle_properties()
    restored = VLEPropertyPackage.from_dict(json.loads(json.dumps(model.to_dict())))
    kwargs = {"temperature_K": 350, "pressure_Pa": 1e5}
    feed = {"light": .5, "heavy": .5}
    assert tp_flash(model, feed, **kwargs).to_dict() == tp_flash(restored, feed, **kwargs).to_dict()


def test_flash_uses_current_vapor_composition_for_fugacity_and_departure_enthalpy():
    from chemworld.physchem.eos import CubicEOSSpec, EOSComponentSpec, evaluate_cubic_eos
    from chemworld.physchem.property_packages import evaluate_property_state

    model = synthetic_vle_properties()
    eos = CubicEOSSpec("synthetic-vapor", "peng_robinson", (
        EOSComponentSpec("light", 500, 5e6, .1), EOSComponentSpec("heavy", 550, 5e6, .1)))
    model = replace(model, vapor_eos=eos)
    result = tp_flash(model, {"light": .5, "heavy": .5}, temperature_K=350, pressure_Pa=1e5)
    state = evaluate_cubic_eos(eos, result.vapor_composition, temperature_K=350,
                               pressure_Pa=1e5, phase="vapor")
    for key, value in state.fugacity_coefficients.items():
        assert result.vapor_fugacity_coefficients[key] == pytest.approx(value, rel=1e-8)
    gas_h = evaluate_property_state(
        model.packages, result.vapor_composition, temperature_K=350, pressure_Pa=1e5,
        phase="gas", quantities=("enthalpy",), references=model.gas_references, eos=eos)
    assert result.vapor_enthalpy_J == pytest.approx(
        result.vapor_fraction * gas_h.properties["enthalpy"].value, rel=1e-8)
    restored = VLEPropertyPackage.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.vapor_eos.to_dict() == eos.to_dict()


def independent_native_h(state):
    return (state.temperature_K - 298.15) + sum(
        amount * (cp * (state.temperature_K - 298.15)
                  - (latent if phase.phase_type == "liquid" else 0))
        for phase in state.phases.phases.values()
        for key, cp, latent in (("light", 80, 30000), ("heavy", 100, 40000))
        for amount in (phase.species_amounts_mol[key],))


def test_real_session_feed_heat_measure_transfer_and_snapshot_share_phase_inventory():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.vle_world import vle_world_spec

    session = ChemWorldRuntime(composition=vle_world_spec(synthetic_vle_properties()))
    session.reset(seed=12)
    continued = None
    try:
        initial_h = independent_native_h(session._state)
        assert len(session._state.phases.phases) == 2
        feed = session.execute({"operation": "add_component", "component": "mixture-feed",
                                "amount_mol": 1e-4})
        assert feed.info["transaction_status"] == "committed"
        packet_h = sum(5e-5 * (cp * (300 - 298.15) - latent)
                       for cp, latent in ((80, 30000), (100, 40000)))
        assert independent_native_h(session._state) - initial_h == pytest.approx(packet_h, abs=1e-7)
        before_h, before_ledger = independent_native_h(session._state), session._state.ledger
        heated = session.execute({"operation": "heat", "target_temperature_K": 410,
                                  "duration_s": 20, "stirring_speed_rpm": 600})
        assert heated.info["transaction_status"] == "committed"
        assert heated.info["outcome_diagnostics"]["advance_strategy"] == (
            "constant-pressure-flash-enthalpy")
        assert session._state.ledger.time_s == 20
        assert list(session._state.phases.phases) == ["vle_gas"]
        energy_in = (session._state.ledger.energy_jacket_J - before_ledger.energy_jacket_J
                     - session._state.ledger.heat_loss_J + before_ledger.heat_loss_J)
        assert independent_native_h(session._state) - before_h == pytest.approx(energy_in, abs=1e-6)
        assert all(p.solvents.volume_L == 0 for p in session._state.phases.phases.values())
        read = session.execute({"operation": "measure", "instrument": "phase_state"})
        assert read.info["transaction_status"] == "committed"
        assert read.info["actual_duration_s"] == 1
        assert session._state.ledger.time_s == 21
        assert session.observe().channels[0].sampled_at.time_s == 21
        assert session.observe().channels[1].value == pytest.approx(1)
        before_total = independent_native_h(session._state) + independent_native_h(
            session._state.inactive_vessels["receiver"])
        routed = session.execute({"operation": "route_material",
                                  "connection": "reactor-to-receiver",
                                  "transfer_fraction": .2, "mixing": 1})
        assert routed.info["transaction_status"] == "committed"
        assert session._state.ledger.time_s == 21
        after_total = independent_native_h(session._state) + independent_native_h(
            session._state.inactive_vessels["receiver"])
        assert after_total == pytest.approx(before_total, abs=1e-6)
        continued = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict())))
        action = {"operation": "wait", "duration_s": 1, "stirring_speed_rpm": 600}
        first, other = session.execute(action), continued.execute(action)
        assert first.info["transaction_status"] == other.info["transaction_status"] == "committed"
        assert session._state.to_dict() == continued._state.to_dict()
    finally:
        session.close()
        if continued is not None:
            continued.close()


def test_flash_thermal_stop_commits_actual_heat_and_time():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.vle_world import vle_world_spec

    spec = vle_world_spec(synthetic_vle_properties())
    model = replace(spec.physics.flash_models["reactor"], stop_temperature_K=360)
    spec = replace(spec, physics=replace(spec.physics, flash_models={
        **spec.physics.flash_models, "reactor": model}))
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=12)
        before = independent_native_h(session._state)
        result = session.execute({"operation": "heat", "target_temperature_K": 410,
                                  "duration_s": 100, "stirring_speed_rpm": 600})
        assert result.info["transaction_status"] == "committed"
        assert result.info["operation_outcome"] == "stopped"
        assert 0 < result.info["actual_duration_s"] < 100
        assert session._state.temperature_K == pytest.approx(360, abs=1e-7)
        assert session._state.ledger.time_s == result.info["actual_duration_s"]
        assert independent_native_h(session._state) - before == pytest.approx(
            session._state.ledger.energy_jacket_J - session._state.ledger.heat_loss_J, abs=1e-6)
    finally:
        session.close()


def test_tp_heat_derivative_matches_independent_binary_raoult_expression():
    from chemworld.physchem.property_reports import R_J_PER_MOL_K

    temperature = 350
    latent = (30000, 40000)
    ks = [exp(heat / R_J_PER_MOL_K * (1 / boiling - 1 / temperature))
          for heat, boiling in zip(latent, (330, 370), strict=True)]
    a, b = [k - 1 for k in ks]
    da, db = [k * heat / R_J_PER_MOL_K / temperature ** 2
              for k, heat in zip(ks, latent, strict=True)]
    s, ds, d, dd = .5 * (a + b), .5 * (da + db), a * b, da * b + a * db
    beta, derivative = -s / d, -ds / d + s * dd / d ** 2
    liquid_derivatives = [
        -derivative * .5 / (1 + beta * value)
        - (1 - beta) * .5 * (derivative * value + beta * rate) / (1 + beta * value) ** 2
        for value, rate in ((a, da), (b, db))]
    expected_cp = 90 - sum(heat * rate for heat, rate in zip(
        latent, liquid_derivatives, strict=True))
    model, step = synthetic_vle_properties(), 1e-3
    low = tp_flash(model, {"light": .5, "heavy": .5}, temperature_K=350-step, pressure_Pa=1e5)
    high = tp_flash(model, {"light": .5, "heavy": .5}, temperature_K=350+step,
                    pressure_Pa=1e5, feed_enthalpy_J=low.total_enthalpy_j)
    assert high.heat_duty_J / (2 * step) == pytest.approx(expected_cp, rel=1e-5)


def test_complete_transfer_preserves_distinct_apparatus_capacities_and_empty_source():
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.runtime.flash_thermal import install_flash
    from chemworld.world.vle_world import vle_world_spec

    spec = vle_world_spec(synthetic_vle_properties())
    receiver = replace(spec.physics.flash_models["receiver"], support_heat_capacity_J_K=2)
    states = dict(spec.physics.initial_states)
    states["receiver"] = install_flash(states["receiver"], receiver,
                                       receiver.at_tp(states["receiver"].species_amounts, 300))
    spec = replace(spec, physics=replace(spec.physics, initial_states=states,
        flash_models={**spec.physics.flash_models, "receiver": receiver}))
    session = ChemWorldRuntime(composition=spec)
    try:
        session.reset(seed=12)
        def energy():
            target = session._state.inactive_vessels["receiver"]
            return (independent_native_h(session._state) + independent_native_h(target)
                    + target.temperature_K - 298.15)
        before = energy()
        result = session.execute({"operation": "route_material",
                                  "connection": "reactor-to-receiver",
                                  "transfer_fraction": 1, "mixing": 1})
        assert result.info["transaction_status"] == "committed"
        assert sum(session._state.species_amounts.values()) == 0
        assert energy() == pytest.approx(before, abs=1e-6)
        assert session._state.temperature_K == pytest.approx(350)
    finally:
        session.close()


def test_native_flash_failure_rolls_back_without_a_legacy_fallback(monkeypatch):
    from chemworld.physchem.flash_units import FlashSystemSpec
    from chemworld.runtime import ChemWorldRuntime
    from chemworld.world.vle_world import vle_world_spec

    session = ChemWorldRuntime(composition=vle_world_spec(synthetic_vle_properties()))
    try:
        session.reset(seed=12)
        phases, temperature = session._state.phases.to_dict(), session._state.temperature_K
        def fail(*args, **kwargs):
            raise RuntimeError("injected algebraic solver failure")
        monkeypatch.setattr(FlashSystemSpec, "at_ph", fail)
        result = session.execute({"operation": "heat", "target_temperature_K": 410,
                                  "duration_s": 20, "stirring_speed_rpm": 600})
        assert result.info["operation_outcome"] == "numerical_failure"
        assert result.info["transaction_status"] == "rolled_back"
        assert session._state.phases.to_dict() == phases
        assert session._state.temperature_K == temperature
        assert session._state.ledger.time_s == session._state.ledger.energy_jacket_J == 0
    finally:
        session.close()


def test_registered_liquid_cannot_duplicate_a_legacy_carrier_inventory():
    from chemworld.foundation.solvents import SolventInventory
    from chemworld.world.vle_world import vle_world_spec

    state = vle_world_spec(synthetic_vle_properties()).physics.initial_states["reactor"]
    phase = state.phases.phases["vle_liquid"]
    with pytest.raises(ValueError, match="cannot also carry"):
        replace(phase, solvents=SolventInventory.pure(0, phase.volume_L))


@pytest.mark.parametrize("temperature", [250,500])
@pytest.mark.parametrize("capacity", [0,1e-4,1])
@pytest.mark.parametrize("amounts", [
    {"light":.001001234567,"heavy":.000999876543},
    {"light":.00000001731,"heavy":.003198375131},
])
def test_ideal_ph_endpoint_retains_roundoff_exact_bounds(temperature,capacity,amounts):
    model = synthetic_vle_properties()
    bounds = model.ideal_enthalpy_bounds(amounts,1e5,capacity,298.15)
    result = ph_flash(model,amounts,pressure_Pa=1e5,
        enthalpy_J=bounds[0 if temperature == 250 else 1],support_heat_capacity_J_K=capacity)
    assert result.temperature_K == temperature
    assert result.energy_balance_error_J < 1e-10


def test_vle_cached_constitutive_correlations_are_owned_immutable_copies():
    model = synthetic_vle_properties()
    original = model.packages[0].correlations[0]
    with pytest.raises(TypeError):
        original.coefficients['a'] = 500
    restored = type(model).from_dict(model.to_dict())
    assert restored.to_dict() == model.to_dict()
