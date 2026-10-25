"""The original session consumes PH/UV heat and replays real molecular stock."""

import json
from dataclasses import replace
from math import exp

import pytest

from chemworld.physchem.heat_transfer_units import ThermalDeviceSpec
from chemworld.physchem.molecular_multiphase import (
    MolecularMultiphaseSeed,
    MolecularMultiphaseSpec,
    solve_molecular_multiphase,
)
from chemworld.runtime import ChemWorldRuntime
from chemworld.runtime.molecular_multiphase_inventory import (
    MolecularMultiphaseInventorySpec,
    molecular_multiphase_inventory_state,
)
from chemworld.runtime.molecular_multiphase_thermal import (
    MolecularMultiphaseThermalRuntime,
    MolecularMultiphaseThermalSpec,
    install_molecular_multiphase_thermal,
)
from chemworld.runtime.molecular_phase_inventory import MolecularPhaseTarget
from chemworld.world.molecular_multiphase_world import build_molecular_multiphase_world
from reference.molecular28_caloric_cases import caloric_spec


def declared(boundary):
    old = caloric_spec("PR")
    core = MolecularMultiphaseSpec(
        old.model,
        old.basins,
        (280.0, 330.0),
        (4e5, 8e5),
        phase_kind_by_basin={"vapor": "gas", "liquid": "liquid"},
    )
    feed = {"propane": 0.006, "butane": 0.004}
    anchor = solve_molecular_multiphase(core, feed, inputs={"T": 300.0, "P": 6e5})
    assert anchor.success, anchor.to_dict()
    inventory = MolecularMultiphaseInventorySpec(
        core,
        (
            MolecularPhaseTarget("vapor", "vapor", "gas", True),
            MolecularPhaseTarget("liquid", "liquid", "liquid"),
        ),
    )
    spec = MolecularMultiphaseThermalSpec(
        inventory, boundary, 6e5 if boundary == "P" else anchor.volume_m3, 285.0, 325.0
    )
    composition = build_molecular_multiphase_world(
        spec,
        feed,
        energy_J=anchor.enthalpy_J if boundary == "P" else anchor.internal_energy_J,
        seed=MolecularMultiphaseSeed(300.0, 6e5, anchor.phases),
        formulas={"propane": "C3H8", "butane": "C4H10"},
        device=ThermalDeviceSpec(
            conductance_W_K=1.0,
            maximum_power_W=0.02,
            maximum_cooling_power_W=0.02,
            setpoint_K=320.0,
            minimum_setpoint_K=280.0,
            maximum_setpoint_K=330.0,
        ),
        max_volume_L=0.1,
        max_pressure_Pa=8e5,
    )
    return spec, composition


def execute(session, action):
    outcome = session.execute(action)
    assert outcome.info["transaction_status"] == "committed", outcome.info
    return outcome


def calorics(state, spec):
    return molecular_multiphase_inventory_state(
        state.phases,
        spec.inventory,
        vessel_id=state.vessel_id,
        source_phase_ids=tuple(state.phases.phases),
        temperature_K=state.temperature_K,
        pressure_Pa=state.pressure_Pa,
    )


@pytest.mark.parametrize("boundary", ["P", "V"])
def test_real_finite_heat_first_law_and_exact_json_continuation(boundary):
    spec, composition = declared(boundary)
    restored = MolecularMultiphaseThermalSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    session = ChemWorldRuntime(composition=composition)
    session.reset(seed=8)
    replay = None
    try:
        before = session._state
        initial = calorics(before, spec)
        execute(session, {"operation": "wait", "duration_s": 1})
        replay = ChemWorldRuntime.from_snapshot(
            json.loads(json.dumps(session.snapshot().to_dict()))
        )
        for runtime in (session, replay):
            execute(runtime, {"operation": "wait", "duration_s": 1})
        after = session._state
        final = calorics(after, spec)
        assert after.to_dict() == replay._state.to_dict()
        assert after.species_amounts == pytest.approx(before.species_amounts, rel=1e-13, abs=0)
        assert after.ledger.time_s == 2.0
        assert after.ledger.energy_jacket_J == pytest.approx(0.04, abs=1e-9)
        work = 6e5 * (final.volume_m3 - initial.volume_m3) if boundary == "P" else 0.0
        assert after.ledger.boundary_work_J == pytest.approx(work, abs=2e-8)
        assert final.internal_energy_J - initial.internal_energy_J == pytest.approx(
            0.04 - work, abs=2e-7
        )
        if boundary == "V":
            assert after.volume_L == pytest.approx(before.volume_L, rel=2e-8)
            assert after.pressure_Pa > before.pressure_Pa
        else:
            assert after.pressure_Pa == pytest.approx(6e5, rel=2e-8)
        assert execute(session, {"operation": "measure", "instrument": "phase_state"}).observation
    finally:
        session.close()
        if replay is not None:
            replay.close()


def test_unqualified_trial_cannot_enter_world_inventory():
    spec, composition = declared("P")
    state = composition.physics.initial_states["reactor"]
    inventory = calorics(state, spec)
    result = solve_molecular_multiphase(
        spec.inventory.equilibrium,
        inventory.feed_amounts_mol,
        inputs={"P": 6e5, "H": inventory.enthalpy_J},
        seed=inventory.seed,
        verify_stability=False,
    )
    assert result.success and not result.stability_checked
    with pytest.raises(ValueError, match="unchecked"):
        install_molecular_multiphase_thermal(state, spec, result)


def test_volume_pressure_boundary_and_operating_domain_validation():
    spec, _ = declared("P")
    for changes in (
        {"boundary": "T"},
        {"boundary_value": 0.0},
        {"boundary_value": 9e5},
        {"minimum_temperature_K": 270.0},
    ):
        with pytest.raises(ValueError):
            replace(spec, **changes)


def ideal_event_world(
    boundary, *, power=1.0, event="temperature_limit", support=0.0, loss=0.0, cooling=False
):
    from chemworld.physchem.helmholtz import HelmholtzFluid
    from chemworld.physchem.helmholtz_terms import HelmholtzTerm
    from chemworld.physchem.molecular_flash import MolecularDensityBasin
    from chemworld.physchem.multifluid import MultifluidModel

    gas, pressure, amount = 8.3145, 101325.0, 0.01
    pure = HelmholtzFluid(
        "a",
        "",
        0.055845,
        gas,
        300.0,
        pressure / (gas * 300.0),
        100.0,
        1000.0,
        1e9,
        (
            HelmholtzTerm("IdealGasHelmholtzLead", {"a1": -1.0, "a2": 0.0}),
            HelmholtzTerm("IdealGasHelmholtzLogTau", {"a": 1.5}),
        ),
        (),
        "authored ideal gas",
    )
    model = MultifluidModel((pure,), (), gas, "source_equal", "event control").caloric_model()
    high_p = pressure * 315 / 300 if event == "pressure_domain_limit" else 2e5
    core = MolecularMultiphaseSpec(
        model,
        (MolecularDensityBasin("gas", (1.0, 1000.0), 40.0),),
        (250.0, 350.0),
        (1e4, high_p),
        phase_kind_by_basin={"gas": "gas"},
    )
    anchor = solve_molecular_multiphase(core, {"a": amount}, inputs={"T": 300.0, "P": pressure})
    assert anchor.success, anchor.to_dict()
    inventory = MolecularMultiphaseInventorySpec(
        core, (MolecularPhaseTarget("gas", "gas", "gas", True),)
    )
    spec = MolecularMultiphaseThermalSpec(
        inventory,
        boundary,
        pressure if boundary == "P" else anchor.volume_m3,
        260.0,
        349.0,
        support_heat_capacity_J_K=support,
        loss_conductance_W_K=loss,
        environment_temperature_K=290.0,
    )
    composition = build_molecular_multiphase_world(
        spec,
        {"a": amount},
        energy_J=(anchor.enthalpy_J if boundary == "P" else anchor.internal_energy_J)
        + support * (300.0 - 298.15),
        seed=MolecularMultiphaseSeed(300.0, pressure, anchor.phases),
        formulas={"a": "Fe"},
        device=ThermalDeviceSpec(
            conductance_W_K=1e6,
            maximum_power_W=power,
            maximum_cooling_power_W=power,
            setpoint_K=250.1 if cooling else 349.9,
            minimum_setpoint_K=250.0,
            maximum_setpoint_K=350.0,
        ),
        max_volume_L=amount * gas * 310 / pressure * 1000 if event == "vessel_capacity" else 1.0,
        max_pressure_Pa=pressure * 310 / 300 if event == "pressure_limit" else 2e5,
    )
    return spec, composition


@pytest.mark.parametrize("power", (1.0, 1000.0))
@pytest.mark.parametrize(
    "boundary,event,temperature",
    (
        ("P", "temperature_limit", 349.0),
        ("P", "vessel_capacity", 310.0),
        ("V", "pressure_limit", 310.0),
    ),
)
def test_large_trial_steps_stop_at_real_energy_temperature_volume_or_pressure_boundary(
    power,
    boundary,
    event,
    temperature,
):
    _spec, composition = ideal_event_world(boundary, power=power, event=event)
    state = composition.physics.initial_states["reactor"]
    output = MolecularMultiphaseThermalRuntime(composition).advance(state, {"duration_s": 30.0})
    assert output.success and output.outcome == "stopped", output.diagnostics
    capacity = 0.01 * 8.3145 * (2.5 if boundary == "P" else 1.5)
    assert output.actual_duration_s == pytest.approx(
        capacity * (temperature - 300) / power, abs=2e-7
    )
    assert output.state.temperature_K == pytest.approx(temperature, abs=2e-6)
    assert output.diagnostics["stopping_event"] == event
    assert output.state.ledger.time_s == output.actual_duration_s
    assert abs(output.diagnostics["first_law_residual_J"]) < 2e-7
    assert output.state.species_amounts == state.species_amounts
    if boundary == "V":
        assert output.state.ledger.boundary_work_J == 0
    else:
        expected = 101325.0 * (output.state.volume_L - state.volume_L) * 1e-3
        assert output.state.ledger.boundary_work_J == pytest.approx(expected, abs=1e-10)


def test_fixed_volume_stops_at_actual_pressure_domain_before_unreachable_temperature():
    _spec, composition = ideal_event_world("V", power=1000.0, event="pressure_domain_limit")
    state = composition.physics.initial_states["reactor"]
    output = MolecularMultiphaseThermalRuntime(composition).advance(state, {"duration_s": 30.0})
    assert output.success and output.outcome == "stopped", output.diagnostics
    assert output.diagnostics["stopping_event"] == "pressure_domain_limit"
    assert output.state.temperature_K == pytest.approx(315.0, abs=2e-6)
    assert output.state.pressure_Pa == pytest.approx(101325.0 * 315 / 300, rel=2e-8)
    assert output.actual_duration_s == pytest.approx(0.01 * 1.5 * 8.3145 * 15 / 1000, abs=2e-7)
    assert any(not a["success"] for a in output.diagnostics["event_energy_enclosure"])


def test_cooling_uses_lower_energy_boundary_and_negative_device_heat():
    _spec, composition = ideal_event_world("V", power=1000.0, cooling=True)
    state = composition.physics.initial_states["reactor"]
    output = MolecularMultiphaseThermalRuntime(composition).advance(state, {"duration_s": 30.0})
    assert output.success and output.outcome == "stopped", output.diagnostics
    assert output.diagnostics["stopping_event"] == "minimum_temperature"
    assert output.state.temperature_K == pytest.approx(260.0, abs=2e-6)
    assert output.state.ledger.energy_jacket_J == pytest.approx(-0.01 * 1.5 * 8.3145 * 40, abs=2e-7)


@pytest.mark.parametrize("boundary", ("P", "V"))
def test_support_and_heat_loss_remain_in_the_same_energy_ledger(boundary):
    _spec, composition = ideal_event_world(boundary, support=0.2, loss=0.01)
    state = composition.physics.initial_states["reactor"]
    output = MolecularMultiphaseThermalRuntime(composition).advance(state, {"duration_s": 5.0})
    assert output.success and output.outcome == "success", output.diagnostics
    capacity = 0.2 + 0.01 * 8.3145 * (2.5 if boundary == "P" else 1.5)
    temperature = 390.0 - 90.0 * exp(-0.01 * 5 / capacity)
    assert output.state.temperature_K == pytest.approx(temperature, abs=2e-6)
    assert output.state.ledger.energy_jacket_J == pytest.approx(5.0, abs=1e-8)
    assert output.state.ledger.heat_loss_J == pytest.approx(
        5.0 - capacity * (temperature - 300.0), abs=2e-7
    )
    assert abs(output.diagnostics["first_law_residual_J"]) < 2e-7
