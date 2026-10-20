"""Shared contributions, real strategy dispatch and single-interval settlement.

Electrical and cache/error fixtures below exercise the existing declared models;
they do not qualify a new electrode, surface-flow model or failure mechanism.
"""

import json
from math import exp

import numpy as np
import pytest

from chemworld.foundation.gas import GasBoundary
from chemworld.physchem.coupled_flow import integrate_flow
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    EnergyPort,
    ProcessContribution,
)
from chemworld.physchem.electrochemistry import FARADAY_C_PER_MOL
from chemworld.physchem.gas_control import advance_gas, advance_gas_with_report
from chemworld.runtime.advance_paths import advance_ledger
from chemworld.runtime.continuous_streams import public_streams
from chemworld.runtime.continuous_streams import settings as stream_settings
from chemworld.runtime.control_program import control_settings
from chemworld.runtime.engine import ChemWorldRuntime
from test_continuous_streams import (
    actions as stream_actions,
)
from test_continuous_streams import (
    advance as stream_advance,
)
from test_continuous_streams import (
    make_env as stream_env,
)
from test_continuous_streams import (
    request as stream_request,
)
from test_continuous_streams import (
    step as stream_step,
)
from test_control_program import config
from test_control_program import make_env as control_env
from test_control_program import step as control_step
from test_flow_coupling import _charged_flow_env, _configure, _run


@pytest.mark.parametrize("power,direction", [(3.0, "in"), (3.0, "out"), (-3.0, "in")])
def test_energy_only_boundaries_have_real_integrals_and_close_u(power, direction):
    initial = CoupledState({}, 350, 1, exchange_energy_J={"utility": 7.0})
    process = CoupledProcess(
        "utility",
        lambda _: ProcessContribution(
            energy_port_rates_W={"utility": power},
        ),
    )
    system = CoupledSystem(
        ConstantCvThermodynamics({}, support_heat_capacity_J_K=20),
        (process,),
        energy_ports=(EnergyPort("utility", "supply", "body", direction),),
    )
    result = system.advance(initial, 2)
    sign = 1 if direction == "in" else -1
    assert result.final_state.temperature_K == pytest.approx(350 + sign * power * 2 / 20)
    assert result.final_state.exchange_energy_J["utility"] == pytest.approx(7 + power * 2)
    assert result.final_state.boundary_heat_J == 0
    assert abs(result.energy_balance_residual_J) < 1e-9
    assert result.actual_exchanges[0]["energy_J"] == pytest.approx(power * 2)
    segmented = system.advance(system.advance(initial, 1).final_state, 1)
    assert segmented.final_state.temperature_K == pytest.approx(result.final_state.temperature_K)


def test_energy_boundary_stops_at_actual_root_between_requested_samples():
    system = CoupledSystem(
        ConstantCvThermodynamics({}, support_heat_capacity_J_K=20),
        (
            CoupledProcess(
                "utility",
                lambda _: ProcessContribution(
                    energy_port_rates_W={"utility": 3.0},
                ),
            ),
        ),
        energy_ports=(EnergyPort("utility", "supply", "body", "in"),),
        events=(CoupledEvent("temperature", lambda s: 350.3 - s.temperature_K),),
    )
    result = system.advance(CoupledState({}, 350, 1), 10, evaluation_times_s=(0, 10))
    assert result.outcome == "stopped"
    assert result.actual_duration_s == pytest.approx(2, abs=1e-9)
    assert result.actual_exchanges[0]["energy_J"] == pytest.approx(6, abs=1e-8)
    assert abs(result.energy_balance_residual_J) < 1e-9


def flow_case(*, inlet=1e-5, outlet=1e-5, current=0.0, duration=100.0):
    inventory = np.array([1e-3, 0, 0, 0.01, 0, 0, 0, 1.0])
    rates = np.array([inlet * 0.01, 0, 0, inlet, 0, 0, 0, inlet / 0.01])
    return integrate_flow(
        inventory=inventory,
        inlet_rates=rates,
        species_count=3,
        inlet_heat_W=inlet * 4000 * (320 - 298.15),
        temperature_K=320,
        duration_s=duration,
        outlet_rate_L_s=outlet,
        inlet_pipes=((inlet, (1.0, 0.0, 0.0, 0.0)),) if inlet else (),
        rho_cp_J_L_K=4000,
        reaction=lambda n, c, t: (np.zeros(3), 0.0),
        target_temperature_K=320,
        ua_W_K=0.0,
        environment_temperature_K=320,
        current_mA=current,
        electrode_indices=(0, 1, 2),
    )


@pytest.mark.parametrize("outlet", [0.0, 1e-5])
def test_open_carrier_and_species_ports_match_analytic_mixed_tank(outlet):
    result = flow_case(outlet=outlet)
    expected_a = (1e-4 + (1e-3 - 1e-4) * exp(-0.1)) if outlet else 1.01e-3
    assert result.tank[0] == pytest.approx(expected_a, rel=1e-9)
    assert result.tank[3] == pytest.approx(0.01 + (1e-5 - outlet) * 100, abs=1e-12)
    assert result.tank[0] + result.outlet[0] == pytest.approx(1.01e-3, abs=1e-12)
    assert result.temperature_K == pytest.approx(320, abs=1e-10)
    assert abs(result.diagnostics["energy_residual_J"]) < 1e-8
    assert result.actual_duration_s == 100
    volumes = [r for r in result.exchanges if r["unit"] == "L"]
    assert sum(r["amount"] for r in volumes if r["source"] == "feed") == pytest.approx(0.001)
    assert sum(r["amount"] for r in volumes if r["source"] == "tank") == pytest.approx(outlet * 100)
    assert result.diagnostics["solver"]["policy"]["method"] == "DOP853"


def test_closed_electrode_channel_conserves_moles_charge_and_sensible_energy():
    result = flow_case(inlet=0, outlet=0, current=25)
    extent = 0.025 * 100 * 0.9 / (2 * FARADAY_C_PER_MOL)
    np.testing.assert_allclose(
        result.tank[:3], (1e-3 - extent, extent * 0.9, extent * 0.1), atol=1e-12
    )
    ledger = result.ledger
    assert ledger["electrical_work_J"] == pytest.approx(
        ledger["chemical_work_J"] + ledger["cell_heat_J"]
    )
    assert ledger["charge_C"] == pytest.approx(2.5)
    # Existing finite cell plus its 4 W/K jacket has an analytic thermal response.
    power = 0.025**2 * 5 + 0.025 * 0.25 - 0.025 * 0.9 * 0.25
    assert result.temperature_K == pytest.approx(
        320 + power / 4 * (1 - exp(-4 * 100 / 40)), abs=1e-9
    )
    assert abs(result.diagnostics["energy_residual_J"]) < 1e-8
    assert sum(r["amount"] for r in result.exchanges if r["unit"] == "mol") == 0


def test_plenum_uses_explicit_molar_ports_and_thermostat_work_balance():
    initial = GasBoundary.initialize(0.02, 298.15, 101325)
    kwargs = {
        "duration_s": 20,
        "final_temperature_K": 315,
        "pressure_setpoint_Pa": 150000,
        "atmosphere": 0,
    }
    report = advance_gas_with_report(initial, **kwargs)
    final = report.boundary
    assert advance_gas(initial, **kwargs) == final
    assert report.actual_duration_s == 20
    assert final.material_residual_mol < 1e-12
    assert abs(final.energy_residual_j) < 1e-8
    assert final.energy_j - initial.energy_j == pytest.approx(
        final.enthalpy_in_J - final.enthalpy_out_J + final.bath_heat_J,
        abs=1e-8,
    )
    incoming = sum(
        row["amount"]
        for row in report.exchanges
        if row["unit"] == "mol" and row["source"] == "gas_supply"
    )
    assert incoming == pytest.approx(sum(final.added_mol))


def prepare_stream(env):
    for action in stream_actions():
        if action["operation"] == "advance_flow":
            break
        stream_step(env, action)


def test_actual_stream_dispatch_settles_once_and_retains_real_source_receipts(monkeypatch):
    env = stream_env()
    prepare_stream(env)
    services = env.runtime.domain_services
    coordinator = services.reaction_thermal.advancer
    assert services.continuous_streams.advancer is coordinator
    results = []
    original = coordinator.mixed_tank

    def capture(**kwargs):
        result = original(**kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(coordinator, "mixed_tank", capture)
    before = env._state
    stream_step(env, stream_advance(60))
    after = env._state
    # Sealed equipment supplies are static, not transported tank contents.
    for name, old in {before.vessel_id: before, **before.inactive_vessels}.items():
        new = after if name == after.vessel_id else after.inactive_vessels[name]
        assert old.state_slots.slots
        assert all(s.spec.time_evolution == "static" for s in old.state_slots.slots)
        assert new.state_slots == old.state_slots
    assert len(results) == 1
    assert after.ledger.time_s - before.ledger.time_s == results[0].actual_duration_s == 60
    assert after.ledger.energy_jacket_J - before.ledger.energy_jacket_J == pytest.approx(
        results[0].ledger["jacket_J"] + results[0].ledger["cell_heat_J"],
    )
    receipt = stream_settings(after)["last_exchanges"]
    for source in ("feed-a", "feed-b"):
        volume = sum(
            row["amount"] for row in receipt if row["unit"] == "L" and row["source"] == source
        )
        assert volume == pytest.approx(
            before.inactive_vessels[source].volume_L - after.inactive_vessels[source].volume_L
        )
    assert "last_exchanges" not in public_streams(after)
    env.close()


@pytest.mark.parametrize("owner, evolution", [("equipment", "coupled"), ("material", "static")])
def test_stream_rejects_unmodelled_dynamic_or_carried_state(owner, evolution):
    from dataclasses import replace

    from chemworld.foundation.state_ownership import StateSlots
    from chemworld.runtime.continuous_streams import stream_error

    env = stream_env()
    prepare_stream(env)
    slot = env._state.state_slots.slots[0]
    modified = replace(slot, spec=replace(slot.spec, owner_kind=owner, time_evolution=evolution))
    candidate = env._state.replace(state_slots=StateSlots((modified,)))
    error = stream_error(candidate, "advance_flow", stream_advance())
    assert "does not transport or evolve" in error
    env.close()


def test_control_feedback_stop_advances_plenum_for_actual_elapsed_time(monkeypatch):
    env = control_env()
    for action in (
        {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
        {"operation": "add_reagent", "amount_mol": 0.015},
        config(),
        {
            "operation": "set_control_feedback",
            "feedback_sensor": "temperature",
            "feedback_threshold": 295.0,
            "feedback_direction": "above",
            "feedback_response": "pause",
        },
    ):
        control_step(env, action)
    coordinator = env.runtime.domain_services.reaction_thermal.advancer
    reports = []
    original = coordinator.control_stage

    def capture(*args, **kwargs):
        report = original(*args, **kwargs)
        reports.append(report)
        return report

    monkeypatch.setattr(coordinator, "control_stage", capture)
    before = env._state
    result = control_step(env, {"operation": "advance_control", "duration_s": 240})
    assert result[-1]["operation_outcome"] == "stopped"
    assert len(reports) == 1
    assert result[-1]["actual_duration_s"] == 10
    assert env._state.ledger.time_s - before.ledger.time_s == 10
    gas = GasBoundary.from_dict(control_settings(env._state)["gas"])
    assert gas == reports[0].gas
    assert env._state.ledger.gas_pump_work_J - before.ledger.gas_pump_work_J == pytest.approx(
        gas.pump_work_J
    )
    env.close()


def test_spatial_pfr_strategy_does_not_add_residence_time_twice(monkeypatch):
    env = _charged_flow_env()
    _configure(env)
    base = env.unwrapped
    coordinator = base.runtime.domain_services.reaction_thermal.advancer
    results = []
    original = coordinator.plug_flow

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        results.append(result)
        return result

    monkeypatch.setattr(coordinator, "plug_flow", capture)
    before = base._state
    info, equipment = _run(env, duration_s=1800)
    after = base._state
    assert info["transaction_status"] == "committed"
    assert len(results) == 1
    assert after.ledger.time_s - before.ledger.time_s == 1800
    assert equipment["spatial_residence_time_s"] != 1800
    assert equipment["material_basis"] == "representative_feed_inventory_no_implicit_addition"
    scale = before.volume_L / equipment["reactor_volume_L"]
    assert after.ledger.energy_jacket_J - before.ledger.energy_jacket_J == pytest.approx(
        results[0].final_state.energy_jacket_J * scale,
    )
    assert (
        after.ledger.flow_pump_work_J - before.ledger.flow_pump_work_J
        == equipment["hydraulic_energy_J"]
    )
    assert after.process.metrics["flow_hydraulic_energy_J"] == after.ledger.flow_pump_work_J
    env.close()


def test_gym_and_native_session_share_the_same_stream_physics_and_accounting():
    gym_env = stream_env(seed=5)
    session = ChemWorldRuntime(composition=stream_request(), seed=5)
    session.reset(seed=5)
    for action in stream_actions():
        stream_step(gym_env, action)
        outcome = session.execute(action)
        assert outcome.info["transaction_status"] == "committed"
        assert session._state.to_dict() == gym_env._state.to_dict()
        if action["operation"] == "advance_flow":
            break
    gym_env.close()
    session.close()


@pytest.mark.parametrize("plant", ["stream", "plenum"])
@pytest.mark.parametrize("fault", [RuntimeError, TypeError])
def test_subsystem_failures_keep_numerical_and_programming_errors_distinct(
    monkeypatch, plant, fault
):
    if plant == "stream":
        env = stream_env()
        prepare_stream(env)
        target = env.runtime.domain_services.reaction_thermal.advancer
        method = "mixed_tank"
        action = stream_advance(60)
    else:
        env = control_env()
        for item in (
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.015},
            config(),
        ):
            control_step(env, item)
        import chemworld.runtime.advance_paths as target

        method = "advance_gas_with_report"
        action = {"operation": "advance_control", "duration_s": 60}

    def fail(*args, **kwargs):
        raise fault("declared failure contract fixture")

    monkeypatch.setattr(target, method, fail)
    before = env._state
    if fault is TypeError:
        with pytest.raises(TypeError, match="failure contract fixture"):
            env.step(action)
        assert env._state is before
        env.close()
        return
    info = env.step(action)[4]
    assert info["operation_outcome"] == "numerical_failure"
    assert info["transaction_status"] == "rolled_back"
    assert env._state.species_amounts == before.species_amounts
    assert env._state.ledger.time_s == before.ledger.time_s
    assert env._state.ledger.cost - before.ledger.cost == pytest.approx(0.03)
    env.close()


def test_accounting_rejects_nonfinite_or_unregistered_energy():
    from chemworld.foundation.state import Ledger

    with pytest.raises(ValueError):
        advance_ledger(Ledger(), float("nan"))
    with pytest.raises(ValueError):
        advance_ledger(Ledger(), 1, energy_deltas={"made_up_energy": 1})


@pytest.mark.parametrize("mode", ["stream", "control"])
def test_serialized_resume_keeps_stage_ports_clock_and_utility_costs(mode):
    if mode == "stream":
        env = stream_env()
        prepare_stream(env)
        action = stream_advance(60)
    else:
        env = control_env()
        for item in (
            {"operation": "add_solvent", "volume_L": 0.03, "solvent": 0},
            {"operation": "add_reagent", "amount_mol": 0.015},
            config(),
        ):
            control_step(env, item)
        action = {"operation": "advance_control", "duration_s": 10}
    assert env.step(action)[4]["transaction_status"] == "committed"
    snapshot = json.loads(json.dumps(env.snapshot().to_dict()))
    restored = ChemWorldRuntime.from_snapshot(snapshot)
    before = env._state.ledger.time_s
    expected = env.step(action)
    actual = restored.execute(action)
    assert expected[4]["transaction_status"] == actual.info["transaction_status"] == "committed"
    assert restored._state.to_dict() == env._state.to_dict()
    assert restored._state.ledger.time_s - before == action["duration_s"]
    env.close()
    restored.close()
