"""Finite analytic/thermodynamic fixtures for the common local ODE assembly.

The adsorption fixture tests shared inventories; the five-step surface mechanism
and reference matrix are separate tasks. No new paper evidence is produced here.
"""

from dataclasses import replace
from math import exp

import numpy as np
import pytest

import chemworld.physchem.coupled_system as coupled
from chemworld.physchem.batch_reactors import BatchReactorModel, DynamicBatchReactorModel
from chemworld.physchem.coupled_system import (
    ConstantCvThermodynamics,
    CoupledEvent,
    CoupledProcess,
    CoupledState,
    CoupledSystem,
    HeatFlux,
    InternalTransfer,
    MaterialPort,
    ProcessContribution,
    SpeciesInternalEnergy,
)
from chemworld.physchem.reaction_network import (
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SpeciesSpec,
)
from chemworld.physchem.reactor_shared import HeatTransferSpec
from chemworld.physchem.reactor_solvers import reaction_network_process
from chemworld.physchem.solver_backend import ODESolverPolicy, solve_ode


def network(rate=0.03):
    return ReactionNetworkSpec(
        "coupled-isomerization",
        (SpeciesSpec("A", "CO"), SpeciesSpec("B", "CO")),
        (
            ReactionSpec.from_equation(
                reaction_id="bulk",
                equation="A => B",
                rate_law=RateLawSpec("first-order", "mass_action", {"k": rate}),
                delta_h_J_per_mol=-999_999.0,  # Native U, not this old model parameter, owns heat.
            ),
        ),
    )


def thermal():
    return ConstantCvThermodynamics(
        {
            "A": SpeciesInternalEnergy(0.0, 10.0),
            "B": SpeciesInternalEnergy(-12_000.0, 20.0),
            "adsorbed_A": SpeciesInternalEnergy(-20_000.0, 5.0),
        },
        support_heat_capacity_J_K=20.0,
    )


def initial(amount=0.01):
    return CoupledState({"A": amount, "B": 0.0, "adsorbed_A": 0.0}, 350.0, 0.01)


def adsorption(rate=0.02):
    return CoupledProcess(
        "surface-fixture",
        lambda state: ProcessContribution(
            internal_transfers=(
                InternalTransfer(
                    "adsorption-fixture",
                    "A",
                    "adsorbed_A",
                    rate * state.amounts_mol["A"],
                ),
            ),
        ),
    )


def shared_system():
    return CoupledSystem(
        thermal(),
        (reaction_network_process(network(0.01)), adsorption()),
        conservation_laws={"carbon": {"A": 1, "B": 1, "adsorbed_A": 1}},
        inventory_scales_mol={"A": 0.01, "B": 0.01, "adsorbed_A": 0.01},
    )


@pytest.mark.parametrize("method", ["BDF", "Radau"])
def test_inert_analytical_heat_response_has_one_time_and_heat_advance(method):
    system = CoupledSystem(
        ConstantCvThermodynamics({}, 20.0),
        heat_flux=lambda state: HeatFlux(
            min(0.2 * (370.0 - state.temperature_K), 5.0),
            0.01 * (state.temperature_K - 300.0),
        ),
        policy=ODESolverPolicy(f"heat-{method}", method=method),
    )
    result = system.advance(CoupledState({}, 350.0, 0.01), 600.0)
    steady = (0.2 * 370 + 0.01 * 300) / 0.21
    expected = steady + (350.0 - steady) * exp(-0.21 / 20.0 * 600.0)
    assert result.final_state.temperature_K == pytest.approx(expected, abs=2e-6)
    assert result.actual_duration_s == 600.0
    assert result.solver_diagnostic["solver_calls"] == 1
    assert result.energy_balance_residual_J == pytest.approx(0.0, abs=1e-8)
    assert 20 * (result.final_state.temperature_K - 350) == pytest.approx(
        result.final_state.boundary_heat_J - result.final_state.heat_loss_J,
        abs=1e-8,
    )


def test_adiabatic_bulk_uses_single_native_internal_energy_and_variable_capacity():
    thermo = thermal()
    system = CoupledSystem(thermo, (reaction_network_process(network()),))
    start = initial()
    result = system.advance(start, 120.0, evaluation_times_s=(0, 10, 60, 120))
    product = 0.01 * (1 - exp(-0.03 * 120))
    expected_temperature = 350 + 12_000 * product / (20 + 10 * (0.01 - product) + 20 * product)
    assert result.final_state.amounts_mol["B"] == pytest.approx(product, rel=1e-7)
    assert result.final_state.temperature_K == pytest.approx(expected_temperature, abs=2e-6)
    for point in result.trajectory:
        assert thermo.internal_energy_j(point) == pytest.approx(
            thermo.internal_energy_j(start),
            abs=2e-6,
        )
    assert result.final_state.boundary_heat_J == 0.0
    assert result.energy_balance_residual_J == pytest.approx(0, abs=2e-6)


def test_bulk_and_surface_read_one_candidate_and_compete_for_same_finite_inventory(monkeypatch):
    calls = []

    def counted(*args, **kwargs):
        calls.append(kwargs["time_span_s"])
        return solve_ode(*args, **kwargs)

    monkeypatch.setattr(coupled, "solve_ode", counted)
    start = initial()
    result = shared_system().advance(start, 120, evaluation_times_s=(0, 20, 120))
    converted = 0.01 * (1 - exp(-0.03 * 120))
    final = result.final_state
    assert final.amounts_mol["A"] == pytest.approx(0.01 - converted, rel=2e-7, abs=1e-10)
    assert final.amounts_mol["B"] == pytest.approx(converted / 3, rel=2e-7)
    assert final.amounts_mol["adsorbed_A"] == pytest.approx(2 * converted / 3, rel=2e-7)
    assert sum(final.amounts_mol.values()) == pytest.approx(0.01, abs=1e-12)
    assert result.conservation_residuals["carbon"] == pytest.approx(0, abs=1e-12)
    assert result.actual_exchanges == ()  # Paired adsorption is internal.
    assert result.solver_diagnostic["solver_calls"] == 1
    assert calls == [(0.0, 120)]
    assert start.amounts_mol["A"] == 0.01


def test_segmented_and_whole_advance_preserve_state_and_cumulative_energy():
    system = replace(
        shared_system(),
        heat_flux=lambda state: HeatFlux(
            1.0,
            0.01 * (state.temperature_K - 300),
        ),
    )
    whole = system.advance(initial(), 200).final_state
    first = system.advance(initial(), 70).final_state
    segmented = system.advance(first, 130).final_state
    assert segmented.time_s == whole.time_s == 200.0
    assert dict(segmented.amounts_mol) == pytest.approx(dict(whole.amounts_mol), abs=2e-10)
    assert segmented.temperature_K == pytest.approx(whole.temperature_K, abs=2e-6)
    assert segmented.boundary_heat_J == pytest.approx(whole.boundary_heat_J, abs=1e-8)
    assert segmented.heat_loss_J == pytest.approx(whole.heat_loss_J, abs=2e-6)


def test_isothermal_boundary_records_compensating_heat_using_same_reaction_function():
    result = replace(shared_system(), isothermal=True).advance(initial(), 100)
    assert result.final_state.temperature_K == 350.0
    assert result.final_state.boundary_heat_J < 0.0
    assert result.final_state.boundary_heat_J == pytest.approx(
        result.final_state.chemical_energy_change_J,
        abs=1e-7,
    )
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-7)


def test_event_root_between_requested_samples_keeps_exact_endpoint_and_heat():
    event = CoupledEvent("temperature-limit", lambda state: 307.0 - state.temperature_K)
    system = CoupledSystem(
        ConstantCvThermodynamics({}, 20),
        heat_flux=lambda state: HeatFlux(2),
        events=(event,),
    )
    result = system.advance(CoupledState({}, 300, 0.01), 200, evaluation_times_s=(0, 100, 200))
    assert result.outcome == "stopped"
    assert result.stopping_event == "temperature-limit"
    assert result.actual_duration_s == pytest.approx(70, abs=1e-8)
    assert result.final_state.temperature_K == pytest.approx(307, abs=1e-10)
    assert result.final_state.boundary_heat_J == pytest.approx(140, abs=1e-7)
    assert result.trajectory[-1].time_s == pytest.approx(70, abs=1e-8)
    assert result.solver_diagnostic["final_time_s"] == pytest.approx(70, abs=1e-8)
    already_stopped = system.advance(result.final_state, 100)
    assert already_stopped.outcome == "stopped"
    assert already_stopped.actual_duration_s == 0
    assert already_stopped.solver_diagnostic["solver_calls"] == 0


def test_backend_reports_actual_terminal_time_when_t_eval_misses_root():
    def stop(time, state):
        return 0.7 - time

    stop.terminal = True
    report = solve_ode(
        lambda time, state: np.asarray([2.0]),
        [0.0],
        time_span_s=(0, 2),
        evaluation_times_s=(0, 1, 2),
        events=(stop,),
        policy=ODESolverPolicy("root-test", method="BDF", event_names=("stop",)),
    )
    assert report.t.tolist() == [0.0]
    assert report.diagnostic.final_time_s == pytest.approx(0.7)
    assert report.terminal_point[1][0] == pytest.approx(1.4)


def test_finite_outlet_stops_at_zero_inventory_and_preserves_actual_exchange():
    port = MaterialPort("out", "A", "reactor", "collector", "out")
    process = CoupledProcess(
        "outflow",
        lambda state: ProcessContribution(
            port_rates_mol_s={"out": 0.001},
            port_enthalpies_J_mol={"out": 1000},
        ),
        events=(CoupledEvent("depleted-A", lambda state: state.amounts_mol["A"]),),
    )
    system = CoupledSystem(
        ConstantCvThermodynamics({"A": SpeciesInternalEnergy(1000, 0)}, 20),
        (process,),
        ports=(port,),
        conservation_laws={"A": {"A": 1}},
    )
    result = system.advance(CoupledState({"A": 0.01}, 350, 0.01), 100, evaluation_times_s=(0, 100))
    assert result.outcome == "stopped"
    assert result.actual_duration_s == pytest.approx(10, abs=1e-7)
    assert result.final_state.amounts_mol["A"] == pytest.approx(0, abs=1e-12)
    assert result.actual_exchanges[0]["amount"] == pytest.approx(0.01)
    assert result.actual_exchanges[0]["energy_J"] == pytest.approx(10)
    assert result.final_state.temperature_K == pytest.approx(350)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=1e-8)


def test_external_ports_integrate_real_amount_and_inlet_enthalpy_into_one_energy_state():
    port = MaterialPort("feed-A", "A", "finite-source:A", "reactor", "in")
    process = CoupledProcess(
        "feed",
        lambda state: ProcessContribution(
            port_rates_mol_s={"feed-A": 0.001},
            port_enthalpies_J_mol={"feed-A": 5000},
        ),
    )
    thermo = ConstantCvThermodynamics({"A": SpeciesInternalEnergy(1000, 10)}, 20)
    system = CoupledSystem(
        thermo,
        (process,),
        ports=(port,),
        conservation_laws={"A": {"A": 1}},
        inventory_scales_mol={"A": 0.02},
    )
    start = CoupledState({"A": 0.01}, 350, 0.01)
    result = system.advance(start, 10)
    assert result.final_state.amounts_mol["A"] == pytest.approx(0.02)
    assert result.final_state.temperature_K == pytest.approx(350 + 40 / 20.2, abs=1e-6)
    assert result.conservation_residuals["A"] == pytest.approx(0, abs=1e-12)
    assert result.energy_balance_residual_J == pytest.approx(0, abs=2e-6)
    assert result.actual_exchanges == (
        {
            "exchange_id": "feed-A",
            "inventory_key": "A",
            "source": "finite-source:A",
            "destination": "reactor",
            "direction": "in",
            "unit": "mol",
            "amount": pytest.approx(0.01),
            "energy_J": pytest.approx(50),
        },
    )
    second = system.advance(result.final_state, 5)
    assert second.actual_exchanges[0]["amount"] == pytest.approx(0.005)


def test_native_thermodynamics_rejects_double_counted_reaction_heat():
    system = CoupledSystem(thermal(), (reaction_network_process(network(), legacy_energy=True),))
    with pytest.raises(ValueError, match="separate reaction heat"):
        system.advance(initial(), 1)


def test_process_receives_an_immutable_state_and_cannot_modify_other_inventory():
    def mutate(state):
        state.amounts_mol["A"] = 10
        return ProcessContribution()

    system = CoupledSystem(thermal(), (CoupledProcess("bad", mutate),))
    with pytest.raises(TypeError):
        system.advance(initial(), 1)


def test_nonconserving_process_is_rejected_even_with_legal_positive_candidate():
    process = CoupledProcess(
        "bad",
        lambda state: ProcessContribution(
            amount_rates_mol_s={"B": 0.001},
        ),
    )
    system = replace(shared_system(), processes=(process,))
    with pytest.raises(RuntimeError, match="conservation law"):
        system.advance(initial(), 1)


def test_duplicate_process_exchange_and_reaction_contributions_are_rejected():
    process = adsorption()
    with pytest.raises(ValueError, match="duplicate process"):
        CoupledSystem(thermal(), (process, process))
    duplicate_transfer = CoupledProcess("other", process.evaluate)
    with pytest.raises(ValueError, match="duplicate internal exchange"):
        CoupledSystem(thermal(), (process, duplicate_transfer)).advance(initial(), 1)
    bulk = reaction_network_process(network())
    with pytest.raises(ValueError, match="duplicate reaction"):
        CoupledSystem(thermal(), (bulk, replace(bulk, process_id="duplicate"))).advance(
            initial(), 1
        )


def test_missing_port_energy_wrong_units_and_unknown_inventory_are_rejected():
    with pytest.raises(ValueError, match="nonzero weights"):
        CoupledSystem(thermal(), conservation_laws={"unmodeled": {}})
    with pytest.raises(ValueError, match="enthalpy"):
        ProcessContribution(port_rates_mol_s={"in": 1})
    with pytest.raises(ValueError, match="mol"):
        MaterialPort("in", "A", "source", "sink", "in", unit="L")
    process = CoupledProcess(
        "bad",
        lambda state: ProcessContribution(
            amount_rates_mol_s={"unknown": 1},
        ),
    )
    with pytest.raises(ValueError, match="unknown inventory"):
        CoupledSystem(thermal(), (process,)).advance(initial(), 1)


def test_numerical_failure_never_returns_a_false_physical_stop(monkeypatch):
    state = initial()

    def fail(*args, **kwargs):
        raise RuntimeError("controlled nonconvergence")

    monkeypatch.setattr(coupled, "solve_ode", fail)
    with pytest.raises(RuntimeError, match="nonconvergence"):
        shared_system().advance(state, 100)
    assert state.amounts_mol["A"] == 0.01
    assert state.temperature_K == 350
    assert state.time_s == 0


def test_zero_duration_is_identity_without_solver_or_new_receipts(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("zero duration must not invoke a solver")

    monkeypatch.setattr(coupled, "solve_ode", forbidden)
    state = initial()
    result = shared_system().advance(state, 0)
    assert result.final_state is state
    assert result.actual_duration_s == 0
    assert result.actual_exchanges == ()


def test_tightening_tolerance_converges_for_different_inventory_scales():
    sites = 2.5e-7
    process = CoupledProcess(
        "bounded-adsorption-fixture",
        lambda state: ProcessContribution(
            internal_transfers=(
                InternalTransfer(
                    "bounded-fixture",
                    "A",
                    "adsorbed_A",
                    max(
                        0,
                        0.02
                        * state.amounts_mol["A"]
                        * (1 - state.amounts_mol["adsorbed_A"] / sites),
                    ),
                ),
            ),
        ),
    )
    system = replace(
        shared_system(),
        processes=(process,),
        inventory_scales_mol={"A": 1e-4, "B": 1e-4, "adsorbed_A": sites},
    )
    start = initial(1e-4)
    baseline = system.advance(start, 600).final_state
    tighter = replace(system, policy=replace(system.policy, rtol=1e-9, atol=1e-13))
    reference = tighter.advance(start, 600).final_state
    assert abs(baseline.temperature_K - reference.temperature_K) / 350 < 1e-6
    for key, scale in system.inventory_scales_mol.items():
        assert abs(baseline.amounts_mol[key] - reference.amounts_mol[key]) / scale < 1e-6
    assert baseline.amounts_mol["adsorbed_A"] == pytest.approx(sites, abs=1e-13)


@pytest.mark.parametrize("model_type", [BatchReactorModel, DynamicBatchReactorModel])
def test_old_independent_batch_entries_use_common_assembly(model_type, monkeypatch):
    from chemworld.physchem import reactor_solvers

    assemble = reactor_solvers.assemble_coupled_rates
    count = 0

    def counted(*args, **kwargs):
        nonlocal count
        count += 1
        return assemble(*args, **kwargs)

    monkeypatch.setattr(reactor_solvers, "assemble_coupled_rates", counted)
    kwargs = {"volume_L": 1.0} if model_type is BatchReactorModel else {"initial_volume_L": 1.0}
    result = model_type(network(0.001)).simulate(
        {"A": 0.01},
        temperature_K=350,
        duration_s=10,
        heat_transfer=HeatTransferSpec(rho_cp_J_per_L_K=1000),
        **kwargs,
    )
    assert count > 0
    assert result.final_state.amounts_mol["B"] > 0
    assert result.material_balance_error_mol < 1e-12
