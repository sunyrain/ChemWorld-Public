from __future__ import annotations

from dataclasses import replace

import pytest

from chemworld.physchem import (
    FeedStreamSpec,
    HeatTransferSpec,
    RateLawSpec,
    ReactionNetworkSpec,
    ReactionSpec,
    SemiBatchFeedSpec,
    SpeciesSpec,
    cstr_multiple_steady_state_reference_case,
    solve_cstr_multiple_steady_states,
    validate_model_card,
)
from chemworld.physchem.reactor_solvers import _amounts_from_vector
from chemworld.physchem.reactors import (
    BatchReactorModel,
    BatchReactorSession,
    CSTRModel,
    DynamicBatchReactorModel,
    PressureBoundarySpec,
    ReactorState,
    ReactorValidityDomain,
    SemiBatchReactorModel,
    WithdrawalSpec,
    reactor_model_cards,
)


def _first_order_network(*, k: float = 0.02, delta_h_j_mol: float = 0.0) -> ReactionNetworkSpec:
    return ReactionNetworkSpec(
        network_id="reactor_reference_first_order",
        species=(SpeciesSpec("A", "C2H4O2"), SpeciesSpec("P", "C2H4O2")),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="target",
                equation="A => P",
                rate_law=RateLawSpec("target", "mass_action", {"k": k}),
                delta_h_J_per_mol=delta_h_j_mol,
            ),
        ),
    )


def _selectivity_network() -> ReactionNetworkSpec:
    species = tuple(SpeciesSpec(species_id, "C2H4O2") for species_id in ("A", "P", "W"))
    return ReactionNetworkSpec(
        network_id="temperature_time_selectivity_window",
        species=species,
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="target",
                equation="A => P",
                rate_law=RateLawSpec(
                    "target_arrhenius",
                    "arrhenius",
                    {"A": 1.0e7, "Ea_J_per_mol": 50_000.0},
                ),
            ),
            ReactionSpec.from_equation(
                reaction_id="degradation",
                equation="P => W",
                rate_law=RateLawSpec(
                    "degradation_arrhenius",
                    "arrhenius",
                    {"A": 1.0e14, "Ea_J_per_mol": 90_000.0},
                ),
            ),
        ),
    )


def test_first_order_isothermal_batch_matches_analytical_closure() -> None:
    result = BatchReactorModel(_first_order_network(k=0.04)).simulate(
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=320.0,
        duration_s=50.0,
        evaluation_times_s=(0.0, 25.0, 50.0),
    )

    assert result.final_state.amounts_mol["A"] == pytest.approx(
        pytest.importorskip("numpy").exp(-2.0),
        rel=2.0e-6,
    )
    assert result.temperatures_K == pytest.approx((320.0, 320.0, 320.0))
    assert result.volumes_L == pytest.approx((1.0, 1.0, 1.0))
    assert result.diagnostics.material_balance_closed


def test_adiabatic_batch_energy_closure_and_runaway_diagnostics() -> None:
    result = BatchReactorModel(
        _first_order_network(k=1.0, delta_h_j_mol=-100_000.0)
    ).simulate(
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=300.0,
        duration_s=2.0,
        heat_transfer=HeatTransferSpec(rho_cp_J_per_L_K=100.0),
        validity_domain=ReactorValidityDomain(
            maximum_temperature_K=700.0,
            maximum_temperature_rate_K_s=10.0,
        ),
        evaluation_times_s=(0.0, 0.25, 0.5, 1.0, 2.0),
    )

    expected = 300.0 - result.final_state.heat_reaction_J / 100.0
    assert result.final_state.temperature_K == pytest.approx(expected, rel=2.0e-5)
    assert result.diagnostics.sensible_energy_input_J == pytest.approx(
        100.0 * (result.final_state.temperature_K - 300.0),
        rel=2.0e-5,
    )
    assert "temperature_outside_validity_domain" in result.diagnostics.warnings
    assert "thermal_runaway_rate_exceeds_limit" in result.diagnostics.warnings
    assert result.diagnostics.within_validity_domain is False


def test_pressure_boundary_is_explicit_and_domain_checked() -> None:
    pressure = PressureBoundarySpec(pressure_Pa=2.0e6)
    result = BatchReactorModel(_first_order_network(k=0.0)).simulate(
        {"A": 1.0},
        volume_L=2.0,
        temperature_K=300.0,
        duration_s=10.0,
        pressure_boundary=pressure,
        validity_domain=ReactorValidityDomain(maximum_pressure_Pa=1.0e6),
        evaluation_times_s=(0.0, 10.0),
    )

    assert result.initial_state.pressure_Pa == pytest.approx(2.0e6)
    assert result.pressures_Pa == pytest.approx((2.0e6, 2.0e6))
    assert result.final_state.concentrations_mol_L["A"] == pytest.approx(0.5)
    assert "pressure_outside_validity_domain" in result.diagnostics.warnings


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("grid", [(0.0, 2.0), (0.0, 0.25, 0.5, 1.0, 2.0), None])
def test_temperature_rate_limit_is_independent_of_output_sampling(dynamic, grid) -> None:
    network = _first_order_network(k=1.0, delta_h_j_mol=-100_000.0)
    model = DynamicBatchReactorModel(network) if dynamic else BatchReactorModel(network)
    result = model.simulate(
        {"A": 1.0},
        **{"initial_volume_L" if dynamic else "volume_L": 1.0},
        temperature_K=300.0,
        duration_s=2.0,
        heat_transfer=HeatTransferSpec(rho_cp_J_per_L_K=100.0),
        validity_domain=ReactorValidityDomain(
            maximum_temperature_K=2000.0, maximum_temperature_rate_K_s=500.0,
        ),
        evaluation_times_s=grid,
    )
    # Exact initial dT/dt = (-delta H) * k * n_A / (rho_cp * V) = 1000 K/s.
    # The endpoint secant is only 432 K/s and used to miss this violation.
    assert result.diagnostics.maximum_temperature_rate_K_s == pytest.approx(1000.0)
    assert result.diagnostics.warnings == ("thermal_runaway_rate_exceeds_limit",)
    assert not result.diagnostics.within_validity_domain
    assert result.diagnostics.material_balance_closed


@pytest.mark.parametrize("dynamic", [False, True])
def test_resolved_temperature_rate_catches_a_peak_after_the_initial_state(dynamic) -> None:
    network = ReactionNetworkSpec(
        network_id="delayed_heat_release",
        species=tuple(SpeciesSpec(name, "C2H4O2") for name in ("A", "B", "P")),
        reactions=(
            ReactionSpec.from_equation(
                reaction_id="first", equation="A => B",
                rate_law=RateLawSpec("first", "mass_action", {"k": 1.0}),
            ),
            ReactionSpec.from_equation(
                reaction_id="second", equation="B => P",
                rate_law=RateLawSpec("second", "mass_action", {"k": 1.0}),
                delta_h_J_per_mol=-100_000.0,
            ),
        ),
    )
    model = DynamicBatchReactorModel(network) if dynamic else BatchReactorModel(network)
    result = model.simulate(
        {"A": 1.0}, **{"initial_volume_L" if dynamic else "volume_L": 1.0},
        temperature_K=300.0, duration_s=10.0,
        heat_transfer=HeatTransferSpec(rho_cp_J_per_L_K=100.0),
        validity_domain=ReactorValidityDomain(
            maximum_temperature_K=2000.0, maximum_temperature_rate_K_s=200.0,
        ),
        evaluation_times_s=(0.0, 10.0),
    )
    # B(t) = t exp(-t), so heat release peaks at t=1 with dT/dt = 1000/e.
    assert result.diagnostics.maximum_temperature_rate_K_s == pytest.approx(367.879441, rel=0.002)
    assert "thermal_runaway_rate_exceeds_limit" in result.diagnostics.warnings


def test_semibatch_feed_and_withdrawal_share_volume_and_material_ledgers() -> None:
    model = SemiBatchReactorModel(
        _first_order_network(k=0.0),
        feeds=(
            SemiBatchFeedSpec(
                FeedStreamSpec({"A": 0.002}, volumetric_flow_L_s=0.001),
                start_s=0.0,
                end_s=100.0,
            ),
        ),
        withdrawals=(WithdrawalSpec(0.001, start_s=0.0, end_s=100.0),),
    )
    result = model.simulate(
        {"A": 1.0},
        initial_volume_L=1.0,
        temperature_K=300.0,
        duration_s=100.0,
        evaluation_times_s=(0.0, 50.0, 100.0),
    )

    assert result.final_state.volume_L == pytest.approx(1.0, rel=1.0e-8)
    assert result.final_state.material_in_mol["A"] == pytest.approx(0.2, rel=2.0e-5)
    assert result.final_state.material_out_mol["A"] > 0.1
    assert result.material_balance_error_mol < 1.0e-8
    assert result.diagnostics.material_balance_closed


def test_cstr_first_order_steady_state_matches_design_equation_and_residual_gate() -> None:
    network = _first_order_network(k=0.08)
    inlet = FeedStreamSpec({"A": 0.01}, volumetric_flow_L_s=0.01, temperature_K=320.0)
    result = CSTRModel(network, inlet=inlet, volume_L=1.0).simulate_to_steady_state(
        temperature_K=320.0,
        residence_times=16.0,
    )

    tau_s = 100.0
    expected_concentration = 1.0 / (1.0 + 0.08 * tau_s)
    assert result.final_state.concentrations_mol_L["A"] == pytest.approx(
        expected_concentration,
        rel=2.0e-6,
    )
    steady = result.metadata["steady_state"]
    assert steady["converged"] is True
    assert steady["maximum_species_residual_mol_s"] <= steady["tolerance_mol_s"]


def test_constant_volume_cstr_rejects_hydraulically_inconsistent_boundary() -> None:
    with pytest.raises(ValueError, match="equal inlet and outlet"):
        CSTRModel(
            _first_order_network(),
            inlet=FeedStreamSpec({"A": 0.01}, volumetric_flow_L_s=0.01),
            outlet_volumetric_flow_L_s=0.02,
            volume_L=1.0,
        )


def test_existing_exothermic_cstr_reference_exposes_multiple_stable_states() -> None:
    result = solve_cstr_multiple_steady_states(cstr_multiple_steady_state_reference_case())

    assert len(result.steady_states) == 3
    assert tuple(state.stability for state in result.steady_states) == (
        "stable",
        "unstable",
        "stable",
    )


def test_temperature_and_time_create_a_nonmonotonic_selectivity_window() -> None:
    model = BatchReactorModel(_selectivity_network())
    temperature_yields = {
        temperature: model.simulate(
            {"A": 1.0},
            volume_L=1.0,
            temperature_K=temperature,
            duration_s=3.0,
        ).yield_on("P", "A")
        for temperature in (280.0, 320.0, 380.0)
    }
    time_yields = {
        duration: model.simulate(
            {"A": 1.0},
            volume_L=1.0,
            temperature_K=300.0,
            duration_s=duration,
        ).yield_on("P", "A")
        for duration in (3.0, 30.0, 300.0)
    }

    assert temperature_yields[320.0] > temperature_yields[280.0]
    assert temperature_yields[320.0] > temperature_yields[380.0]
    assert time_yields[30.0] > time_yields[3.0]
    assert time_yields[30.0] > time_yields[300.0]


def test_batch_session_retries_do_not_double_advance_or_double_book_energy() -> None:
    session = BatchReactorSession.create(
        DynamicBatchReactorModel(_first_order_network(k=0.01)),
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=300.0,
    )
    configured = session.configure(
        "cfg-1",
        heat_transfer=HeatTransferSpec(
            rho_cp_J_per_L_K=1000.0,
            fixed_heat_W=10.0,
        ),
    )
    first = session.advance("advance-1", duration_s=10.0, operation_type="heat")
    state_after_first = session.state
    retry = session.advance("advance-1", duration_s=10.0, operation_type="heat")

    assert configured.start_time_s == configured.end_time_s == 0.0
    assert first.applied is True
    assert retry.applied is False
    assert session.state == state_after_first
    assert session.state.time_s == pytest.approx(10.0)
    assert session.state.energy_jacket_J == pytest.approx(100.0)
    configure_retry = session.configure("cfg-1")
    wait = session.advance("wait-1", duration_s=5.0, operation_type="wait")
    reaction = session.advance(
        "reaction-1",
        duration_s=5.0,
        operation_type="reaction_advance",
    )
    assert configure_retry.applied is False
    assert wait.start_time_s == pytest.approx(10.0)
    assert reaction.start_time_s == pytest.approx(15.0)
    assert session.state.time_s == pytest.approx(20.0)
    assert session.state.energy_jacket_J == pytest.approx(200.0)
    with pytest.raises(ValueError, match="different payload"):
        session.advance("advance-1", duration_s=20.0, operation_type="wait")


def test_negative_initial_amount_and_exhaustive_withdrawal_fail_explicitly() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        BatchReactorModel(_first_order_network()).simulate(
            {"A": -1.0},
            volume_L=1.0,
            temperature_K=300.0,
            duration_s=1.0,
        )
    model = SemiBatchReactorModel(
        _first_order_network(k=0.0),
        feeds=(),
        withdrawals=(WithdrawalSpec(0.02, start_s=0.0, end_s=100.0),),
    )
    with pytest.raises(ValueError, match="exhaust"):
        model.simulate(
            {"A": 1.0},
            initial_volume_L=1.0,
            temperature_K=300.0,
            duration_s=100.0,
        )
    with pytest.raises(RuntimeError, match="negative amount encountered"):
        _amounts_from_vector(_first_order_network(), (-1.0e-4, 0.0))
    with pytest.raises(ValueError, match="finite"):
        HeatTransferSpec(fixed_heat_W=float("nan"))
    with pytest.raises(ValueError, match="finite nonnegative"):
        ReactorState(
            amounts_mol={"A": float("nan")},
            volume_L=1.0,
            temperature_K=300.0,
            time_s=0.0,
        )


def test_conservation_failure_is_machine_readable() -> None:
    valid = BatchReactorModel(_first_order_network(k=0.0)).simulate(
        {"A": 1.0},
        volume_L=1.0,
        temperature_K=300.0,
        duration_s=1.0,
    )
    corrupted = replace(valid, material_balance_error_mol=0.1)

    assert corrupted.diagnostics.material_balance_closed is False
    assert "material_balance_residual_exceeds_tolerance" in corrupted.diagnostics.warnings


def test_reactor_reference_cards_are_valid_and_do_not_claim_full_pressure_dynamics() -> None:
    cards = {
        card.model_id: card
        for card in reactor_model_cards()
        if card.model_id
        in {
            "dynamic_batch_heat_release_jacket_sampling",
            "dynamic_cstr_startup_shutdown",
        }
    }

    assert set(cards) == {
        "dynamic_batch_heat_release_jacket_sampling",
        "dynamic_cstr_startup_shutdown",
    }
    assert all(card.maturity.value == "reference_validated" for card in cards.values())
    assert all(validate_model_card(card) == [] for card in cards.values())
    assert any(
        "fixed-liquid boundary" in note
        for note in cards["dynamic_batch_heat_release_jacket_sampling"].validity_limits
    )
    assert any(
        evidence.evidence_id == "semibatch-feed-withdrawal-ledger-closure"
        for evidence in cards[
            "dynamic_batch_heat_release_jacket_sampling"
        ].validation_evidence
    )
