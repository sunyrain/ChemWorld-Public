from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.foundation import equipment_settings
from chemworld.physchem.crystallization_adapter_manifest import (
    RUNTIME_MODEL_ID,
    ValidatedCrystallizationRuntimeProvider,
    crystallization_runtime_adapter_manifest,
)
from chemworld.physchem.crystallization_units import (
    CrystallizationExecutionSpec,
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.crystallization_validation import CrystallizationGridCase
from chemworld.runtime.kernel_contracts import ModelProviderResult

REACTION_STEPS = (
    {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
    {"operation": "add_reagent", "amount_mol": 0.010},
    {
        "operation": "add_catalyst",
        # Explicitly preserve the 0.00035 mol catalyst condition used by this
        # characterization before catalyst inventory was separated from the
        # reagent-charge policy.
        "catalyst_amount_mol": 0.00035,
        "catalyst": 1,
    },
    {
        "operation": "heat",
        "target_temperature_K": 385.0,
        "duration_s": 1500.0,
        "stirring_speed_rpm": 720.0,
    },
    {"operation": "wait", "duration_s": 900.0, "stirring_speed_rpm": 720.0},
    {"operation": "measure", "instrument": "hplc"},
)


class RecordingProvider(ValidatedCrystallizationRuntimeProvider):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def evaluate(self, inputs: Any) -> ModelProviderResult:
        self.calls.append(dict(inputs))
        return super().evaluate(inputs)


def test_seed_crystal_limit_is_cumulative_across_repeated_calls() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        _, _, _, _, first = env.step({"operation": "seed_crystals", "seed_mass_g": 0.049})
        _, _, _, _, second = env.step({"operation": "seed_crystals", "seed_mass_g": 0.002})

        assert first["transaction_status"] == "committed"
        assert second["transaction_status"] == "validation_failed"
        assert second["preconditions"]["payload_bounds:seed_mass_g"] is False
        settings = equipment_settings(env.unwrapped._state.equipment, "crystallizer")
        assert settings["crystal_seed_mass_g"] == pytest.approx(0.049)
    finally:
        env.close()


def test_runtime_uses_actual_temperature_population_without_prescribed_profile_provider() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        base: Any = env.unwrapped
        for action in REACTION_STEPS:
            env.step(action)
        env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        provider = RecordingProvider()
        base.runtime.domain_services.crystallization.runtime_provider = provider
        before = base._state
        info = env.step(
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 1800.0}
        )[-1]
        assert info["transaction_status"] == "committed"
        assert provider.calls == []
        execution = base.runtime.domain_services.crystallization.last_provider_execution
        assert execution["success"] is True
        assert execution["model_id"] == "finite-crystal-cohorts-temperature-v1"
        state = base._state
        settings = equipment_settings(state.equipment, "crystallizer")
        assert settings["crystallization_model_id"] == execution["model_id"]
        assert state.ledger.time_s - before.ledger.time_s == 1800
        assert 278.15 < state.temperature_K < before.temperature_K
        assert abs(state.metadata["last_energy_transition"]["energy_balance_residual_J"]) < 1e-6
        assert settings["material_balance_error_mol"] <= 1e-10
        assert settings["particle_target_balance_error_mol"] <= 1e-10
        assert settings["csd_number_moment_0"] == pytest.approx(
            settings["csd_total_particle_count"]
        )
        assert settings["csd_volume_moment_3_m3"] > 0
        assert settings["temperature_history_K"][0] == before.temperature_K
        assert settings["temperature_history_K"][-1] == state.temperature_K
        assert len(settings["execution_history"]) == 1
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
    finally:
        env.close()


def test_crystallization_slurry_can_redissolve_and_recool_without_recreating_solids() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        _, _, _, _, first_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1800.0,
            }
        )
        assert first_info["transaction_status"] == "committed"
        base: Any = env.unwrapped
        view = base.runtime.domain_services.crystallization.species_view
        target = view.primary_target_species
        first = base._state
        first_solid = first.phases.phases["solid"].species_amounts_mol[target]

        # Simultaneous reaction/crystal calorics are undeclared in this legacy world.
        env.step({"operation": "quench"})
        _, _, _, _, heat_info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": 300.0,
                "duration_s": 300.0,
                "stirring_speed_rpm": 600.0,
            }
        )
        heated = base._state
        heated_solid = heated.phases.phases["solid"].species_amounts_mol[target]
        assert heat_info["transaction_status"] == "committed"
        assert heated.process.metrics["crystal_redissolved_target_mol"] > 0.0
        assert heated_solid < first_solid
        assert heated.phases.total_amounts_mol() == pytest.approx(heated.species_amounts)

        _, _, _, _, second_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1800.0,
            }
        )
        state = base._state
        settings = equipment_settings(state.equipment, "crystallizer")
        history = settings["execution_history"]
        assert second_info["transaction_status"] == "committed"
        assert len(history) == 3
        assert history[-1]["prior_solid_target_mol"] == pytest.approx(heated_solid)
        assert history[-1]["prior_solid_target_mol"] > 0.0
        assert state.phases.phases["solid"].species_amounts_mol[target] >= (heated_solid)
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        assert base.constitution.check_state(state).passed
    finally:
        env.close()


def test_quench_stops_reaction_but_allows_later_thermal_holds() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-purification", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS[:5]:
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"
        _, _, _, _, quench_info = env.step({"operation": "quench"})
        assert quench_info["transaction_status"] == "committed"
        quenched = env.unwrapped._state
        amounts = quenched.species_amounts.copy()
        reaction_heat = quenched.ledger.heat_reaction_J
        reaction_count = quenched.process.metrics["reaction_advance_count"]

        _, _, _, _, heat_info = env.step(
            {
                "operation": "heat",
                "target_temperature_K": 350.0,
                "duration_s": 300.0,
                "stirring_speed_rpm": 600.0,
            }
        )
        state = env.unwrapped._state
        reactor = equipment_settings(state.equipment, "batch_reactor")
        assert heat_info["transaction_status"] == "committed"
        assert state.temperature_K == pytest.approx(350.0)
        assert state.species_amounts == pytest.approx(amounts)
        assert state.ledger.heat_reaction_J == pytest.approx(reaction_heat)
        assert state.process.metrics["reaction_advance_count"] == reaction_count
        assert reactor["reaction_stopped"] is True
        assert reactor["reaction_chemistry_advanced"] is False
        assert reactor["last_operation_semantic"] == "thermal_hold_after_quench"
    finally:
        env.close()


def test_cooling_seed_and_time_perturbations_change_yield_purity_size_tradeoff() -> None:
    mild = _runtime_outcome(seed_mass_g=0.006, target_temperature_K=305.0, duration_s=1200.0)
    deep = _runtime_outcome(
        seed_mass_g=0.006,
        target_temperature_K=278.15,
        duration_s=1800.0,
    )
    long = _runtime_outcome(
        seed_mass_g=0.006,
        target_temperature_K=278.15,
        duration_s=3600.0,
    )
    low_seed = _runtime_outcome(
        seed_mass_g=0.002,
        target_temperature_K=278.15,
        duration_s=1800.0,
    )
    high_seed = _runtime_outcome(
        seed_mass_g=0.012,
        target_temperature_K=278.15,
        duration_s=1800.0,
    )

    assert deep["yield"] > mild["yield"] + 0.30
    assert long["time_s"] > deep["time_s"]
    # The finite apparatus has already reached its stationary saturation at
    # 1800 s. Continuing a hold need not strictly improve a plateaued outcome.
    for key in ("yield", "purity", "d50_m", "fines_fraction"):
        assert long[key] == pytest.approx(deep[key], rel=1e-6, abs=1e-12)
    assert high_seed["purity"] > low_seed["purity"] + 5.0e-4, (high_seed, low_seed)
    assert high_seed["maximum_supersaturation"] < low_seed["maximum_supersaturation"], (
        high_seed,
        low_seed,
    )


def test_filter_consumes_validated_solid_and_closes_retention_ledger() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1800.0,
            }
        )
        env.step({"operation": "measure", "instrument": "hplc"})
        before = env.unwrapped._state
        species_view = env.unwrapped.runtime.domain_services.crystallization.species_view
        target = species_view.primary_target_species
        impurity = species_view.primary_impurity_species
        solid_before = before.phases.phases["solid"].species_amounts_mol

        _obs, _reward, _terminated, _truncated, info = env.step({"operation": "filter_crystals"})

        assert info["transaction_status"] == "committed"
        state = env.unwrapped._state
        settings = equipment_settings(state.equipment, "crystal_filter")
        solid_after = state.phases.phases["solid"].species_amounts_mol
        assert solid_after[target] == pytest.approx(solid_before[target] * 0.96)
        assert solid_after[impurity] == pytest.approx(solid_before[impurity] * 0.92)
        assert settings["target_returned_to_filtrate_mol"] == pytest.approx(
            solid_before[target] * 0.04
        )
        assert settings["impurity_returned_to_filtrate_mol"] == pytest.approx(
            solid_before[impurity] * 0.08
        )
        assert settings["component_balance_error_mol"] <= 1.0e-12
        assert state.phases.total_amounts_mol() == pytest.approx(state.species_amounts)
        metrics = state.process.metrics
        assert metrics["crystal_yield"] == pytest.approx(metrics["seed_excluded_recovery"])
        assert metrics["filtered_product_from_solution_mol"] == pytest.approx(
            settings["filtered_product_from_solution_mol"]
        )
        assert settings["filtered_product_mol"] == pytest.approx(
            settings["retained_seed_mol"] + settings["filtered_product_from_solution_mol"]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "seed_mass_g,target_temperature_K,duration_s,expected_reason,expected_status",
    [
        (
            0.006,
            278.15,
            60.0,
            "payload_coupling:maximum_cooling_rate_K_s",
            "validation_failed",
        ),
        (
            1.0e-6,
            278.15,
            1800.0,
            "cool_crystallize_seed_population_effective",
            "rolled_back",
        ),
    ],
)
def test_invalid_cooling_proposals_roll_back_without_crystal_state(
    seed_mass_g: float,
    target_temperature_K: float,
    duration_s: float,
    expected_reason: str,
    expected_status: str,
) -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        env.step({"operation": "seed_crystals", "seed_mass_g": seed_mass_g})
        base: Any = env.unwrapped
        before = base._state

        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": target_temperature_K,
                "duration_s": duration_s,
            }
        )

        assert info["transaction_status"] == expected_status
        assert info["measurement_cost"] == 0.0
        assert base._state.temperature_K == pytest.approx(before.temperature_K)
        assert base._state.species_amounts == pytest.approx(before.species_amounts)
        assert base._state.phases.to_dict() == before.phases.to_dict()
        execution = base.runtime.domain_services.crystallization.last_provider_execution
        assert execution == {}
        assert info["preconditions"][expected_reason] is False
        assert (
            equipment_settings(base._state.equipment, "crystallizer").get(
                "crystallization_model_id"
            )
            is None
        )
    finally:
        env.close()


def test_cooling_above_solubility_reference_domain_is_publicly_rejected() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        env.step(
            {
                "operation": "heat",
                "target_temperature_K": 450.0,
                "duration_s": 1000.0,
                "stirring_speed_rpm": 500.0,
            }
        )
        base: Any = env.unwrapped
        assert base._state.temperature_K > 430.0
        provider = RecordingProvider()
        base.runtime.domain_services.crystallization.runtime_provider = provider

        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 330.0,
                "duration_s": 1800.0,
            }
        )

        assert info["transaction_status"] == "rolled_back"
        assert not provider.calls
        assert (
            info["preconditions"]["cool_crystallize_reference_temperature_in_solubility_domain"]
            is False
        )
        assert info["preconditions"].get("runtime_domain_valid") is None
    finally:
        env.close()


def test_exhausted_target_rolls_back_and_does_not_fabricate_crystals() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        env.step({"operation": "add_solvent", "volume_L": 0.028, "solvent": 2})
        env.step({"operation": "add_reagent", "amount_mol": 0.010})
        base: Any = env.unwrapped
        before = base._state
        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 278.15,
                "duration_s": 1800.0,
            }
        )
        assert info["transaction_status"] == "rolled_back"
        assert base._state.species_amounts == pytest.approx(before.species_amounts)
        assert base._state.phases.to_dict() == before.phases.to_dict()
        assert base.runtime.domain_services.crystallization.last_provider_execution == {}
        assert (
            "cool_crystallize_requires_reaction_or_seed"
            in info["world_events"][0]["payload"]["failed_preconditions"]
        )
    finally:
        env.close()


def test_isothermal_negative_crystallization_is_a_committed_observation() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        base: Any = env.unwrapped
        # At the world-bound material solubility, 330 K is undersaturated for
        # this feed; an isothermal call therefore remains a valid negative
        # crystallization observation without redefining the material curve
        # from the current feed concentration.
        base._state = base._state.replace(temperature_K=330.0)
        initial_temperature = base._state.temperature_K

        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": initial_temperature,
                "duration_s": 1200.0,
            }
        )

        assert info["transaction_status"] == "committed"
        execution = base.runtime.domain_services.crystallization.last_provider_execution
        assert execution["success"] is True
        assert execution["diagnostics"]["crystal_population_formed"] is False
        assert execution["diagnostics"]["meaningful_transfer"] is False
        assert base._state.process.metrics["crystal_yield"] == pytest.approx(0.0)
    finally:
        env.close()


def test_empty_crystal_population_remains_a_valid_formal_recooling_state() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        base: Any = env.unwrapped

        _, _, _, _, first_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 330.0,
                "duration_s": 1200.0,
            }
        )
        empty_settings = equipment_settings(base._state.equipment, "crystallizer")
        assert first_info["transaction_status"] == "committed"
        assert empty_settings["csd_total_particle_count"] == pytest.approx(0.0)
        assert empty_settings["csd_d50_m"] == pytest.approx(0.0)
        assert "cool_crystallize" in {action["operation"] for action in base.available_actions()}

        provider = RecordingProvider()
        base.runtime.domain_services.crystallization.runtime_provider = provider
        _, _, _, _, second_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 280.0,
                "duration_s": 1800.0,
            }
        )

        assert second_info["transaction_status"] == "committed"
        assert second_info["preconditions"].get("runtime_domain_valid") is None
        assert provider.calls[-1]["case"].seed_mass_g == pytest.approx(0.0)
        assert provider.calls[-1]["case"].seed_diameter_m == pytest.approx(100.0e-6)
        settings = equipment_settings(base._state.equipment, "crystallizer")
        assert len(settings["execution_history"]) == 2
    finally:
        env.close()


def test_seeding_after_empty_population_remains_a_valid_formal_path() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        base: Any = env.unwrapped
        _, _, _, _, first_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 330.0,
                "duration_s": 1200.0,
            }
        )
        assert first_info["transaction_status"] == "committed"
        assert equipment_settings(base._state.equipment, "crystallizer")[
            "csd_d50_m"
        ] == pytest.approx(0.0)

        _, _, _, _, assay_info = env.step({"operation": "measure", "instrument": "hplc"})
        _, _, _, _, seed_info = env.step({"operation": "seed_crystals", "seed_mass_g": 0.006})
        assert assay_info["transaction_status"] == "committed"
        assert seed_info["transaction_status"] == "committed"

        provider = RecordingProvider()
        base.runtime.domain_services.crystallization.runtime_provider = provider
        _, _, _, _, second_info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": 280.0,
                "duration_s": 1800.0,
            }
        )

        case = provider.calls[-1]["case"]
        assert second_info["transaction_status"] == "committed"
        assert second_info["preconditions"].get("runtime_domain_valid") is None
        assert case.seed_mass_g > 0.0
        assert case.seed_diameter_m == pytest.approx(100.0e-6)
        assert (
            equipment_settings(base._state.equipment, "crystallizer")["csd_total_particle_count"]
            > 0.0
        )
    finally:
        env.close()


def test_provider_fails_closed_for_no_population_no_transfer_and_nonconvergence() -> None:
    provider = ValidatedCrystallizationRuntimeProvider()
    policy = CrystallizationExecutionSpec.strict_runtime()

    no_population = provider.evaluate(
        {
            "case": _case(seed_mass_g=0.0, nucleation_coefficient=0.0),
            "time_steps": 60,
            "execution_spec": policy,
        }
    )
    no_transfer_case = _case(seed_mass_g=0.006)
    no_transfer = provider.evaluate(
        {
            "case": replace(
                no_transfer_case,
                final_temperature_K=320.0,
                solubility_curve=replace(
                    no_transfer_case.solubility_curve,
                    reference_solubility_mol_L=0.40,
                ),
            ),
            "time_steps": 60,
            "execution_spec": policy,
        }
    )
    nonconverged = provider.evaluate(
        {
            "case": _case(
                seed_mass_g=0.006,
                nucleation_coefficient=0.0,
                growth_coefficient=1.0e-4,
            ),
            "time_steps": 24,
            "execution_spec": replace(policy, max_growth_solver_iterations=1),
        }
    )

    assert no_population.success is False
    assert "no crystal population" in (no_population.failure_reason or "")
    assert no_transfer.success is False
    assert "no meaningful crystallization transfer" in (no_transfer.failure_reason or "")
    assert nonconverged.success is False
    assert "solver did not converge" in (nonconverged.failure_reason or "")
    assert not no_population.outputs and not no_transfer.outputs and not nonconverged.outputs


def test_runtime_manifest_is_truthful() -> None:
    manifest = crystallization_runtime_adapter_manifest()
    assert manifest.status == "integrated"
    assert manifest.provider_contract.role.value == "runtime"
    assert manifest.provider_contract.model_id == RUNTIME_MODEL_ID
    assert manifest.provider_contract.maturity.value == "professional_candidate"


def _runtime_outcome(
    *,
    seed_mass_g: float,
    target_temperature_K: float,
    duration_s: float,
) -> dict[str, float]:
    env = gym.make("ChemWorld", task_id="reaction-to-crystallization", seed=0)
    try:
        env.reset(seed=0)
        for action in REACTION_STEPS:
            env.step(action)
        env.step({"operation": "seed_crystals", "seed_mass_g": seed_mass_g})
        _obs, _reward, _terminated, _truncated, info = env.step(
            {
                "operation": "cool_crystallize",
                "target_temperature_K": target_temperature_K,
                "duration_s": duration_s,
            }
        )
        assert info["transaction_status"] == "committed"
        state = env.unwrapped._state
        settings = equipment_settings(state.equipment, "crystallizer")
        return {
            "yield": state.process.metrics["crystal_yield"],
            "purity": state.process.metrics["crystal_purity"],
            "d50_m": settings["csd_d50_m"],
            "fines_fraction": settings["csd_fines_number_fraction"],
            "maximum_supersaturation": settings["maximum_supersaturation_ratio"],
            "time_s": state.ledger.time_s,
        }
    finally:
        env.close()


def _case(
    *,
    seed_mass_g: float,
    nucleation_coefficient: float = 2.0e7,
    growth_coefficient: float = 2.0e-8,
) -> CrystallizationGridCase:
    return CrystallizationGridCase(
        feed_amounts_mol={"P": 0.030, "B": 0.003},
        target_component="P",
        impurity_component="B",
        solvent_volume_L=0.10,
        initial_temperature_K=320.0,
        final_temperature_K=280.0,
        duration_s=1800.0,
        solubility_curve=SolubilityCurveSpec(
            model_id="test_solubility",
            reference_solubility_mol_L=0.20,
            reference_temperature_K=320.0,
            dissolution_enthalpy_J_mol=18_000.0,
            minimum_temperature_K=260.0,
            maximum_temperature_K=340.0,
            provenance_id="test_solubility_reference",
        ),
        kinetics=CrystallizationKineticsSpec(
            model_id="test_kinetics",
            primary_nucleation_coefficient_per_L_s=nucleation_coefficient,
            primary_nucleation_exponent=2.0,
            growth_coefficient_m_s=growth_coefficient,
            growth_exponent=1.0,
            crystal_density_kg_m3=1200.0,
            target_molecular_weight_kg_mol=0.10,
            nucleus_diameter_m=8.0e-6,
            impurity_occlusion_mol_per_mol=0.02,
            supersaturation_occlusion_factor=0.5,
            fines_threshold_m=20.0e-6,
            provenance_id="test_kinetics_reference",
        ),
        seed_mass_g=seed_mass_g,
        seed_diameter_m=100.0e-6,
    )
