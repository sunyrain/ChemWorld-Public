from dataclasses import replace
from math import pi

import gymnasium as gym
import pytest
from tests.fixtures.process_recipes import TASKS, reference_actions

import chemworld  # noqa: F401
from chemworld.foundation import equipment_settings
from chemworld.physchem.crystallization_units import (
    CrystallizationKineticsSpec,
    SolubilityCurveSpec,
)
from chemworld.physchem.crystallization_validation import CrystallizationGridCase
from chemworld.runtime.full_process_contract import (
    FULL_PROCESS_POPULATION_CONTRACT,
    FULL_PROCESS_SEED_CONTRACT,
    shrink_population,
)


def test_same_isothermal_time_grid_survives_operation_boundary():
    kinetics = CrystallizationKineticsSpec(
        model_id="test",
        primary_nucleation_coefficient_per_L_s=2e7,
        primary_nucleation_exponent=2,
        growth_coefficient_m_s=2e-8,
        growth_exponent=1,
        crystal_density_kg_m3=1200,
        target_molecular_weight_kg_mol=0.06,
        nucleus_diameter_m=8e-6,
        impurity_occlusion_mol_per_mol=0.02,
        supersaturation_occlusion_factor=0.5,
        fines_threshold_m=20e-6,
        provenance_id="test",
    )
    curve = SolubilityCurveSpec(
        model_id="test",
        reference_solubility_mol_L=0.06,
        reference_temperature_K=298.15,
        dissolution_enthalpy_J_mol=20000,
        minimum_temperature_K=250,
        maximum_temperature_K=430,
        provenance_id="test",
    )
    case = CrystallizationGridCase(
        feed_amounts_mol={"P": 0.003, "B": 0.0002},
        target_component="P",
        impurity_component="B",
        solvent_volume_L=0.028,
        initial_temperature_K=285,
        final_temperature_K=285,
        duration_s=1200,
        solubility_curve=curve,
        kinetics=kinetics,
        seed_mass_g=0.006,
        retain_population=True,
    )
    full = case.run(40)
    first = replace(case, duration_s=600).run(20)
    second = replace(
        case,
        duration_s=600,
        feed_amounts_mol=first.mother_liquor_amounts_mol,
        seed_mass_g=first.crystals_amounts_mol["P"] * 60,
        initial_cohorts=first.population_cohorts,
    ).run(20)
    assert second.crystals_amounts_mol["P"] == pytest.approx(
        full.crystals_amounts_mol["P"], abs=1e-12
    )
    assert second.crystal_size_distribution.to_dict() == pytest.approx(
        full.crystal_size_distribution.to_dict()
    )


def test_radial_dissolution_preserves_mass_and_removes_smallest_particles():
    original = ((1e6, 8e-6), (2e4, 100e-6))
    remaining = shrink_population(original, 0.25)
    assert sum(n * d**3 for n, d in remaining) == pytest.approx(
        sum(n * d**3 for n, d in original) * 0.25, rel=1e-12
    )
    assert len(remaining) == 1
    assert remaining[0][0] == 2e4


def test_independent_distillation_cut_limits_molar_receiver_and_conserves_carriers():
    from chemworld.agent_interface import action_schema

    env = gym.make(
        "ChemWorld",
        task_id=TASKS[2],
        budget_override=90,
        full_process_contract_id=FULL_PROCESS_POPULATION_CONTRACT,
    )
    try:
        env.reset(seed=0)
        for action in reference_actions(TASKS[2])[:8]:
            assert env.step(action)[-1]["transaction_status"] == "committed"
        initial = env.unwrapped._state.volume_L
        assert "cut_fraction" in str(action_schema(env, "distill"))
        info = env.step(
            {
                "operation": "distill",
                "target_temperature_K": 365,
                "duration_s": 3600,
                "reflux_ratio": 10,
                "cut_fraction": 0.05,
            }
        )[-1]
        assert info["transaction_status"] == "committed"
        # The column cut is a fraction of all feed moles (solute + carrier),
        # not a fraction of the solvent volume. Check its actual material basis.
        state = env.unwrapped._state
        kernel = equipment_settings(state.equipment, "distillation_column")["distillation_kernel"]
        feed = kernel["feed_amounts_mol"]
        receiver = kernel["outlets"]["distillate"]
        bottoms = kernel["outlets"]["bottoms"]
        assert sum(receiver.values()) <= sum(feed.values()) * 0.05 + 1e-12
        assert kernel["actual_distillate_cut_fraction"] <= 0.05 + 1e-12
        for component, amount in feed.items():
            assert receiver[component] + bottoms[component] == pytest.approx(amount, abs=1e-12)
        assert (state.phases.phases["distillate"].volume_L
                + state.phases.phases["bottoms"].volume_L) == pytest.approx(initial, abs=1e-12)
        assert not env.unwrapped.operation_validator.validate(
            {
                "operation": "distill",
                "target_temperature_K": 365,
                "duration_s": 3600,
                "reflux_ratio": 10,
                "cut_fraction": -0.2,
            },
            env.unwrapped._state,
        ).is_valid
    finally:
        env.close()


def test_runtime_population_matches_solid_through_seed_cool_sample_reheat_filter():
    env = gym.make(
        "ChemWorld",
        task_id=TASKS[1],
        budget_override=90,
        full_process_contract_id=FULL_PROCESS_SEED_CONTRACT,
    )
    try:
        env.reset(seed=0)
        actions = [
            *reference_actions(TASKS[1])[:9],
            {"operation": "measure", "instrument": "particle_size"},
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 7200},
            {"operation": "measure", "instrument": "hplc"},
            {
                "operation": "heat",
                "target_temperature_K": 295,
                "duration_s": 300,
                "stirring_speed_rpm": 600,
            },
            {"operation": "measure", "instrument": "particle_size"},
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 7200},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "filter_crystals"},
        ]
        for a in actions:
            before = env.unwrapped._state
            old_settings = equipment_settings(before.equipment, "crystallizer")
            old_seed = sum(
                float(old_settings.get(k, 0))
                for k in ("seed_target_mol", "dissolved_seed_target_mol")
            )
            _, _, _, _, info = env.step(a)
            assert info["transaction_status"] == "committed", (a, info)
            state = env.unwrapped._state
            if a["operation"] == "heat" and before.quenched:
                assert state.phases.phases["solid"].species_amounts_mol["P"] < (
                    before.phases.phases["solid"].species_amounts_mol["P"] - 1e-8
                )
                # Phase redistribution integrates floating-point inventories;
                # 1 pmol is stricter than the solver's material-balance tolerance.
                assert state.species_amounts == pytest.approx(
                    before.species_amounts, rel=0, abs=1e-12
                )
                transition = state.metadata["last_energy_transition"]
                assert transition["phase_change_heat_J"] > 0
                assert (
                    state.ledger.energy_jacket_J - before.ledger.energy_jacket_J
                    == pytest.approx(transition["jacket_energy_J"])
                )
                assert (
                    state.ledger.heat_reaction_J - before.ledger.heat_reaction_J
                    == pytest.approx(-transition["phase_change_heat_J"])
                )
                assert state.ledger.cost - before.ledger.cost == pytest.approx(
                    0.01
                    + a["duration_s"] / 3600 * 0.015
                    + abs(transition["jacket_energy_J"]) / 250000
                )
                assert (
                    equipment_settings(state.equipment, "crystallizer")["dissolved_seed_target_mol"]
                    > 0
                )
            if old_seed and a["operation"] != "seed_crystals":
                settings = equipment_settings(state.equipment, "crystallizer")
                actual = settings.get("seed_target_mol", 0) + settings.get(
                    "dissolved_seed_target_mol", 0
                )
                sampled = (
                    state.metadata.get("last_sample_fraction", 0)
                    if (a.get("instrument") == "hplc")
                    else 0
                )
                assert actual == pytest.approx(old_seed * (1 - sampled), abs=1e-12)
            cohorts = equipment_settings(state.equipment, "crystallizer").get("population_cohorts")
            if cohorts is not None:
                from chemworld.runtime.crystallization_services import (
                    ChemWorldCrystallizationServices,
                )
                from chemworld.runtime.species import MechanismSpeciesView

                mw = ChemWorldCrystallizationServices(
                    env.unwrapped.world,
                    MechanismSpeciesView(env.unwrapped.scenario_instance.compiled_mechanism),
                )._target_molecular_weight_kg_mol()
                mass = sum(n * pi / 6 * d**3 * 1200 / mw for n, d in cohorts)
                assert mass == pytest.approx(
                    state.phases.phases["solid"].species_amounts_mol["P"], abs=1e-10
                )
        assert env.unwrapped.observation_kernel._truth_values(state)["crystal_yield"] >= 0
    finally:
        env.close()


def test_recrystallized_seed_alone_cannot_be_reported_as_new_product():
    from chemworld.foundation.state import SpeciesLedger
    from chemworld.runtime.crystallization_services import ChemWorldCrystallizationServices
    from chemworld.runtime.reaction_thermal_services import ChemWorldReactionThermalServices
    from chemworld.runtime.species import MechanismSpeciesView

    env = gym.make(
        "ChemWorld", task_id=TASKS[1], full_process_contract_id=FULL_PROCESS_SEED_CONTRACT
    )
    try:
        env.reset(seed=0)
        base = env.unwrapped
        view = MechanismSpeciesView(base.scenario_instance.compiled_mechanism)
        thermal = ChemWorldReactionThermalServices(base.world, view)
        crystals = ChemWorldCrystallizationServices(base.world, view, thermal=thermal)
        state = base._state.replace(
            species_amounts={**dict.fromkeys(base._state.species_amounts, 0), "A": 0.01},
            species=SpeciesLedger(initial_amounts_mol={"A": 0.01}),
            phases=None,
            volume_L=0.001,
            temperature_K=278.15,
            quenched=True,
        )
        from chemworld.foundation.state import PhaseLedger, PhaseRecord
        state = state.replace(phases=PhaseLedger({
            "reactor_liquid": PhaseRecord(
                "reactor_liquid", state.vessel_id, "liquid", state.volume_L,
                dict(state.species_amounts),
            ),
        }))
        state = crystals.seed_crystals(state, {"seed_mass_g": 0.005})
        def heat_slurry(state):
            # This seed-only material fixture deliberately has no prior reaction
            # or assay. Exercise the current thermal interface, not preset-task
            # preconditions or the removed instantaneous dissolution helper.
            result = thermal.integrate(
                state, {"operation": "heat", "target_temperature_K": 350,
                        "duration_s": 600, "stirring_speed_rpm": 600}, heat=True,
            )
            assert result.success and result.state is not None, result.diagnostics
            return result.state

        old_solid = state.phases.phases["solid"].species_amounts_mol["P"]
        state = heat_slurry(state)
        assert state.phases.phases["solid"].species_amounts_mol["P"] < old_solid
        def cool_slurry(state):
            result = crystals.cool_crystallize(
                state, {"operation": "cool_crystallize", "duration_s": 14400,
                        "target_temperature_K": 250},
            )
            assert result.success and result.state is not None, result.diagnostics
            return result.state

        state = cool_slurry(state)
        assert state.phases.phases["solid"].species_amounts_mol["P"] > 0
        assert base.observation_kernel._truth_values(state)["crystal_yield"] == pytest.approx(
            0, abs=1e-12
        )
        state = crystals.filter_crystals(state)
        state = crystals.resuspend_crystals(state, {"volume_L": 0.001, "solvent": 0})
        state = crystals.seed_crystals(state, {"seed_mass_g": 0.001})
        assert base.observation_kernel._truth_values(state)["crystal_yield"] == pytest.approx(
            0, abs=1e-12
        )
        state = heat_slurry(state)
        state = cool_slurry(state)
        assert base.observation_kernel._truth_values(state)["crystal_yield"] == pytest.approx(
            0, abs=1e-12
        )
    finally:
        env.close()
