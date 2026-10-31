"""Assay and preset contracts formerly mixed with retired research applications."""

import gymnasium as gym
import pytest
from tests.fixtures.process_recipes import reference_actions

import chemworld  # noqa: F401 -- registers the environment
from chemworld.agent_interface import full_process_operational_state
from chemworld.envs.chemworld_env import ChemWorldEnv
from chemworld.foundation import upsert_equipment_record
from chemworld.foundation.state import PhaseLedger, PhaseRecord, ProcessLedger, SpeciesLedger
from chemworld.runtime.full_process_contract import (
    FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    FULL_PROCESS_SEED_CONTRACT,
)
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.phase_kernel import INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1
from chemworld.world.scoring import (
    ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2,
    PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
    PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
    TASK_DERIVED_SCORING_CONTRACT,
    TaskScoringContract,
)
from chemworld.world.separation_kernel import downstream_truth_values


@pytest.mark.parametrize(
    "contract_id",
    [
        TASK_DERIVED_SCORING_CONTRACT,
        PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
        PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
    ],
)
def test_partition_assay_retains_initial_product_charge(contract_id: str) -> None:
    """An extraction-only feed has product, but no initial reaction reagent."""
    import numpy as np

    from chemworld.agents.task_recipes import (
        task_recipe_dimension,
        task_recipe_from_unit_vector,
    )

    scores, observed = [], []
    for interventions in (
        [],
        [{"kind": "mechanism_family", "mode": "constitutive_law_family", "severity": 0.8}],
    ):
        with gym.make(
            "ChemWorld", task_id="partition-discovery", seed=0,
            scoring_contract_id=contract_id, world_interventions=interventions,
        ) as env:
            env.reset(seed=0)
            base = env.unwrapped
            view = MechanismSpeciesView(base.scenario_instance.compiled_mechanism)
            initial = base._state.species.initial_amounts_mol
            charge = sum(initial.get(key, 0.0) for key in view.target_species)
            assert charge > 0.0
            assert view.initial_reactant_amount(base._state) == 0.0
            task_info = base.task_info()
            recipe = task_recipe_from_unit_vector(
                task_info, np.full(task_recipe_dimension(task_info), 0.5),
            )
            for action in recipe["steps"]:
                observation, _, _, _, info = env.step(action)
                assert info["transaction_status"] == "committed"
                if action["operation"] not in {"mix", "separate_phase"}:
                    continue
                state = base._state
                organic = state.phases.phases["organic"].species_amounts_mol
                # Sampling carries its share of the reference charge away;
                # use the retained material's ledger, not its current product yield.
                retained_charge = sum(
                    state.species.initial_amounts_mol.get(key, 0.0)
                    for key in view.target_species
                )
                assert 0.0 < retained_charge <= charge
                expected = (
                    sum(organic.get(key, 0.0) for key in view.target_species) / retained_charge
                )
                truth = base.observation_kernel._truth_values(state)
                assert 0.0 < expected < 1.0
                assert truth["product_in_organic"] == pytest.approx(expected, abs=1e-12)
                assert truth["recovery"] == pytest.approx(expected, abs=1e-12)
                if contract_id != PARTITION_S0_EXTRACTION_EFFICIENCY_V3:
                    # Preserve the existing preset's P_org-only observable scope.
                    assert truth["product_in_aqueous"] == 0.0
                assert state.process.metrics["process_mass_balance_error"] <= 1e-8
            scores.append(info["leaderboard_score"])
            observed.append(float(observation["product_in_organic"][0]))
    assert abs(scores[1] - scores[0]) > 1e-8
    assert abs(observed[1] - observed[0]) > 1e-8


def test_rejected_empty_assay_does_not_evaluate_physical_truth(monkeypatch):
    with gym.make("ChemWorld", task_id="equilibrium-characterization", seed=0) as env:
        env.reset(seed=0)
        monkeypatch.setattr(
            env.unwrapped.observation_kernel,
            "_equilibrium_truth_values",
            lambda state: pytest.fail("Rejected empty assay requested physical truth"),
        )
        _, _, _, _, info = env.step({"operation": "measure", "instrument": "final_assay"})
        assert info["transaction_status"] == "rolled_back"


def test_aqueous_selection_preserves_other_phase_inventory():
    with gym.make(
        "ChemWorld",
        task_id="reaction-to-purification",
        seed=0,
        full_process_contract_id=FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    ) as env:
        env.reset(seed=0)
        base = env.unwrapped
        state = base._state
        phases = PhaseLedger(
            {
                "organic": PhaseRecord(
                    phase_id="organic",
                    vessel_id=state.vessel_id,
                    phase_type="organic",
                    volume_L=0.01,
                    species_amounts_mol={"P_org": 0.002},
                ),
                "aqueous": PhaseRecord(
                    phase_id="aqueous",
                    vessel_id=state.vessel_id,
                    phase_type="aqueous",
                    volume_L=0.01,
                    selected=True,
                    species_amounts_mol={"P_aq": 0.003, "B_aq": 0.001, "A": 0.0},
                ),
            }
        )
        state = state.replace(
            phases=phases,
            species_amounts=phases.total_amounts_mol(),
            species=SpeciesLedger(initial_amounts_mol={"A": 0.02}),
            volume_L=0.02,
        )
        # The retired study supplied both phase-specific product identities.
        # Test the product kernel with those roles explicitly; the default preset
        # targets P_org only and is not that historical study adapter.
        values = downstream_truth_values(
            state,
            product_amount_mol=0.005,
            initial_product_mol=0.02,
            target_species=("P_org", "P_aq"),
            impurity_species=("B_aq",),
        )
        assert values["purity"] == pytest.approx(0.75, abs=1e-12)
        assert values["recovery"] == pytest.approx(0.15, abs=1e-12)
        assert values["process_mass_balance_error"] < 1e-12
        assert state.phases.phases["organic"].species_amounts_mol == {"P_org": 0.002}


def test_electrochemical_s0_v2_score_has_no_composite_double_counting() -> None:
    from chemworld.tasks import get_task

    task = get_task("electrochemical-conversion")
    contract = TaskScoringContract.from_success_metrics(
        objective=task.objective,
        success_metrics=task.success_metrics,
        contract_id=ELECTROCHEMICAL_S0_BALANCED_EFFICIENCY_V2,
    )

    assert "reaction_score" not in contract.component_weights
    assert "cost" not in contract.component_weights
    assert "safety_risk" not in contract.component_weights
    assert sum(contract.component_weights.values()) == pytest.approx(1.0)
    assert contract.component_weights == {
        "selective_product_yield": 0.30,
        "electrochemical_selectivity": 0.15,
        "electrochemical_conversion": 0.10,
        "faradaic_efficiency": 0.12,
        "transport_efficiency": 0.10,
        "ohmic_efficiency": 0.08,
        "energy_efficiency": 0.15,
    }


def test_partition_nominal_pair_law_is_scoped_to_v3_contract() -> None:
    environments = {
        "default": ChemWorldEnv(task_id="partition-discovery", seed=0),
        "v2": ChemWorldEnv(
            task_id="partition-discovery",
            seed=0,
            scoring_contract_id=PARTITION_S0_EXTRACTION_EFFICIENCY_V2,
        ),
        "v3": ChemWorldEnv(
            task_id="partition-discovery",
            seed=0,
            scoring_contract_id=PARTITION_S0_EXTRACTION_EFFICIENCY_V3,
        ),
    }
    try:
        assert (
            environments["default"].runtime.domain_services.phase_separation.nominal_pair_contract
            is None
        )
        assert (
            environments["v2"].runtime.domain_services.phase_separation.nominal_pair_contract
            is None
        )
        assert (
            environments["v3"].runtime.domain_services.phase_separation.nominal_pair_contract
            == INDEPENDENT_NOMINAL_SOLVENT_EXTRACTANT_PAIR_V1
        )
    finally:
        for environment in environments.values():
            environment.close()


def test_actual_assay_uses_initial_charge_not_internal_stage_recovery():
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        full_process_contract_id=FULL_PROCESS_FREE_RESEARCH_CONTRACT,
    )
    try:
        env.reset(seed=0)
        base = env.unwrapped
        species = MechanismSpeciesView(base.scenario_instance.compiled_mechanism)
        target, feed = species.target_species[0], species.feed_species[0]
        state = base._state
        # Partial conversion: 0.01 mol available before separation from 0.02 mol feed.
        # Solid contains 0.006 mol, of which 0.001 mol is purchased seed.
        base._state = state.replace(
            species=SpeciesLedger(initial_amounts_mol={feed: 0.02}),
            process=ProcessLedger(
                metrics={
                    "crystal_yield": 0.5,
                    "pre_separation_product_mol": 0.01,
                }
            ),
            phases=PhaseLedger(
                {
                    "solid": PhaseRecord(
                        phase_id="solid",
                        vessel_id="test",
                        phase_type="solid",
                        volume_L=0.001,
                        species_amounts_mol={target: 0.006},
                    )
                }
            ),
            equipment=upsert_equipment_record(
                state.equipment,
                equipment_id="crystallizer",
                equipment_type="crystallizer",
                attached_vessel_id="batch_reactor",
                status="configured",
                settings={"seed_target_mol": 0.001},
            ),
        )
        truth = base.observation_kernel._truth_values(base._state)
        assert truth["crystal_yield"] == pytest.approx(0.25)
        assert base._state.process.metrics["crystal_yield"] == 0.5
        assert (
            "tracked initial reagent charge"
            in full_process_operational_state(base, {})["sampling_contract"]
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "contract,open_measurements",
    [
        (FULL_PROCESS_SEED_CONTRACT, False),
        (FULL_PROCESS_FREE_RESEARCH_CONTRACT, True),
    ],
)
def test_new_contract_frees_measurement_order_preserving_physical_preconditions(
    contract,
    open_measurements,
):
    env = gym.make(
        "ChemWorld",
        task_id="reaction-to-crystallization",
        seed=0,
        budget_override=100,
        full_process_contract_id=contract,
    )
    try:
        env.reset(seed=0)
        _, _, _, _, invalid = env.step({"operation": "filter_crystals"})
        assert invalid["transaction_status"] == "rolled_back"
        for action in [
            a
            for a in reference_actions("reaction-to-crystallization")[:8]
            if a["operation"] != "measure"
        ]:
            assert env.step(action)[4]["transaction_status"] == "committed"
        seed = {"operation": "seed_crystals", "seed_mass_g": 0.006}
        info = env.step(seed)[4]
        assert (info["transaction_status"] == "committed") == open_measurements
        if not open_measurements:
            assert not info["preconditions"]["seed_crystals_requires_current_reaction_assay"]
            env.step({"operation": "measure", "instrument": "hplc"})
            assert env.step(seed)[4]["transaction_status"] == "committed"
        assert (
            env.step(
                {
                    "operation": "cool_crystallize",
                    "target_temperature_K": 278.15,
                    "duration_s": 1800,
                }
            )[4]["transaction_status"]
            == "committed"
        )
        filtered = env.step({"operation": "filter_crystals"})[4]
        assert (filtered["transaction_status"] == "committed") == open_measurements
    finally:
        env.close()


def test_v5_preserves_v4_thermal_and_seed_accounting_for_identical_legal_actions():
    truths = []
    for contract in (FULL_PROCESS_SEED_CONTRACT, FULL_PROCESS_FREE_RESEARCH_CONTRACT):
        env = gym.make(
            "ChemWorld",
            task_id="reaction-to-crystallization",
            seed=0,
            budget_override=100,
            full_process_contract_id=contract,
        )
        try:
            env.reset(seed=0)
            for action in reference_actions("reaction-to-crystallization")[:-2]:
                assert env.step(action)[4]["transaction_status"] == "committed"
            base = env.unwrapped
            truths.append(base.observation_kernel._truth_values(base._state))
        finally:
            env.close()
    assert truths[0] == truths[1]
