from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.envs.chemworld_env import OBSERVATION_KEYS
from chemworld.tasks import TaskSpec, list_tasks

REACTION_STEPS: tuple[dict[str, Any], ...] = (
    {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
    {"operation": "add_reagent", "amount_mol": 0.010},
    {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
    {
        "operation": "heat",
        "target_temperature_K": 385.0,
        "duration_s": 1500.0,
        "stirring_speed_rpm": 720.0,
    },
    {"operation": "wait", "duration_s": 900.0, "stirring_speed_rpm": 720.0},
)

REACTION_FINAL = (
    *REACTION_STEPS,
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

PURIFICATION_FINAL = (
    *REACTION_STEPS,
    {"operation": "quench"},
    {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
    {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
    {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
    {"operation": "settle", "duration_s": 420.0},
    {"operation": "separate_phase", "target_phase": "organic"},
    {"operation": "wash", "wash_volume_L": 0.008},
    {"operation": "dry"},
    {"operation": "concentrate", "duration_s": 600.0},
    {"operation": "transfer", "transfer_fraction": 0.97},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

CRYSTALLIZATION_FINAL = (
    *REACTION_STEPS,
    {"operation": "measure", "instrument": "hplc"},
    {"operation": "seed_crystals", "seed_mass_g": 0.006},
    {
        "operation": "cool_crystallize",
        "target_temperature_K": 278.15,
        "duration_s": 1800.0,
    },
    {"operation": "measure", "instrument": "hplc"},
    {"operation": "filter_crystals"},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

DISTILLATION_FINAL = (
    *REACTION_STEPS,
    {"operation": "evaporate", "target_temperature_K": 335.0, "duration_s": 600.0},
    {
        "operation": "distill",
        "target_temperature_K": 360.0,
        "duration_s": 1500.0,
        "reflux_ratio": 2.0,
    },
    {"operation": "collect_fraction", "transfer_fraction": 0.92},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

FLOW_FINAL = (
    {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2},
    {"operation": "add_reagent", "amount_mol": 0.010},
    {"operation": "add_catalyst", "catalyst_amount_mol": 0.00022, "catalyst": 1},
    {"operation": "set_flow_rate", "flow_rate_mL_min": 1.2, "residence_time_s": 900.0},
    {"operation": "run_flow", "target_temperature_K": 382.0, "duration_s": 1800.0},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

ELECTROCHEMISTRY_FINAL = (
    {"operation": "add_solvent", "volume_L": 0.026, "solvent": 0},
    {"operation": "add_reagent", "amount_mol": 0.010},
    {
        "operation": "set_potential",
        "potential_V": 1.15,
        "current_mA": 75.0,
        "electrolyte_profile": 1,
    },
    {"operation": "electrolyze", "duration_s": 1800.0},
    {"operation": "measure", "instrument": "ph_meter"},
    {"operation": "measure", "instrument": "uvvis"},
    {
        "operation": "set_potential",
        "potential_V": 1.25,
        "current_mA": 75.0,
        "electrolyte_profile": 1,
    },
    {"operation": "electrolyze", "duration_s": 1800.0},
    {"operation": "measure", "instrument": "uvvis"},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

EQUILIBRIUM_FINAL = (
    {"operation": "add_solvent", "volume_L": 0.030, "solvent": 0},
    {"operation": "add_reagent", "amount_mol": 0.006},
    {"operation": "measure", "instrument": "ph_meter"},
    {"operation": "add_reagent", "amount_mol": 0.004},
    {"operation": "measure", "instrument": "ph_meter"},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

PARTITION_FINAL = (
    {"operation": "add_solvent", "volume_L": 0.025, "solvent": 1},
    {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.018},
    {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
    {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 750.0},
    {"operation": "settle", "duration_s": 360.0},
    {"operation": "separate_phase", "target_phase": "organic"},
    {"operation": "terminate"},
    {"operation": "measure", "instrument": "final_assay"},
)

# Native-foundation regression baseline (2026-10-23), attributed against 7a6b9e75.
# The old scores and parent/child replays are retained in
# workstreams/world_foundation/reports/convergence/NATIVE_GOLDEN_ATTRIBUTION_RESULTS.json.
# See NATIVE_CONVERGENCE_ACCEPTANCE.md, NATIVE-REDUNDANCY-05, before migrating again.
GOLDEN: dict[str, dict[str, Any]] = {
    "electrochemical-conversion": {
        "mechanism_id": "electrochemical_conversion",
        "steps": 11,
        "score": 0.44142083220631545,
    },
    "equilibrium-characterization": {
        "mechanism_id": "simple_batch_reaction",
        "steps": 7,
        "score": 0.29453447974364416,
    },
    "flow-reaction-optimization": {
        "mechanism_id": "pfr_hotspot",
        "steps": 7,
        "score": 0.17941706640113322,
    },
    "low-budget-characterization": {
        "mechanism_id": "autocatalytic_reaction",
        "steps": 7,
        # The unchanged fourth action reaches 2.439 K/s, above the existing
        # 2 K/s limit. Resolving the accepted solver mesh now rejects it;
        # only the following 900 s wait advances physics. The old score
        # 0.2254381139292035 included the incorrectly committed heating.
        "rejected_steps": (4,),
        "score": 0.41831820592731117,
    },
    "partition-discovery": {
        "mechanism_id": "reaction_extraction",
        "steps": 8,
        # 2026-10-31: direct product feed uses its initial charge as the assay
        # denominator (77a7193), not its recovered amount. The eight actions,
        # physical states and score weights are unchanged. Recovery is
        # 0.002732299944564943 / 0.008805788504423727 = 0.31028452968093984;
        # the noisy organic reading is 0.30324717732286993, so the score is
        # 0.85 * reading - 0.10 * phase_ratio (1.0). The former 0.7440182504956406
        # expectation normalized incomplete recovery to 1.0 before noise.
        "score": 0.15776010072443944,
    },
    "public-private-generalization": {
        "mechanism_id": "parallel_series_reaction",
        "steps": 7,
        "score": 0.3105405119459594,
    },
    "purity-yield-tradeoff": {
        "mechanism_id": "reaction_extraction",
        "steps": 17,
        "score": 0.2798456245937464,
    },
    "reaction-mechanism-explanation": {
        "mechanism_id": "autocatalytic_reaction",
        "steps": 7,
        # Same fixed recipe: its fourth action reaches 2.335 K/s and rolls
        # back. The former 0.22979798353945266 included invalid heating.
        "rejected_steps": (4,),
        "score": 0.4253052081209714,
    },
    "reaction-optimization-standard": {
        "mechanism_id": "simple_batch_reaction",
        "steps": 7,
        "score": 0.38286650005401207,
    },
    "reaction-safety-constrained": {
        "mechanism_id": "catalyst_deactivation",
        "steps": 7,
        "score": 0.11650468116438889,
    },
    "reaction-to-assay": {
        "mechanism_id": "simple_batch_reaction",
        "steps": 7,
        "score": 0.34336651316585537,
        # Default reactor LSODA uses rtol=1e-8, atol=1e-12. At that policy,
        # SciPy 1.17.1 differs from the converged score by 5.60e-9 (1.18.0:
        # 5.07e-10). The two tighter policies below retain the 1e-9 reference
        # check; their cross-environment difference is below 5e-13.
        "score_abs_tolerance": 1e-8,
    },
    "reaction-to-crystallization": {
        "mechanism_id": "simple_batch_reaction",
        "steps": 12,
        # 2026-10-31 (008fd2709): finite crystal cooling now settles its
        # previously omitted cost. Physical states and other final readings
        # are unchanged; cost increases by 0.0452268971600068, reducing the
        # former 0.4095446795240686 score by 0.25 * 0.15 * that amount.
        "score": 0.4078486708806845,
    },
    "reaction-to-distillation": {
        "mechanism_id": "reactive_distillation_lite",
        "steps": 10,
        "score": 0.24615426316584232,
    },
    "reaction-to-purification": {
        "mechanism_id": "reaction_extraction",
        "steps": 17,
        "score": 0.29884907139394523,
    },
    "tool-agent-planning": {
        "mechanism_id": "reaction_extraction",
        "steps": 17,
        "score": 0.016028418138510518,
    },
}


def _scripted_final_actions(task_id: str) -> tuple[dict[str, Any], ...]:
    if task_id in {
        "reaction-to-purification",
        "purity-yield-tradeoff",
        "tool-agent-planning",
    }:
        return PURIFICATION_FINAL
    if task_id == "reaction-to-crystallization":
        return CRYSTALLIZATION_FINAL
    if task_id == "reaction-to-distillation":
        return DISTILLATION_FINAL
    if task_id == "flow-reaction-optimization":
        return FLOW_FINAL
    if task_id == "electrochemical-conversion":
        return ELECTROCHEMISTRY_FINAL
    if task_id == "equilibrium-characterization":
        return EQUILIBRIUM_FINAL
    if task_id == "partition-discovery":
        return PARTITION_FINAL
    return REACTION_FINAL


def test_golden_registry_covers_current_formal_tasks() -> None:
    assert {task.task_id for task in list_tasks()} == set(GOLDEN)


@pytest.mark.parametrize("task", list_tasks(), ids=lambda task: task.task_id)
def test_runtime_v2_golden_scripted_final_assay(
    task: TaskSpec,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHEMWORLD_PRIVATE_EVAL_SALT", raising=False)
    expected = GOLDEN[task.task_id]
    actions = _scripted_final_actions(task.task_id)
    assert len(actions) == expected["steps"]

    env = gym.make("ChemWorld", task_id=task.task_id, seed=task.seeds[0])
    try:
        observation, _reset_info = env.reset(seed=task.seeds[0])
        provenance = env.unwrapped.evaluator_provenance()
        assert provenance["mechanism_id"] == expected["mechanism_id"]
        assert provenance["mechanism_hash"]

        final_info: dict[str, Any] = {}
        final_terminated = False
        final_truncated = False
        rejected_steps = expected.get("rejected_steps", ())
        for step_index, action in enumerate(actions, start=1):
            before = env.unwrapped._state
            observation, _, final_terminated, final_truncated, final_info = env.step(action)
            assert not final_info["constraint_flags"]["precondition_failed"], action
            if step_index in rejected_steps:
                assert action["operation"] == "heat"
                assert final_info["transaction_status"] == "rolled_back"
                assert final_info["operation_outcome"] == "rejected"
                assert final_info["actual_duration_s"] == 0.0
                diagnostic = final_info["outcome_diagnostics"]
                assert diagnostic["failure_reason"] == "reaction_validity_domain_exceeded"
                assert diagnostic["validity_violations"] == ["thermal_runaway_rate_exceeds_limit"]
                assert diagnostic["reactor_diagnostic"]["maximum_temperature_rate_K_s"] > 2.0
                assert diagnostic["validity_domain"]["maximum_temperature_rate_K_s"] == 2.0
                after = env.unwrapped._state
                assert after.species_amounts == before.species_amounts
                assert after.temperature_K == before.temperature_K
                assert after.ledger.time_s == before.ledger.time_s
                assert after.ledger.energy_jacket_J == before.ledger.energy_jacket_J
            else:
                assert final_info["transaction_status"] == "committed"
            if rejected_steps and action["operation"] == "wait":
                assert env.unwrapped._state.ledger.time_s == 900.0
            assert final_info["kernel_id"].startswith("chemworld.operation.")
            assert "mechanism_hash" not in final_info

        assert final_info["operation_type"] == "measure"
        assert final_info["instrument"] == "final_assay"
        assert final_info["kernel_id"] == "chemworld.operation.measure"
        assert final_info["leaderboard_score"] == pytest.approx(
            expected["score"],
            abs=expected.get("score_abs_tolerance", 1e-9),
            rel=0.0,
        )
        assert final_info["remaining_budget"] == task.budget - expected["steps"]
        assert final_info["world_events"][0]["event_type"] == "operation_applied"
        assert final_info["state_patches_summary"][0]["patch_type"] == "replace_state"
        assert "process" in final_info["affected_ledgers"]
        assert "observation" in final_info["affected_ledgers"]
        assert "equipment" in final_info["affected_ledgers"]
        assert {
            "phase_system",
            "phase_settled",
            "selected_phase",
            "final_assay_done",
            "final_assay_time_s",
            "crystal_seeded",
            "crystal_seed_mass_g",
            "crystallization_active",
            "crystal_product_mol",
            "crystal_impurity_mol",
            "distillation_active",
            "distillate_product_mol",
            "distillate_impurity_mol",
            "solvent_loss",
            "initial_reactant_mol",
            "crystals_filtered",
            "distillation_model",
            "distillation_kernel",
            "fraction_collected",
        }.isdisjoint(env.unwrapped._state.metadata)
        assert not any(
            key.startswith("initial_") and key.endswith("_mol")
            for key in env.unwrapped._state.metadata
        )
        if env.unwrapped._state.phases is not None:
            assert all(
                {"product_mol", "impurity_mol", "solvent_loss"}.isdisjoint(phase.metadata)
                for phase in env.unwrapped._state.phases.phases.values()
            )

        if task.episode_mode == "campaign":
            assert not final_terminated
            assert not final_truncated
            assert final_info["experiment_ended"] is True
            assert final_info["next_experiment_ready"] is True
        else:
            assert final_terminated
            assert not final_truncated
            assert final_info["experiment_ended"] is False
            assert "next_experiment_ready" not in final_info

        observable_success_metrics = set(OBSERVATION_KEYS).intersection(task.success_metrics)
        for key in observable_success_metrics:
            assert math.isfinite(float(observation[key][0])), key
    finally:
        env.close()


@pytest.mark.parametrize("rtol,atol", [(1e-10, 1e-14), (1e-12, 1e-16)])
def test_assay_score_converges_to_the_numeric_reference(
    monkeypatch: pytest.MonkeyPatch, rtol: float, atol: float,
) -> None:
    """Distinguish default solver precision from changes to model or scoring."""
    from chemworld.physchem import reactor_solvers

    monkeypatch.delenv("CHEMWORLD_PRIVATE_EVAL_SALT", raising=False)
    monkeypatch.setattr(
        reactor_solvers, "DEFAULT_REACTOR_ODE_POLICY",
        replace(reactor_solvers.DEFAULT_REACTOR_ODE_POLICY, rtol=rtol, atol=atol),
    )
    task = next(task for task in list_tasks() if task.task_id == "reaction-to-assay")
    with gym.make("ChemWorld", task_id=task.task_id, seed=task.seeds[0]) as env:
        env.reset(seed=task.seeds[0])
        for action in REACTION_FINAL:
            _, _, _, _, info = env.step(action)
            assert info["transaction_status"] == "committed"
        assert info["leaderboard_score"] == pytest.approx(
            GOLDEN[task.task_id]["score"], abs=1e-9, rel=0.0,
        )
