from __future__ import annotations

import gymnasium as gym
import pytest

import chemworld  # noqa: F401
from chemworld.agent_interface import agent_view_bundle
from chemworld.agents.interaction import build_decision_context


def _public_view(*operations: str) -> dict:
    final_only = operations == ("measure",)
    return {
        "tool_json": {
            "available_actions": [
                {
                    "operation": operation,
                    "valid": True,
                    "schema": {
                        "fields": [
                            {
                                "field": "instrument",
                                "choices": ["final_assay"] if final_only else ["hplc"],
                            }
                        ]
                        if operation == "measure"
                        else []
                    },
                }
                for operation in operations
            ],
            "lab_report": {"visible_metrics": {}, "spectra_summary": {}},
        }
    }


@pytest.mark.parametrize("previous_event", ["operation_result", "measurement_result"])
def test_measure_only_affordance_is_persistently_labeled_closeout(
    previous_event: str,
) -> None:
    context = build_decision_context(
        step=8,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 12},
        public_view=_public_view("measure"),
        previous_event_type=previous_event,
    )

    assert context.available_operations == ("measure",)
    assert context.decision_stage == "experiment_closeout"


def test_measurement_before_termination_remains_an_evidence_update() -> None:
    context = build_decision_context(
        step=5,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 15},
        public_view=_public_view("measure", "terminate"),
        previous_event_type="measurement_result",
    )

    assert context.decision_stage == "evidence_update"


def test_measure_only_nonfinal_assay_is_not_misclassified_as_closeout() -> None:
    public_view = _public_view("measure")
    public_view["tool_json"]["available_actions"][0]["schema"]["fields"][0][
        "choices"
    ] = ["hplc"]
    context = build_decision_context(
        step=8,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 12},
        public_view=public_view,
        previous_event_type="operation_result",
    )

    assert context.available_operations == ("measure",)
    assert context.decision_stage == "experiment_control"


def test_measure_without_terminate_is_not_misclassified_as_closeout() -> None:
    context = build_decision_context(
        step=2,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 18},
        public_view=_public_view("add_reagent", "sample", "measure"),
        previous_event_type="operation_result",
    )

    assert context.available_operations == ("add_reagent", "sample", "measure")
    assert context.decision_stage == "experiment_control"


def test_retained_instrument_estimate_is_not_a_fresh_spectral_packet() -> None:
    public_view = _public_view("heat", "measure", "terminate")
    public_view["tool_json"].update(
        {
            "processed_estimate": {"byproduct_signal": 0.31},
            "historical_spectrum_catalog": [
                {
                    "spectrum_id": "spectrum-e001-s0006",
                    "measurement_step": 6,
                    "instrument_id": "gc",
                }
            ],
        }
    )
    public_view["tool_json"]["lab_report"] = {
        "visible_metrics": {"byproduct_signal": 0.31},
        "spectra_summary": {
            "has_spectral_packet": False,
            "processed_estimate": {"byproduct_signal": 0.31},
        },
    }

    context = build_decision_context(
        step=8,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 15},
        public_view=public_view,
        previous_event_type="operation_result",
    )
    payload = context.to_dict()

    assert context.visible_metrics == {"byproduct_signal": 0.31}
    assert context.latest_spectra["processed_estimate"] == {}
    assert payload["observation_provenance"] == {
        "current_event_type": "operation_result",
        "current_spectral_packet": False,
        "latest_cataloged_spectrum_id": "spectrum-e001-s0006",
        "latest_spectrum_measurement_step": 6,
        "operations_since_latest_spectrum": 1,
    }


def test_new_experiment_setup_takes_priority_over_affordance_shape() -> None:
    context = build_decision_context(
        step=9,
        task_info={"task_id": "reaction-to-crystallization"},
        campaign_state={"remaining_budget": 11},
        public_view=_public_view("measure"),
        previous_event_type="experiment_end",
    )

    assert context.decision_stage == "experiment_setup"


def test_real_environment_reports_closeout_after_termination() -> None:
    env = gym.make("ChemWorld", task_id="reaction-to-assay", seed=0)
    try:
        observation, _ = env.reset(seed=0)
        observation, _, _, _, info = env.step(
            {"operation": "add_solvent", "volume_L": 0.02, "solvent": 0}
        )
        public_view = agent_view_bundle(env, observation, info)
        before_reagent = build_decision_context(
            step=2,
            task_info={"task_id": "reaction-to-assay"},
            campaign_state=env.unwrapped.campaign_state(),
            public_view=public_view,
            previous_event_type="operation_result",
        )

        assert "measure" in before_reagent.available_operations
        assert "terminate" not in before_reagent.available_operations
        assert before_reagent.decision_stage == "experiment_control"

        observation, _, _, _, info = env.step({"operation": "add_reagent", "amount_mol": 0.01})
        observation, _, _, _, info = env.step({"operation": "terminate"})
        public_view = agent_view_bundle(env, observation, info)

        context = build_decision_context(
            step=4,
            task_info={"task_id": "reaction-to-assay"},
            campaign_state=env.unwrapped.campaign_state(),
            public_view=public_view,
            previous_event_type="operation_result",
        )

        assert context.available_operations == ("measure",)
        assert context.decision_stage == "experiment_closeout"

        measure = public_view["tool_json"]["available_actions"][0]
        instrument = measure["schema"]["fields"][0]
        assert instrument["choices"] == ["final_assay"]
    finally:
        env.close()
