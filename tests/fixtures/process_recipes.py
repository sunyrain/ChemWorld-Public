"""Fixed physical regression recipes, independent of retired study runners."""

from contextlib import suppress

import gymnasium as gym

from chemworld.action_codec import ActionCodec
from chemworld.campaign_resources import (
    CampaignResourceCard,
    CampaignResourceLedger,
    campaign_resource_event_id,
)
from chemworld.data.records import trajectory_value
from chemworld.runtime.phase_ledger_services import ChemWorldPhaseLedgerServices
from chemworld.runtime.species import MechanismSpeciesView
from chemworld.world.instruments import chemworld_instruments
from chemworld.world.operations import CAMPAIGN_OPERATION_TYPES

TASKS = ("reaction-to-purification", "reaction-to-crystallization", "reaction-to-distillation")


def reference_actions(task):
    steps = [
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
        {"operation": "measure", "instrument": "hplc"},
        {"operation": "quench"},
        {"operation": "measure", "instrument": "hplc"},
    ]
    if task == TASKS[0]:
        steps += [
            {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
            {"operation": "add_extractant", "extractant": 3, "volume_L": 0.018},
            {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
            {"operation": "settle", "duration_s": 420.0},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "separate_phase", "target_phase": "organic"},
            {"operation": "wash", "wash_volume_L": 0.008},
            {"operation": "dry"},
            {"operation": "concentrate", "duration_s": 600.0},
            {"operation": "transfer", "transfer_fraction": 0.97},
            {"operation": "measure", "instrument": "hplc"},
        ]
    elif task == TASKS[1]:
        steps += [
            {"operation": "seed_crystals", "seed_mass_g": 0.006},
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 1800.0},
            {
                "operation": "heat",
                "target_temperature_K": 300.0,
                "duration_s": 300.0,
                "stirring_speed_rpm": 600.0,
            },
            {"operation": "cool_crystallize", "target_temperature_K": 278.15, "duration_s": 1800.0},
            {"operation": "measure", "instrument": "hplc"},
            {"operation": "filter_crystals"},
        ]
    elif task == TASKS[2]:
        steps += [
            {"operation": "evaporate", "target_temperature_K": 335.0, "duration_s": 600.0},
            {
                "operation": "distill",
                "target_temperature_K": 360.0,
                "duration_s": 1500.0,
                "reflux_ratio": 2.0,
            },
            {"operation": "collect_fraction", "transfer_fraction": 0.5},
            {"operation": "measure", "instrument": "gc"},
            {
                "operation": "distill",
                "target_temperature_K": 370.0,
                "duration_s": 1200.0,
                "reflux_ratio": 3.0,
            },
            {"operation": "collect_fraction", "transfer_fraction": 0.92},
            {"operation": "measure", "instrument": "gc"},
        ]
    else:
        raise ValueError(task)
    return [
        *steps,
        {"operation": "terminate"},
        {"operation": "measure", "instrument": "final_assay"},
    ]


class EvaluatorCapture(gym.Wrapper):
    """Observe committed PA sample bookkeeping without returning it to the agent."""

    def __init__(self, env, records):
        super().__init__(env)
        self.evaluator_records = records

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        base = self.unwrapped
        original = base.runtime.apply_transaction
        phases = ChemWorldPhaseLedgerServices(
            MechanismSpeciesView(base.scenario_instance.compiled_mechanism)
        )
        species = phases.product_candidate_species()

        def target_total(state):
            if state.phases is not None:
                return sum(
                    p.species_amounts_mol.get(s, 0)
                    for p in state.phases.phases.values()
                    for s in species
                )
            return sum(state.species_amounts.get(s, 0) for s in species)

        def capture(state, payload):
            result = original(state, payload)
            if payload.get("instrument") == "final_assay":
                initial = sum(state.species.initial_amounts_mol.get(s, 0) for s in species)
                before, after = target_total(state), target_total(result.state)
                volume = chemworld_instruments()["final_assay"].sample_volume_L
                # PA assay withdraws from the selected receiver (organic by
                # default); the unselected aqueous phase is not sampled.
                phase_map = state.phases.phases
                selected = next((p for p in phase_map.values() if p.selected),
                                phase_map["organic"])
                expected_loss = sum(selected.species_amounts_mol.get(s, 0)
                                    for s in species) * volume / selected.volume_L
                self.evaluator_records.append(
                    {
                        "initial_target_mol": initial,
                        "before_assay_target_mol": before,
                        "after_assay_target_mol": after,
                        "expected_sample_target_mol": expected_loss,
                        "before_error_mol": abs(before - initial),
                        "after_plus_sample_error_mol": abs(after + expected_loss - initial),
                    }
                )
            return result

        base.runtime.apply_transaction = capture
        return result


def verify_resources(records):
    ledger = CampaignResourceLedger(
        CampaignResourceCard.from_dict(records[0]["campaign_resource_card"])
    )
    codec = ActionCodec(operation_types=CAMPAIGN_OPERATION_TYPES)
    for index, record in enumerate(records, 1):
        campaign = trajectory_value(record, "agent_view")["tool_json"]["campaign_state"]
        resources = campaign["campaign_resources"]
        receipt = resources["latest_receipt"]
        event_id = receipt["event_id"]
        assert event_id == campaign_resource_event_id(campaign["campaign_id"], index)
        starts = receipt["preflight"]["proposed_delta"]["vessel_starts"] == 1
        # The runner retains the requested action; env.step canonicalizes it before
        # resource admission. In particular integer-valued control fields become
        # floats, which changes their canonical JSON hash without changing physics.
        # Match the environment's invalid-action fallback as well as valid actions.
        action = record["action"]
        with suppress(IndexError, OverflowError, TypeError, ValueError):
            action = codec.canonicalize(action)
        preflight = ledger.preflight(event_id, action, starts_vessel=starts)
        assert preflight.to_dict() == receipt["preflight"]
        delta = ledger.record_outcome(
            event_id,
            action,
            {
                "operation_committed": receipt["operation_committed"],
                "campaign_resource_report_delta": receipt["outcome_delta"]["report_only"],
            },
            starts_vessel=starts,
        )
        assert delta.to_dict() == receipt["outcome_delta"]
        snapshot = ledger.snapshot()
        assert snapshot["ledger_sha256"] == resources["ledger_sha256"]
        assert snapshot["state"] == resources["state"]
    return {"checked_receipts": len(records), "verified": True}
