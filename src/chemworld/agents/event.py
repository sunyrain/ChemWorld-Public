"""Event-sequence baseline agents for foundation-backed environments."""

from __future__ import annotations

from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord


class ScriptedChemistryAgent(BaseAgent):
    """A simple chemistry-aware event planner for ChemWorld tasks."""

    name = "scripted_chemistry"

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        step = len(history)
        allowed_operations = set(self.task_info.get("allowed_operations", []))
        purification_enabled = "separate_phase" in allowed_operations
        crystallization_enabled = "cool_crystallize" in allowed_operations
        distillation_enabled = "distill" in allowed_operations
        flow_enabled = "run_flow" in allowed_operations
        electrochemistry_enabled = "electrolyze" in allowed_operations
        partition_only = "add_phase" in allowed_operations and "heat" not in allowed_operations
        allowed_instruments = set(self.task_info.get("allowed_instruments", []))
        reaction_diagnostic = "uvvis" if "uvvis" in allowed_instruments else "hplc"
        if electrochemistry_enabled:
            electrochemistry_sequence: list[dict[str, Any]] = [
                {"operation": "add_solvent", "volume_L": 0.026, "solvent": 0},
                {"operation": "add_reagent", "amount_mol": 0.010},
                {
                    "operation": "set_potential",
                    "potential_V": 0.95,
                    "current_mA": 52.0,
                    "electrolyte_profile": 1,
                },
                {"operation": "electrolyze", "duration_s": 540.0},
                {"operation": "measure", "instrument": "ph_meter"},
                {"operation": "measure", "instrument": "uvvis"},
                {
                    "operation": "set_potential",
                    "potential_V": 1.35,
                    "current_mA": 92.0,
                    "electrolyte_profile": 1,
                },
                {"operation": "electrolyze", "duration_s": 2100.0},
                {"operation": "measure", "instrument": "uvvis"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
            return self._sequence_action(electrochemistry_sequence, step)
        if "ph_meter" in allowed_instruments:
            equilibrium_sequence: list[dict[str, Any]] = [
                {"operation": "add_solvent", "volume_L": 0.034, "solvent": 0},
                {"operation": "add_reagent", "amount_mol": 0.0045},
                {"operation": "measure", "instrument": "ph_meter"},
                {"operation": "add_reagent", "amount_mol": 0.0075},
                {"operation": "measure", "instrument": "ph_meter"},
                {"operation": "measure", "instrument": "uvvis"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
            return self._sequence_action(equilibrium_sequence, step)
        if flow_enabled:
            flow_sequence: list[dict[str, Any]] = [
                {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2},
                {"operation": "add_reagent", "amount_mol": 0.010},
                {"operation": "add_catalyst", "catalyst_amount_mol": 0.00022, "catalyst": 1},
                {
                    "operation": "set_flow_rate",
                    "flow_rate_mL_min": 0.85,
                    "residence_time_s": 1250.0,
                },
                {"operation": "run_flow", "target_temperature_K": 395.0, "duration_s": 2100.0},
                {"operation": "measure", "instrument": "uvvis"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
            return self._sequence_action(flow_sequence, step)
        if partition_only:
            partition_sequence: list[dict[str, Any]] = [
                {"operation": "add_solvent", "volume_L": 0.018, "solvent": 2},
                {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.018},
                {"operation": "add_extractant", "extractant": 3, "volume_L": 0.024},
                {"operation": "mix", "duration_s": 360.0, "stirring_speed_rpm": 920.0},
                {"operation": "settle", "duration_s": 720.0},
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "separate_phase", "target_phase": "organic"},
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
            return self._sequence_action(partition_sequence, step)
        sequence: list[dict[str, Any]] = [
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
            {"operation": "wait", "duration_s": 600.0, "stirring_speed_rpm": 720.0},
            {"operation": "quench"},
            {"operation": "measure", "instrument": reaction_diagnostic},
        ]
        if purification_enabled:
            sequence.extend(
                [
                    {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
                    {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
                    {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
                    {"operation": "settle", "duration_s": 420.0},
                    {"operation": "separate_phase", "target_phase": "organic"},
                    {"operation": "wash", "wash_volume_L": 0.008},
                    {"operation": "dry"},
                    {"operation": "concentrate", "duration_s": 600.0},
                    {"operation": "transfer", "transfer_fraction": 0.97},
                    {"operation": "measure", "instrument": "hplc"},
                ]
            )
        if crystallization_enabled:
            sequence.extend(
                [
                    {"operation": "seed_crystals", "seed_mass_g": 0.006},
                    {
                        "operation": "cool_crystallize",
                        "target_temperature_K": 278.15,
                        "duration_s": 1800.0,
                    },
                    {"operation": "measure", "instrument": "hplc"},
                    {"operation": "filter_crystals"},
                    {"operation": "measure", "instrument": "hplc"},
                ]
            )
        if distillation_enabled:
            sequence.extend(
                [
                    {
                        "operation": "evaporate",
                        "target_temperature_K": 335.0,
                        "duration_s": 600.0,
                    },
                    {
                        "operation": "distill",
                        "target_temperature_K": 360.0,
                        "duration_s": 1500.0,
                        "reflux_ratio": 2.0,
                    },
                    {"operation": "collect_fraction", "transfer_fraction": 0.92},
                    {"operation": "measure", "instrument": "gc"},
                ]
            )
        sequence.extend(
            [
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        )
        return self._sequence_action(sequence, step)

    @staticmethod
    def _sequence_action(
        sequence: list[dict[str, Any]],
        step: int,
    ) -> dict[str, Any]:
        return sequence[step % len(sequence)]
