"""LLM adapter interfaces.

This module deliberately avoids direct API dependencies. Reproducible LLM
experiments should declare model metadata, prompt templates, cache policy, and
offline replay files in the agent manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from chemworld.agents.base import BaseAgent, HistoryRecord
from chemworld.agents.recipe_sequence import RecipeSequenceMixin
from chemworld.data.logging import to_builtin
from chemworld.world.actions import canonicalize_action

DEFAULT_PROMPT_TEMPLATE = """You are planning the next virtual ChemWorld reaction experiment.
Task info:
{task_info}

History:
{history}

Return one JSON object with keys temperature, time, initial_concentration,
stirring_speed, catalyst, solvent, hypothesis, rationale.
"""


class LLMPlannerAgent(RecipeSequenceMixin, BaseAgent):
    """Abstract adapter for online or cached LLM planners."""

    name = "llm_planner"

    def __init__(self, prompt_template: str = DEFAULT_PROMPT_TEMPLATE) -> None:
        self.prompt_template = prompt_template

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        pending = self._pop_pending_event()
        if pending is not None:
            return pending

        prompt = self.build_prompt(self._recipe_history)
        response = self.complete(prompt)
        payload = json.loads(response)
        return self._start_recipe(canonicalize_action(payload))

    def build_prompt(self, history: list[HistoryRecord]) -> str:
        compact_history = [
            {
                "step": record.step,
                "action": record.action,
                "observation": record.observation,
                "reward": record.reward,
            }
            for record in history
        ]
        return self.prompt_template.format(
            task_info=json.dumps(self.task_info, sort_keys=True),
            history=json.dumps(compact_history, sort_keys=True),
        )

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "LLMPlannerAgent is an adapter. Implement complete() or use "
            "LLMCompletionReplayAgent."
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "prompt_template": self.prompt_template,
                "requires_online_model": True,
            }
        )
        return manifest


class LLMCompletionReplayAgent(LLMPlannerAgent):
    """Replay cached planner completions before recipe expansion."""

    name = "llm_completion_replay"

    def __init__(
        self,
        replay_path: str | Path,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ) -> None:
        super().__init__(prompt_template=prompt_template)
        self.replay_path = Path(replay_path)

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        with self.replay_path.open("r", encoding="utf-8") as handle:
            self._responses = [json.loads(line) for line in handle if line.strip()]
        self._index = 0

    def complete(self, prompt: str) -> str:
        del prompt
        if self._index >= len(self._responses):
            raise RuntimeError(f"Replay file exhausted: {self.replay_path}")
        payload = self._responses[self._index]
        self._index += 1
        return json.dumps(payload)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "requires_online_model": False,
                "replay_path": str(self.replay_path),
            }
        )
        return manifest


class LLMReplayAgent(BaseAgent):
    """Replay a fixed tool/action trace without requiring an online LLM."""

    name = "llm_replay"

    def __init__(self, replay_path: str | Path | None = None) -> None:
        self.replay_path = None if replay_path is None else Path(replay_path)

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        if self.replay_path is None:
            self._records = self._default_records_for_task(task_info)
        else:
            with self.replay_path.open("r", encoding="utf-8") as handle:
                self._records = [json.loads(line) for line in handle if line.strip()]
        self._index = 0
        self._trace: list[dict[str, Any]] = []

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        if self._index >= len(self._records):
            return {"operation": "measure", "instrument": "final_assay"}
        record = self._records[self._index]
        self._index += 1
        action = dict(record.get("action", record))
        self._trace.append(
            {
                "prompt_input": record.get("prompt_input", {}),
                "selected_action": action,
                "reasoning_summary": record.get("reasoning_summary", "offline replay action"),
                "hypothesis_note": record.get("hypothesis_note"),
                "memory_note": record.get("memory_note"),
            }
        )
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        if self._trace:
            self._trace[-1]["validator_result"] = {
                "constraint_flags": info.get("constraint_flags", {}),
                "error_message": info.get("error_message"),
            }
            self._trace[-1]["observation_summary"] = {
                "reward": reward,
                "observed_keys": info.get("observed_keys", []),
                "leaderboard_score": info.get("leaderboard_score"),
            }
        del action, observation

    def agent_trace(self) -> list[dict[str, Any]]:
        return to_builtin(self._trace)

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "requires_online_model": False,
                "replay_path": None if self.replay_path is None else str(self.replay_path),
                "uses_builtin_trace": self.replay_path is None,
                "trace_format": "chemworld-agent-trace-0.1",
            }
        )
        return manifest

    def _default_records_for_task(self, task_info: dict[str, Any]) -> list[dict[str, Any]]:
        allowed_operations = set(task_info.get("allowed_operations", []))
        allowed_instruments = set(task_info.get("allowed_instruments", []))
        records: list[dict[str, Any]] = [
            self._record(
                {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
                "Start with a moderate solvent charge to create a stable liquid medium.",
            ),
            self._record(
                {"operation": "add_reagent", "amount_mol": 0.010},
                "Add a conservative reactant amount before choosing energy input.",
            ),
        ]
        if "ph_meter" in allowed_instruments:
            records.extend(
                [
                    self._record(
                        {"operation": "measure", "instrument": "ph_meter"},
                        "Measure pH early to characterize acid/base behavior.",
                    ),
                    self._record(
                        {"operation": "add_reagent", "amount_mol": 0.004},
                        "Perturb concentration to test the equilibrium response.",
                    ),
                    self._record(
                        {"operation": "measure", "instrument": "ph_meter"},
                        "Compare the second pH observation against the first.",
                    ),
                    self._record(
                        {"operation": "terminate"},
                        "Terminate before leaderboard-grade final assay.",
                    ),
                    self._record(
                        {"operation": "measure", "instrument": "final_assay"},
                        "Use final assay to score equilibrium characterization.",
                    ),
                ]
            )
            return records
        if "add_catalyst" in allowed_operations:
            records.append(
                self._record(
                    {
                        "operation": "add_catalyst",
                        "catalyst_amount_mol": 0.00025,
                        "catalyst": 1,
                    },
                    "Use a moderate catalyst loading to expose catalytic acceleration.",
                )
            )
        if "heat" in allowed_operations:
            records.append(
                self._record(
                    {
                        "operation": "heat",
                        "target_temperature_K": 378.0,
                        "duration_s": 1350.0,
                        "stirring_speed_rpm": 720.0,
                    },
                    "Advance conversion without pushing the high-risk temperature range.",
                )
            )
        allowed_instruments = set(task_info.get("allowed_instruments", []))
        if "measure" in allowed_operations and "uvvis" in allowed_instruments:
            records.append(
                self._record(
                    {"operation": "measure", "instrument": "uvvis"},
                    "Take a low-cost proxy measurement before committing to endpoint handling.",
                )
            )
        if "quench" in allowed_operations:
            records.append(
                self._record(
                    {"operation": "quench"},
                    "Stop reaction chemistry before downstream phase operations.",
                )
            )
        if "add_phase" in allowed_operations:
            records.extend(
                [
                    self._record(
                        {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
                        "Create a second phase so partitioning can reveal downstream behavior.",
                    ),
                    self._record(
                        {
                            "operation": "add_extractant",
                            "extractant": "organic",
                            "volume_L": 0.018,
                        },
                        "Add an organic extractant to improve product recovery.",
                    ),
                    self._record(
                        {
                            "operation": "mix",
                            "duration_s": 240.0,
                            "stirring_speed_rpm": 850.0,
                        },
                        "Mix long enough for interphase mass transfer.",
                    ),
                    self._record(
                        {"operation": "settle", "duration_s": 420.0},
                        "Let the phases settle before measuring or separating.",
                    ),
                ]
            )
            if "measure" in allowed_operations and "hplc" in allowed_instruments:
                records.append(
                    self._record(
                        {"operation": "measure", "instrument": "hplc"},
                        "Use HPLC to observe the public product/byproduct signal.",
                    )
                )
            records.append(
                self._record(
                    {"operation": "separate_phase", "target_phase": "organic"},
                    "Separate the phase expected to contain the product-rich fraction.",
                )
            )
        if "wash" in allowed_operations:
            records.append(
                self._record(
                    {"operation": "wash", "wash_volume_L": 0.006},
                    "Wash the isolated phase to improve purity at modest recovery cost.",
                )
            )
        if "dry" in allowed_operations:
            records.append(
                self._record(
                    {"operation": "dry"},
                    "Remove residual water before final concentration or assay.",
                )
            )
        if "concentrate" in allowed_operations:
            records.append(
                self._record(
                    {"operation": "concentrate", "duration_s": 450.0},
                    "Concentrate the product fraction before final assay.",
                )
            )
        records.extend(
            [
                self._record(
                    {"operation": "terminate"},
                    "Terminate the experiment so final assay becomes valid.",
                ),
                self._record(
                    {"operation": "measure", "instrument": "final_assay"},
                    "Use the leaderboard instrument after termination.",
                ),
            ]
        )
        return records

    def _record(self, action: dict[str, Any], reasoning_summary: str) -> dict[str, Any]:
        return {
            "action": action,
            "prompt_input": {
                "baseline": "builtin deterministic LLM replay",
                "hidden_truth_policy": "public observations only",
            },
            "reasoning_summary": reasoning_summary,
            "hypothesis_note": (
                "Moderate conditions should expose informative score and spectra signals."
            ),
        }


class CodexSubagentReplayAgent(LLMReplayAgent):
    """Replay a trace produced by a Codex sub-agent planner."""

    name = "codex_subagent_replay"

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "agent_family": "codex_subagent",
                "requires_online_model": False,
                "online_counterpart": "codex_subagent_online",
                "trace_format": "chemworld-codex-subagent-trace-0.1",
            }
        )
        return manifest


class CodexSubagentOnlineAgent(BaseAgent):
    """Protocol placeholder for live Codex sub-agent experiments.

    Repository code cannot spawn the conversation-level Codex sub-agent tool by
    itself. Live runs are orchestrated by the evaluator, written as JSONL traces,
    and then replayed by :class:`CodexSubagentReplayAgent`.
    """

    name = "codex_subagent_online"

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        del history
        raise RuntimeError(
            "codex_subagent_online requires an external evaluator harness. "
            "Run live Codex planning from the host session and replay the saved "
            "trace with codex_subagent_replay."
        )

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "agent_family": "codex_subagent",
                "requires_online_model": True,
                "execution_policy": "external_codex_subagent_orchestrator",
                "replay_agent": "codex_subagent_replay",
                "trace_format": "chemworld-codex-subagent-trace-0.1",
            }
        )
        return manifest


class ToolUsingLLMStubAgent(BaseAgent):
    """Deterministic tool-using LLM stub for offline benchmark plumbing."""

    name = "tool_using_llm_stub"

    def reset(self, task_info: dict[str, Any], seed: int) -> None:
        super().reset(task_info, seed)
        self._plan = self._plan_for_task(task_info)
        self._trace: list[dict[str, Any]] = []
        self._memory: list[str] = []

    def act(self, history: list[HistoryRecord]) -> dict[str, Any]:
        step = len(history)
        action = dict(self._plan[step % len(self._plan)])
        self._trace.append(
            {
                "prompt_input": {
                    "task_id": self.task_info.get("task_id"),
                    "allowed_operations": self.task_info.get("allowed_operations", []),
                    "history_length": len(history),
                    "memory_summary": list(self._memory[-3:]),
                },
                "selected_action": action,
                "reasoning_summary": self._reason_for_action(action),
                "hypothesis_note": self._hypothesis_note(step),
                "memory_note": self._memory[-1] if self._memory else "no prior observations",
            }
        )
        return action

    def update(
        self,
        action: dict[str, Any],
        observation: dict[str, float | None],
        reward: float,
        info: dict[str, Any],
    ) -> None:
        flags = info.get("constraint_flags", {})
        memory = (
            f"{action.get('operation')} -> reward={reward:.3f}, "
            f"precondition_failed={bool(flags.get('precondition_failed', False))}, "
            f"observed={','.join(info.get('observed_keys', [])) or 'none'}"
        )
        self._memory.append(memory)
        if self._trace:
            self._trace[-1]["validator_result"] = {
                "constraint_flags": flags,
                "error_message": info.get("error_message"),
                "preconditions": info.get("preconditions", {}),
            }
            self._trace[-1]["observation_summary"] = {
                "reward": reward,
                "observation": observation,
                "observed_keys": info.get("observed_keys", []),
                "leaderboard_score": info.get("leaderboard_score"),
            }

    def agent_trace(self) -> list[dict[str, Any]]:
        return to_builtin(self._trace)

    def _plan_for_task(self, task_info: dict[str, Any]) -> list[dict[str, Any]]:
        allowed_operations = set(task_info.get("allowed_operations", []))
        allowed_instruments = set(task_info.get("allowed_instruments", []))
        if "ph_meter" in allowed_instruments:
            return [
                {"operation": "add_solvent", "volume_L": 0.030, "solvent": 0},
                {"operation": "add_reagent", "amount_mol": 0.006},
                {"operation": "measure", "instrument": "ph_meter"},
                {"operation": "add_reagent", "amount_mol": 0.004},
                {"operation": "measure", "instrument": "ph_meter"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        if "run_flow" in allowed_operations:
            return [
                {"operation": "add_solvent", "volume_L": 0.026, "solvent": 2},
                {"operation": "add_reagent", "amount_mol": 0.010},
                {"operation": "add_catalyst", "catalyst_amount_mol": 0.00022, "catalyst": 1},
                {
                    "operation": "set_flow_rate",
                    "flow_rate_mL_min": 1.2,
                    "residence_time_s": 900.0,
                },
                {"operation": "run_flow", "target_temperature_K": 382.0, "duration_s": 1800.0},
                {"operation": "measure", "instrument": "uvvis"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        if "electrolyze" in allowed_operations:
            return [
                {"operation": "add_solvent", "volume_L": 0.026, "solvent": 1},
                {"operation": "add_reagent", "amount_mol": 0.010},
                {"operation": "set_potential", "potential_V": 1.15, "current_mA": 75.0},
                {"operation": "electrolyze", "duration_s": 1800.0},
                {"operation": "measure", "instrument": "uvvis"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        if "add_phase" in allowed_operations and "heat" not in allowed_operations:
            return [
                {"operation": "add_solvent", "volume_L": 0.024, "solvent": 2},
                {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.014},
                {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.020},
                {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
                {"operation": "settle", "duration_s": 420.0},
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "separate_phase", "target_phase": "organic"},
                {"operation": "measure", "instrument": "hplc"},
                {"operation": "terminate"},
                {"operation": "measure", "instrument": "final_assay"},
            ]
        sequence: list[dict[str, Any]] = [
            {"operation": "add_solvent", "volume_L": 0.028, "solvent": 2},
            {"operation": "add_reagent", "amount_mol": 0.010},
            {"operation": "add_catalyst", "catalyst_amount_mol": 0.00025, "catalyst": 1},
            {
                "operation": "heat",
                "target_temperature_K": 382.0,
                "duration_s": 1350.0,
                "stirring_speed_rpm": 720.0,
            },
            {"operation": "quench"},
        ]
        if "separate_phase" in allowed_operations:
            sequence.extend(
                [
                    {"operation": "add_phase", "phase": "aqueous", "volume_L": 0.012},
                    {"operation": "add_extractant", "extractant": "organic", "volume_L": 0.018},
                    {"operation": "mix", "duration_s": 240.0, "stirring_speed_rpm": 850.0},
                    {"operation": "settle", "duration_s": 420.0},
                    {"operation": "separate_phase", "target_phase": "organic"},
                    {"operation": "wash", "wash_volume_L": 0.006},
                    {"operation": "dry"},
                    {"operation": "concentrate", "duration_s": 450.0},
                ]
            )
        if "cool_crystallize" in allowed_operations:
            sequence.extend(
                [
                    {"operation": "seed_crystals", "seed_mass_g": 0.006},
                    {
                        "operation": "cool_crystallize",
                        "target_temperature_K": 278.15,
                        "duration_s": 1800.0,
                    },
                    {"operation": "filter_crystals"},
                    {"operation": "measure", "instrument": "hplc"},
                ]
            )
        if "distill" in allowed_operations:
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
        return sequence

    def _reason_for_action(self, action: dict[str, Any]) -> str:
        operation = str(action.get("operation"))
        reasons = {
            "add_solvent": "charge a stable liquid medium before adding material",
            "add_reagent": "create observable reactant inventory",
            "add_catalyst": "increase target pathway rate before heating",
            "heat": "advance reaction while monitoring safety-cost tradeoff",
            "wait": "allow additional conversion before an intermediate instrument check",
            "measure": "use an allowed instrument to update the public belief state",
            "quench": "stop reaction chemistry before downstream handling",
            "add_phase": "create a phase system for partition or purification",
            "add_extractant": "improve product transfer into the recoverable phase",
            "mix": "promote interphase mass transfer",
            "settle": "let phases separate before taking a product-rich phase",
            "separate_phase": "isolate the phase expected to contain product",
            "terminate": "mark the experiment ready for final assay",
        }
        return reasons.get(operation, "execute the next task-allowed operation in the plan")

    def _hypothesis_note(self, step: int) -> str:
        if step == 0:
            return "Moderate solvent/catalyst choices should expose a stable baseline."
        if step < 5:
            return "Build enough reaction progress for informative spectroscopy."
        return "Use downstream operations to improve public purity/recovery signals."

    def manifest(self) -> dict[str, Any]:
        manifest = super().manifest()
        manifest.update(
            {
                "requires_online_model": False,
                "uses_recipe_compiler": True,
                "uses_validator_feedback": True,
                "llm_role": "offline_tool_agent_stub",
                "agent_trace_schema_version": "chemworld-agent-trace-0.1",
            }
        )
        return manifest
