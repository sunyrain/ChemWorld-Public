"""Demo: deterministic LLM-style replay harness."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from chemworld.data.logging import load_jsonl
from chemworld.data.records import trajectory_value
from chemworld.eval.runner import run_agent
from chemworld.legacy.agents.llm import LLMReplayAgent, ToolUsingLLMStubAgent


def main() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        trace_path = (
            Path(__file__).resolve().parent
            / "fixtures"
            / "llm_replay"
            / "reaction_to_assay_public_trace.jsonl"
        )
        output_path = root / "run.jsonl"

        history = run_agent(
            env_id="ChemWorld",
            agent=LLMReplayAgent(trace_path),
            world_split="public-dev",
            budget=18,
            objective="balanced",
            seed=0,
            task_id="reaction-to-assay",
            output_path=output_path,
        )
        print("Replay final reward:", history[-1].reward)
        records = load_jsonl(output_path)
        print("Logged agent trace entries:", len(records[-1]["agent_trace"]))
        print("Final lab report:")
        print(trajectory_value(records[-1], 'agent_view')["lab_report"]["text"])

        stub_history = run_agent(
            env_id="ChemWorld",
            agent=ToolUsingLLMStubAgent(),
            world_split="public-dev",
            budget=18,
            objective="balanced",
            seed=0,
            task_id="reaction-to-assay",
        )
        print("ToolUsingLLMStub final reward:", stub_history[-1].reward)


if __name__ == "__main__":
    main()
