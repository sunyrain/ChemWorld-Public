"""Historical preset factory, explicitly opt-in."""

from collections.abc import Callable

from chemworld.agents.base import Agent
from chemworld.legacy.agents import (
    CodexSubagentReplayAgent,
    GaussianProcessBOAgent,
    GaussianProcessPIAgent,
    GaussianProcessUCBAgent,
    GreedyLocalAgent,
    LatinHypercubeAgent,
    LLMReplayAgent,
    RandomAgent,
    RandomForestEIAgent,
    RandomRecipeAgent,
    SafetyConstrainedBOAgent,
    ScriptedChemistryAgent,
    StructuredGaussianProcessBOAgent,
    StructuredGaussianProcessPIAgent,
    StructuredGaussianProcessUCBAgent,
    StructuredRandomForestEIAgent,
    StructuredSafetyConstrainedBOAgent,
    ToolUsingLLMStubAgent,
)

AGENT_REGISTRY: dict[str, Callable[[], Agent]] = {
    "random": RandomAgent,
    "lhs": LatinHypercubeAgent,
    "latin_hypercube": LatinHypercubeAgent,
    "greedy": GreedyLocalAgent,
    "greedy_local": GreedyLocalAgent,
    "gp_bo": GaussianProcessBOAgent,
    "gp_pi": GaussianProcessPIAgent,
    "gp_ucb": GaussianProcessUCBAgent,
    "rf_ei": RandomForestEIAgent,
    "safe_gp_bo": SafetyConstrainedBOAgent,
    "structured_gp_bo": StructuredGaussianProcessBOAgent,
    "structured_gp_pi": StructuredGaussianProcessPIAgent,
    "structured_gp_ucb": StructuredGaussianProcessUCBAgent,
    "structured_rf_ei": StructuredRandomForestEIAgent,
    "structured_safe_gp_bo": StructuredSafetyConstrainedBOAgent,
    "random_recipe": RandomRecipeAgent,
    "scripted_chemistry": ScriptedChemistryAgent,
    "scripted_reaction_to_purification": ScriptedChemistryAgent,
    "partition_discovery_heuristic": ScriptedChemistryAgent,
    "heuristic": ScriptedChemistryAgent,
    "tool_using_llm_stub": ToolUsingLLMStubAgent,
    "llm_replay": LLMReplayAgent,
    "codex_subagent_replay": CodexSubagentReplayAgent,
}


def make_agent(name: str) -> Agent:
    if name not in AGENT_REGISTRY:
        allowed = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Unknown agent={name!r}. Allowed: {allowed}")
    return AGENT_REGISTRY[name]()
