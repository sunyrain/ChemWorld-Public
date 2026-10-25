"""Archived preset agents for historical applications."""

from chemworld.agents.base import Agent, HistoryRecord
from chemworld.agents.interaction import (
    INTERACTION_CONTRACT_VERSION,
    AgentDecisionContext,
    DecisionAuditRecord,
    InteractionCapabilities,
)
from chemworld.legacy.agents.bo import (
    GaussianProcessBOAgent,
    GaussianProcessPIAgent,
    GaussianProcessUCBAgent,
    RandomForestEIAgent,
    SafetyConstrainedBOAgent,
    StructuredGaussianProcessBOAgent,
    StructuredGaussianProcessPIAgent,
    StructuredGaussianProcessUCBAgent,
    StructuredRandomForestEIAgent,
    StructuredSafetyConstrainedBOAgent,
)
from chemworld.legacy.agents.diagnostic_live_llm import MechanismDiagnosticLiveLLMAgent
from chemworld.legacy.agents.event import ScriptedChemistryAgent
from chemworld.legacy.agents.greedy import GreedyLocalAgent
from chemworld.legacy.agents.lhs import LatinHypercubeAgent
from chemworld.legacy.agents.live_llm import LiveLLMAgent
from chemworld.legacy.agents.llm import (
    CodexSubagentOnlineAgent,
    CodexSubagentReplayAgent,
    LLMCompletionReplayAgent,
    LLMPlannerAgent,
    LLMReplayAgent,
    ToolUsingLLMStubAgent,
)
from chemworld.legacy.agents.random import RandomAgent, RandomRecipeAgent
from chemworld.legacy.agents.rl import FrozenSB3Agent

__all__ = [
    "INTERACTION_CONTRACT_VERSION",
    "Agent",
    "AgentDecisionContext",
    "CodexSubagentOnlineAgent",
    "CodexSubagentReplayAgent",
    "DecisionAuditRecord",
    "FrozenSB3Agent",
    "GaussianProcessBOAgent",
    "GaussianProcessPIAgent",
    "GaussianProcessUCBAgent",
    "GreedyLocalAgent",
    "HistoryRecord",
    "InteractionCapabilities",
    "LLMCompletionReplayAgent",
    "LLMPlannerAgent",
    "LLMReplayAgent",
    "LatinHypercubeAgent",
    "LiveLLMAgent",
    "MechanismDiagnosticLiveLLMAgent",
    "RandomAgent",
    "RandomForestEIAgent",
    "RandomRecipeAgent",
    "SafetyConstrainedBOAgent",
    "ScriptedChemistryAgent",
    "StructuredGaussianProcessBOAgent",
    "StructuredGaussianProcessPIAgent",
    "StructuredGaussianProcessUCBAgent",
    "StructuredRandomForestEIAgent",
    "StructuredSafetyConstrainedBOAgent",
    "ToolUsingLLMStubAgent",
]
