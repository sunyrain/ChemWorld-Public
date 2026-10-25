"""Interfaces and tools for user-owned agents.

Historical preset policies live in ``chemworld.legacy.agents``. Importing this
package does not load them or select an agent, objective, or reward function.
"""

from chemworld.agents.base import Agent, BaseAgent, HistoryRecord
from chemworld.agents.interaction import (
    INTERACTION_CONTRACT_VERSION,
    AgentDecisionContext,
    DecisionAuditRecord,
    InteractionCapabilities,
)

__all__ = [
    "INTERACTION_CONTRACT_VERSION", "Agent", "AgentDecisionContext", "BaseAgent",
    "DecisionAuditRecord", "HistoryRecord", "InteractionCapabilities",
]
