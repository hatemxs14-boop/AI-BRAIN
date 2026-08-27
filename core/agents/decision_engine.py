from __future__ import annotations

from abc import ABC, abstractmethod

from core.agents.agent_action import AgentAction
from core.agents.agent_context import AgentContext


class AgentDecisionEngine(ABC):
    """
    Abstract decision engine for an AI-BRAIN Agent.

    The decision engine receives the current AgentContext
    and returns exactly one AgentAction.

    It does not:

    - execute tools
    - access executors
    - perform authorization
    - make security decisions
    - modify the execution state directly
    """

    @abstractmethod
    def decide(
        self,
        context: AgentContext,
    ) -> AgentAction:
        """
        Produce the next action from the current context.
        """
        raise NotImplementedError