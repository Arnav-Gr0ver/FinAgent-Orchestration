# agents/delegation_agent.py
"""Agent responsible for delegating tasks."""

from typing import Any

from base.base_agent import BaseAgent


class TaskDelegationAgent(BaseAgent):
    """An agent that analyzes a query and delegates it to the appropriate specialist agent."""

    def __init__(self, model: Any, **kwargs) -> None:
        # This agent does not use external tools
        super().__init__(model=model, tools=[], **kwargs)

    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for this agent.

        Returns:
            Default system instruction string
        """
        return (
            "You are a task router. Based on the user's query, you must decide which "
            "specialist agent is best suited to handle the request. The available "
            "specialists are: 'crypto', 'equity', 'news'.\n"
            "Respond with only a single word: the name of the specialist. For example, "
            "if the query is about stock prices, respond with 'equity'."
        )