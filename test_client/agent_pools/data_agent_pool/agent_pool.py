# agent_pool.py
"""A pool of specialized agents for handling various data-related tasks."""

import logging
from typing import Any, Dict

from agent_pools.data_agent_pool.agents import (
    CryptoAgent,
    EquityAgent,
    NewsAgent,
    TaskDelegationAgent,
    ValidationAgent,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataAgentPool:
    """A pool of agents that work together to process complex queries."""

    def __init__(self, model: Any):
        """Initialize the agent pool with all required agents.

        Args:
            model: The language model instance to be used by all agents.
        """
        logger.info("Initializing Data Agent Pool...")
        self.delegator = TaskDelegationAgent(model)
        self.validator = ValidationAgent(model)
        self.specialists: Dict[str, Any] = {
            "crypto": CryptoAgent(model),
            "equity": EquityAgent(model),
            "news": NewsAgent(model),
        }
        logger.info("Data Agent Pool initialized successfully.")

    async def process_query(self, query: str, session_id: str = None) -> str:
        """Process a query by delegating, executing, and validating.

        Args:
            query: The user's query.
            session_id: An optional session ID for conversation history.

        Returns:
            A validated response to the query.
        """
        logger.info(f"Received query: '{query}'")

        # 1. Delegate the task to the appropriate specialist
        delegation_response = await self.delegator.invoke(query, session_id)
        delegate_name = delegation_response.content.strip().lower()
        logger.info(f"Delegating task to '{delegate_name}' agent.")

        # 2. Get the specialist agent
        specialist_agent = self.specialists.get(delegate_name)
        if not specialist_agent:
            return f"Error: Could not find a specialist agent named '{delegate_name}'."

        # 3. Invoke the specialist agent to get the primary response
        logger.info(f"Invoking '{delegate_name}' agent...")
        specialist_response = await specialist_agent.invoke(query, session_id)
        initial_content = specialist_response.content
        logger.info(f"Specialist response: '{initial_content}'")

        # 4. Invoke the validation agent to check the response
        logger.info("Invoking 'validation' agent...")
        validation_query = (
            f"Please validate the following information: '{initial_content}'"
        )
        final_response = await self.validator.invoke(validation_query, session_id)
        logger.info(f"Final validated response: '{final_response.content}'")

        return final_response.content