# agents/crypto_agent.py
"""Agent specialized in cryptocurrency data."""

from typing import Any, List, Optional

from langchain_core.tools import BaseTool

from base.base_agent import BaseAgent
from tools.crypto_tools import CryptoPriceTool


class CryptoAgent(BaseAgent):
    """An agent that specializes in providing cryptocurrency information."""

    def __init__(
        self,
        model: Any,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ) -> None:
        # If no tools are provided, initialize with the default crypto tool
        if tools is None:
            tools = [CryptoPriceTool()]
        super().__init__(model=model, tools=tools, **kwargs)

    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for this agent.

        Returns:
            Default system instruction string.
        """
        return (
            "You are a cryptocurrency expert. Your primary function is to provide "
            "accurate and up-to-date price information for various cryptocurrencies. "
            "Use the available tools to answer questions about crypto prices."
        )