# agents/equity_agent.py
"""Agent specialized in equity data."""

from typing import Any, List, Optional

from langchain_core.tools import BaseTool

from base.base_agent import BaseAgent
from tools.equity_tools import EquityPriceTool


class EquityAgent(BaseAgent):
    """An agent that specializes in providing equity (stock) information."""

    def __init__(
        self,
        model: Any,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ) -> None:
        # If no tools are provided, initialize with the default equity tool
        if tools is None:
            tools = [EquityPriceTool()]
        super().__init__(model=model, tools=tools, **kwargs)

    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for this agent.

        Returns:
            Default system instruction string.
        """
        return (
            "You are a financial analyst specializing in equities. Your main role is to "
            "provide stock market prices and data. Use your tools to find information "
            "on stock tickers."
        )