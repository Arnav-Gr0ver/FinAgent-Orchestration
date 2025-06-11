# tools/equity_tools.py
"""Tools for the Equity Agent."""

import random
from typing import Type

from pydantic import BaseModel, Field

from base_tool import BaseAsyncTool, ToolContext, ToolInput, ToolOutput


class EquityPriceInput(ToolInput):
    """Input for getting equity prices."""

    ticker: str = Field(description="The stock ticker symbol (e.g., AAPL, GOOGL).")


class EquityPriceTool(BaseAsyncTool):
    """Tool to get the current price of a stock."""

    name: str = "get_equity_price"
    description: str = "Fetches the current price for a given stock ticker symbol."
    args_schema: Type[BaseModel] = EquityPriceInput

    async def _arun(self, ticker: str, context: ToolContext = None) -> ToolOutput:
        """Simulates fetching a stock price."""
        # In a real implementation, this would call a financial data API
        price = random.uniform(100, 3000)
        return ToolOutput(
            success=True,
            result=f"The current price of {ticker.upper()} is ${price:,.2f}.",
        )