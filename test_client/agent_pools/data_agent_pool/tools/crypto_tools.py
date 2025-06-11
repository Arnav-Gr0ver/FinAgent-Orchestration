import random
from typing import Type

from pydantic import BaseModel, Field

# Updated import path
from base.base_tool import BaseAsyncTool, ToolContext, ToolInput, ToolOutput


class CryptoPriceInput(ToolInput):
    symbol: str = Field(description="The cryptocurrency symbol (e.g., BTC, ETH).")


class CryptoPriceTool(BaseAsyncTool):
    name: str = "get_crypto_price"
    description: str = "Fetches the current price for a given cryptocurrency symbol."
    args_schema: Type[BaseModel] = CryptoPriceInput

    async def _arun(self, symbol: str, context: ToolContext = None) -> ToolOutput:
        price = random.uniform(2000, 70000)
        return ToolOutput(
            success=True,
            result=f"The current price of {symbol.upper()} is ${price:,.2f}.",
        )