# tools/news_tools.py
"""Tools for the News Agent."""

from typing import Type

from pydantic import BaseModel, Field

from base_tool import BaseAsyncTool, ToolContext, ToolInput, ToolOutput


class NewsSearchInput(ToolInput):
    """Input for searching news."""

    query: str = Field(description="The topic to search for in the news.")


class NewsSearchTool(BaseAsyncTool):
    """Tool to search for recent news articles."""

    name: str = "search_news"
    description: str = "Searches for recent news articles on a given topic."
    args_schema: Type[BaseModel] = NewsSearchInput

    async def _arun(self, query: str, context: ToolContext = None) -> ToolOutput:
        """Simulates searching for news."""
        # In a real implementation, this would call a news API
        return ToolOutput(
            success=True,
            result=f"Found 3 news articles about '{query}'. The top headline is: '{query.title()} Stocks Surge Amidst Economic Optimism'.",
        )