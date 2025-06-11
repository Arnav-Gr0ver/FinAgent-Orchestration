# agents/news_agent.py
"""Agent specialized in news data."""

from typing import Any, List, Optional

from langchain_core.tools import BaseTool

from base.base_agent import BaseAgent
from tools.news_tools import NewsSearchTool


class NewsAgent(BaseAgent):
    """An agent that specializes in finding and reporting news."""

    def __init__(
        self,
        model: Any,
        tools: Optional[List[BaseTool]] = None,
        **kwargs,
    ) -> None:
        # If no tools are provided, initialize with the default news tool
        if tools is None:
            tools = [NewsSearchTool()]
        super().__init__(model=model, tools=tools, **kwargs)

    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for this agent.

        Returns:
            Default system instruction string.
        """
        return (
            "You are a news reporter. Your job is to find and summarize the latest "
            "news on a given topic. Use the news search tool to answer user queries."
        )