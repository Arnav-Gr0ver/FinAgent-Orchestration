"""Async BaseAgent class for LangGraph with A2A protocol integration.

This module provides a base agent implementation using LangGraph framework
with support for Google's Agent-to-Agent (A2A) protocol communication.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, AsyncIterable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    """Structured response format for agent operations."""
    
    content: str = Field(description="Response content")
    is_task_complete: bool = Field(default=False, description="Task completion status")
    require_user_input: bool = Field(default=False, description="Input requirement flag")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class BaseAgent(ABC):
    """Abstract base agent class with async LangGraph and A2A protocol support.
    
    Provides a foundation for building production-ready agents with:
    - Async LangGraph orchestration
    - A2A protocol compatibility  
    - Memory management
    - Tool integration
    - Streaming support
    """
    
    SUPPORTED_CONTENT_TYPES: List[str] = ["text/plain"]
    
    def __init__(
        self,
        model: Any,
        tools: Optional[List[BaseTool]] = None,
        system_instruction: Optional[str] = None,
        memory_store: Optional[MemorySaver] = None,
    ) -> None:
        """Initialize the base agent.
        
        Args:
            model: Language model instance (e.g., ChatOpenAI, ChatGoogleGenerativeAI)
            tools: List of tools available to the agent
            system_instruction: System prompt for the agent
            memory_store: Memory store for conversation persistence
        """
        self._model = model
        self._tools = tools or []
        self._system_instruction = system_instruction or self._get_default_system_instruction()
        self._memory = memory_store or MemorySaver()
        self._graph = self._create_agent_graph()
    
    @property
    def model(self) -> Any:
        """Get the language model."""
        return self._model
    
    @property
    def tools(self) -> List[BaseTool]:
        """Get available tools."""
        return self._tools.copy()
    
    @property
    def graph(self) -> StateGraph:
        """Get the agent graph."""
        return self._graph
    
    def _create_agent_graph(self) -> StateGraph:
        """Create the LangGraph agent with ReAct pattern.
        
        Returns:
            Configured agent graph
        """
        return create_react_agent(
            model=self._model,
            tools=self._tools,
            checkpointer=self._memory,
            prompt=self._system_instruction,
        )
    
    @abstractmethod
    def _get_default_system_instruction(self) -> str:
        """Get the default system instruction for this agent.
        
        Returns:
            Default system instruction string
        """
        pass
    
    async def invoke(self, query: str, session_id: Optional[str] = None) -> AgentResponse:
        """Invoke the agent asynchronously.
        
        Args:
            query: User query string
            session_id: Optional session identifier for context persistence
            
        Returns:
            AgentResponse with the result
        """
        session_id = session_id or str(uuid.uuid4())
        config = self._create_config(session_id)
        
        result = await self._graph.ainvoke(
            {"messages": [HumanMessage(content=query)]}, 
            config
        )
        
        return self._process_graph_result(result, config)
    
    async def stream(
        self, 
        query: str, 
        session_id: Optional[str] = None
    ) -> AsyncIterable[AgentResponse]:
        """Stream agent responses asynchronously.
        
        Args:
            query: User query string
            session_id: Optional session identifier for context persistence
            
        Yields:
            AgentResponse objects as they become available
        """
        session_id = session_id or str(uuid.uuid4())
        config = self._create_config(session_id)
        inputs = {"messages": [HumanMessage(content=query)]}
        
        async for chunk in self._graph.astream(inputs, config, stream_mode="values"):
            if "messages" in chunk and chunk["messages"]:
                message = chunk["messages"][-1]
                response = self._process_streaming_message(message)
                if response:
                    yield response
        
        # Final response
        final_result = await self._graph.ainvoke(inputs, config)
        yield self._process_graph_result(final_result, config, is_final=True)
    
    def _create_config(self, session_id: str) -> RunnableConfig:
        """Create configuration for graph execution.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Runnable configuration
        """
        return RunnableConfig(
            configurable={"thread_id": session_id},
            metadata={"session_id": session_id, "timestamp": datetime.now().isoformat()}
        )
    
    def _process_graph_result(
        self, 
        result: Dict[str, Any], 
        config: RunnableConfig,
        is_final: bool = True
    ) -> AgentResponse:
        """Process the graph execution result.
        
        Args:
            result: Graph execution result
            config: Execution configuration
            is_final: Whether this is the final response
            
        Returns:
            Processed AgentResponse
        """
        messages = result.get("messages", [])
        if not messages:
            return AgentResponse(
                content="No response generated",
                is_task_complete=True,
                metadata={"error": "empty_response"}
            )
        
        last_message = messages[-1]
        return self._create_agent_response(last_message, is_final)
    
    def _process_streaming_message(self, message: BaseMessage) -> Optional[AgentResponse]:
        """Process a streaming message.
        
        Args:
            message: Message from stream
            
        Returns:
            AgentResponse if message should be yielded, None otherwise
        """
        if isinstance(message, AIMessage):
            if hasattr(message, 'tool_calls') and message.tool_calls:
                return AgentResponse(
                    content="Processing with tools...",
                    is_task_complete=False,
                    require_user_input=False,
                    metadata={"stage": "tool_execution"}
                )
            elif message.content:
                return self._create_agent_response(message, is_final=False)
        
        return None
    
    def _create_agent_response(
        self, 
        message: BaseMessage, 
        is_final: bool = True
    ) -> AgentResponse:
        """Create an AgentResponse from a message.
        
        Args:
            message: Source message
            is_final: Whether this is the final response
            
        Returns:
            AgentResponse object
        """
        content = str(message.content) if message.content else ""
        
        return AgentResponse(
            content=content,
            is_task_complete=is_final,
            require_user_input=self._requires_user_input(content),
            metadata={
                "message_type": type(message).__name__,
                "timestamp": datetime.now().isoformat(),
                "is_final": is_final
            }
        )
    
    def _requires_user_input(self, content: str) -> bool:
        """Determine if the response requires user input.
        
        Args:
            content: Response content
            
        Returns:
            True if user input is required
        """
        # Override in subclasses for specific logic
        return "?" in content and any(
            keyword in content.lower() 
            for keyword in ["which", "what", "specify", "clarify", "need more"]
        )
    
    async def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the agent.
        
        Args:
            tool: Tool to add
        """
        if tool not in self._tools:
            self._tools.append(tool)
            self._graph = self._create_agent_graph()
    
    async def remove_tool(self, tool: BaseTool) -> bool:
        """Remove a tool from the agent.
        
        Args:
            tool: Tool to remove
            
        Returns:
            True if tool was removed, False if not found
        """
        try:
            self._tools.remove(tool)
            self._graph = self._create_agent_graph()
            return True
        except ValueError:
            return False
    
    async def get_conversation_history(self, session_id: str) -> List[BaseMessage]:
        """Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of messages in the conversation
        """
        config = self._create_config(session_id)
        state = await self._graph.aget_state(config)
        return state.values.get("messages", []) if state.values else []
    
    async def clear_conversation_history(self, session_id: str) -> bool:
        """Clear conversation history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            True if history was cleared successfully
        """
        try:
            config = self._create_config(session_id)
            # Reset the conversation state
            await self._graph.aupdate_state(config, {"messages": []})
            return True
        except Exception:
            return False