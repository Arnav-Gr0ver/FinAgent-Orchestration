"""Base Tool implementation for LangGraph agents.

This module provides a base tool class that integrates with both LangGraph
and MCP protocols for agent-to-agent communication.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, validator


logger = logging.getLogger(__name__)


class ToolInput(BaseModel):
    """Base input model for tools."""
    
    class Config:
        """Pydantic config."""
        extra = "forbid"
        validate_assignment = True


class ToolOutput(BaseModel):
    """Base output model for tools."""
    
    success: bool = Field(description="Whether the tool execution was successful")
    result: Any = Field(description="Tool execution result")
    error: Optional[str] = Field(default=None, description="Error message if execution failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    execution_time_ms: Optional[float] = Field(default=None, description="Execution time in milliseconds")


class ToolContext(BaseModel):
    """Context information for tool execution."""
    
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier")
    agent_id: Optional[str] = Field(default=None, description="Agent identifier")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")


class BaseAsyncTool(BaseTool, ABC):
    """Abstract base class for async tools with enhanced error handling and logging.
    
    Provides a foundation for building production-ready tools with:
    - Async execution support
    - Input/output validation
    - Error handling and logging
    - Performance monitoring
    - Context management
    """
    
    # Tool configuration
    return_direct: bool = False
    handle_tool_error: bool = True
    
    # Input/output models
    args_schema: Type[ToolInput] = ToolInput
    output_schema: Type[ToolOutput] = ToolOutput
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the tool."""
        super().__init__(**kwargs)
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    async def _arun(
        self,
        *args: Any,
        context: Optional[ToolContext] = None,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute the tool asynchronously.
        
        Args:
            *args: Positional arguments
            context: Execution context
            **kwargs: Keyword arguments
            
        Returns:
            ToolOutput with execution result
        """
        pass
    
    def _run(self, *args: Any, **kwargs: Any) -> ToolOutput:
        """Synchronous wrapper for async execution."""
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, create a new task
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, self._arun(*args, **kwargs))
                        return future.result()
                else:
                    return loop.run_until_complete(self._arun(*args, **kwargs))
            except RuntimeError:
                # No event loop, create a new one
                return asyncio.run(self._arun(*args, **kwargs))
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution failed: {str(e)}")
            return ToolOutput(
                success=False,
                result=None,
                error=str(e),
                metadata={"error_type": type(e).__name__}
            )
    
    async def arun(
        self,
        *args: Any,
        context: Optional[ToolContext] = None,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute the tool asynchronously with monitoring.
        
        Args:
            *args: Positional arguments
            context: Execution context
            **kwargs: Keyword arguments
            
        Returns:
            ToolOutput with execution result
        """
        start_time = datetime.now()
        self._execution_count += 1
        
        try:
            logger.debug(f"Executing tool '{self.name}' (run #{self._execution_count})")
            
            # Execute the tool
            result = await self._arun(*args, context=context, **kwargs)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._total_execution_time += execution_time
            
            # Update result metadata
            if result.execution_time_ms is None:
                result.execution_time_ms = execution_time
            
            result.metadata.update({
                "execution_count": self._execution_count,
                "average_execution_time_ms": self._total_execution_time / self._execution_count,
            })
            
            logger.debug(f"Tool '{self.name}' completed successfully in {execution_time:.2f}ms")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"Tool '{self.name}' failed after {execution_time:.2f}ms: {str(e)}")
            
            return ToolOutput(
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=execution_time,
                metadata={
                    "error_type": type(e).__name__,
                    "execution_count": self._execution_count,
                }
            )
    
    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            Validated input data
            
        Raises:
            ValueError: If validation fails
        """
        try:
            if self.args_schema and self.args_schema != ToolInput:
                validated = self.args_schema(**input_data)
                return validated.dict()
            return input_data
        except Exception as e:
            raise ValueError(f"Input validation failed for tool '{self.name}': {str(e)}")
    
    def get_input_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool input.
        
        Returns:
            JSON schema dictionary
        """
        if self.args_schema and self.args_schema != ToolInput:
            return self.args_schema.schema()
        return {}
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tool output.
        
        Returns:
            JSON schema dictionary
        """
        return self.output_schema.schema()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time,
            "average_execution_time_ms": (
                self._total_execution_time / self._execution_count 
                if self._execution_count > 0 else 0
            ),
        }
    
    def reset_statistics(self) -> None:
        """Reset execution statistics."""
        self._execution_count = 0
        self._total_execution_time = 0.0
    
    def to_mcp_tool_definition(self) -> Dict[str, Any]:
        """Convert to MCP tool definition format.
        
        Returns:
            MCP tool definition dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.get_input_schema(),
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Override in subclasses if cleanup is needed
        pass


class SimpleAsyncTool(BaseAsyncTool):
    """Simple async tool implementation for quick tool creation."""
    
    def __init__(
        self,
        name: str,
        description: str,
        func: callable,
        args_schema: Optional[Type[ToolInput]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize simple async tool.
        
        Args:
            name: Tool name
            description: Tool description
            func: Async function to execute
            args_schema: Input schema class
            **kwargs: Additional tool arguments
        """
        self._name = name
        self._description = description
        self._func = func
        
        if args_schema:
            self.args_schema = args_schema
        
        super().__init__(**kwargs)
    
    @property
    def name(self) -> str:
        """Tool name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Tool description."""
        return self._description
    
    async def _arun(
        self,
        *args: Any,
        context: Optional[ToolContext] = None,
        **kwargs: Any,
    ) -> ToolOutput:
        """Execute the provided function.
        
        Args:
            *args: Positional arguments
            context: Execution context
            **kwargs: Keyword arguments
            
        Returns:
            ToolOutput with execution result
        """
        try:
            # Call the function
            if asyncio.iscoroutinefunction(self._func):
                result = await self._func(*args, **kwargs)
            else:
                result = self._func(*args, **kwargs)
            
            return ToolOutput(
                success=True,
                result=result,
                metadata={
                    "context": context.dict() if context else {},
                }
            )
            
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=str(e),
                metadata={
                    "context": context.dict() if context else {},
                    "error_type": type(e).__name__,
                }
            )


class ToolRegistry:
    """Registry for managing tool instances."""
    
    def __init__(self) -> None:
        """Initialize the registry."""
        self._tools: Dict[str, BaseAsyncTool] = {}
        self._categories: Dict[str, List[str]] = {}
    
    def register(
        self, 
        tool: BaseAsyncTool, 
        category: Optional[str] = None
    ) -> None:
        """Register a tool.
        
        Args:
            tool: Tool instance to register
            category: Optional category for organization
        """
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' is already registered, overwriting")
        
        self._tools[tool.name] = tool
        
        if category:
            if category not in self._categories:
                self._categories[category] = []
            if tool.name not in self._categories[category]:
                self._categories[category].append(tool.name)
        
        logger.info(f"Registered tool '{tool.name}' in category '{category or 'default'}'")
    
    def unregister(self, name: str) -> bool:
        """Unregister a tool.
        
        Args:
            name: Tool name to unregister
            
        Returns:
            True if tool was found and removed, False otherwise
        """
        if name not in self._tools:
            return False
        
        # Remove from tools
        del self._tools[name]
        
        # Remove from categories
        for category, tool_names in self._categories.items():
            if name in tool_names:
                tool_names.remove(name)
        
        logger.info(f"Unregistered tool '{name}'")
        return True
    
    def get_tool(self, name: str) -> Optional[BaseAsyncTool]:
        """Get a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[str] = None) -> List[str]:
        """List tool names.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool names
        """
        if category:
            return self._categories.get(category, [])
        return list(self._tools.keys())
    
    def list_categories(self) -> List[str]:
        """List all categories.
        
        Returns:
            List of category names
        """
        return list(self._categories.keys())
    
    def get_tool_definitions(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get MCP tool definitions.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of MCP tool definitions
        """
        tool_names = self.list_tools(category)
        definitions = []
        
        for name in tool_names:
            tool = self._tools[name]
            definitions.append(tool.to_mcp_tool_definition())
        
        return definitions
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Statistics dictionary
        """
        tool_stats = {}
        for name, tool in self._tools.items():
            tool_stats[name] = tool.get_statistics()
        
        return {
            "total_tools": len(self._tools),
            "categories": {
                category: len(tools) 
                for category, tools in self._categories.items()
            },
            "tool_statistics": tool_stats,
        }
    
    def reset_all_statistics(self) -> None:
        """Reset statistics for all tools."""
        for tool in self._tools.values():
            tool.reset_statistics()
        logger.info("Reset statistics for all tools")


# Global registry instance
default_registry = ToolRegistry()


def register_tool(
    tool: BaseAsyncTool, 
    category: Optional[str] = None,
    registry: Optional[ToolRegistry] = None
) -> None:
    """Register a tool in the default or specified registry.
    
    Args:
        tool: Tool to register
        category: Optional category
        registry: Optional registry instance (defaults to global registry)
    """
    target_registry = registry or default_registry
    target_registry.register(tool, category)


def create_simple_tool(
    name: str,
    description: str,
    func: callable,
    args_schema: Optional[Type[ToolInput]] = None,
    category: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
) -> SimpleAsyncTool:
    """Create and register a simple async tool.
    
    Args:
        name: Tool name
        description: Tool description
        func: Function to execute
        args_schema: Optional input schema
        category: Optional category
        registry: Optional registry instance
        
    Returns:
        Created tool instance
    """
    tool = SimpleAsyncTool(
        name=name,
        description=description,
        func=func,
        args_schema=args_schema,
    )
    
    register_tool(tool, category, registry)
    return tool