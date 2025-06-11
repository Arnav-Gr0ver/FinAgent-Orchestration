"""Base MCP Tool Server for agent-to-agent communication.

This module provides a base MCP server implementation for exposing tools
and resources to other agents via the Model Context Protocol.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base
from mcp.types import Tool as MCPTool, Resource as MCPResource
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


@dataclass
class ServerContext:
    """Context container for server lifespan resources."""
    metadata: Dict[str, Any]
    initialized_at: str
    

class ToolDefinition(BaseModel):
    """Definition for an MCP tool."""
    name: str = Field(description="Tool name")
    description: str = Field(description="Tool description")
    input_schema: Dict[str, Any] = Field(description="JSON schema for input parameters")


class ResourceDefinition(BaseModel):
    """Definition for an MCP resource."""
    uri_template: str = Field(description="URI template for the resource")
    name: str = Field(description="Resource name")
    description: str = Field(description="Resource description")
    mime_type: str = Field(default="text/plain", description="MIME type of resource")


class BaseMCPToolServer(ABC):
    """Abstract base class for MCP tool servers with A2A support.
    
    Provides a foundation for creating MCP servers that expose tools and resources
    to other agents for agent-to-agent communication.
    """
    
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> None:
        """Initialize the MCP tool server.
        
        Args:
            name: Server name
            description: Server description
            dependencies: List of Python dependencies
        """
        self._name = name
        self._description = description or f"{name} MCP Tool Server"
        self._dependencies = dependencies or []
        self._server: Optional[FastMCP] = None
        self._is_initialized = False
    
    @property
    def name(self) -> str:
        """Get server name."""
        return self._name
    
    @property
    def description(self) -> str:
        """Get server description."""
        return self._description
    
    @property
    def server(self) -> FastMCP:
        """Get the FastMCP server instance."""
        if not self._server:
            raise RuntimeError("Server not initialized. Call initialize() first.")
        return self._server
    
    async def initialize(self) -> None:
        """Initialize the MCP server with lifespan management."""
        if self._is_initialized:
            return
        
        self._server = FastMCP(
            name=self._name,
            dependencies=self._dependencies,
            lifespan=self._create_lifespan,
        )
        
        # Register tools and resources
        await self._register_tools()
        await self._register_resources()
        await self._register_prompts()
        
        self._is_initialized = True
        logger.info(f"MCP server '{self._name}' initialized successfully")
    
    @asynccontextmanager
    async def _create_lifespan(self, server: FastMCP) -> AsyncIterator[ServerContext]:
        """Manage server lifespan with context."""
        logger.info(f"Starting MCP server: {self._name}")
        
        # Initialize context
        context = ServerContext(
            metadata=await self._initialize_context(),
            initialized_at=str(datetime.now().isoformat()),
        )
        
        try:
            yield context
        finally:
            logger.info(f"Shutting down MCP server: {self._name}")
            await self._cleanup_context(context)
    
    @abstractmethod
    async def _initialize_context(self) -> Dict[str, Any]:
        """Initialize server context.
        
        Returns:
            Context metadata dictionary
        """
        pass
    
    @abstractmethod
    async def _cleanup_context(self, context: ServerContext) -> None:
        """Clean up server context.
        
        Args:
            context: Server context to clean up
        """
        pass
    
    @abstractmethod
    async def _register_tools(self) -> None:
        """Register tools with the MCP server."""
        pass
    
    @abstractmethod
    async def _register_resources(self) -> None:
        """Register resources with the MCP server."""
        pass
    
    async def _register_prompts(self) -> None:
        """Register prompts with the MCP server. Override if needed."""
        # Default implementation - no prompts
        pass
    
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: callable,
    ) -> None:
        """Register a tool with the server.
        
        Args:
            name: Tool name
            description: Tool description
            input_schema: JSON schema for input validation
            handler: Async function to handle tool calls
        """
        if not self._server:
            raise RuntimeError("Server not initialized")
        
        # Create tool decorator and apply to handler
        tool_decorator = self._server.tool()
        decorated_handler = tool_decorator(handler)
        
        # Store tool metadata
        decorated_handler.__annotations__ = self._create_annotations_from_schema(input_schema)
        decorated_handler.__doc__ = description
        
        logger.debug(f"Registered tool: {name}")
    
    def register_resource(
        self,
        uri_template: str,
        handler: callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Register a resource with the server.
        
        Args:
            uri_template: URI template for the resource
            handler: Function to handle resource requests
            name: Resource name (defaults to URI template)
            description: Resource description
        """
        if not self._server:
            raise RuntimeError("Server not initialized")
        
        # Create resource decorator and apply to handler  
        resource_decorator = self._server.resource(uri_template)
        decorated_handler = resource_decorator(handler)
        
        # Add metadata if provided
        if description:
            decorated_handler.__doc__ = description
        
        logger.debug(f"Registered resource: {uri_template}")
    
    def register_prompt(
        self,
        name: str,
        handler: callable,
        description: Optional[str] = None,
    ) -> None:
        """Register a prompt with the server.
        
        Args:
            name: Prompt name
            handler: Function to handle prompt requests
            description: Prompt description
        """
        if not self._server:
            raise RuntimeError("Server not initialized")
        
        # Create prompt decorator and apply to handler
        prompt_decorator = self._server.prompt()
        decorated_handler = prompt_decorator(handler)
        
        if description:
            decorated_handler.__doc__ = description
        
        logger.debug(f"Registered prompt: {name}")
    
    def _create_annotations_from_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Create function annotations from JSON schema.
        
        Args:
            schema: JSON schema dictionary
            
        Returns:
            Function annotations dictionary
        """
        annotations = {}
        properties = schema.get("properties", {})
        
        for param_name, param_def in properties.items():
            param_type = param_def.get("type", "string")
            
            # Map JSON schema types to Python types
            type_mapping = {
                "string": str,
                "integer": int,
                "number": float,
                "boolean": bool,
                "array": List[Any],
                "object": Dict[str, Any],
            }
            
            annotations[param_name] = type_mapping.get(param_type, Any)
        
        return annotations
    
    async def start_stdio(self) -> None:
        """Start the server with stdio transport."""
        if not self._is_initialized:
            await self.initialize()
        
        logger.info(f"Starting MCP server '{self._name}' with stdio transport")
        self._server.run()
    
    async def start_sse(self, host: str = "localhost", port: int = 8000) -> None:
        """Start the server with SSE transport.
        
        Args:
            host: Server host
            port: Server port
        """
        if not self._is_initialized:
            await self.initialize()
        
        logger.info(f"Starting MCP server '{self._name}' with SSE transport on {host}:{port}")
        # SSE server would be started here - implementation depends on deployment strategy
        # For now, log the configuration
        logger.info(f"SSE server configured for {host}:{port}")
    
    def get_sse_app(self):
        """Get the ASGI app for SSE transport mounting.
        
        Returns:
            ASGI application for mounting
        """
        if not self._is_initialized:
            raise RuntimeError("Server not initialized")
        
        return self._server.sse_app()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check.
        
        Returns:
            Health status dictionary
        """
        return {
            "status": "healthy" if self._is_initialized else "not_initialized",
            "name": self._name,
            "description": self._description,
            "initialized": self._is_initialized,
        }


# Import datetime after avoiding conflicts
from datetime import datetime