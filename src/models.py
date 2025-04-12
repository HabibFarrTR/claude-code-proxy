"""
Pydantic models for the AIplatform proxy.
These models are used to validate and convert between Anthropic and AIplatform formats.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator


# Content Block Models
class ContentBlockText(BaseModel):
    """Text content block in Anthropic API format."""

    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    """Image content block in Anthropic API format."""

    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    """Tool use content block in Anthropic API format."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    """Tool result content block in Anthropic API format."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class SystemContent(BaseModel):
    """System message content in Anthropic API format."""

    type: Literal["text"]
    text: str


# Message Models
class Message(BaseModel):
    """Message in Anthropic API format."""

    role: Literal["user", "assistant"]
    content: Union[str, List[Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]]]


class Tool(BaseModel):
    """Tool definition in Anthropic API format."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    """Thinking configuration in Anthropic API format."""

    enabled: bool


# Request Models
class MessagesRequest(BaseModel):
    """Request for /v1/messages endpoint in Anthropic API format."""

    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    def validate_model_field(cls, v, info):  # Renamed to avoid conflict
        from src.utils import map_model_name

        return map_model_name(v, info.data)


class TokenCountRequest(BaseModel):
    """Request for /v1/messages/count_tokens endpoint in Anthropic API format."""

    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None  # Will store the original model name

    @field_validator("model")
    def validate_model_token_count(cls, v, info):  # Renamed to avoid conflict
        from src.utils import map_model_name

        return map_model_name(v, info.data)


# Response Models
class TokenCountResponse(BaseModel):
    """Response for /v1/messages/count_tokens endpoint."""

    input_tokens: int


class Usage(BaseModel):
    """Usage information in Anthropic API response format."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    """Response for /v1/messages endpoint in Anthropic API format."""

    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[Union[ContentBlockText, ContentBlockToolUse]]
    type: Literal["message"] = "message"
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage
