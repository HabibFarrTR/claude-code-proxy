from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, field_validator

# --- Pydantic Models (Anthropic Format) ---
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str
    data: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: ContentBlockImageSource


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]]]
    is_error: Optional[bool] = False


# Use Field alias for Pydantic v2 compatibility if needed, though Union should work
ContentBlock = Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]


class ToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: Optional[List[str]] = None


class ToolDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema


class MessagesRequest(BaseModel):
    model: str  # This will hold the *mapped* Gemini model ID after validation
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    original_model_name: Optional[str] = None  # Internal field to store original name pre-mapping

    @field_validator("model")
    def validate_and_map_model(cls, v, info):
        # The validator now just returns the mapped Gemini model ID
        from src.utils import map_model_name
        return map_model_name(v)


class TokenCountRequest(BaseModel):
    model: str  # Mapped Gemini model ID
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    original_model_name: Optional[str] = None  # Internal field

    @field_validator("model")
    def validate_and_map_model_token_count(cls, v, info):
        from src.utils import map_model_name
        return map_model_name(v)


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str  # Original Anthropic model name
    content: List[ContentBlock]
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage