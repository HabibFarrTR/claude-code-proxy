"""Pydantic models for API request/response validation.

Defines data models for the proxy API, including models for:
- Content blocks (text, images, tool use)
- Messages
- API requests and responses
- Model name mapping between Claude and Gemini
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, field_validator, model_validator

from src.config import GEMINI_BIG_MODEL, GEMINI_SMALL_MODEL
from src.utils import get_logger

logger = get_logger()


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

    @model_validator(mode='before')
    def store_original_model(cls, values):
        # Store the original model name before field validation
        if isinstance(values, dict) and 'model' in values:
            values['original_model_name'] = values['model']
        return values

    @field_validator("model")
    def validate_and_map_model(cls, v) -> str:
        # Return the mapped Gemini model ID
        mapped_model = map_model_name(v)
        logger.debug(f"Model mapped: '{v}' -> '{mapped_model}'")
        return mapped_model


class TokenCountRequest(BaseModel):
    model: str  # Mapped Gemini model ID
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    original_model_name: Optional[str] = None  # Internal field

    @model_validator(mode='before')
    def store_original_model_token_count(cls, values):
        # Store the original model name before field validation
        if isinstance(values, dict) and 'model' in values:
            values['original_model_name'] = values['model']
        return values

    @field_validator("model")
    def validate_and_map_model_token_count(cls, v):
        # Return the mapped Gemini model ID
        mapped_model = map_model_name(v)
        logger.debug(f"Token count model mapped: '{v}' -> '{mapped_model}'")
        return mapped_model


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


def map_model_name(anthropic_model_name: str) -> str:
    """Map Anthropic model names to equivalent Gemini models.

    Translates Claude model names (haiku, sonnet, opus) to their appropriate
    Gemini counterparts based on model capabilities and size.

    Args:
        anthropic_model_name: The original Claude model name requested

    Returns:
        str: The corresponding Gemini model ID to use
    """
    original_model = anthropic_model_name
    mapped_gemini_model = GEMINI_BIG_MODEL  # Default to the larger model

    logger.debug(
        f"Attempting to map model: '{original_model}' -> Target Gemini BIG='{GEMINI_BIG_MODEL}', SMALL='{GEMINI_SMALL_MODEL}'"
    )

    clean_name = anthropic_model_name.lower().split("@")[0]
    if clean_name.startswith("anthropic/"):
        clean_name = clean_name[10:]
    elif clean_name.startswith("gemini/"):
        clean_name = clean_name[7:]  # Allow direct gemini model names like 'gemini/gemini-2.5-pro-latest'

    if "haiku" in clean_name:
        mapped_gemini_model = GEMINI_SMALL_MODEL
        logger.info(f"Mapping '{original_model}' (Haiku) -> Target Gemini SMALL '{mapped_gemini_model}'")
    elif "sonnet" in clean_name or "opus" in clean_name:
        mapped_gemini_model = GEMINI_BIG_MODEL
        logger.info(f"Mapping '{original_model}' (Sonnet/Opus) -> Target Gemini BIG '{mapped_gemini_model}'")
    elif clean_name == GEMINI_BIG_MODEL.lower() or clean_name == GEMINI_SMALL_MODEL.lower():
        mapped_gemini_model = clean_name  # Use the directly specified Gemini model
        logger.info(f"Using directly specified target Gemini model: '{mapped_gemini_model}'")
    else:
        logger.warning(
            f"Unrecognized Anthropic model name '{original_model}'. Defaulting to BIG model '{mapped_gemini_model}'."
        )

    # Return just the Gemini model ID (e.g., "gemini-2.5-pro-latest") for the SDK
    return mapped_gemini_model
