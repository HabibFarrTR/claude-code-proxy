import asyncio
import json
import logging
import os
import sys
import time
import traceback  # For detailed error logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

import requests
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.oauth2.credentials import Credentials as OAuth2Credentials
from pydantic import BaseModel, field_validator
from vertexai.generative_models import (
    Content,
    FinishReason,
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)
from vertexai.generative_models._generative_models import (
    FinishReason,
    ResponseBlockedError,
)

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Using DEBUG level for more detailed logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure uvicorn and other libraries to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.auth.compute_engine._metadata").setLevel(logging.WARNING)


# List of supported AI Platform models
AIPLATFORM_MODELS = [
    "gemini-2.5-pro-preview-03-25",  # Main model for high-quality outputs
    "gemini-2.0-flash",  # Faster, smaller model for simpler queries
]


# Define ANSI color codes for terminal output (optional, for log_request_beautifully)
class Colors:
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    DIM = "\033[2m"


# --- FastAPI App Initialization ---
app = FastAPI(title="Anthropic to Gemini Proxy")


# --- Custom Exceptions ---
class AuthenticationError(Exception):
    """Custom exception for authentication failures."""

    pass


# --- Authentication ---
def get_gemini_credentials():
    """
    Authenticates with the custom endpoint and returns Vertex AI credentials.

    Returns:
        tuple: Contains project_id, location, and OAuth2Credentials object.

    Raises:
        AuthenticationError: If authentication fails for any reason.
    """
    workspace_id = os.getenv("WORKSPACE_ID")
    # Default to the more capable model if MODEL_NAME isn't set
    model_name = os.getenv("MODEL_NAME", AIPLATFORM_MODELS[0])
    auth_url = os.getenv("AUTH_URL")

    if not all([workspace_id, auth_url]):
        logger.error("Missing required environment variables: WORKSPACE_ID and/or AUTH_URL")
        raise AuthenticationError("Missing required environment variables: WORKSPACE_ID, AUTH_URL")

    logger.info(f"Attempting authentication for workspace '{workspace_id}' and model '{model_name}'")
    logger.debug(f"Authentication URL: {auth_url}")
    logger.info("NOTE: Authentication requires AWS credentials configured ('mltools-cli aws-login').")

    payload = {
        "workspace_id": workspace_id,
        "model_name": model_name,
    }

    logger.info(f"Requesting temporary token from {auth_url}")

    try:
        resp = requests.post(auth_url, headers=None, json=payload, timeout=30)

        if resp.status_code != 200:
            error_msg = f"Authentication request failed with status {resp.status_code}"
            try:
                # Try to get more details from the response body
                resp_content = resp.text
            except Exception:
                resp_content = "(Could not decode response body)"
            logger.error(f"{error_msg}. Response: {resp_content}")
            raise AuthenticationError(f"{error_msg}. See server logs for response details.")

        try:
            credentials_data = resp.json()  # Use resp.json() for better error handling
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from auth server: {e}"
            logger.error(f"{error_msg}. Raw Response: {resp.text}")
            raise AuthenticationError(error_msg)

        if "token" not in credentials_data:
            error_msg = credentials_data.get("message", "Token not found in authentication response.")
            logger.error(f"Authentication failed: {error_msg}. Response Data: {credentials_data}")
            raise AuthenticationError(f"Failed to retrieve token. Server response: {error_msg}")

        project_id = credentials_data.get("project_id")
        location = credentials_data.get("region")
        token = credentials_data["token"]
        expires_on_str = credentials_data.get("expires_on", "N/A")

        if not project_id or not location:
            logger.error(f"Missing 'project_id' or 'region' in auth response. Data: {credentials_data}")
            raise AuthenticationError("Missing 'project_id' or 'region' in authentication response.")

        logger.info(
            f"Successfully fetched Gemini credentials. Project: {project_id}, Location: {location}, Valid until: {expires_on_str}"
        )

        temp_creds = OAuth2Credentials(token)
        return project_id, location, temp_creds

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Failed to connect to authentication server: {e}. Check network and URL.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Authentication request timed out: {e}. Auth server might be down or overloaded.")
    except requests.exceptions.RequestException as e:
        response_text = getattr(e.response, "text", "(No response text available)")
        logger.error(
            f"Network error during authentication to {auth_url}: {e}. Response: {response_text}", exc_info=True
        )
        raise AuthenticationError(f"Network error connecting to auth endpoint: {e}")
    except AuthenticationError:  # Re-raise specific auth errors
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during authentication: {e}", exc_info=True)
        raise AuthenticationError(f"An unexpected error during authentication: {e}")


# --- AI Platform Client ---
class AIPlatformClient:
    """Client for direct interaction with Thomson Reuters AI Platform via Vertex AI SDK."""

    def __init__(self):
        """Initialize the AI Platform client."""
        self.project_id: Optional[str] = None
        self.location: Optional[str] = None
        self.credentials: Optional[OAuth2Credentials] = None
        self.initialized: bool = False
        logger.info("AIPlatformClient created.")

    def initialize(self):
        """Initialize AI Platform credentials and Vertex AI SDK."""
        if self.initialized:
            logger.debug("AIPlatformClient already initialized.")
            return True
        try:
            logger.info("Initializing AIPlatformClient...")
            self.project_id, self.location, self.credentials = get_gemini_credentials()
            logger.info(
                f"Credentials obtained. Initializing Vertex AI for project '{self.project_id}' in '{self.location}'."
            )
            vertexai.init(project=self.project_id, location=self.location, credentials=self.credentials)
            logger.info("Vertex AI SDK initialized successfully.")
            self.initialized = True
            return True
        except AuthenticationError as e:
            logger.critical(f"CRITICAL: Failed to initialize AI Platform credentials due to AuthenticationError: {e}")
            self.initialized = False  # Ensure state reflects failure
            raise  # Re-raise to prevent server from operating without credentials
        except Exception as e:
            logger.critical(f"CRITICAL: Unexpected error initializing AI Platform credentials: {e}", exc_info=True)
            self.initialized = False  # Ensure state reflects failure
            raise  # Re-raise to prevent server from operating without credentials

    def ensure_initialized(self):
        """Ensure the client is initialized, attempting initialization if not."""
        if not self.initialized:
            logger.warning("AIPlatformClient accessed before initialization. Attempting to initialize now.")
            self.initialize()  # This will raise an exception if it fails

    async def completion(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        system_message: Optional[str] = None,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        # TODO: Add tools parameter here if/when tool mapping is implemented
    ) -> Union[Dict[str, Any], AsyncGenerator[Any, None]]:
        """
        Send a completion request to AI Platform using Vertex AI SDK.
        Handles both streaming and non-streaming requests.
        """
        self.ensure_initialized()  # Make sure we have credentials

        logger.info(f"Received completion request: model={model_name}, stream={stream}")
        logger.debug(
            f"Parameters: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}, top_k={top_k}, stop_sequences={stop_sequences}"
        )

        try:
            # --- Model and Config Setup ---
            generation_config_dict = {}
            if max_tokens is not None:
                generation_config_dict["max_output_tokens"] = max_tokens
            if temperature is not None:
                generation_config_dict["temperature"] = temperature
            if top_p is not None:
                generation_config_dict["top_p"] = top_p
            if top_k is not None:
                generation_config_dict["top_k"] = top_k
            if stop_sequences:
                generation_config_dict["stop_sequences"] = stop_sequences

            # Create the Vertex AI model instance
            # System instruction is handled differently in Gemini
            vertex_model_args = {}
            if system_message:
                vertex_model_args["system_instruction"] = Content(role="system", parts=[Part.from_text(system_message)])
                logger.debug(f"Using system instruction: '{system_message[:100]}...'")

            # TODO: Add 'tools' argument here when tool mapping is implemented
            # vertex_model_args["tools"] = vertex_tools

            model = GenerativeModel(model_name, **vertex_model_args)

            # --- Message Conversion ---
            # Convert Anthropic message format to Vertex AI Content objects
            vertex_messages: List[Content] = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                vertex_role = "user" if role == "user" else "model"  # Gemini uses 'user' and 'model'

                parts = []
                if isinstance(content, str):
                    parts.append(Part.from_text(content))
                elif isinstance(content, list):
                    # Convert complex content blocks to text parts for now
                    # TODO: Enhance this to handle images, tool_calls, etc. if needed
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            item_type = item.get("type")
                            if item_type == "text":
                                text_parts.append(item.get("text", ""))
                            elif item_type == "tool_result":
                                tool_id = item.get("tool_use_id", "unknown_tool_id")
                                result_content = parse_tool_result_content(item.get("content", ""))
                                # Format tool results clearly for the model
                                text_parts.append(
                                    f"[Tool Result for ID: {tool_id}]\nOutput:\n{result_content}\n[/Tool Result]"
                                )
                            elif item_type == "tool_use":  # This case is less likely in input but handle defensively
                                tool_name = item.get("name", "unknown_tool")
                                tool_id = item.get("id", "unknown_tool_id")
                                tool_input_str = json.dumps(item.get("input", {}))
                                text_parts.append(
                                    f"[Tool Call ID: {tool_id}, Name: {tool_name}]\nInput: {tool_input_str}\n[/Tool Call]"
                                )
                            # Add handling for 'image' type if needed
                        elif isinstance(content, str):  # Handle lists containing simple strings
                            text_parts.append(content)

                    full_text = "\n\n".join(text_parts).strip()
                    if full_text:
                        parts.append(Part.from_text(full_text))

                # Only add message if it has content parts
                if parts:
                    vertex_messages.append(Content(role=vertex_role, parts=parts))

            logger.debug(f"Converted messages for Vertex AI: {vertex_messages}")

            # --- API Call ---
            if stream:
                logger.info("Streaming mode: Calling model.generate_content(stream=True) in thread.")
                try:
                    # Run the synchronous SDK call in a thread pool executor
                    stream_iterator = await asyncio.to_thread(
                        model.generate_content,
                        contents=vertex_messages,
                        generation_config=generation_config_dict if generation_config_dict else None,
                        stream=True,
                    )

                    # Adapt the synchronous iterator to an async generator
                    async def async_stream_adapter():
                        logger.debug("Async stream adapter started.")
                        try:
                            for chunk in stream_iterator:
                                yield chunk
                                await asyncio.sleep(0.005)  # Small sleep to prevent blocking event loop entirely
                        except ResponseBlockedError as rbe_inner:
                            logger.error(f"STREAMING Response Blocked during iteration: {rbe_inner}", exc_info=True)
                            # Re-raise to be caught by the outer handler
                            raise rbe_inner
                        except Exception as e_inner:
                            logger.error(f"STREAMING Error during iteration: {e_inner}", exc_info=True)
                            # Re-raise to be caught by the outer handler
                            raise e_inner
                        finally:
                            logger.debug("Async stream adapter finished.")

                    return async_stream_adapter()

                except ResponseBlockedError as rbe:
                    # This might catch blocks that happen *before* streaming starts
                    logger.error(f"STREAMING Response Blocked immediately: {rbe}", exc_info=True)
                    raise HTTPException(
                        status_code=400, detail=f"Response blocked by safety filters (Initial): {rbe.block_reason}"
                    )
                except Exception as e:
                    logger.error(f"Error initializing stream from Vertex AI: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to start streaming from Vertex AI: {str(e)}")
            else:
                # Non-streaming request
                logger.info("Non-streaming mode: Calling model.generate_content(stream=False) in thread.")
                try:
                    response = await asyncio.to_thread(
                        model.generate_content,
                        contents=vertex_messages,
                        generation_config=generation_config_dict if generation_config_dict else None,
                        stream=False,
                    )
                    logger.debug(f"Raw non-streaming Vertex AI response: {response}")

                    # --- Non-Streaming Response Parsing ---
                    response_text = ""
                    tool_calls = []  # For OpenAI compatible format
                    finish_reason_raw = None
                    prompt_tokens = 0
                    completion_tokens = 0

                    # Extract usage metadata
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                        completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                        logger.debug(f"Usage metadata found: prompt={prompt_tokens}, completion={completion_tokens}")

                    if not response.candidates:
                        logger.warning("Non-streaming response has no candidates.")
                        # Handle case where response might be blocked or empty
                        if response.prompt_feedback and response.prompt_feedback.block_reason:
                            raise ResponseBlockedError(
                                f"Prompt blocked: {response.prompt_feedback.block_reason_message}"
                            )
                        # Otherwise return an empty response structure
                        finish_reason_raw = FinishReason.STOP  # Assume normal stop if no block reason
                        response_text = ""
                    else:
                        candidate = response.candidates[0]
                        finish_reason_raw = candidate.finish_reason

                        # Extract content parts (text, function_call, executable_code)
                        if candidate.content and candidate.content.parts:
                            for part in candidate.content.parts:
                                try:
                                    if hasattr(part, "text") and part.text:
                                        response_text += part.text
                                except (ValueError, AttributeError) as e:
                                    logger.debug(f"Could not get part.text: {e}")

                                if hasattr(part, "function_call") and part.function_call:
                                    fc = part.function_call
                                    logger.info(f"Non-streaming: Found native function_call: {fc.name}")
                                    try:
                                        args_str = json.dumps(dict(fc.args))  # Gemini args are dict-like
                                    except Exception:
                                        logger.warning(
                                            f"Could not serialize args for tool {fc.name}, using empty dict.",
                                            exc_info=True,
                                        )
                                        args_str = "{}"
                                    tool_calls.append(
                                        {
                                            "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {"name": fc.name, "arguments": args_str},
                                        }
                                    )

                                # Handle executable_code (less common in non-streaming, but possible)
                                elif hasattr(part, "executable_code") and part.executable_code:
                                    logger.info(
                                        f"Non-streaming: Found executable_code: {part.executable_code.language}"
                                    )
                                    code = part.executable_code.code
                                    logger.debug(f"Executable code content: {code}")
                                    # Attempt to parse tool calls from executable_code (similar to streaming logic)
                                    # This is a simplified version for non-streaming
                                    import re

                                    # Look for common patterns
                                    tool_name_match = re.search(r"(?:print\()?\s*(\w+Tool)\(", code)
                                    if tool_name_match:
                                        tool_name = tool_name_match.group(1)
                                        # Simple arg extraction (might need refinement)
                                        args_match = re.search(r"\((.*)\)", code)
                                        args_str = args_match.group(1) if args_match else ""
                                        tool_calls.append(
                                            {
                                                "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                                "type": "function",
                                                "function": {
                                                    "name": tool_name,
                                                    "arguments": json.dumps(
                                                        {"raw_code_args": args_str}
                                                    ),  # Basic representation
                                                },
                                            }
                                        )
                                        logger.info(
                                            f"Non-streaming: Extracted tool call '{tool_name}' from executable_code."
                                        )

                    # Map finish reason to OpenAI/Anthropic compatible format
                    openai_finish_reason = "stop"  # Default
                    if finish_reason_raw:
                        logger.debug(
                            f"Mapping finish reason (non-streaming): {finish_reason_raw} (type: {type(finish_reason_raw)})"
                        )
                        # Handle enum or int/string representation
                        if (
                            finish_reason_raw == FinishReason.MAX_TOKENS
                            or str(finish_reason_raw) == "MAX_TOKENS"
                            or finish_reason_raw == 3
                        ):
                            openai_finish_reason = "length"
                        elif (
                            finish_reason_raw == FinishReason.SAFETY
                            or str(finish_reason_raw) == "SAFETY"
                            or finish_reason_raw == 4
                        ):
                            openai_finish_reason = "content_filter"
                        elif (
                            finish_reason_raw == FinishReason.RECITATION
                            or str(finish_reason_raw) == "RECITATION"
                            or finish_reason_raw == 5
                        ):
                            openai_finish_reason = "content_filter"  # Treat recitation as a content filter issue
                        elif (
                            finish_reason_raw == FinishReason.STOP
                            or str(finish_reason_raw) == "STOP"
                            or finish_reason_raw == 1
                        ):
                            openai_finish_reason = "stop"
                        # Note: Vertex FUNCTION_CALL finish reason (value 2) is handled below

                    # If tool calls were generated, the finish reason should reflect that
                    if tool_calls:
                        openai_finish_reason = "tool_calls"
                        logger.info("Setting finish_reason to 'tool_calls' due to extracted tool calls.")

                    # Estimate tokens if not provided by the API (basic approximation)
                    if prompt_tokens == 0:
                        prompt_text_combined = system_message or "" + "".join([str(m.parts) for m in vertex_messages])
                        prompt_tokens = len(prompt_text_combined) // 4  # Rough estimate
                        logger.debug(f"Estimating prompt tokens: ~{prompt_tokens}")
                    if completion_tokens == 0:
                        completion_tokens = len(response_text) // 4  # Rough estimate
                        logger.debug(f"Estimating completion tokens: ~{completion_tokens}")

                    # Construct OpenAI-compatible response structure
                    openai_response = {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,  # Use the actual model called
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": (
                                        response_text if response_text else None
                                    ),  # Use None if empty, as per OpenAI spec
                                    "tool_calls": tool_calls if tool_calls else None,  # Use None if no tool calls
                                },
                                "finish_reason": openai_finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                        # Add system_fingerprint if available/needed (Gemini doesn't provide this directly)
                        # "system_fingerprint": None
                    }
                    logger.info("Non-streaming request completed successfully.")
                    return openai_response

                except ResponseBlockedError as rbe:
                    logger.error(f"NON-STREAMING Response Blocked: {rbe}", exc_info=True)
                    # Attempt to get block reason details
                    block_reason = "Unknown"
                    block_message = str(rbe)
                    if hasattr(rbe, "block_reason"):
                        block_reason = str(rbe.block_reason)
                    if hasattr(rbe, "block_reason_message"):
                        block_message = rbe.block_reason_message
                    raise HTTPException(
                        status_code=400,
                        detail=f"Response blocked by safety filters. Reason: {block_reason}. Message: {block_message}",
                    )
                except Exception as e:
                    logger.error(f"Error during non-streaming Vertex AI completion: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Vertex AI completion error: {str(e)}")

        except Exception as e:
            # Catch-all for unexpected errors during setup or dispatch
            logger.error(f"Unexpected error in AIPlatformClient.completion: {e}", exc_info=True)
            # Determine status code if possible, default to 500
            status_code = getattr(e, "status_code", 500)
            detail = getattr(e, "detail", str(e))
            raise HTTPException(status_code=status_code, detail=f"Internal client error: {detail}")


# --- Singleton Client Instance ---
aiplatform_client = AIPlatformClient()


# --- Pydantic Models for Anthropic API Compatibility ---


# Content Block Models
class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImageSource(BaseModel):
    type: Literal["base64"]
    media_type: str  # e.g., "image/jpeg", "image/png"
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
    content: Union[str, List[Dict[str, Any]]]  # Anthropic allows string or list of text blocks
    is_error: Optional[bool] = False  # Optional field


# Allow any valid content block type
ContentBlock = Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]


class SystemContent(BaseModel):  # Anthropic allows list of content blocks for system
    type: Literal["text"]  # For now, only support text system prompts
    text: str


# Message Models
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[ContentBlock]]  # Content can be simple string or list of blocks


# Tool Models
class ToolInputSchema(BaseModel):
    type: Literal["object"] = "object"
    properties: Dict[str, Any]
    required: Optional[List[str]] = None


class ToolDefinition(BaseModel):  # Renamed from Tool to avoid conflict with Vertex AI Tool
    name: str
    description: Optional[str] = None
    input_schema: ToolInputSchema


# Request Models
class MessagesRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None  # Allow string or list
    max_tokens: int
    metadata: Optional[Dict[str, Any]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None  # Default is often 1.0 in Anthropic
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Dict[str, Any]] = None  # e.g., {"type": "any"}, {"type": "tool", "name": "..."}

    # Internal field to store original model name after mapping
    original_model_name: Optional[str] = None

    @field_validator("model")
    def validate_and_map_model(cls, v, info):
        """Validates and maps the model name, storing the original."""
        original_name = v
        # Store original name in the instance's __dict__ directly during validation
        # We can't easily access 'values' dict like in older Pydantic versions
        # We'll handle storing it in the main endpoint logic instead.
        # This validator just performs the mapping.
        return map_model_name(v)  # Pass only the model name


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None

    # Internal field
    original_model_name: Optional[str] = None

    @field_validator("model")
    def validate_and_map_model_token_count(cls, v, info):
        """Validates and maps the model name for token counting."""
        # Similar to MessagesRequest, store original name in endpoint logic
        return map_model_name(v)


# Response Models (Anthropic Format)
class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    # Anthropic includes cache tokens, Gemini doesn't expose this
    # cache_creation_input_tokens: Optional[int] = 0
    # cache_read_input_tokens: Optional[int] = 0


class MessagesResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str  # Should be the *original* requested model name
    content: List[ContentBlock]  # Now uses the ContentBlock union
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# --- Streaming Handler ---
async def handle_vertex_streaming(response_generator: AsyncGenerator, original_request: MessagesRequest):
    """
    Processes the async generator from Vertex AI and yields Anthropic-compatible SSE events.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    # Use the original model name provided by the client for the response events
    response_model_name = original_request.original_model_name or original_request.model
    request_id = f"req_{uuid.uuid4().hex[:12]}"  # Unique ID for this stream processing
    logger.info(f"[{request_id}] Starting Anthropic SSE stream for message {message_id} (model: {response_model_name})")

    # Initial usage estimate (will be updated)
    input_tokens = 0  # We don't get this reliably per chunk from Vertex stream
    output_tokens = 0

    # State tracking for content blocks
    current_content_block_index = -1
    current_block_type = None  # 'text' or 'tool_use'
    current_tool_id = None
    current_tool_name = None
    accumulated_tool_args_json = ""  # Store the *full* JSON string for comparison
    text_block_started = False
    tool_blocks_generated = []  # Keep track of generated tool_use blocks
    accumulated_full_text = ""  # Accumulate all text for final token count estimate

    try:
        # 1. Send message_start event
        message_start_payload = {
            "type": "message_start",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": response_model_name,  # Use original model name
                "content": [],  # Content starts empty
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},  # Placeholder
            },
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_payload)}\n\n"
        logger.debug(f"[{request_id}] Sent message_start")

        # 2. Send initial ping event
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        logger.debug(f"[{request_id}] Sent ping")

        # 3. Start processing the stream from Vertex AI generator
        async for chunk in response_generator:
            logger.debug(f"[{request_id}] Raw Vertex Chunk: {chunk}")

            # --- Variables for current chunk ---
            chunk_text = ""
            chunk_function_call = None  # Native Vertex FunctionCall object
            chunk_finish_reason_raw = None
            chunk_safety_ratings = None
            is_last_chunk = False  # Assume not the last chunk initially
            chunk_usage_metadata = None

            try:
                # --- Start: Chunk Parsing Logic ---
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    chunk_usage_metadata = chunk.usage_metadata
                    # Update output tokens if available (often only in the last chunk)
                    if hasattr(chunk_usage_metadata, "candidates_token_count"):
                        output_tokens = chunk_usage_metadata.candidates_token_count
                    if hasattr(chunk_usage_metadata, "prompt_token_count"):  # Less common in stream
                        input_tokens = chunk_usage_metadata.prompt_token_count
                    logger.debug(f"[{request_id}] Chunk Usage: input={input_tokens}, output={output_tokens}")

                # Check for blocked response within the chunk
                if hasattr(chunk, "prompt_feedback") and chunk.prompt_feedback.block_reason:
                    logger.warning(
                        f"[{request_id}] Stream chunk indicates prompt blocked: {chunk.prompt_feedback.block_reason_message}"
                    )
                    # We'll handle this by setting stop_reason later
                    chunk_finish_reason_raw = FinishReason.SAFETY  # Treat as safety block
                    is_last_chunk = True  # End the stream

                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]

                    # Check safety ratings (can indicate content filtering)
                    if hasattr(candidate, "safety_ratings") and candidate.safety_ratings:
                        chunk_safety_ratings = candidate.safety_ratings
                        # Check if any rating indicates blocking
                        for rating in chunk_safety_ratings:
                            if rating.blocked:
                                logger.warning(
                                    f"[{request_id}] Stream chunk blocked due to safety rating: Category={rating.category}, Probability={rating.probability_score}"
                                )
                                chunk_finish_reason_raw = FinishReason.SAFETY
                                is_last_chunk = True
                                break  # Stop checking ratings

                    # Get finish reason if present (usually in the last chunk)
                    if (
                        hasattr(candidate, "finish_reason")
                        and candidate.finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED
                    ):
                        chunk_finish_reason_raw = candidate.finish_reason
                        logger.debug(f"[{request_id}] Finish reason found in chunk: {chunk_finish_reason_raw}")
                        is_last_chunk = True  # A finish reason means it's the last chunk

                    # Extract content parts if not blocked
                    if not is_last_chunk or chunk_finish_reason_raw != FinishReason.SAFETY:
                        if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                            for part in candidate.content.parts:
                                # Prioritize native function_call
                                if hasattr(part, "function_call") and part.function_call:
                                    chunk_function_call = part.function_call
                                    logger.info(
                                        f"[{request_id}] Found native function_call in chunk: {chunk_function_call.name}"
                                    )
                                    chunk_text = ""  # Clear any text if we have a function call
                                    break  # Process only the function call from this part list

                                # Check for executable_code (alternative tool format)
                                elif hasattr(part, "executable_code") and part.executable_code:
                                    logger.info(
                                        f"[{request_id}] Found executable_code in chunk: Lang={part.executable_code.language}"
                                    )
                                    code = part.executable_code.code
                                    logger.debug(f"[{request_id}] Executable code content: {code}")
                                    # Attempt to parse tool call from code (using simplified logic)
                                    import re

                                    # Example: Look for ReadFileTool().run(...) or similar
                                    tool_match = re.search(
                                        r"(?:print\()?\s*(\w+Tool)\(.*\)\.run\((.*)\)", code, re.IGNORECASE | re.DOTALL
                                    )
                                    if tool_match:
                                        tool_name = tool_match.group(1)
                                        args_str = tool_match.group(2)
                                        try:
                                            # Attempt a very basic parse of args like paths=['...']
                                            paths_match = re.search(r"paths=\[(.*?)\]", args_str)
                                            if paths_match:
                                                path_list_str = paths_match.group(1)
                                                paths = re.findall(r"'(.*?)'|\"(.*?)\"", path_list_str)
                                                parsed_args = {"paths": [p[0] or p[1] for p in paths]}
                                            else:
                                                parsed_args = {"raw_code_args": args_str}
                                        except Exception:
                                            parsed_args = {"raw_code_args": args_str}  # Fallback

                                        # Create a synthetic FunctionCall object
                                        class SyntheticFunctionCall:
                                            def __init__(self, name, args):
                                                self.name = name
                                                self.args = args  # Store parsed args dict

                                        chunk_function_call = SyntheticFunctionCall(tool_name, parsed_args)
                                        logger.info(
                                            f"[{request_id}] Extracted synthetic tool call from executable_code: {tool_name} with args {parsed_args}"
                                        )
                                        chunk_text = ""  # Clear text
                                        break  # Prioritize this extracted call
                                    else:
                                        logger.warning(
                                            f"[{request_id}] executable_code found but no tool pattern matched. Treating as text."
                                        )
                                        # Fall through to treat as text if parsing fails

                                # Accumulate text if no function call found yet in this part
                                if not chunk_function_call and hasattr(part, "text"):
                                    try:
                                        part_text = part.text
                                        if part_text:
                                            chunk_text += part_text
                                            logger.debug(f"[{request_id}] Found text in part: '{part_text[:100]}...'")
                                    except (ValueError, AttributeError) as e:
                                        logger.debug(f"[{request_id}] Could not get part.text: {e}")

                # --- End: Chunk Parsing Logic ---

                # --- Start: Event Generation Logic ---

                # A. Process Function Calls (Native or Extracted)
                if chunk_function_call:
                    logger.debug(f"[{request_id}] Processing function call: {chunk_function_call.name}")
                    # Close any open text block *before* starting tool block
                    if current_block_type == "text" and text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent content_block_stop for text block {current_content_block_index}"
                        )
                        text_block_started = False
                        current_block_type = None

                    # Start a *new* tool_use block for this call
                    # (Gemini generally sends the whole call at once in streaming)
                    current_content_block_index += 1
                    current_block_type = "tool_use"
                    current_tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                    current_tool_name = chunk_function_call.name
                    # Gemini args are dict-like, convert to dict for JSON serialization
                    current_tool_input_dict = {}
                    if hasattr(chunk_function_call, "args"):
                        try:
                            # Handle potential Struct/dict variations in args
                            current_tool_input_dict = dict(chunk_function_call.args)
                        except Exception as e:
                            logger.warning(
                                f"[{request_id}] Could not convert tool args to dict for {current_tool_name}: {e}. Using empty input."
                            )
                            current_tool_input_dict = {}

                    tool_block_start_payload = {
                        "type": "content_block_start",
                        "index": current_content_block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": current_tool_id,
                            "name": current_tool_name,
                            "input": current_tool_input_dict,  # Anthropic expects the full input dict here
                        },
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(tool_block_start_payload)}\n\n"
                    logger.debug(
                        f"[{request_id}] Sent content_block_start for tool_use block {current_content_block_index} ({current_tool_name})"
                    )

                    # Anthropic's streaming format for tool_use doesn't use input_json_delta.
                    # The input is sent entirely within the content_block_start event.
                    # We just need to mark that we generated this tool block.
                    tool_blocks_generated.append(
                        {"id": current_tool_id, "name": current_tool_name, "index": current_content_block_index}
                    )

                    # Immediately close the tool_use block as Gemini sends it whole
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                    logger.debug(
                        f"[{request_id}] Sent content_block_stop for tool_use block {current_content_block_index}"
                    )
                    current_block_type = None  # Reset block type after closing

                # B. Process Text Content (if no function call in *this* chunk)
                elif chunk_text:
                    logger.debug(f"[{request_id}] Processing text content: '{chunk_text[:100]}...'")
                    # --- FIX IMPLEMENTATION ---
                    tool_call_extracted_from_text = False
                    looks_like_tool_code_attempt = False

                    # Check if text contains patterns indicating a potential tool call
                    tool_patterns = ["```tool_code", "print(ReadFileTool", "ReadFileTool().run", "print(BatchTool"]
                    if any(pattern in chunk_text for pattern in tool_patterns):
                        looks_like_tool_code_attempt = True
                        logger.info(f"[{request_id}] Potential tool code pattern detected in text chunk.")

                        # Try to extract tool call *if* it looks like a complete one in this chunk
                        import re

                        # Refined regex looking for a complete ```tool_code ... ``` block
                        tool_code_match = re.search(r"```tool_code\s+(.*?)\s*```", chunk_text, re.DOTALL)
                        if tool_code_match:
                            # ... [Your existing/refined extraction logic for backtick format] ...
                            # If extraction is successful:
                            #   chunk_function_call = SyntheticFunctionCall(...)
                            #   tool_call_extracted_from_text = True
                            #   logger.info(f"[{request_id}] Extracted tool call from text (backtick format).")
                            pass  # Placeholder for extraction logic

                        # Try direct print pattern if backticks didn't match or weren't present
                        elif not tool_call_extracted_from_text:
                            # Refined regex for print(...) pattern
                            direct_match = re.search(
                                r"print\(\s*(\w+Tool)\(.*\)\.run\(.*\)\)", chunk_text, re.IGNORECASE | re.DOTALL
                            )
                            if direct_match:
                                # ... [Your existing/refined extraction logic for print format] ...
                                # If extraction is successful:
                                #   chunk_function_call = SyntheticFunctionCall(...)
                                #   tool_call_extracted_from_text = True
                                #   logger.info(f"[{request_id}] Extracted tool call from text (print format).")
                                pass  # Placeholder for extraction logic

                    # --- Decision Logic ---
                    if tool_call_extracted_from_text and chunk_function_call:
                        # If we successfully extracted a *complete* tool call from the text
                        logger.info(
                            f"[{request_id}] Handling tool call extracted from text: {chunk_function_call.name}"
                        )
                        # Close any open text block
                        if current_block_type == "text" and text_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                            logger.debug(
                                f"[{request_id}] Sent content_block_stop for text block {current_content_block_index}"
                            )
                            text_block_started = False
                            current_block_type = None

                        # Start and immediately stop the tool_use block (as done for native calls)
                        current_content_block_index += 1
                        current_block_type = "tool_use"
                        current_tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        current_tool_name = chunk_function_call.name
                        current_tool_input_dict = dict(
                            chunk_function_call.args
                        )  # Assuming synthetic call has .args dict

                        tool_block_start_payload = {...}  # Same as above
                        yield f"event: content_block_start\ndata: {json.dumps(tool_block_start_payload)}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent content_block_start for extracted tool {current_content_block_index}"
                        )

                        tool_blocks_generated.append({...})  # Track it

                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent content_block_stop for extracted tool {current_content_block_index}"
                        )
                        current_block_type = None

                    elif looks_like_tool_code_attempt and not tool_call_extracted_from_text:
                        # If it looked like tool code but we couldn't extract a complete call
                        logger.warning(
                            f"[{request_id}] Suppressing text delta: Detected potential partial tool code but failed to extract complete call. Chunk: '{chunk_text[:100]}...'"
                        )
                        # *** DO NOTHING - Suppress this chunk ***
                        pass

                    else:
                        # If it's just normal text (no tool patterns detected)
                        logger.debug(f"[{request_id}] Processing as normal text delta.")
                        # Close any open tool block *before* starting text
                        if current_block_type == "tool_use":
                            # This case is less likely now, but handle defensively
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                            logger.debug(
                                f"[{request_id}] Sent content_block_stop for tool block {current_content_block_index} before text"
                            )
                            current_block_type = None

                        # Ensure a text block is active
                        if not text_block_started:
                            current_content_block_index += 1
                            current_block_type = "text"
                            text_block_start_payload = {
                                "type": "content_block_start",
                                "index": current_content_block_index,
                                "content_block": {"type": "text", "text": ""},  # Start with empty text
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(text_block_start_payload)}\n\n"
                            logger.debug(
                                f"[{request_id}] Sent content_block_start for text block {current_content_block_index}"
                            )
                            text_block_started = True

                        # Send the text delta
                        accumulated_full_text += chunk_text  # Accumulate for token estimation
                        text_delta_payload = {
                            "type": "content_block_delta",
                            "index": current_content_block_index,
                            "delta": {"type": "text_delta", "text": chunk_text},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(text_delta_payload)}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent text_delta for block {current_content_block_index}: '{chunk_text[:100]}...'"
                        )

                # C. Process Finish Reason / End of Stream
                if is_last_chunk:
                    logger.info(f"[{request_id}] Processing last chunk. Finish Reason Raw: {chunk_finish_reason_raw}")
                    # Close the last open content block (could be text)
                    if current_block_type == "text" and text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent content_block_stop for final text block {current_content_block_index}"
                        )
                    # Tool blocks are stopped immediately after starting in this logic

                    # Map Vertex finish reason to Anthropic stop_reason
                    anthropic_stop_reason = "end_turn"  # Default
                    stop_sequence = None  # Usually None unless stop_sequences matched

                    if chunk_finish_reason_raw:
                        logger.debug(f"[{request_id}] Mapping final finish reason: {chunk_finish_reason_raw}")
                        if chunk_finish_reason_raw == FinishReason.MAX_TOKENS:
                            anthropic_stop_reason = "max_tokens"
                        elif chunk_finish_reason_raw == FinishReason.STOP:
                            anthropic_stop_reason = "end_turn"
                        elif chunk_finish_reason_raw == FinishReason.SAFETY:
                            anthropic_stop_reason = "content_filtered"  # Use Anthropic's specific reason
                        elif chunk_finish_reason_raw == FinishReason.RECITATION:
                            anthropic_stop_reason = "content_filtered"  # Treat recitation as filtered
                        elif chunk_finish_reason_raw == FinishReason.OTHER:
                            anthropic_stop_reason = "end_turn"  # Map 'OTHER' to default end_turn
                        # Note: FinishReason.FUNCTION_CALL (value 2) from Vertex implies a tool call *was* generated.
                        # We rely on tool_blocks_generated list instead of the reason itself.

                    # If any tool_use blocks were generated during the stream, override stop_reason
                    if tool_blocks_generated:
                        anthropic_stop_reason = "tool_use"
                        logger.info(
                            f"[{request_id}] Overriding stop_reason to 'tool_use' because tool blocks were generated."
                        )

                    # Final usage update (use accumulated estimate if API didn't provide)
                    if output_tokens == 0:
                        output_tokens = len(accumulated_full_text) // 4  # Estimate from accumulated text
                        logger.debug(f"[{request_id}] Estimating final output tokens: ~{output_tokens}")

                    # Send final message_delta with stop reason and usage
                    message_delta_payload = {
                        "type": "message_delta",
                        "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": stop_sequence},
                        "usage": {"output_tokens": output_tokens},  # Only output tokens in delta
                    }
                    yield f"event: message_delta\ndata: {json.dumps(message_delta_payload)}\n\n"
                    logger.debug(f"[{request_id}] Sent final message_delta with stop_reason: {anthropic_stop_reason}")

                    # Send message_stop event
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                    logger.debug(f"[{request_id}] Sent message_stop")

                    # Although not standard SSE, some clients look for this
                    # yield "data: [DONE]\n\n"
                    logger.info(f"[{request_id}] Anthropic SSE stream completed.")
                    return  # Exit the generator loop

            except Exception as chunk_error:
                logger.error(f"[{request_id}] Error processing chunk: {chunk_error}", exc_info=True)
                # Attempt to gracefully end the stream with an error state if possible
                try:
                    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}})}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                except Exception:
                    pass  # Ignore errors during error reporting
                logger.warning(f"[{request_id}] Stream terminated due to chunk processing error.")
                return  # Stop processing stream

        # Fallback if the loop finishes without is_last_chunk being set (shouldn't happen with Vertex)
        logger.warning(f"[{request_id}] Stream generator finished unexpectedly without a final chunk/finish reason.")
        if current_block_type == "text" and text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
        # Estimate tokens
        if output_tokens == 0:
            output_tokens = len(accumulated_full_text) // 4
        # Send closing events
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        logger.info(f"[{request_id}] Anthropic SSE stream completed (fallback exit).")

    except ResponseBlockedError as rbe:
        logger.error(f"[{request_id}] STREAMING Response Blocked (outer handler): {rbe}", exc_info=True)
        try:
            # Try to send Anthropic error format if possible
            error_payload = {
                "type": "error",
                "error": {
                    "type": "overloaded_error",  # Or map based on rbe.block_reason?
                    "message": f"Response blocked by upstream safety filters: {rbe.block_reason_message or rbe.block_reason or 'Unknown reason'}",
                },
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            # Also send message_stop to formally end
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except Exception as e_report:
            logger.error(f"[{request_id}] Failed to send error event to client: {e_report}")
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error during streaming: {e}", exc_info=True)
        try:
            # Generic error event
            error_payload = {
                "type": "error",
                "error": {"type": "internal_server_error", "message": f"Streaming error: {str(e)}"},
            }
            yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        except Exception as e_report:
            logger.error(f"[{request_id}] Failed to send generic error event to client: {e_report}")


# --- Helper Functions ---


def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini Tools."""
    if isinstance(schema, dict):
        # Remove keys unsupported by Gemini FunctionDeclaration parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        # Gemini might support 'enum' within string, but complex validation isn't needed here yet.
        # schema.pop("pattern", None) # Keep pattern? Check Gemini docs.

        # Recursively clean nested schemas
        for key, value in list(schema.items()):
            if key in ["properties", "items"] or isinstance(value, (dict, list)):
                schema[key] = clean_gemini_schema(value)
            # Remove null values potentially left by recursive calls, if necessary
            # if schema[key] is None:
            #    del schema[key]
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema


def convert_anthropic_tools_to_gemini(anthropic_tools: List[ToolDefinition]) -> Optional[List[Tool]]:
    """Converts Anthropic tool definitions to Gemini Tool objects."""
    if not anthropic_tools:
        return None

    gemini_tools = []
    for tool_def in anthropic_tools:
        try:
            # Clean the schema first
            cleaned_schema = clean_gemini_schema(tool_def.input_schema.dict())  # Convert Pydantic model to dict

            # Ensure 'properties' exists, even if empty, as Gemini expects it
            if "properties" not in cleaned_schema:
                cleaned_schema["properties"] = {}

            func_decl = FunctionDeclaration(
                name=tool_def.name,
                description=tool_def.description or "",  # Gemini requires description
                parameters=cleaned_schema,  # Pass the cleaned schema dict
            )
            gemini_tools.append(Tool(function_declarations=[func_decl]))
            logger.debug(f"Converted tool '{tool_def.name}' to Gemini FunctionDeclaration.")
        except Exception as e:
            logger.error(f"Failed to convert tool '{tool_def.name}' to Gemini format: {e}", exc_info=True)
            # Decide whether to skip the tool or raise an error
            # Skipping for now to allow other tools to work
            continue

    return gemini_tools if gemini_tools else None


def parse_tool_result_content(content: Any) -> str:
    """
    Helper function to parse tool result content (which can be complex) into a string
    suitable for Gemini's 'model' role message parts.
    """
    if content is None:
        return "Tool returned no content."
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Anthropic tool results can be lists of text blocks
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif isinstance(item, str):  # Handle lists of strings too
                text_parts.append(item)
            else:
                # Try to serialize other list items as JSON
                try:
                    text_parts.append(json.dumps(item))
                except Exception:
                    text_parts.append(str(item))  # Fallback to string representation
        return "\n".join(text_parts)
    if isinstance(content, dict):
        # Try to serialize dicts as JSON
        try:
            return json.dumps(content)
        except Exception:
            return str(content)  # Fallback
    # Fallback for other types
    try:
        return str(content)
    except Exception:
        return "[Unparseable tool result content]"


def map_model_name(model_name: str) -> str:
    """Maps common Claude model names to specific AI Platform Gemini models."""
    original_model = model_name
    new_model = model_name  # Default if no mapping applies

    logger.debug(f"Attempting to map model name: '{original_model}'")

    # Normalize: lowercase and remove potential provider prefixes
    clean_model_name = model_name.lower()
    if clean_model_name.startswith("anthropic/"):
        clean_model_name = clean_model_name[10:]
    elif clean_model_name.startswith("gemini/"):
        clean_model_name = clean_model_name[7:]
    elif clean_model_name.startswith("aiplatform/"):
        clean_model_name = clean_model_name[11:]
    # Keep only the core name part (e.g., "claude-3-haiku-...")
    clean_model_name = clean_model_name.split("@")[0]  # Remove version specifiers if any

    # --- Mapping Logic ---
    mapped = False
    if "haiku" in clean_model_name:
        new_model = AIPLATFORM_MODELS[1]  # gemini-2.0-flash
        mapped = True
        logger.info(f"Mapping '{original_model}' (Haiku) to '{new_model}'")
    elif "sonnet" in clean_model_name or "opus" in clean_model_name:
        new_model = AIPLATFORM_MODELS[0]  # gemini-2.5-pro-preview...
        mapped = True
        logger.info(f"Mapping '{original_model}' (Sonnet/Opus) to '{new_model}'")
    elif clean_model_name in AIPLATFORM_MODELS:
        # If the user specified a supported Gemini model directly (without prefix)
        new_model = clean_model_name
        mapped = True
        logger.info(f"Using directly specified supported model: '{new_model}'")
    elif original_model in AIPLATFORM_MODELS:
        # If the user specified a supported Gemini model directly (without prefix)
        # and it wasn't caught by lowercase check
        new_model = original_model
        mapped = True
        logger.info(f"Using directly specified supported model: '{new_model}'")

    else:
        # Default mapping for unrecognized models
        new_model = AIPLATFORM_MODELS[0]  # Default to the most capable model
        mapped = True
        logger.warning(f"Unrecognized model name: '{original_model}'. Defaulting to '{new_model}'.")

    # Return the *Gemini* model name (without any prefix)
    return new_model


def log_request_beautifully(method, path, original_model, mapped_model, num_messages, num_tools, status_code):
    """Log requests in a colorized format."""
    try:
        # Format the original (Claude) model name
        claude_display = f"{Colors.CYAN}{original_model}{Colors.RESET}"

        # Extract endpoint
        endpoint = path.split("?")[0]

        # Format the mapped (Gemini) model name
        gemini_display = f"{Colors.GREEN}{mapped_model}{Colors.RESET}"

        # Format tools and messages
        tools_str = (
            f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
            if num_tools > 0
            else f"{Colors.DIM}{num_tools} tools{Colors.RESET}"
        )
        messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

        # Format status code
        status_color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
        status_symbol = "✓" if 200 <= status_code < 300 else "✗"
        status_str = f"{status_color}{status_symbol} {status_code}{Colors.RESET}"

        log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
        model_line = f"  {claude_display} → {gemini_display} ({messages_str}, {tools_str})"

        print(log_line)
        print(model_line)
        sys.stdout.flush()  # Ensure logs appear immediately
    except Exception as e:
        logger.error(f"Error during beautiful logging: {e}")
        # Fallback to simple logging
        print(
            f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools"
        )


# --- Middleware ---
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    """Middleware to log request basics and timing."""
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Incoming request: {request.method} {request.url.path} from {client_host}")
    logger.debug(f"Request Headers: {dict(request.headers)}")

    response = await call_next(request)

    process_time = time.time() - start_time
    response_log_detail = f"status={response.status_code}, time={process_time:.3f}s"

    # Add mapped model info to the response log if available (set by endpoint)
    mapped_model = getattr(response, "X-Mapped-Model", None)
    if mapped_model:
        response_log_detail += f", mapped_model={mapped_model}"

    logger.info(f"Response completed: {request.method} {request.url.path} ({response_log_detail})")
    logger.debug(f"Response Headers: {dict(response.headers)}")

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: Status={exc.status_code}, Detail={exc.detail}")
    # Log traceback for server errors
    if exc.status_code >= 500:
        logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"type": "api_error", "message": exc.detail}},  # Basic error structure
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.critical(f"Unhandled Exception: {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_server_error", "message": "An unexpected error occurred."}},
    )


# --- API Endpoints ---


@app.post("/v1/messages", response_model=None)  # response_model=None allows StreamingResponse or MessagesResponse
async def create_message(request_data: MessagesRequest, raw_request: Request):
    """
    Handles Anthropic's /v1/messages endpoint for completions.
    Supports both streaming and non-streaming.
    """
    try:
        # Ensure client is initialized (will raise if auth fails)
        aiplatform_client.ensure_initialized()

        # --- Request Processing ---
        # Get original model name *before* Pydantic validation/mapping runs
        try:
            raw_body = await raw_request.body()
            raw_body_json = json.loads(raw_body.decode("utf-8"))
            original_model_name = raw_body_json.get("model", "unknown_original")
        except Exception:
            logger.warning("Could not parse raw request body to get original model name.")
            original_model_name = request_data.model  # Fallback, might be mapped already

        # Store the original name in the validated request object for later use
        request_data.original_model_name = original_model_name

        # Get the *mapped* Gemini model name from the validated request
        gemini_model_name = request_data.model  # This has been mapped by the validator

        client_ip = raw_request.client.host if raw_request.client else "unknown"
        logger.info(
            f"Processing '/v1/messages': OriginalModel='{original_model_name}', MappedModel='{gemini_model_name}', Stream={request_data.stream}, Client={client_ip}"
        )

        # Process system message
        system_message_text: Optional[str] = None
        if request_data.system:
            if isinstance(request_data.system, str):
                system_message_text = request_data.system
            elif isinstance(request_data.system, list):
                # Concatenate text from system content blocks
                system_message_text = "\n".join(
                    [
                        block.text
                        for block in request_data.system
                        if isinstance(block, SystemContent) and block.type == "text"
                    ]
                )
        logger.debug(f"Processed system message: '{system_message_text[:100]}...'")

        # Convert messages to format expected by AIPlatformClient.completion (simple dict list)
        client_messages: List[Dict[str, Any]] = []
        for msg in request_data.messages:
            # Handle content based on type (string or list of blocks)
            processed_content: Any
            if isinstance(msg.content, str):
                processed_content = msg.content
            else:  # List of content blocks
                content_parts = []
                for block in msg.content:
                    if block.type == "text":
                        content_parts.append(block.text)
                    elif block.type == "tool_result":
                        # Format tool results clearly for the Gemini model
                        result_str = parse_tool_result_content(block.content)
                        status = "ERROR" if getattr(block, "is_error", False) else "OK"
                        content_parts.append(
                            f"[Tool Result ({status}) for ID: {block.tool_use_id}]\n{result_str}\n[/Tool Result]"
                        )
                    elif block.type == "tool_use":
                        # Include tool use info if present in input (less common)
                        input_str = json.dumps(block.input)
                        content_parts.append(
                            f"[Tool Call ID: {block.id}, Name: {block.name}]\nInput: {input_str}\n[/Tool Call]"
                        )
                    elif block.type == "image":
                        # Basic placeholder - image data isn't passed to Gemini text model here
                        content_parts.append(f"[Image Content: {block.source.media_type}]")
                processed_content = "\n\n".join(content_parts)

            client_messages.append({"role": msg.role, "content": processed_content})

        logger.debug(f"Messages prepared for AIPlatformClient: {client_messages}")

        # Convert Anthropic tools to Gemini format (if provided)
        # TODO: This is where tool conversion needs to happen
        # gemini_tools = convert_anthropic_tools_to_gemini(request_data.tools)
        # if gemini_tools:
        #     logger.info(f"Converted {len(gemini_tools)} tools for Gemini.")

        # Log the request details beautifully
        num_tools = len(request_data.tools) if request_data.tools else 0
        log_request_beautifully(
            raw_request.method,
            raw_request.url.path,
            original_model_name,
            gemini_model_name,
            len(client_messages),
            num_tools,
            200,  # Log 200 initially, middleware logs final status
        )

        # --- Call AI Platform Client ---
        completion_result = await aiplatform_client.completion(
            model_name=gemini_model_name,
            messages=client_messages,
            system_message=system_message_text,
            stream=request_data.stream or False,
            max_tokens=request_data.max_tokens,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
            top_k=request_data.top_k,
            stop_sequences=request_data.stop_sequences,
            # tools=gemini_tools # Pass converted tools here
        )

        # --- Handle Response ---
        if request_data.stream:
            logger.info("Streaming response requested. Returning StreamingResponse.")
            if not isinstance(completion_result, AsyncGenerator):
                logger.error("Expected AsyncGenerator for streaming response, but got different type.")
                raise HTTPException(status_code=500, detail="Internal server error: Streaming setup failed.")

            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Useful for Nginx proxies
                "X-Mapped-Model": gemini_model_name,  # Custom header for logging
            }
            return StreamingResponse(
                handle_vertex_streaming(completion_result, request_data),
                media_type="text/event-stream",
                headers=headers,
            )
        else:
            # Non-streaming: Convert OpenAI-compatible response to Anthropic format
            logger.info("Non-streaming response requested. Converting to Anthropic format.")
            if not isinstance(completion_result, dict):
                logger.error("Expected dict for non-streaming response, but got different type.")
                raise HTTPException(status_code=500, detail="Internal server error: Completion processing failed.")

            # Extract info from the OpenAI-like structure returned by completion()
            response_id = completion_result.get("id", f"msg_{uuid.uuid4().hex[:24]}")
            choices = completion_result.get("choices", [])
            usage_info = completion_result.get("usage", {})

            anthropic_content: List[ContentBlock] = []
            anthropic_stop_reason: Optional[
                Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]
            ] = None
            final_message = {}
            finish_reason_openai = "stop"  # Default

            if choices:
                choice = choices[0]
                final_message = choice.get("message", {})
                finish_reason_openai = choice.get("finish_reason", "stop")

                # Extract text content
                text_content = final_message.get("content")
                if text_content:
                    anthropic_content.append(ContentBlockText(type="text", text=text_content))

                # Extract tool calls
                tool_calls = final_message.get("tool_calls")
                if tool_calls:
                    for tool_call in tool_calls:
                        if tool_call.get("type") == "function":
                            function_info = tool_call.get("function", {})
                            tool_name = function_info.get("name", "unknown_tool")
                            tool_id = tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                            try:
                                # Arguments should be a JSON string from our completion method
                                args_str = function_info.get("arguments", "{}")
                                tool_input = json.loads(args_str)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode tool arguments for {tool_name}: {args_str}")
                                tool_input = {"error": "Failed to parse arguments", "raw_arguments": args_str}

                            anthropic_content.append(
                                ContentBlockToolUse(type="tool_use", id=tool_id, name=tool_name, input=tool_input)
                            )

            # Ensure there's at least one content block if the response was empty
            if not anthropic_content:
                anthropic_content.append(ContentBlockText(type="text", text=""))

            # Map OpenAI finish_reason to Anthropic stop_reason
            if finish_reason_openai == "stop":
                anthropic_stop_reason = "end_turn"
            elif finish_reason_openai == "length":
                anthropic_stop_reason = "max_tokens"
            elif finish_reason_openai == "content_filter":
                anthropic_stop_reason = "content_filtered"
            elif finish_reason_openai == "tool_calls":
                anthropic_stop_reason = "tool_use"
            # 'stop_sequence' needs specific handling if stop sequences matched

            # Create final Anthropic response object
            anthropic_response = MessagesResponse(
                id=response_id,
                model=original_model_name,  # Use the original requested model name
                role="assistant",
                content=anthropic_content,
                stop_reason=anthropic_stop_reason,
                stop_sequence=None,  # TODO: Populate if stop sequence matched
                usage=Usage(
                    input_tokens=usage_info.get("prompt_tokens", 0),
                    output_tokens=usage_info.get("completion_tokens", 0),
                ),
            )
            logger.debug(f"Final Anthropic non-streaming response: {anthropic_response.dict()}")

            # Add custom header for logging middleware
            response = JSONResponse(content=anthropic_response.dict())
            response.headers["X-Mapped-Model"] = gemini_model_name
            return response

    except HTTPException:
        raise  # Re-raise HTTPExceptions directly
    except AuthenticationError as e:
        logger.error(f"Authentication failed during request processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"Authentication Error: {e}"
        )  # 500 or 401? 500 suggests proxy config issue.
    except Exception as e:
        logger.error(f"Error processing '/v1/messages' request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens(request_data: TokenCountRequest, raw_request: Request):
    """
    Estimates token count for a given prompt based on character count.
    Note: This is a rough approximation. Vertex AI has a specific count_tokens API.
    """
    try:
        # Get original model name
        try:
            raw_body = await raw_request.body()
            raw_body_json = json.loads(raw_body.decode("utf-8"))
            original_model_name = raw_body_json.get("model", "unknown_original")
        except Exception:
            original_model_name = request_data.model  # Fallback

        request_data.original_model_name = original_model_name
        gemini_model_name = request_data.model  # Mapped name

        logger.info(
            f"Processing '/v1/messages/count_tokens': OriginalModel='{original_model_name}', MappedModel='{gemini_model_name}'"
        )

        # --- Basic Text Concatenation for Estimation ---
        prompt_text = ""
        if request_data.system:
            if isinstance(request_data.system, str):
                prompt_text += request_data.system + "\n\n"
            elif isinstance(request_data.system, list):
                prompt_text += "\n".join([block.text for block in request_data.system if block.type == "text"]) + "\n\n"

        for msg in request_data.messages:
            if isinstance(msg.content, str):
                prompt_text += msg.content + "\n\n"
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if block.type == "text":
                        prompt_text += block.text + "\n"
                    # Ignore other block types for simple estimation

        # --- Estimation Logic ---
        # Very rough estimate: 1 token ~ 4 characters.
        # TODO: Consider using vertexai.get_generative_model(gemini_model_name).count_tokens(prompt_text)
        # This would require making the endpoint async and ensuring client initialization.
        token_estimate = len(prompt_text) // 4
        logger.info(f"Estimated token count for ~{len(prompt_text)} chars: {token_estimate}")

        # Log request details
        log_request_beautifully(
            raw_request.method,
            raw_request.url.path,
            original_model_name,
            gemini_model_name,
            len(request_data.messages),
            0,  # No tools considered in this basic estimation
            200,
        )

        response = TokenCountResponse(input_tokens=token_estimate)
        # Add custom header for logging middleware
        response_headers = {"X-Mapped-Model": gemini_model_name}
        return JSONResponse(content=response.dict(), headers=response_headers)

    except Exception as e:
        logger.error(f"Error counting tokens: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint providing basic service info."""
    return {
        "message": "Anthropic API Compatible Proxy for Gemini on Thomson Reuters AI Platform",
        "status": "running",
        "vertex_initialized": aiplatform_client.initialized,
        "vertex_project": aiplatform_client.project_id,
        "vertex_location": aiplatform_client.location,
        "supported_gemini_models": AIPLATFORM_MODELS,
    }


# --- Server Startup ---
@app.on_event("startup")
async def startup_event():
    """Attempt to initialize the AI Platform client on server startup."""
    logger.info("Server starting up. Initializing AIPlatformClient...")
    try:
        # Use asyncio.to_thread for the potentially blocking auth call
        await asyncio.to_thread(aiplatform_client.initialize)
        if aiplatform_client.initialized:
            logger.info("AIPlatformClient initialized successfully on startup.")
        else:
            # This case shouldn't be reached if initialize raises on failure
            logger.warning("AIPlatformClient initialization reported no error but status is not initialized.")
    except Exception as e:
        # Log critical failure but allow server to start. Requests will fail until fixed.
        logger.critical(
            f"CRITICAL FAILURE during startup initialization: {e}. Server will start but API calls will likely fail.",
            exc_info=True,
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8082))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()  # Control uvicorn log level

    print(f"Starting server on {host}:{port} with log level {log_level}")

    # Run Uvicorn
    # Note: --reload is useful for development but removed for clarity
    uvicorn.run(app, host=host, port=port, log_level=log_level)
