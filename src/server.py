import asyncio
import enum
import json
import logging
import os
import sys
import time
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
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models._generative_models import ResponseBlockedError, FinishReason

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Using DEBUG level for more detailed logs during testing
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configure uvicorn to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# List of supported AI Platform models
AIPLATFORM_MODELS = [
    "gemini-2.5-pro-preview-03-25",  # Main model for high-quality outputs
    "gemini-2.0-flash",  # Faster, smaller model for simpler queries
]

app = FastAPI()


class AuthenticationError(Exception):
    """Custom exception for authentication failures."""

    pass


def get_gemini_credentials():
    """
    Authenticates with the custom endpoint and returns Vertex AI credentials.

    Returns:
        tuple: Contains project_id, location, and OAuth2Credentials object.
        Returns None, None, None on failure.

    Raises:
        AuthenticationError: If authentication fails for any reason.
    """
    workspace_id = os.getenv("WORKSPACE_ID")
    model_name = os.getenv("MODEL_NAME", "gemini-2.5-pro-preview-03-25")  # Use BIG_MODEL as default
    auth_url = os.getenv("AUTH_URL")

    if not all([workspace_id, auth_url]):
        raise AuthenticationError("Missing required environment variables: WORKSPACE_ID, AUTH_URL")

    logging.info(f"Authenticating with AI Platform for workspace {workspace_id} and model {model_name}")
    logging.info(
        "NOTE: Authentication requires AWS credentials to be set up via 'mltools-cli aws-login' before running"
    )

    payload = {
        "workspace_id": workspace_id,
        "model_name": model_name,
    }

    logging.info(f"Requesting temporary token from {auth_url} for workspace {workspace_id}")

    try:
        # Send the request with timeout
        resp = requests.post(auth_url, headers=None, json=payload, timeout=30)

        # Check status code before proceeding
        if resp.status_code != 200:
            error_msg = f"Authentication request failed with status {resp.status_code}"
            logging.error(f"{error_msg}: {resp.text}")
            raise AuthenticationError(error_msg)

        # Parse the response as JSON
        try:
            credentials_data = json.loads(resp.content)
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response: {e}"
            logging.error(f"{error_msg}. Response: {resp.text}")
            raise AuthenticationError(error_msg)

        # Check if token exists in response (matching README example)
        if "token" not in credentials_data:
            error_msg = credentials_data.get("message", "Token not found in response.")
            logging.error(f"Authentication failed: {error_msg}")
            raise AuthenticationError(f"Failed to retrieve token. Server response: {error_msg}")

        project_id = credentials_data.get("project_id")
        location = credentials_data.get("region")
        token = credentials_data["token"]
        expires_on_str = credentials_data.get("expires_on", "N/A")

        if not project_id or not location:
            raise AuthenticationError("Missing 'project_id' or 'region' in authentication response.")

        # Optional: Add check for token expiry if needed for long-running sessions,
        # but typically these tokens are short-lived and re-fetched per session.
        logging.info(f"Successfully fetched Gemini credentials. Valid until: {expires_on_str}")

        # Create credentials using the simplified constructor as in the README example
        temp_creds = OAuth2Credentials(token)
        return project_id, location, temp_creds

    except requests.exceptions.ConnectionError as e:
        logging.error(f"Connection error during authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Failed to connect to authentication server: {e}. Check your network connection.")
    except requests.exceptions.Timeout as e:
        logging.error(f"Timeout during authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Authentication request timed out: {e}. Server might be overloaded.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Network error during authentication: {e}", exc_info=True)
        # Include response text in the error message if available
        if "resp" in locals() and hasattr(resp, "text"):
            error_details = f"{e}. Response: {resp.text}"
        else:
            error_details = str(e)
        raise AuthenticationError(f"Network error connecting to auth endpoint: {error_details}")
    except json.JSONDecodeError as e:
        # Get the response text if 'resp' exists
        response_text = resp.text if "resp" in locals() and hasattr(resp, "text") else "No response available"
        error_msg = f"Failed to parse authentication response: {e}"
        logging.error(f"{error_msg}. Response text: {response_text}", exc_info=True)
        raise AuthenticationError(f"{error_msg}. Check server response format.")
    except Exception as e:
        logging.error(f"An unexpected error occurred during authentication: {e}", exc_info=True)
        # Re-raise specific AuthenticationError or a generic one
        if isinstance(e, AuthenticationError):
            raise
        else:
            raise AuthenticationError(f"An unexpected error during authentication: {e}")


class AIPlatformClient:
    """Client for direct interaction with Thomson Reuters AI Platform."""

    def __init__(self):
        """Initialize the AI Platform client."""
        # AI Platform (Thomson Reuters) credentials
        self.project_id = None
        self.location = None
        self.credentials = None
        self.initialized = False

    def initialize(self):
        """Initialize AI Platform credentials."""
        try:
            self.project_id, self.location, self.credentials = get_gemini_credentials()
            logging.info(
                f"Successfully initialized AI Platform credentials for project {self.project_id} in {self.location}"
            )

            # Initialize Vertex AI with our authenticated credentials
            vertexai.init(project=self.project_id, location=self.location, credentials=self.credentials)
            logging.info(f"Initialized Vertex AI with project={self.project_id}, location={self.location}")

            self.initialized = True
            return True
        except AuthenticationError as e:
            logging.error(f"Failed to initialize AI Platform credentials: {e}")
            raise  # Fail immediately if we can't get credentials - AI Platform is our only provider
        except Exception as e:
            logging.error(f"Unexpected error initializing AI Platform credentials: {e}")
            raise  # Fail immediately if we can't get credentials - AI Platform is our only provider

    def ensure_initialized(self):
        """Ensure the client is initialized."""
        if not self.initialized:
            self.initialize()

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
    ) -> Union[Dict[str, Any], AsyncGenerator[Any, None]]:
        """
        Send a completion request to AI Platform using Vertex AI SDK.
        Handles both streaming and non-streaming requests.

        Args:
            model_name: The AI Platform model name
            messages: List of message objects with role and content
            system_message: Optional system message for the model
            stream: Whether to stream the response
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop_sequences: List of sequences to stop generation

        Returns:
            For non-streaming: Dict with response data
            For streaming: AsyncGenerator yielding response chunks
        """
        self.ensure_initialized()

        try:
            logger.info(f"Vertex AI request: model={model_name}, stream={stream}")

            # Create the Vertex AI model instance with system instruction if provided
            generation_config = {}
            if system_message:
                # For Gemini, system instruction is passed directly
                model = GenerativeModel(model_name, system_instruction=system_message)
            else:
                model = GenerativeModel(model_name)
            
            # Configure generation parameters
            if max_tokens is not None:
                generation_config["max_output_tokens"] = max_tokens
            if temperature is not None:
                generation_config["temperature"] = temperature
            if top_p is not None:
                generation_config["top_p"] = top_p
            if top_k is not None:
                generation_config["top_k"] = top_k
            if stop_sequences:
                generation_config["stop_sequences"] = stop_sequences

            # Process the messages to build conversation history for Vertex AI
            vertex_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                # Gemini uses 'user' and 'model' roles
                vertex_role = "user" if role == "user" else "model"
                
                # Handle different content formats
                if isinstance(content, str):
                    vertex_messages.append({
                        "role": vertex_role,
                        "parts": [{"text": content}]
                    })
                elif isinstance(content, list):
                    # Simplify complex content to text for now
                    text_content = ""
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text_content += item.get("text", "") + "\n"
                        # Basic handling for tool results
                        elif isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_id = item.get("tool_use_id", "unknown")
                            result_content = parse_tool_result_content(item.get("content", ""))
                            text_content += f"[Tool Result for {tool_id}]:\n{result_content}\n"
                    
                    if text_content.strip():
                        vertex_messages.append({
                            "role": vertex_role,
                            "parts": [{"text": text_content.strip()}]
                        })
                        
            # Initialize chat session if needed
            # Note: We'll use generate_content directly instead of chat for better streaming control

            # If streaming is enabled, return a generator
            if stream:
                logger.info("Streaming mode: Using generate_content with stream=True")
                try:
                    # Use generate_content with streaming - make it asynchronous
                    # Vertex AI's generate_content is synchronous and blocking,
                    # so we'll run it in a thread and convert to an async iterator
                    
                    # Get the stream generator from Vertex AI in a separate thread
                    stream_generator = await asyncio.to_thread(
                        model.generate_content,
                        vertex_messages,
                        stream=True,
                        generation_config=generation_config if generation_config else None
                    )
                    
                    # Create an async generator adapter for the synchronous iterator
                    async def async_stream_adapter():
                        # We're using a regular for loop with the synchronous iterator
                        # but yielding asynchronously to avoid blocking the event loop
                        for chunk in stream_generator:
                            yield chunk
                            # Small delay to avoid overwhelming the event loop
                            await asyncio.sleep(0.01)
                    
                    response_stream = async_stream_adapter()
                    
                    return response_stream
                except Exception as e:
                    logger.error(f"Error initializing stream from Vertex AI: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")
            else:
                # For non-streaming requests
                logger.info("Non-streaming mode: Using generate_content")
                try:
                    # Run the blocking call in a thread to avoid blocking the FastAPI event loop
                    response = await asyncio.to_thread(
                        model.generate_content,
                        vertex_messages,
                        generation_config=generation_config if generation_config else None
                    )
                    
                    # Extract response content
                    response_text = ""
                    tool_calls = []
                    finish_reason = None
                    prompt_tokens = 0
                    completion_tokens = 0
                    
                    # Extract usage metadata if available
                    if hasattr(response, "usage_metadata") and response.usage_metadata:
                        prompt_tokens = response.usage_metadata.prompt_token_count
                        completion_tokens = response.usage_metadata.candidates_token_count
                    
                    # Extract finish reason
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason = response.candidates[0].finish_reason
                    
                    # Extract content and function calls
                    if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                        for part in response.candidates[0].content.parts:
                            try:
                                if hasattr(part, "text") and part.text:
                                    response_text += part.text
                            except (ValueError, AttributeError) as e:
                                logger.debug(f"Could not get part.text: {e}")
                                
                            if hasattr(part, "function_call") and part.function_call:
                                # Handle function call
                                fc = part.function_call
                                tool_calls.append({
                                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                    "type": "function",
                                    "function": {
                                        "name": fc.name,
                                        "arguments": json.dumps(fc.args)
                                    }
                                })
                            
                            # Handle executable_code (Gemini's way of representing tool calls)
                            elif hasattr(part, "executable_code") and part.executable_code:
                                logger.info(f"Found executable_code in non-streaming response: {part.executable_code.language}")
                                
                                # Parse the tool code from executable_code
                                code = part.executable_code.code
                                logger.debug(f"Executable code: {code}")
                                
                                # Try to extract tool call info from the code
                                import re
                                
                                # Extract all tool code blocks
                                tool_blocks = re.findall(r'<tool_code>(.*?)</tool_code>', code, re.DOTALL)
                                
                                for tool_block in tool_blocks:
                                    # Extract tool name
                                    tool_name_match = re.search(r'<tool_name>(.*?)</tool_name>', tool_block, re.DOTALL)
                                    if tool_name_match:
                                        tool_name = tool_name_match.group(1).strip()
                                        
                                        # Extract args
                                        args = {}
                                        args_section = re.search(r'<args>(.*?)</args>', tool_block, re.DOTALL)
                                        if args_section:
                                            args_text = args_section.group(1).strip()
                                            # Parse individual arg tags
                                            arg_matches = re.findall(r'<(\w+)>(.*?)</\1>', args_text, re.DOTALL)
                                            for arg_name, arg_value in arg_matches:
                                                args[arg_name] = arg_value.strip()
                                        
                                        # Add as a tool call
                                        tool_calls.append({
                                            "id": f"toolu_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {
                                                "name": tool_name,
                                                "arguments": json.dumps(args)
                                            }
                                        })
                                        logger.info(f"Extracted tool call from executable_code: {tool_name} with args {args}")
                    
                    # Map finish reason to OpenAI/Anthropic format
                    openai_finish_reason = "stop"
                    if finish_reason:
                        logger.debug(f"Mapping finish reason (non-streaming): {finish_reason}")
                        # Handle both enum values and string representations
                        if finish_reason == "MAX_TOKENS" or (hasattr(FinishReason, "MAX_TOKENS") and finish_reason == FinishReason.MAX_TOKENS):
                            openai_finish_reason = "length"
                        elif finish_reason == "SAFETY" or (hasattr(FinishReason, "SAFETY") and finish_reason == FinishReason.SAFETY):
                            openai_finish_reason = "content_filter"
                        elif finish_reason == "FUNCTION_CALL" or (hasattr(FinishReason, "FUNCTION_CALL") and finish_reason == FinishReason.FUNCTION_CALL):
                            openai_finish_reason = "tool_calls"
                        elif finish_reason == "RECITATION" or (hasattr(FinishReason, "RECITATION") and finish_reason == FinishReason.RECITATION):
                            # Recitation issues are similar to safety
                            openai_finish_reason = "content_filter"
                        elif finish_reason == "STOP" or (hasattr(FinishReason, "STOP") and finish_reason == FinishReason.STOP):
                            openai_finish_reason = "stop"
                        # All other reasons map to "stop" by default
                        
                    # If we have tool_calls, force the finish reason
                    if tool_calls:
                        openai_finish_reason = "tool_calls"
                        logger.info(f"Setting finish_reason to 'tool_calls' due to tool_calls presence")
                    
                    # Estimate tokens if not provided by the API
                    if prompt_tokens == 0:
                        # Rough estimation based on character count
                        prompt_text = ""
                        for msg in messages:
                            content = msg.get("content", "")
                            if isinstance(content, str):
                                prompt_text += content
                            elif isinstance(content, list):
                                for item in content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        prompt_text += item.get("text", "") + "\n"
                        prompt_tokens = len(prompt_text) // 4
                    
                    if completion_tokens == 0:
                        completion_tokens = len(response_text) // 4
                    
                    # Return in OpenAI-compatible format
                    return {
                        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": response_text,
                                    "tool_calls": tool_calls if tool_calls else None
                                },
                                "finish_reason": openai_finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    }
                
                except ResponseBlockedError as rbe:
                    logger.error(f"Response blocked by Vertex AI: {rbe}", exc_info=True)
                    raise HTTPException(status_code=400, detail=f"Response blocked by safety filters: {rbe.block_reason}")
                except Exception as e:
                    logger.error(f"Error in Vertex AI completion: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Completion error: {str(e)}")

        except Exception as e:
            logger.error(f"Error in direct Vertex AI integration: {e}", exc_info=True)
            raise


# Create a singleton instance
aiplatform_client = AIPlatformClient()

# Initialize AI Platform client
try:
    aiplatform_client.initialize()
except Exception as e:
    logger.critical(f"CRITICAL: Failed to initialize AIPlatformClient. Server cannot function. Error: {e}", exc_info=True)
    # The server will still start, but requests will fail until authentication works


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


async def handle_vertex_streaming(response_generator, original_request: MessagesRequest):
    """
    Handles the raw streaming response from Vertex AI model.generate_content(stream=True)
    and converts it to Anthropic-compatible Server-Sent Events.
    
    The response_generator should be an async generator that yields chunks from Vertex AI.
    These chunks are processed and converted to the Anthropic streaming format.
    """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    model_name = original_request.original_model or original_request.model  # Use original name for client
    input_tokens = 0
    output_tokens = 0

    try:
        # 1. Send message_start event
        message_start_data = {
            'type': 'message_start',
            'message': {
                'id': message_id,
                'type': 'message',
                'role': 'assistant',
                'model': model_name,
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
                'usage': {'input_tokens': 0, 'output_tokens': 0}  # Placeholder
            }
        }
        yield f"event: message_start\ndata: {json.dumps(message_start_data)}\n\n"
        logger.debug("Sent message_start")

        # 2. Send ping event
        yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
        logger.debug("Sent ping")

        # State tracking
        current_content_block_index = -1
        current_block_type = None  # 'text' or 'tool_use'
        current_tool_id = None
        current_tool_name = None
        accumulated_tool_args = ""
        text_block_started = False
        tool_blocks = []  # Keep track of tool blocks used
        accumulated_text = ""

        # 3. Process the stream from Vertex AI
        # Start with a text block for initial content
        current_content_block_index = 0
        current_block_type = 'text'
        text_block_start_data = {
            'type': 'content_block_start',
            'index': current_content_block_index,
            'content_block': {'type': 'text', 'text': ""}
        }
        yield f"event: content_block_start\ndata: {json.dumps(text_block_start_data)}\n\n"
        logger.debug(f"Sent content_block_start for initial text block {current_content_block_index}")
        text_block_started = True

        async for chunk in response_generator:
            logger.debug(f"Raw Vertex Chunk: {chunk}")

            chunk_text = ""
            chunk_function_call = None # Reset for each chunk
            chunk_finish_reason = None
            is_last_chunk = False

            try:
                # --- Start: Parsing Logic ---
                # Start with empty text and function call
                chunk_text = ""
                chunk_function_call = None
                
                # Check candidates for parts (text, function_call, executable_code)
                if hasattr(chunk, "candidates") and chunk.candidates:
                    candidate = chunk.candidates[0]

                    if hasattr(candidate, "content") and candidate.content and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            # 1. Check for explicit function_call attribute
                            if hasattr(part, "function_call") and part.function_call:
                                chunk_function_call = part.function_call
                                logger.info(f"Found native function_call: {chunk_function_call.name}")
                                chunk_text = "" # Clear any text found if we have a function call
                                break # Prioritize function_call over text/executable_code in the same part

                            # 2. Check for executable_code and parse it ****** NEW LOGIC ******
                            elif hasattr(part, "executable_code") and part.executable_code:
                                logger.info(f"Found executable_code: {part.executable_code.language}")
                                code = part.executable_code.code
                                logger.debug(f"Executable code content: {code}")

                                import re
                                # First try XML-style tool code pattern
                                tool_blocks = re.findall(r'<tool_code>(.*?)</tool_code>', code, re.DOTALL)

                                if tool_blocks:
                                    tool_block = tool_blocks[0] # Assume one for now
                                    tool_name_match = re.search(r'<tool_name>(.*?)</tool_name>', tool_block, re.DOTALL)
                                    if tool_name_match:
                                        tool_name = tool_name_match.group(1).strip()
                                        args = {}
                                        args_section = re.search(r'<args>(.*?)</args>', tool_block, re.DOTALL)
                                        if args_section:
                                            args_text = args_section.group(1).strip()
                                            arg_matches = re.findall(r'<(\w+)>(.*?)</\1>', args_text, re.DOTALL)
                                            for arg_name, arg_value in arg_matches:
                                                # Attempt to decode JSON strings within args
                                                try:
                                                    # Handle potential double escaping
                                                    decoded_value = json.loads(f'"{arg_value.strip()}"')
                                                    args[arg_name] = decoded_value
                                                except json.JSONDecodeError:
                                                    args[arg_name] = arg_value.strip() # Fallback to raw string

                                        # Create a synthetic FunctionCall object for consistent handling
                                        class SyntheticFunctionCall:
                                            def __init__(self, name, args):
                                                self.name = name
                                                self.args = args
                                        chunk_function_call = SyntheticFunctionCall(tool_name, args)
                                        logger.info(f"Extracted tool call from executable_code XML format: {tool_name} with args {args}")
                                        chunk_text = "" # Clear any text found
                                        break # Prioritize this parsed call
                                
                                # Try backtick-style tool code pattern if XML pattern didn't match
                                elif "```tool_code" in code:
                                    logger.info("Found ```tool_code pattern in executable_code")
                                    # Try to match ```tool_code format
                                    tool_code_match = re.search(r'```tool_code\s+(.*?)```', code, re.DOTALL)
                                    if tool_code_match:
                                        tool_content = tool_code_match.group(1).strip()
                                        logger.debug(f"Found tool_code content: {tool_content}")
                                        
                                        # Extract tool name from typical pattern like ReadFileTool().run(paths=["README.md"])
                                        # or print(ReadFileTool().run(paths=["README.md"]))
                                        tool_call_pattern = r'(?:print\()?\s*(\w+)(?:Tool)?\(\)\.run\('
                                        tool_name_match = re.search(tool_call_pattern, tool_content, re.IGNORECASE)
                                        
                                        if tool_name_match:
                                            tool_name = tool_name_match.group(1)
                                            logger.debug(f"Found tool name from backtick code: {tool_name}")
                                            
                                            # Extract arguments
                                            args = {}
                                            
                                            # Handle paths argument
                                            paths_pattern = r'paths=\[(.*?)\]'
                                            paths_match = re.search(paths_pattern, tool_content)
                                            if paths_match:
                                                # Extract individual paths from the list
                                                path_content = paths_match.group(1)
                                                paths = []
                                                # Match quoted strings
                                                for path in re.findall(r'"([^"]+)"', path_content):
                                                    paths.append(path)
                                                if paths:
                                                    args["paths"] = paths
                                                    logger.debug(f"Extracted paths: {paths}")
                                            
                                            # Create synthetic function call
                                            class SyntheticFunctionCall:
                                                def __init__(self, name, args):
                                                    self.name = name
                                                    self.args = args
                                            
                                            chunk_function_call = SyntheticFunctionCall(tool_name, args)
                                            logger.info(f"Extracted tool call from executable_code backtick format: {tool_name} with args {args}")
                                            chunk_text = "" # Clear any text found
                                            break # Prioritize this parsed call
                                
                                else:
                                    # If executable_code doesn't match any tool format, treat as text
                                    logger.warning("executable_code found but no tool format matched. Treating as text.")
                                    try:
                                        if hasattr(part, "text") and part.text:
                                           chunk_text += part.text
                                    except (ValueError, AttributeError): pass


                            # 3. Accumulate text if no function call found in this part
                            elif not chunk_function_call:
                                try:
                                    if hasattr(part, "text") and part.text:
                                        part_text = part.text
                                        if part_text:
                                            chunk_text += part_text
                                            logger.debug(f"Found text in part: {part_text}")
                                except (ValueError, AttributeError) as e:
                                    logger.debug(f"Could not get part.text: {e}")
                
                # If no text found in parts, try to get from the chunk directly
                # Only do this if we didn't already find text in parts to avoid duplication
                if not chunk_text and not chunk_function_call:
                    try:
                        if hasattr(chunk, "text") and chunk.text:
                            chunk_text = chunk.text
                            logger.debug(f"Chunk Text (direct): '{chunk_text}'")
                    except (ValueError, AttributeError) as e:
                        logger.debug(f"Could not get chunk.text: {e}")

                # Check for finish reason from candidate
                if hasattr(chunk, "candidates") and chunk.candidates and hasattr(chunk.candidates[0], "finish_reason") and chunk.candidates[0].finish_reason:
                         # Convert enum to string if needed, or handle string directly
                        raw_reason = chunk.candidates[0].finish_reason
                        try:
                            if isinstance(raw_reason, enum.Enum): # Check if it's an enum
                                chunk_finish_reason = raw_reason.name # Get the string name
                            else:
                                chunk_finish_reason = str(raw_reason) # Assume it might be string/int
                            
                            logger.debug(f"Finish reason found: {chunk_finish_reason} (Raw type: {type(raw_reason).__name__})")
                        except Exception as e:
                            # Fallback for any enum handling issues
                            chunk_finish_reason = "STOP"  # Default to STOP if we can't determine
                            logger.debug(f"Error handling finish reason, using default: {e}")
                        is_last_chunk = True

                # Get usage metrics if available (keep this from original code)
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    if hasattr(chunk.usage_metadata, "prompt_token_count"):
                        input_tokens = chunk.usage_metadata.prompt_token_count
                    if hasattr(chunk.usage_metadata, "candidates_token_count"):
                        output_tokens = chunk.usage_metadata.candidates_token_count
                    logger.debug(f"Usage: input={input_tokens}, output={output_tokens}")
                
                # --- End: Parsing Logic ---

                # --- Start: Event Generation Logic (largely unchanged but now uses parsed chunk_function_call) ---
                # Process function calls (now includes those parsed from executable_code)
                if chunk_function_call:
                    # Close text block if open
                    if current_block_type == 'text' and text_block_started:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                        logger.debug(f"Closed text block {current_content_block_index} before tool")
                        text_block_started = False
                        current_block_type = None

                    # Start a new tool_use block if needed
                    if current_block_type != 'tool_use' or current_tool_name != chunk_function_call.name:
                        current_content_block_index += 1
                        current_block_type = 'tool_use'
                        current_tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        current_tool_name = chunk_function_call.name
                        accumulated_tool_args = ""
                        tool_block_data = { # payload for content_block_start
                            'type': 'content_block_start',
                            'index': current_content_block_index,
                            'content_block': {
                                'type': 'tool_use',
                                'id': current_tool_id,
                                'name': current_tool_name,
                                'input': {}
                            }
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(tool_block_data)}\n\n"
                        logger.debug(f"Started tool block {current_content_block_index} for {current_tool_name}")
                        tool_blocks.append(current_content_block_index)

                    # Process tool arguments
                    if hasattr(chunk_function_call, "args") and chunk_function_call.args:
                        # Gemini args are dicts, Anthropic expects JSON string fragments
                        args_json_str = json.dumps(chunk_function_call.args)
                        # Avoid sending empty "{}" args delta if that's all we got
                        if args_json_str != "{}" and args_json_str != accumulated_tool_args:
                            tool_delta_data = { # payload for content_block_delta
                                'type': 'content_block_delta',
                                'index': current_content_block_index,
                                'delta': {'type': 'input_json_delta', 'partial_json': args_json_str}
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(tool_delta_data)}\n\n"
                            logger.debug(f"Sent tool args delta: {args_json_str[:100]}...")
                            accumulated_tool_args = args_json_str

                # Process text content (only if no function call was processed in this chunk)
                elif chunk_text:
                    # Check for tool code pattern directly in text content
                    tool_call_extracted = False
                    
                    # Check if text contains ```tool_code or some form of print(ReadFileTool)
                    if ("```tool_code" in chunk_text or 
                        "print(ReadFileTool" in chunk_text or 
                        "ReadFileTool().run" in chunk_text):
                        
                        logger.info("Found potential tool code in text chunk - attempting to extract")
                        
                        import re
                        # Try to extract from backtick format first
                        tool_code_match = re.search(r'```tool_code\s+(.*?)```', chunk_text, re.DOTALL)
                        
                        if tool_code_match:
                            # Extract code from backticks
                            tool_content = tool_code_match.group(1).strip()
                            logger.debug(f"Found ```tool_code content in text: {tool_content}")
                            
                            # Extract tool name
                            tool_call_pattern = r'(?:print\()?\s*(\w+)(?:Tool)?\(\)\.run\('
                            tool_name_match = re.search(tool_call_pattern, tool_content, re.IGNORECASE)
                            
                            if tool_name_match:
                                tool_name = tool_name_match.group(1)
                                args = {}
                                
                                # Extract paths argument
                                paths_pattern = r'paths=\[(.*?)\]'
                                paths_match = re.search(paths_pattern, tool_content)
                                if paths_match:
                                    path_content = paths_match.group(1)
                                    paths = []
                                    for path in re.findall(r'"([^"]+)"', path_content):
                                        paths.append(path)
                                    if paths:
                                        args["paths"] = paths
                                
                                # Create synthetic function call
                                class SyntheticFunctionCall:
                                    def __init__(self, name, args):
                                        self.name = name
                                        self.args = args
                                
                                chunk_function_call = SyntheticFunctionCall(tool_name, args)
                                logger.info(f"Extracted tool call from text backtick format: {tool_name} with args {args}")
                                tool_call_extracted = True
                        
                        # If not found in backticks, try for direct print pattern 
                        if not tool_call_extracted:
                            # Look for print(ReadFileTool().run(paths=["README.md"]))
                            direct_call_pattern = r'print\(\s*(\w+)(?:Tool)?\(\)\.run\((?:paths=\[(.*?)\])\)\)'
                            direct_match = re.search(direct_call_pattern, chunk_text, re.DOTALL)
                            
                            if direct_match:
                                tool_name = direct_match.group(1)
                                path_content = direct_match.group(2)
                                args = {}
                                
                                # Extract paths
                                paths = []
                                for path in re.findall(r'"([^"]+)"', path_content):
                                    paths.append(path)
                                if paths:
                                    args["paths"] = paths
                                
                                # Create synthetic function call
                                class SyntheticFunctionCall:
                                    def __init__(self, name, args):
                                        self.name = name
                                        self.args = args
                                
                                chunk_function_call = SyntheticFunctionCall(tool_name, args)
                                logger.info(f"Extracted tool call from direct text pattern: {tool_name} with args {args}")
                                tool_call_extracted = True
                    
                    # If we extracted a tool call from text, process it as a function call
                    if tool_call_extracted and chunk_function_call:
                        # Close text block if open
                        if current_block_type == 'text' and text_block_started:
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                            logger.debug(f"Closed text block {current_content_block_index} before extracted tool")
                            text_block_started = False
                            current_block_type = None
                        
                        # Start a new tool_use block
                        current_content_block_index += 1
                        current_block_type = 'tool_use'
                        current_tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        current_tool_name = chunk_function_call.name
                        accumulated_tool_args = ""
                        
                        tool_block_data = {
                            'type': 'content_block_start',
                            'index': current_content_block_index,
                            'content_block': {
                                'type': 'tool_use',
                                'id': current_tool_id,
                                'name': current_tool_name,
                                'input': {}
                            }
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(tool_block_data)}\n\n"
                        logger.debug(f"Started tool block {current_content_block_index} for extracted tool {current_tool_name}")
                        tool_blocks.append(current_content_block_index)
                        
                        # Send tool arguments
                        if hasattr(chunk_function_call, "args") and chunk_function_call.args:
                            args_json_str = json.dumps(chunk_function_call.args)
                            if args_json_str != "{}" and args_json_str != accumulated_tool_args:
                                tool_delta_data = {
                                    'type': 'content_block_delta',
                                    'index': current_content_block_index,
                                    'delta': {'type': 'input_json_delta', 'partial_json': args_json_str}
                                }
                                yield f"event: content_block_delta\ndata: {json.dumps(tool_delta_data)}\n\n"
                                logger.debug(f"Sent tool args for extracted tool: {args_json_str[:100]}...")
                                accumulated_tool_args = args_json_str
                    
                    # Process as normal text if no tool call was extracted
                    else:
                        # Close tool block if open
                        if current_block_type == 'tool_use':
                            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                            logger.debug(f"Closed tool block {current_content_block_index} before text")
                            current_block_type = None

                        # Ensure text block is started
                        if not text_block_started:
                            # If the first block wasn't text, or was closed, start a new one
                            if current_content_block_index != 0 or current_block_type is None:
                                 current_content_block_index += 1
                            current_block_type = 'text'
                            text_block_start_data = {
                                'type': 'content_block_start',
                                'index': current_content_block_index,
                                'content_block': {'type': 'text', 'text': ""}
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(text_block_start_data)}\n\n"
                            logger.debug(f"Started/reopened text block {current_content_block_index}")
                            text_block_started = True

                        # Send text delta
                        accumulated_text += chunk_text
                        text_delta_data = {
                            'type': 'content_block_delta',
                            'index': current_content_block_index, # Use the current text block index
                            'delta': {'type': 'text_delta', 'text': chunk_text}
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(text_delta_data)}\n\n"
                        logger.debug(f"Sent text delta for block {current_content_block_index}: '{chunk_text}'")

                # Process finish reason / End of stream
                if is_last_chunk:
                    # Close the last open block
                    if current_block_type is not None and current_content_block_index >= 0:
                         yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                         logger.debug(f"Closed final block {current_content_block_index} (type: {current_block_type})")

                    # Map finish reason
                    anthropic_stop_reason = "end_turn" # Default
                    if chunk_finish_reason:
                        logger.debug(f"Mapping final finish reason: {chunk_finish_reason}")
                        # Just check string values directly - safer than enum comparisons
                        if chunk_finish_reason == "MAX_TOKENS":
                            anthropic_stop_reason = "max_tokens"
                        # If the *reason* is FUNCTION_CALL OR if we processed a function call in the *last* chunk
                        elif chunk_finish_reason == "FUNCTION_CALL" or chunk_function_call:
                            anthropic_stop_reason = "tool_use"
                        elif chunk_finish_reason == "STOP":
                            anthropic_stop_reason = "end_turn"
                        elif chunk_finish_reason == "SAFETY":
                            anthropic_stop_reason = "end_turn" # Or map to error?
                        elif chunk_finish_reason == "RECITATION":
                            anthropic_stop_reason = "end_turn" # Or map to error?

                    # If we've used any tools during this stream, set the reason
                    if len(tool_blocks) > 0:
                        anthropic_stop_reason = "tool_use"
                        logger.info(f"Setting final stop_reason to 'tool_use' due to tool blocks used ({len(tool_blocks)} blocks)")

                    # Estimate output tokens if needed
                    if output_tokens == 0:
                        output_tokens = len(accumulated_text) // 4

                    # Send final events
                    message_delta_data = {
                        'type': 'message_delta',
                        'delta': {'stop_reason': anthropic_stop_reason, 'stop_sequence': None},
                        'usage': {'output_tokens': output_tokens}
                    }
                    yield f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
                    logger.debug(f"Sent final message_delta with stop_reason: {anthropic_stop_reason}")

                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
                    logger.debug("Sent final message_stop")

                    yield "data: [DONE]\n\n"
                    logger.info("Stream completed.")
                    return # Exit the generator

            except Exception as chunk_error:
                logger.error(f"Error processing chunk: {chunk_error}", exc_info=True)
                # Decide whether to continue or terminate stream based on error severity

        # Fallback if stream ends without a finish reason (less likely with Vertex)
        logger.warning("Stream ended without explicit finish reason.")
        if current_block_type is not None and current_content_block_index >= 0:
             yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
        
        # Estimate tokens if not provided
        if output_tokens == 0:
            output_tokens = len(accumulated_text) // 4
        
        # Send closing events
        message_delta_data = {
            'type': 'message_delta',
            'delta': {'stop_reason': "end_turn", 'stop_sequence': None},
            'usage': {'output_tokens': output_tokens}
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta_data)}\n\n"
        
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        yield "data: [DONE]\n\n"

    # Keep existing top-level exception handling
    except ResponseBlockedError as rbe:
        logger.error(f"Response blocked: {rbe}", exc_info=True)
        # Try to send an error message in a format the client can understand
        try:
            # Close any open block
            if 'current_block_type' in locals() and 'current_content_block_index' in locals():
                if current_block_type == 'text' and text_block_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                elif current_block_type == 'tool_use':
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
            
            error_message = {
                'type': 'message_delta',
                'delta': {'stop_reason': 'error', 'stop_sequence': None},
                'usage': {'output_tokens': output_tokens}
            }
            yield f"event: message_delta\ndata: {json.dumps(error_message)}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Failed to send error to client: {e}")
            yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        try:
            # Try to send error and close stream properly with appropriate message
            # First, send error content block if we can
            if 'current_content_block_index' in locals():
                try:
                    # If we have an open text block, add error text to it
                    if 'text_block_started' in locals() and text_block_started:
                        error_delta = {
                            'type': 'content_block_delta',
                            'index': current_content_block_index,
                            'delta': {'type': 'text_delta', 'text': f"\n\n[Error: Processing was interrupted]"}
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(error_delta)}\n\n"
                        
                        # Close the text block
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': current_content_block_index})}\n\n"
                except Exception:
                    pass  # If adding error message fails, continue with stream closing
                    
            # Send message stop events to properly close the stream
            yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'error', 'stop_sequence': None}})}\n\n"
            yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception:
            # Last resort - just send DONE marker
            yield "data: [DONE]\n\n"


def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes unsupported fields from a JSON schema for Gemini."""
    if isinstance(schema, dict):
        # Remove specific keys unsupported by Gemini tool parameters
        schema.pop("additionalProperties", None)
        schema.pop("default", None)

        # Check for unsupported 'format' in string types
        if schema.get("type") == "string" and "format" in schema:
            allowed_formats = {"enum", "date-time"}
            if schema["format"] not in allowed_formats:
                logger.debug(f"Removing unsupported format '{schema['format']}' for string type in Gemini schema.")
                schema.pop("format")

        # Recursively clean nested schemas (properties, items, etc.)
        for key, value in list(schema.items()):  # Use list() to allow modification during iteration
            schema[key] = clean_gemini_schema(value)
    elif isinstance(schema, list):
        # Recursively clean items in a list
        return [clean_gemini_schema(item) for item in schema]
    return schema


def parse_tool_result_content(content):
    """Helper function to properly parse and normalize tool result content."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                result += item.get("text", "") + "\n"
            elif isinstance(item, str):
                result += item + "\n"
            elif isinstance(item, dict):
                if "text" in item:
                    result += item.get("text", "") + "\n"
                else:
                    try:
                        result += json.dumps(item) + "\n"
                    except Exception:
                        result += str(item) + "\n"
            else:
                try:
                    result += str(item) + "\n"
                except Exception:
                    result += "Unparseable content\n"
        return result.strip()

    if isinstance(content, dict):
        if content.get("type") == "text":
            return content.get("text", "")
        try:
            return json.dumps(content)
        except Exception:
            return str(content)

    # Fallback for any other type
    try:
        return str(content)
    except Exception:
        return "Unparseable content"


def map_model_name(model_name: str, data: Dict[str, Any]) -> str:
    """Map model names from Claude to AI Platform."""
    original_model = model_name
    new_model = model_name  # Default to original value

    logger.debug(f"📋 MODEL VALIDATION: Original='{original_model}'")

    # Extract model name, removing any provider prefix
    clean_model = model_name
    if clean_model.startswith("anthropic/"):
        clean_model = clean_model[10:]
    elif clean_model.startswith("openai/"):
        clean_model = clean_model[7:]
    elif clean_model.startswith("gemini/"):
        clean_model = clean_model[7:]
    elif clean_model.startswith("aiplatform/"):
        clean_model = clean_model[11:]
    elif clean_model.startswith("claude-"):
        # Direct Claude model reference
        pass

    # --- Mapping Logic ---
    mapped = False

    # Map Claude models to corresponding AI Platform models
    if "haiku" in clean_model.lower():
        # Map Claude Haiku to the small model
        new_model = f"aiplatform/{AIPLATFORM_MODELS[1]}"  # gemini-2.0-flash
        mapped = True
        logger.info(f"Mapping Claude Haiku to {new_model}")
    elif "sonnet" in clean_model.lower() or "opus" in clean_model.lower():
        # Map Claude Sonnet/Opus to the big model
        new_model = f"aiplatform/{AIPLATFORM_MODELS[0]}"  # gemini-2.5-pro-preview
        mapped = True
        logger.info(f"Mapping Claude Sonnet/Opus to {new_model}")
    elif clean_model in AIPLATFORM_MODELS:
        # Direct model reference, ensure it has aiplatform/ prefix
        new_model = f"aiplatform/{clean_model}"
        mapped = True
        logger.info(f"Using directly specified model with AI Platform prefix: {new_model}")
    else:
        # For any unrecognized model, default to the most capable one
        new_model = f"aiplatform/{AIPLATFORM_MODELS[0]}"  # Default to the most capable model
        mapped = True
        logger.warning(f"⚠️ Unrecognized model: '{original_model}'. Defaulting to {new_model}")

    if mapped:
        logger.info(f"📌 MODEL MAPPING: '{original_model}' ➡️ '{new_model}'")

    # Store the original model in the values dictionary
    if isinstance(data, dict):
        data["original_model"] = original_model

    return new_model


# Define ANSI color codes for terminal output
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


def log_request_beautifully(method, path, claude_model, aiplatform_model, num_messages, num_tools, status_code):
    """Log requests in a beautiful format showing Claude to AI Platform mapping."""
    # Format the Claude model name nicely
    claude_display = f"{Colors.CYAN}{claude_model}{Colors.RESET}"

    # Extract endpoint name
    endpoint = path
    if "?" in endpoint:
        endpoint = endpoint.split("?")[0]

    # Extract just the AI Platform model name without provider prefix
    display = aiplatform_model
    if "/" in display:
        display = display.split("/")[-1]
    display = f"{Colors.GREEN}{display}{Colors.RESET}"

    # Format tools and messages
    tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
    messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

    # Format status code
    status_str = (
        f"{Colors.GREEN}✓ {status_code} OK{Colors.RESET}"
        if status_code == 200
        else f"{Colors.RED}✗ {status_code}{Colors.RESET}"
    )

    # Put it all together in a clear, beautiful format
    log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
    model_line = f"{claude_display} → {display} {tools_str} {messages_str}"

    # Print to console
    print(log_line)
    print(model_line)
    sys.stdout.flush()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log basic request details at debug level."""
    # Get request details
    method = request.method
    path = request.url.path
    client = request.client.host if request.client else "unknown"
    headers = {k: v for k, v in request.headers.items()}

    # Log request details
    logger.info(f"Incoming request: {method} {path} from {client}")
    logger.debug(f"Request headers: {headers}")

    # Process the request and get the response
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # Log response details
    logger.info(f"Response: {response.status_code} in {duration:.2f}s")
    logger.debug(f"Response headers: {response.headers}")

    return response


@app.post("/v1/messages")
async def create_message(request: MessagesRequest, raw_request: Request):
    """
    Handle the /v1/messages endpoint, which handles both streaming and non-streaming requests.
    This endpoint is compatible with the Anthropic API format.
    """
    try:
        # Parse the raw body as JSON since it's bytes
        body = await raw_request.body()
        body_json = json.loads(body.decode("utf-8"))
        original_model = body_json.get("model", "unknown")

        # Get client info for logging
        client_ip = raw_request.client.host if raw_request.client else "unknown"
        
        # Get the display name for logging, just the model name without provider prefix
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for proper handling
        clean_model = request.model
        if clean_model.startswith("aiplatform/"):
            clean_model_name = clean_model.replace("aiplatform/", "")
        else:
            # Get the model name portion after the last slash if it exists
            clean_model_name = clean_model.split("/")[-1]

        logger.info(f"📊 PROCESSING REQUEST: Model={clean_model_name}, Stream={request.stream}, Client={client_ip}")

        # Process system message if present
        system_message = None
        if request.system:
            if isinstance(request.system, str):
                system_message = request.system
            elif isinstance(request.system, list):
                # Combine text blocks for system message
                system_text = []
                for block in request.system:
                    if hasattr(block, "type") and block.type == "text":
                        system_text.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        system_text.append(block.get("text", ""))

                if system_text:
                    system_message = "\n\n".join(system_text)

        # Convert messages to format expected by API client
        messages = []
        for msg in request.messages:
            # Handle content as string or list of content blocks
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
            else:
                # Convert content blocks to simple text
                text_content = []
                for block in msg.content:
                    if hasattr(block, "type"):
                        if block.type == "text":
                            text_content.append(block.text)
                        elif block.type == "tool_result":
                            tool_id = getattr(block, "tool_use_id", "unknown")
                            # Extract text from tool result content
                            result_content = getattr(block, "content", "")
                            processed_content = parse_tool_result_content(result_content)
                            text_content.append(f"[Tool Result ID: {tool_id}]\n{processed_content}")
                        elif block.type == "tool_use":
                            tool_name = getattr(block, "name", "unknown")
                            tool_id = getattr(block, "id", "unknown")
                            tool_input = json.dumps(getattr(block, "input", {}))
                            text_content.append(f"[Tool: {tool_name} (ID: {tool_id})]\nInput: {tool_input}")

                messages.append({"role": msg.role, "content": "\n\n".join(text_content) or "..."})

        # Log request information
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            clean_model_name,
            len(messages),
            num_tools,
            200,  # Assuming success at this point
        )

        # Handle streaming mode
        if request.stream:
            logger.info("Using Vertex AI integration for streaming")
            
            # Convert Anthropic tools to Vertex AI tools if needed
            vertex_tools = None
            if request.tools:
                # This would need further implementation for tool mapping
                logger.info(f"Request includes {len(request.tools)} tools - preparing for Vertex AI")
                # TODO: Implement tool conversion
            
            # Get streaming response from Vertex AI
            response_generator = await aiplatform_client.completion(
                model_name=clean_model_name,
                messages=messages,
                system_message=system_message,
                stream=True,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
            )

            # Convert to Anthropic streaming format
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            
            logger.info("Returning streaming response with Anthropic-compatible headers")
            return StreamingResponse(
                handle_vertex_streaming(response_generator, request),
                media_type="text/event-stream",
                headers=headers
            )
        else:
            # For non-streaming requests
            logger.info("Using direct Vertex AI integration for completion")
            start_time = time.time()

            # Get response from Vertex AI
            response = await aiplatform_client.completion(
                model_name=clean_model_name,
                messages=messages,
                system_message=system_message,
                stream=False,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                stop_sequences=request.stop_sequences,
            )

            logger.debug(f"✅ RESPONSE RECEIVED: Model={clean_model_name}, Time={time.time() - start_time:.2f}s")

            # Extract key information from response
            response_id = response.get("id", f"msg_{uuid.uuid4()}")
            choices = response.get("choices", [{}])
            message = choices[0].get("message", {}) if choices else {}
            content_text = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            finish_reason = choices[0].get("finish_reason", "stop") if choices else "stop"
            usage_info = response.get("usage", {})

            # Map OpenAI finish_reason to Anthropic stop_reason
            anthropic_stop_reason = "end_turn"  # Default
            if finish_reason == "length":
                anthropic_stop_reason = "max_tokens"
            elif finish_reason == "tool_calls":
                anthropic_stop_reason = "tool_use"
            
            # Create content blocks
            content_blocks = []
            
            # Add text content if present
            if content_text:
                content_blocks.append(ContentBlockText(type="text", text=content_text))
            
            # Add tool_use blocks if present and finish_reason indicates tool use
            if tool_calls and anthropic_stop_reason == "tool_use":
                for tool_call in tool_calls:
                    if tool_call.get("type") == "function":
                        function = tool_call.get("function", {})
                        try:
                            # Parse arguments (handle both string and dict formats)
                            if isinstance(function.get("arguments"), str):
                                arguments = json.loads(function.get("arguments", "{}"))
                            else:
                                arguments = function.get("arguments", {})
                        except json.JSONDecodeError:
                            # If JSON parsing fails, use as raw string
                            arguments = {"raw_arguments": function.get("arguments", "{}")}
                        
                        content_blocks.append(
                            ContentBlockToolUse(
                                type="tool_use",
                                id=tool_call.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
                                name=function.get("name", "unknown_function"),
                                input=arguments
                            )
                        )
            
            # Ensure we have at least one content block (empty text if nothing else)
            if not content_blocks:
                content_blocks.append(ContentBlockText(type="text", text=""))

            # Create Anthropic-compatible response
            anthropic_response = MessagesResponse(
                id=response_id,
                model=original_model,  # Use the original model name requested
                role="assistant",
                content=content_blocks,
                stop_reason=anthropic_stop_reason,
                stop_sequence=None,
                usage=Usage(
                    input_tokens=usage_info.get("prompt_tokens", 0),
                    output_tokens=usage_info.get("completion_tokens", 0),
                ),
            )

            return anthropic_response

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()

        # Capture error details
        error_details = {"error": str(e), "type": type(e).__name__, "traceback": error_traceback}

        # Add additional exception details if available
        if hasattr(e, "__dict__"):
            for key, value in e.__dict__.items():
                if key not in error_details and key not in ["args", "__traceback__"]:
                    error_details[key] = str(value)

        # Log all error details
        logger.error(f"Error processing request: {json.dumps(error_details, indent=2)}")

        # Return detailed error
        status_code = getattr(e, "status_code", 500)
        raise HTTPException(status_code=status_code, detail=f"Error: {str(e)}")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: TokenCountRequest, raw_request: Request):
    """
    Handle the /v1/messages/count_tokens endpoint, which estimates token counts
    for requests in Anthropic API format.
    """
    try:
        # Get original and mapped model names
        original_model = request.original_model or request.model

        # Get the display name for logging
        display_model = original_model
        if "/" in display_model:
            display_model = display_model.split("/")[-1]

        # Clean model name for AI Platform
        if request.model.startswith("aiplatform/"):
            clean_model_name = request.model.replace("aiplatform/", "")
        else:
            clean_model_name = request.model.split("/")[-1]

        # Collect all text from messages for character count
        prompt_text = ""

        # Process system message if present
        if request.system:
            if isinstance(request.system, str):
                prompt_text += request.system + "\n\n"
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        prompt_text += block.text + "\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        prompt_text += block.get("text", "") + "\n"

        # Process each message
        for msg in request.messages:
            content = msg.content
            if isinstance(content, str):
                prompt_text += content + "\n\n"
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "type") and block.type == "text":
                        prompt_text += block.text + "\n"
                    elif isinstance(block, dict) and block.get("type") == "text":
                        prompt_text += block.get("text", "") + "\n"

        # Log the request
        num_tools = len(request.tools) if request.tools else 0
        log_request_beautifully(
            "POST",
            raw_request.url.path,
            display_model,
            clean_model_name,
            len(request.messages),
            num_tools,
            200,  # Assuming success at this point
        )

        # Simple approximation: 1 token ≈ 4 characters for Vertex AI models
        token_estimate = len(prompt_text) // 4

        # Return Anthropic-style response
        return TokenCountResponse(input_tokens=token_estimate)

    except Exception as e:
        import traceback

        error_traceback = traceback.format_exc()
        logger.error(f"Error counting tokens: {str(e)}\n{error_traceback}")
        raise HTTPException(status_code=500, detail=f"Error counting tokens: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint that describes the service."""
    return {"message": "Anthropic API Compatible Proxy for Thomson Reuters AI Platform"}


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Run with: uvicorn server:app --reload --host 0.0.0.0 --port 8082")
        sys.exit(0)

    # Configure uvicorn to run with minimal logs
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="error")