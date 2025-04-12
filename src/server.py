import asyncio
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

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
    ) -> Dict[str, Any]:
        """
        Send a completion request to AI Platform.

        Args:
            model_name: The AI Platform model name without the prefix
            messages: List of message objects with role and content
            system_message: Optional system message to prepend
            stream: Whether to stream the response

        Returns:
            The response from AI Platform in a format compatible with our processing
        """
        self.ensure_initialized()

        try:
            logger.info(f"Direct Vertex AI request for model: {model_name}")

            # Create the Vertex AI model
            model = GenerativeModel(model_name)

            # Start a chat session
            chat = model.start_chat(response_validation=False)

            # Process the messages to build the conversation history
            history = []

            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")

                # Build conversation history
                if role in ["user", "assistant"]:
                    if isinstance(content, str):
                        history.append({"role": role, "content": content})
                    else:
                        # Convert complex content to text
                        if isinstance(content, list):
                            text_content = ""
                            for item in content:
                                if isinstance(item, dict):
                                    if item.get("type") == "text":
                                        text_content += item.get("text", "") + "\n"
                            history.append({"role": role, "content": text_content.strip()})

            # Find the last user message which we'll send to the model
            last_user_msg = None
            for msg in reversed(history):
                if msg["role"] == "user":
                    last_user_msg = msg["content"]
                    break

            if not last_user_msg:
                raise ValueError("No user message found in the conversation")

            # Construct the complete prompt with system message if available
            prompt = last_user_msg
            if system_message:
                prompt = f"{system_message}\n\n{prompt}"

            # If streaming is enabled
            if stream:
                return self._stream_response(model_name, chat, prompt)
            else:
                # For non-streaming requests
                response = chat.send_message(prompt)

                # Extract response content (text or executable code)
                response_text = ""
                try:
                    response_text = response.text
                except (AttributeError, ValueError) as e:
                    logger.warning(f"Could not get text from response: {e}")

                    # Log detailed diagnostics about the response structure
                    if hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]
                        part = None
                        if (
                            hasattr(candidate, "content")
                            and candidate.content
                            and hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            part = candidate.content.parts[0]

                        # Try extracting raw format details
                        part_str = str(part) if part else "No parts available"
                        candidate_str = str(candidate) if candidate else "No candidate available"
                        response_str = str(response) if response else "No response available"

                        logger.info("Received response with executable code. Processing format.")
                        logger.debug(
                            f"Response structure - Part: {part_str}, Candidate: {candidate_str}, Response: {response_str}"
                        )

                    # Check if it's executable code - handle both dictionary and object access patterns
                    if hasattr(response, "candidates") and response.candidates:
                        candidate = response.candidates[0]

                        # Try to extract from raw response if available
                        if hasattr(response, "_raw_response"):
                            try:
                                raw_str = str(response._raw_response)
                                # Extract code with basic string parsing for prototype
                                if "executable_code" in raw_str and "code:" in raw_str:
                                    try:
                                        code_start = raw_str.find("code:") + 6  # Skip "code: "
                                        code_end = raw_str.find('"', code_start + 1)
                                        while raw_str[code_end - 1] == "\\":  # Handle escaped quotes
                                            code_end = raw_str.find('"', code_end + 1)
                                        code = raw_str[code_start:code_end]
                                        # Unescape the code
                                        code = code.replace('\\"', '"').replace("\\n", "\n")

                                        language_start = raw_str.find("language:") + 10
                                        language_end = raw_str.find("\n", language_start)
                                        language = raw_str[language_start:language_end].strip()

                                        response_text = f"```{language.lower()}\n{code}\n```"
                                        logger.info(f"Successfully extracted code: {code[:50]}... as {language}")
                                    except Exception as e:
                                        logger.warning(f"Error parsing raw executable code: {e}")
                            except Exception as e:
                                logger.warning(f"Error accessing or processing raw response: {e}")

                        # If we couldn't extract from raw, try the object approach or direct string parsing from part
                        if not response_text:
                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content
                                if hasattr(content, "parts") and content.parts:
                                    for part in content.parts:
                                        # First try direct attribute access
                                        if hasattr(part, "executable_code") and part.executable_code:
                                            try:
                                                code = part.executable_code.code
                                                language = part.executable_code.language
                                                response_text = f"```{language.lower()}\n{code}\n```"
                                                logger.info(
                                                    f"Successfully extracted code: {code[:50]}... as {language}"
                                                )
                                                break
                                            except Exception as e:
                                                logger.warning(f"Error accessing executable_code properties: {e}")

                                        # If direct access fails, try string parsing from part representation
                                        if not response_text:
                                            part_str = str(part)
                                            if "executable_code" in part_str and "code:" in part_str:
                                                try:
                                                    code_start = part_str.find("code:") + 6
                                                    code_end = part_str.find('"', code_start + 1)
                                                    while part_str[code_end - 1] == "\\":
                                                        code_end = part_str.find('"', code_end + 1)
                                                    code = part_str[code_start:code_end]
                                                    code = code.replace('\\"', '"').replace("\\n", "\n")

                                                    language_start = part_str.find("language:") + 10
                                                    language_end = part_str.find("\n", language_start)
                                                    language = part_str[language_start:language_end].strip()

                                                    response_text = f"```{language.lower()}\n{code}\n```"
                                                    logger.info(
                                                        f"Successfully extracted code: {code[:50]}... as {language}"
                                                    )
                                                    break
                                                except Exception as e:
                                                    logger.warning(f"Error parsing part string for code: {e}")
                    # If we still couldn't extract content, use a fallback message
                    if not response_text:
                        response_text = (
                            "I attempted to use a tool, but this feature is not fully supported in this environment."
                        )

                # Convert to formatted response
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response_text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt) // 4,  # Rough estimation
                        "completion_tokens": len(response_text) // 4,  # Rough estimation
                        "total_tokens": (len(prompt) + len(response_text)) // 4,  # Rough estimation
                    },
                }

        except Exception as e:
            logger.error(f"Error in direct Vertex AI integration: {e}", exc_info=True)
            raise

    async def _stream_response(self, model_name: str, chat, prompt: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate streaming responses in Anthropic-compatible format.

        Args:
            model_name: The AI Platform model name
            chat: The VertexAI chat session
            prompt: The prompt to send

        Returns:
            An async generator yielding response chunks
        """
        try:
            response = chat.send_message(prompt)

            # Get the response content
            # Handle different response types (text or executable code)
            response_text = ""
            try:
                response_text = response.text
            except (AttributeError, ValueError) as e:
                logger.warning(f"Could not get text from response: {e}")

                # Log detailed diagnostics about the response structure
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    part = None
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and hasattr(candidate.content, "parts")
                        and candidate.content.parts
                    ):
                        part = candidate.content.parts[0]

                    # Try extracting raw format details
                    part_str = str(part) if part else "No parts available"
                    candidate_str = str(candidate) if candidate else "No candidate available"
                    response_str = str(response) if response else "No response available"

                    logger.info("Stream received response with executable code. Processing format.")
                    logger.debug(
                        f"Stream response structure - Part: {part_str}, Candidate: {candidate_str}, Response: {response_str}"
                    )

                # Check if it's executable code - handle both dictionary and object access patterns
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]

                    # Try to extract from raw response if available
                    if hasattr(response, "_raw_response"):
                        try:
                            raw_str = str(response._raw_response)
                            # Extract code with basic string parsing for prototype
                            if "executable_code" in raw_str and "code:" in raw_str:
                                try:
                                    code_start = raw_str.find("code:") + 6  # Skip "code: "
                                    code_end = raw_str.find('"', code_start + 1)
                                    while raw_str[code_end - 1] == "\\":  # Handle escaped quotes
                                        code_end = raw_str.find('"', code_end + 1)
                                    code = raw_str[code_start:code_end]
                                    # Unescape the code
                                    code = code.replace('\\"', '"').replace("\\n", "\n")

                                    language_start = raw_str.find("language:") + 10
                                    language_end = raw_str.find("\n", language_start)
                                    language = raw_str[language_start:language_end].strip()

                                    response_text = f"```{language.lower()}\n{code}\n```"
                                    logger.info(f"Successfully extracted code: {code[:50]}... as {language}")
                                except Exception as e:
                                    logger.warning(f"Error parsing raw executable code: {e}")
                        except Exception as e:
                            logger.warning(f"Error accessing or processing raw response: {e}")

                    # If we couldn't extract from raw, try the object approach or direct string parsing from part
                    if not response_text:
                        if hasattr(candidate, "content") and candidate.content:
                            content = candidate.content
                            if hasattr(content, "parts") and content.parts:
                                for part in content.parts:
                                    # First try direct attribute access
                                    if hasattr(part, "executable_code") and part.executable_code:
                                        try:
                                            code = part.executable_code.code
                                            language = part.executable_code.language
                                            response_text = f"```{language.lower()}\n{code}\n```"
                                            logger.info(f"Successfully extracted code: {code[:50]}... as {language}")
                                            break
                                        except Exception as e:
                                            logger.warning(f"Error accessing executable_code properties: {e}")

                                    # If direct access fails, try string parsing from part representation
                                    if not response_text:
                                        part_str = str(part)
                                        if "executable_code" in part_str and "code:" in part_str:
                                            try:
                                                code_start = part_str.find("code:") + 6
                                                code_end = part_str.find('"', code_start + 1)
                                                while part_str[code_end - 1] == "\\":
                                                    code_end = part_str.find('"', code_end + 1)
                                                code = part_str[code_start:code_end]
                                                code = code.replace('\\"', '"').replace("\\n", "\n")

                                                language_start = part_str.find("language:") + 10
                                                language_end = part_str.find("\n", language_start)
                                                language = part_str[language_start:language_end].strip()

                                                response_text = f"```{language.lower()}\n{code}\n```"
                                                logger.info(
                                                    f"Successfully extracted code: {code[:50]}... as {language}"
                                                )
                                                break
                                            except Exception as e:
                                                logger.warning(f"Error parsing part string for code: {e}")
                # If we still couldn't extract content, use a fallback message
                if not response_text:
                    response_text = (
                        "I attempted to use a tool, but this feature is not fully supported in this environment."
                    )

            # Simulate Anthropic streaming format for the handle_streaming function
            logger.info("Beginning to yield streaming response chunks")

            # 1. Generate message_start event
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            chunk = {
                "choices": [
                    {
                        "delta": {
                            "anthropic_event": "message_start",
                            "message": {
                                "id": message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": model_name,
                            },
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding message_start chunk: {chunk}")
            yield chunk

            # 2. Generate content_block_start event
            block_id = 0
            block_chunk = {
                "choices": [
                    {
                        "delta": {
                            "anthropic_event": "content_block_start",
                            "index": block_id,
                            "content_block": {"type": "text", "text": ""},
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding content_block_start chunk: {block_chunk}")
            yield block_chunk

            # 3. Generate ping event (common in Anthropic streams)
            ping_chunk = {
                "choices": [{"delta": {"anthropic_event": "ping"}, "index": 0, "finish_reason": None}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding ping chunk: {ping_chunk}")
            yield ping_chunk

            # 4. Stream the response in small chunks with content_block_delta events
            # Add debugging to see response content
            logger.info(f"Preparing to stream response. Text content: '{response_text}'")

            # For non-empty responses, stream the text in small chunks to simulate streaming
            if response_text.strip():
                logger.info(f"Streaming non-empty response text ({len(response_text)} chars) in chunks")
                # Stream in smaller chunks to better simulate streaming
                chunk_size = 10  # Smaller chunks look more like real streaming
                for i in range(0, len(response_text), chunk_size):
                    text_chunk = response_text[i : i + chunk_size]
                    delta_chunk = {
                        "choices": [
                            {
                                "delta": {
                                    "anthropic_event": "content_block_delta",
                                    "index": block_id,
                                    "delta": {"type": "text_delta", "text": text_chunk},
                                },
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                        "created": int(time.time()),
                        "model": model_name,
                        "object": "chat.completion.chunk",
                    }
                    logger.info(
                        f"Yielding content chunk {i//chunk_size+1}/{(len(response_text)//chunk_size)+1}: '{text_chunk}'"
                    )
                    yield delta_chunk

                    # Small delay for natural feeling
                    await asyncio.sleep(0.02)
            else:
                logger.info("Empty response text, sending fallback content block")
                # If empty response, still emit a content_block_delta
                empty_chunk = {
                    "choices": [
                        {
                            "delta": {
                                "anthropic_event": "content_block_delta",
                                "index": block_id,
                                "delta": {"type": "text_delta", "text": "No response text available."},
                            },
                            "index": 0,
                            "finish_reason": None,
                        }
                    ],
                    "created": int(time.time()),
                    "model": model_name,
                    "object": "chat.completion.chunk",
                }
                logger.info(f"Yielding empty content chunk: {empty_chunk}")
                yield empty_chunk

            # 5. Generate content_block_stop event
            logger.info("Sending content_block_stop event")
            content_stop_chunk = {
                "choices": [
                    {
                        "delta": {"anthropic_event": "content_block_stop", "index": block_id},
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding content_block_stop chunk: {content_stop_chunk}")
            yield content_stop_chunk

            # 6. Generate message_delta event
            logger.info("Sending message_delta event with stop_reason")
            output_tokens = len(response_text) // 4
            message_delta_chunk = {
                "choices": [
                    {
                        "delta": {
                            "anthropic_event": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": output_tokens},
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding message_delta chunk with {output_tokens} tokens: {message_delta_chunk}")
            yield message_delta_chunk

            # 7. Generate message_stop event
            logger.info("Sending message_stop event")
            message_stop_chunk = {
                "choices": [{"delta": {"anthropic_event": "message_stop"}, "index": 0, "finish_reason": "stop"}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.info(f"Yielding message_stop chunk: {message_stop_chunk}")
            yield message_stop_chunk
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            # If there's an error, still try to emit some events to avoid hanging clients
            error_chunk = {
                "choices": [{"delta": {"content": f"Error: {str(e)}"}, "index": 0, "finish_reason": "stop"}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
            logger.error(f"Yielding error chunk: {error_chunk}")
            yield error_chunk

            # Even with an error, try to properly close the stream with a [DONE] marker
            logger.info("Sending [DONE] marker after error")
            yield "data: [DONE]\n\n"


# Create a singleton instance
aiplatform_client = AIPlatformClient()

# Initialize AI Platform client
aiplatform_client.initialize()


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


async def handle_streaming(response_generator, original_request):
    """
    Handle streaming responses and convert to Anthropic format.

    Takes a generator that yields response chunks and converts them to
    Server-Sent Events (SSE) in Anthropic's format.

    Args:
        response_generator: An async generator yielding response chunks
        original_request: The original request object

    Returns:
        An async generator yielding SSE formatted events
    """
    logger.info("Beginning to yield streaming response chunks")
    try:
        # Send message_start event
        message_id = f"msg_{uuid.uuid4().hex[:24]}"  # Format similar to Anthropic's IDs

        message_data = {
            "choices": [
                {
                    "delta": {
                        "anthropic_event": "message_start",
                        "message": {
                            "id": message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": original_request.model,
                        },
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding message_start chunk: {message_data}")
        yield f"data: {json.dumps(message_data)}\n\n"
        logger.info(f"Processing chunk: {message_data}")

        # Content block start event
        content_block_start = {
            "choices": [
                {
                    "delta": {
                        "anthropic_event": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding content_block_start chunk: {content_block_start}")
        yield f"data: {json.dumps(content_block_start)}\n\n"
        logger.info(f"Processing chunk: {content_block_start}")

        # Send a ping to keep the connection alive
        ping_data = {
            "choices": [{"delta": {"anthropic_event": "ping"}, "index": 0, "finish_reason": None}],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding ping chunk: {ping_data}")
        yield f"data: {json.dumps(ping_data)}\n\n"
        logger.info(f"Processing chunk: {ping_data}")

        # Process the streaming responses
        content_text = ""
        output_tokens = 0
        content_chunks = []

        # Process content from the AI response
        async for resp in response_generator:
            # Extract text from the AI response
            text_content = None
            try:
                # Try to get text from Vertex AI response
                if hasattr(resp, "candidates") and resp.candidates:
                    candidate = resp.candidates[0]
                    # Handle code blocks specifically for Gemini
                    if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                text_content = part.text
                            elif hasattr(part, "executable_code") and part.executable_code:
                                # Extract code as text
                                logger.info("Stream received response with executable code. Processing format.")
                                code = part.executable_code.code
                                language = part.executable_code.language.lower()
                                logger.info(f'Successfully extracted code: "{code[:20]}... as {language}')
                                text_content = f"```{language}\n{code}\n```"

                # If we couldn't get text directly, log an error
                if text_content is None and hasattr(resp, "candidates"):
                    logger.warning(f"Could not get text from response: {resp}")
            except Exception as ex:
                logger.error(f"Error extracting text from response: {ex}")
                continue

            # If we found text content, stream it in chunks
            if text_content:
                logger.info(f"Preparing to stream response. Text content: '{text_content}'")
                content_text += text_content

                # Stream the content in reasonable-sized chunks
                chunk_size = 10  # Characters per chunk
                num_chunks = (len(text_content) + chunk_size - 1) // chunk_size
                logger.info(f"Streaming non-empty response text ({len(text_content)} chars) in chunks")

                for i in range(num_chunks):
                    start = i * chunk_size
                    end = min(start + chunk_size, len(text_content))
                    text_chunk = text_content[start:end]
                    content_chunks.append(text_chunk)

                    # Create the content block delta event
                    content_delta = {
                        "choices": [
                            {
                                "delta": {
                                    "anthropic_event": "content_block_delta",
                                    "index": 0,
                                    "delta": {"type": "text_delta", "text": text_chunk},
                                },
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                        "created": int(uuid.uuid1().time // 1000),
                        "model": original_request.model,
                        "object": "chat.completion.chunk",
                    }

                    logger.info(f"Yielding content chunk {i+1}/{num_chunks}: '{text_chunk}'")
                    yield f"data: {json.dumps(content_delta)}\n\n"
                    logger.info(f"Processing chunk: {content_delta}")

        # Calculate output tokens from the content length (rough approximation)
        output_tokens = max(1, len(content_text) // 4)

        # Send content_block_stop event
        logger.info("Sending content_block_stop event")
        content_block_stop = {
            "choices": [
                {"delta": {"anthropic_event": "content_block_stop", "index": 0}, "index": 0, "finish_reason": None}
            ],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding content_block_stop chunk: {content_block_stop}")
        yield f"data: {json.dumps(content_block_stop)}\n\n"
        logger.info(f"Processing chunk: {content_block_stop}")

        # Send message_delta with usage
        logger.info("Sending message_delta event with stop_reason")
        message_delta = {
            "choices": [
                {
                    "delta": {
                        "anthropic_event": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": output_tokens},
                    },
                    "index": 0,
                    "finish_reason": None,
                }
            ],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding message_delta chunk with {output_tokens} tokens: {message_delta}")
        yield f"data: {json.dumps(message_delta)}\n\n"
        logger.info(f"Processing chunk: {message_delta}")

        # Send message_stop event
        logger.info("Sending message_stop event")
        message_stop = {
            "choices": [{"delta": {"anthropic_event": "message_stop"}, "index": 0, "finish_reason": "stop"}],
            "created": int(uuid.uuid1().time // 1000),
            "model": original_request.model,
            "object": "chat.completion.chunk",
        }
        logger.info(f"Yielding message_stop chunk: {message_stop}")
        yield f"data: {json.dumps(message_stop)}\n\n"
        logger.info(f"Processing chunk: {message_stop}")

        # IMPORTANT: Send the final [DONE] marker exactly as shown here - critical for Claude client
        logger.info("Sending [DONE] marker at end of stream - normal completion")
        yield "data: [DONE]\n\n"

    except Exception as e:
        # In case of any unexpected error
        logger.error(f"Error in handle_streaming: {e}")
        # Send [DONE] marker to gracefully close the connection
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

        # IMPORTANT: Client protection - Claude Code sends multiple requests for different models
        # We need to prioritize one model to avoid confusing the client
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        # Skip processing for haiku model streams from the same client to prevent conflicts
        if "haiku" in original_model.lower() and body_json.get("stream", False):
            logger.info(f"⏭️ SKIPPING STREAMING REQUEST FOR HAIKU MODEL: Client={client_ip}")

            # Return a simple empty response that satisfies the client without conflicting streams
            return JSONResponse(
                content={
                    "id": f"msg_{uuid.uuid4()}",
                    "model": original_model,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": ""}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 0,
                    },
                },
                status_code=200,
            )

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

        # IMPORTANT: Client protection - only process requests for the primary model
        # Claude client sends multiple requests but we only need to process the main one
        client_ip = raw_request.client.host if raw_request.client else "unknown"

        # For Claude clients, only process the Sonnet model
        # Skip processing for Haiku models from the same client
        if "haiku" in display_model.lower() and request.stream:
            # Instead of processing the haiku model, return an empty response
            # This allows the client to focus on a single stream
            logger.info(f"⏭️ SKIPPING STREAMING REQUEST FOR HAIKU MODEL: Client={client_ip}")

            # Create a MessagesResponse that satisfies the client but doesn't do real work
            empty_response = MessagesResponse(
                id=f"msg_{uuid.uuid4()}",
                model=request.model,
                role="assistant",
                content=[{"type": "text", "text": ""}],
                stop_reason="end_turn",
                stop_sequence=None,
                usage=Usage(
                    input_tokens=1,
                    output_tokens=0,
                ),
            )

            # Return a non-streaming response to avoid conflicts
            return empty_response

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
            logger.info("Using direct AI Platform integration for streaming")
            # Get streaming response from AI Platform
            response_generator = await aiplatform_client.completion(
                model_name=clean_model_name, messages=messages, system_message=system_message, stream=True
            )

            # Convert to Anthropic streaming format with EXACT matching headers
            # Claude client expects these exact headers
            headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"}
            logger.info("Returning streaming response with Anthropic-compatible headers")
            # Make sure both Content-Type header and media_type are set to text/event-stream
            return StreamingResponse(
                handle_streaming(response_generator, request), media_type="text/event-stream", headers=headers
            )
        else:
            # For non-streaming requests
            logger.info("Using direct AI Platform integration for completion")
            start_time = time.time()

            # Get response from AI Platform
            response = await aiplatform_client.completion(
                model_name=clean_model_name, messages=messages, system_message=system_message, stream=False
            )

            logger.debug(f"✅ RESPONSE RECEIVED: Model={clean_model_name}, Time={time.time() - start_time:.2f}s")

            # Extract key information from response
            response_id = response.get("id", f"msg_{uuid.uuid4()}")
            choices = response.get("choices", [{}])
            message = choices[0].get("message", {}) if choices else {}
            content_text = message.get("content", "")
            usage_info = response.get("usage", {})

            # Create Anthropic-compatible response
            anthropic_response = MessagesResponse(
                id=response_id,
                model=request.model,
                role="assistant",
                content=[{"type": "text", "text": content_text}],
                stop_reason="end_turn",
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
