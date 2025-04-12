"""
AI Platform API integration module.
Handles direct interaction with Vertex AI.
"""

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

import vertexai
from vertexai.generative_models import GenerativeModel

from src.authenticator import AuthenticationError, get_gemini_credentials

logger = logging.getLogger(__name__)


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
