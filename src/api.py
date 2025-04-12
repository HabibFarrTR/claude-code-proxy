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

                # Convert to formatted response
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": response.text},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt) // 4,  # Rough estimation
                        "completion_tokens": len(response.text) // 4,  # Rough estimation
                        "total_tokens": (len(prompt) + len(response.text)) // 4,  # Rough estimation
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

            # Get the response text
            response_text = response.text

            # Simulate Anthropic streaming format for the handle_streaming function

            # 1. Generate message_start event
            message_id = f"msg_{uuid.uuid4().hex[:24]}"
            yield {
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

            # 2. Generate content_block_start event
            block_id = 0
            yield {
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

            # 3. Generate ping event (common in Anthropic streams)
            yield {
                "choices": [{"delta": {"anthropic_event": "ping"}, "index": 0, "finish_reason": None}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }

            # 4. Stream the response in small chunks with content_block_delta events
            # For non-empty responses, stream the text in small chunks to simulate streaming
            if response_text.strip():
                # Stream in smaller chunks to better simulate streaming
                chunk_size = 10  # Smaller chunks look more like real streaming
                for i in range(0, len(response_text), chunk_size):
                    chunk = response_text[i : i + chunk_size]
                    yield {
                        "choices": [
                            {
                                "delta": {
                                    "anthropic_event": "content_block_delta",
                                    "index": block_id,
                                    "delta": {"type": "text_delta", "text": chunk},
                                },
                                "index": 0,
                                "finish_reason": None,
                            }
                        ],
                        "created": int(time.time()),
                        "model": model_name,
                        "object": "chat.completion.chunk",
                    }

                    # Small delay for natural feeling
                    await asyncio.sleep(0.02)
            else:
                # If empty response, still emit a content_block_delta
                yield {
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

            # 5. Generate content_block_stop event
            yield {
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

            # 6. Generate message_delta event
            yield {
                "choices": [
                    {
                        "delta": {
                            "anthropic_event": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"output_tokens": len(response_text) // 4},
                        },
                        "index": 0,
                        "finish_reason": None,
                    }
                ],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }

            # 7. Generate message_stop event
            yield {
                "choices": [{"delta": {"anthropic_event": "message_stop"}, "index": 0, "finish_reason": "stop"}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }
        except Exception as e:
            logger.error(f"Error in streaming generation: {e}", exc_info=True)
            # If there's an error, still try to emit some events to avoid hanging clients
            yield {
                "choices": [{"delta": {"content": f"Error: {str(e)}"}, "index": 0, "finish_reason": "stop"}],
                "created": int(time.time()),
                "model": model_name,
                "object": "chat.completion.chunk",
            }


# Create a singleton instance
aiplatform_client = AIPlatformClient()
