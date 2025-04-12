"""
Streaming handler for Anthropic API responses.
"""

import json
import logging
import uuid

logger = logging.getLogger(__name__)


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
