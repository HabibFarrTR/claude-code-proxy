import json
import logging
import sys
import time
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from src.api import aiplatform_client
from src.models import MessagesRequest, MessagesResponse, TokenCountRequest, TokenCountResponse, Usage
from src.streaming import handle_streaming
from src.utils import log_request_beautifully, parse_tool_result_content

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

app = FastAPI()

# Initialize AI Platform client
aiplatform_client.initialize()


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
                    }
                },
                status_code=200
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
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
            logger.info(f"Returning streaming response with Anthropic-compatible headers")
            # Make sure both Content-Type header and media_type are set to text/event-stream
            return StreamingResponse(
                handle_streaming(response_generator, request), 
                media_type="text/event-stream",
                headers=headers
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

            # Import the response conversion function
            from src.models import MessagesResponse, Usage

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
