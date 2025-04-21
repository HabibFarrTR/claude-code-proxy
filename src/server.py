"""FastAPI application implementing Anthropic-compatible API endpoints using Vertex AI Gemini models.

This server acts as a proxy between Anthropic Claude clients and Google's Vertex AI, enabling
seamless integration with Thomson Reuters AI Platform's Gemini models. The server provides:

1. Authentication with Thomson Reuters AI Platform for Vertex AI access
2. Model mapping from Anthropic model names to appropriate Gemini models
3. Request/response format conversion between Anthropic and Vertex AI formats
4. Support for both streaming and non-streaming responses
5. Tool/function calling capability with cross-API conversion
6. Token counting endpoint compatible with Anthropic's API

The main endpoints include:
- POST /v1/messages: For chat completions with Gemini models
- POST /v1/messages/count_tokens: For token counting
- GET /: Service health and information

The implementation uses the native Vertex AI SDK for direct model access, with
custom authentication and error handling designed for high reliability.
"""

import asyncio
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager

import google.api_core.exceptions  # To catch API call errors
import google.auth  # To catch auth errors during vertexai.init
import litellm  # Still used for format definitions, fallback token counting
import uvicorn
import vertexai
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from vertexai.generative_models import (
    FinishReason,
    GenerationConfig,
    GenerativeModel,
    Part,
)

from src.authenticator import AuthenticationError, CredentialManager
from src.config import (
    GEMINI_BIG_MODEL,
    GEMINI_SMALL_MODEL,
    OVERRIDE_TEMPERATURE,
    TEMPERATURE_OVERRIDE,
    TOOL_CALL_TEMPERATURE_OVERRIDE,
)
from src.converters import (
    adapt_vertex_stream_to_litellm,
    convert_anthropic_to_litellm,
    convert_litellm_messages_to_vertex_content,
    convert_litellm_to_anthropic,
    convert_litellm_to_anthropic_sse,
    convert_litellm_tools_to_vertex_tools,
    convert_vertex_response_to_litellm,
)
from src.models import MessagesRequest, TokenCountRequest, TokenCountResponse
from src.utils import Colors, get_logger, log_request_beautifully

logger = get_logger()


# Disable LiteLLM's default logging behavior
litellm.success_callback = []
litellm.failure_callback = []

# Configure uvicorn and other libraries to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.auth.compute_engine._metadata").setLevel(logging.WARNING)
logging.getLogger("google.api_core.bidi").setLevel(logging.WARNING)
logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING)  # Quieten Vertex SDK logs if needed


# Create a credential manager instance for the application
credential_manager = CredentialManager()


# Define application lifespan using the contextmanager-based approach
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for the application.

    Manages the application startup and shutdown events:
    - On startup: Initialize credential manager and Vertex AI SDK
    - On shutdown: Clean up any resources
    """
    # --- Startup actions ---
    logger.info("Application startup: Initializing credential manager...")
    try:
        # Initialize credentials and SDK - one time on startup
        success = await credential_manager.initialize()
        if success:
            logger.info("Application startup complete: Credential manager initialized successfully.")
        else:
            logger.error("Application startup issue: Failed to initialize credential manager.")
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
        # We don't exit here - the credential manager will retry automatically

    # Yield control back to FastAPI to run the application
    yield

    # --- Shutdown actions ---
    logger.info("Application shutting down: Cleaning up resources...")
    # No specific cleanup needed for credential manager, but we could add if necessary


# Create the FastAPI application with the lifespan context manager
app = FastAPI(title="Anthropic to Custom Gemini Proxy (Native SDK Call)", lifespan=lifespan)


@app.post("/v1/messages", response_model=None)  # response_model=None for StreamingResponse
async def create_message(request_data: MessagesRequest, raw_request: Request):
    """
    Process chat completion requests in Anthropic format using Vertex AI Gemini models.

    This endpoint handles the core chat completion functionality, supporting both streaming
    and non-streaming responses. It performs multi-stage conversion between Anthropic and
    Vertex AI formats, with custom authentication and error handling.

    Args:
        request_data (MessagesRequest): Pydantic model containing the structured request data
            including messages, system prompt, model name, and generation parameters
        raw_request (Request): FastAPI Request object providing access to the raw HTTP request,
            used for extracting additional details and headers

    Returns:
        StreamingResponse: For streaming requests, returns SSE stream compatible with Anthropic's API
        JSONResponse: For non-streaming requests, returns a standard JSON response in Anthropic format

    Raises:
        HTTPException: For authentication, validation, or upstream API failures with appropriate status codes

    Note:
        This endpoint handles model mapping, temperature adjustments for tool calls, and retry logic
        for certain error conditions. It includes comprehensive logging for debugging and monitoring.
    """
    # Use request ID from middleware
    request_id = raw_request.state.request_id
    start_time = time.time()

    try:
        # --- Use Validated Request Data ---
        # We now get everything directly from the Pydantic model which has been validated
        # No need to parse the raw request body again
        original_model_name = request_data.original_model_name
        actual_gemini_model_id = request_data.model

        logger.info(
            f"[{request_id}] Processing '/v1/messages': Original='{original_model_name}', Target SDK Model='{actual_gemini_model_id}', Stream={request_data.stream}"
        )

        # --- Get credentials from central credential manager ---
        try:
            project_id, location, temp_creds = await credential_manager.get_credentials()
            logger.debug(
                f"[{request_id}] Using centrally managed credentials. Project: {project_id}, Location: {location}"
            )
        except AuthenticationError as e:
            logger.error(f"[{request_id}] Credential manager failed to provide credentials: {e}")
            raise HTTPException(status_code=503, detail=f"Authentication Service Error: {e}")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error getting credentials: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected credential error")

        # --- Convert Anthropic Request -> Intermediate LiteLLM/OpenAI Format ---
        litellm_request_dict = convert_anthropic_to_litellm(request_data)
        litellm_messages = litellm_request_dict.get("messages", [])
        litellm_tools = litellm_request_dict.get("tools")  # Tools in OpenAI format
        system_prompt_text = litellm_request_dict.get("system_prompt")  # Extracted system prompt

        # --- Convert Intermediate Format -> Vertex AI SDK Format ---
        vertex_history = convert_litellm_messages_to_vertex_content(litellm_messages)
        vertex_tools = convert_litellm_tools_to_vertex_tools(litellm_tools)  # Will be None if no tools
        vertex_system_instruction = Part.from_text(system_prompt_text) if system_prompt_text else None

        # --- Prepare Generation Config for Vertex AI ---
        # *** MODIFICATION START: Use configured override temp ***
        effective_temperature = request_data.temperature

        if OVERRIDE_TEMPERATURE:
            if vertex_tools is not None:
                if effective_temperature is None:
                    # Override temperature if tools are present and no temp was provided
                    effective_temperature = TOOL_CALL_TEMPERATURE_OVERRIDE
                    logger.info(
                        f"[{request_id}] Tools are present. Defaulting temperature to configured override: {TOOL_CALL_TEMPERATURE_OVERRIDE}"
                    )
                elif effective_temperature > TOOL_CALL_TEMPERATURE_OVERRIDE:
                    effective_temperature = TOOL_CALL_TEMPERATURE_OVERRIDE
                    logger.info(
                        f"[{request_id}] Tools are present. Overriding requested temperature {effective_temperature} to configured override: {TOOL_CALL_TEMPERATURE_OVERRIDE}"
                    )
            else:
                if effective_temperature is None:
                    # Override temperature if no tools and no temp was provided
                    effective_temperature = TEMPERATURE_OVERRIDE
                    logger.info(
                        f"[{request_id}] No tools present. Defaulting temperature to configured override: {TEMPERATURE_OVERRIDE}"
                    )
                elif effective_temperature > TEMPERATURE_OVERRIDE:
                    effective_temperature = TEMPERATURE_OVERRIDE
                    logger.info(
                        f"[{request_id}] No tools present. Overriding requested temperature {effective_temperature} to configured override: {TEMPERATURE_OVERRIDE}"
                    )

        generation_config = GenerationConfig(
            max_output_tokens=request_data.max_tokens,
            temperature=effective_temperature,  # Use the potentially adjusted temperature
            top_p=request_data.top_p,
            top_k=request_data.top_k,
            stop_sequences=request_data.stop_sequences if request_data.stop_sequences else None,
        )
        # *** MODIFICATION END ***
        logger.debug(f"[{request_id}] Vertex GenerationConfig: {generation_config}")  # Log the config being used

        safety_settings = None

        # --- Tool Config ---
        intermediate_tool_choice = litellm_request_dict.get("tool_choice")
        if intermediate_tool_choice == "none":
            if vertex_tools:
                logger.warning(f"[{request_id}] Tool choice is 'none', but tools were provided. Ignoring tools.")
                vertex_tools = None
        elif intermediate_tool_choice == "auto":
            pass
        elif isinstance(intermediate_tool_choice, dict) and intermediate_tool_choice.get("type") == "function":
            forced_tool_name = intermediate_tool_choice.get("function", {}).get("name")
            logger.warning(
                f"[{request_id}] Forcing specific tool '{forced_tool_name}' not fully implemented for Vertex SDK. Proceeding with auto tool choice."
            )

        # Log request details before calling API
        num_vertex_content = len(vertex_history) if vertex_history else 0
        num_vertex_tools = len(vertex_tools) if vertex_tools else 0
        log_request_beautifully(
            raw_request.method,
            raw_request.url.path,
            original_model_name,
            actual_gemini_model_id,
            num_vertex_content,
            num_vertex_tools,
            200,
        )

        # --- Instantiate Vertex AI Model ---
        model = GenerativeModel(actual_gemini_model_id, system_instruction=vertex_system_instruction)

        # --- Call Native Vertex AI SDK ---
        if request_data.stream:
            # --- Streaming Call ---
            logger.info(
                f"[{request_id}] Calling Vertex AI generate_content_async (streaming) with effective_temperature={effective_temperature}"
            )  # Log temp used
            try:
                vertex_stream_generator = await model.generate_content_async(
                    contents=vertex_history,
                    generation_config=generation_config,  # Pass the potentially modified config
                    safety_settings=safety_settings,
                    tools=vertex_tools,
                    # tool_config=vertex_tool_config,
                    stream=True,
                )
            # ... (rest of streaming try/except block remains the same) ...
            except google.api_core.exceptions.InvalidArgument as e:
                logger.error(
                    f"[{request_id}] Vertex API Invalid Argument Error (Check Request Structure/Schema): {e}",
                    exc_info=True,
                )
                raise HTTPException(status_code=400, detail=f"Upstream API Invalid Argument: {e.message or str(e)}")
            except google.api_core.exceptions.GoogleAPICallError as e:
                logger.error(f"[{request_id}] Vertex API Call Error (Streaming): {e}", exc_info=True)
                http_status = getattr(e, "code", 502)  # Default to 502 Bad Gateway
                raise HTTPException(
                    status_code=http_status, detail=f"Upstream API Error (Streaming): {e.message or str(e)}"
                )

            adapted_stream = adapt_vertex_stream_to_litellm(vertex_stream_generator, request_id, actual_gemini_model_id)
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Mapped-Model": actual_gemini_model_id,
                "X-Request-ID": request_id,
            }
            return StreamingResponse(
                convert_litellm_to_anthropic_sse(adapted_stream, request_data, request_id),
                media_type="text/event-stream",
                headers=headers,
            )

        else:
            # --- Non-Streaming Call ---
            logger.info(
                f"[{request_id}] Calling Vertex AI generate_content_async (non-streaming) with effective_temperature={effective_temperature}"
            )  # Log temp used
            # *** Using the retry logic proposed previously for non-streaming ***
            max_retries = 1
            retries = 0
            vertex_response = None
            last_error = None

            while retries <= max_retries:
                logger.info(f"[{request_id}] Non-streaming attempt {retries + 1}/{max_retries + 1}")
                try:
                    vertex_response = await model.generate_content_async(
                        contents=vertex_history,
                        generation_config=generation_config,  # Pass the potentially modified config
                        safety_settings=safety_settings,
                        tools=vertex_tools,
                        # tool_config=vertex_tool_config,
                        stream=False,
                    )
                    logger.debug(
                        f"[{request_id}] Raw Vertex AI Non-Streaming Response (Attempt {retries + 1}): {vertex_response}"
                    )

                    is_malformed = False
                    if vertex_response.candidates:
                        candidate = vertex_response.candidates[0]
                        if candidate.finish_reason == FinishReason.MALFORMED_FUNCTION_CALL:
                            is_malformed = True
                            last_error = f"Malformed function call detected by upstream API. Message: {getattr(candidate, 'finish_message', 'N/A')}"
                            logger.warning(f"[{request_id}] {last_error} (Attempt {retries + 1})")

                    if is_malformed and retries < max_retries:
                        retries += 1
                        logger.info(f"[{request_id}] Retrying non-streaming call due to malformed function call...")
                        await asyncio.sleep(1)
                        continue
                    else:
                        break  # Exit loop on success, non-retryable error, or max retries reached

                except google.api_core.exceptions.InvalidArgument as e:
                    last_error = e
                    logger.error(
                        f"[{request_id}] Vertex API Invalid Argument Error (Non-Streaming Attempt {retries + 1}): {e}",
                        exc_info=True,
                    )
                    break
                except google.api_core.exceptions.GoogleAPICallError as e:
                    last_error = e
                    logger.error(
                        f"[{request_id}] Vertex API Call Error (Non-Streaming Attempt {retries + 1}): {e}",
                        exc_info=True,
                    )
                    break
                except Exception as e:
                    last_error = e
                    logger.critical(
                        f"[{request_id}] Unexpected error during non-streaming call attempt {retries + 1}: {e}",
                        exc_info=True,
                    )
                    break

            if vertex_response is None or (
                vertex_response.candidates
                and vertex_response.candidates[0].finish_reason == FinishReason.MALFORMED_FUNCTION_CALL
            ):
                error_detail = f"Upstream model failed after {retries} retries."
                if last_error:
                    error_detail += f" Last error: {str(last_error)}"
                status_code = 502
                if isinstance(last_error, google.api_core.exceptions.InvalidArgument):
                    status_code = 400
                elif isinstance(last_error, google.api_core.exceptions.GoogleAPICallError):
                    status_code = getattr(last_error, "code", 502)
                raise HTTPException(status_code=status_code, detail=error_detail)

            # --- Convert Vertex AI Response -> Intermediate LiteLLM/OpenAI Format ---
            litellm_like_response = convert_vertex_response_to_litellm(
                vertex_response, actual_gemini_model_id, request_id
            )
            # --- Convert Intermediate Format -> Final Anthropic Format ---
            anthropic_response = convert_litellm_to_anthropic(litellm_like_response, original_model_name)

            if not anthropic_response:
                logger.error(f"[{request_id}] Failed to convert final response to Anthropic format.")
                raise HTTPException(status_code=500, detail="Failed to convert response to Anthropic format")

            response = JSONResponse(content=anthropic_response.dict())
            response.headers["X-Mapped-Model"] = actual_gemini_model_id
            response.headers["X-Request-ID"] = request_id
            processing_time = time.time() - start_time
            logger.info(f"[{request_id}] Non-streaming request completed in {processing_time:.3f}s")
            return response

    # --- General Exception Handling ---
    except HTTPException as e:
        # Always use the consistent request ID from middleware
        logger.error(
            f"[{request_id}] HTTPException during '/v1/messages': Status={e.status_code}, Detail={e.detail}",
            exc_info=(e.status_code >= 500),
        )
        # Add request_id to exception detail for traceability
        if not str(e.detail).startswith(f"[{request_id}]"):
            e.detail = f"[{request_id}] {e.detail}"
        raise e
    except Exception as e:
        logger.critical(
            f"[{request_id}] Unhandled Exception during '/v1/messages': {type(e).__name__}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Internal Server Error: [{request_id}] {str(e)}")


@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens(request_data: TokenCountRequest, raw_request: Request):
    """
    Count tokens for a given message in Anthropic format using Vertex AI's native tokenizer.

    This endpoint provides accurate token counting via the Vertex AI SDK's count_tokens method,
    ensuring consistent token counts between token estimation and actual inference. It includes
    a fallback estimation method for rare failure cases.

    Args:
        request_data (TokenCountRequest): Pydantic model containing messages and system prompt
            to count tokens for, along with the target model name
        raw_request (Request): FastAPI Request object providing access to the raw HTTP request,
            used for extracting additional details and headers

    Returns:
        JSONResponse: Contains TokenCountResponse with the input_tokens count, along with
            headers indicating the mapped model and request ID

    Raises:
        HTTPException: For authentication failures or upstream API errors with appropriate
            status codes and error messages

    Note:
        The implementation uses the same credential acquisition and message format conversion
        pipeline as the main completion endpoint to ensure consistency. In extremely rare cases
        where token counting fails, it falls back to a simple character-based estimation.
    """
    # Use request ID from middleware
    request_id = raw_request.state.request_id
    start_time = time.time()
    token_count = 0  # Default
    status_code = 200  # Assume success initially

    # --- Use Validated Request Data ---
    # We now get everything directly from the Pydantic model which has been validated
    # No need to parse the raw request body again
    original_model_name = request_data.original_model_name
    actual_gemini_model_id = request_data.model  # Contains mapped ID

    logger.info(
        f"[{request_id}] Processing '/v1/messages/count_tokens': Original='{original_model_name}', Target SDK='{actual_gemini_model_id}'"
    )

    try:
        # --- Get credentials from central credential manager ---
        try:
            project_id, location, temp_creds = await credential_manager.get_credentials()
            logger.debug(
                f"[{request_id}] Using centrally managed credentials for token count. Project: {project_id}, Location: {location}"
            )
        except AuthenticationError as e:
            logger.error(f"[{request_id}] Credential manager failed to provide credentials for token count: {e}")
            status_code = 503
            raise HTTPException(status_code=status_code, detail=f"Authentication Service Error: {e}")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error getting credentials for token count: {e}", exc_info=True)
            status_code = 500
            raise HTTPException(status_code=status_code, detail=f"Credential Error: {e}")

        # --- Convert messages for counting (using intermediate format) ---
        # Need to simulate a MessagesRequest for the conversion function
        simulated_msg_request = MessagesRequest(
            model=actual_gemini_model_id,  # Pass the mapped name
            messages=request_data.messages,
            system=request_data.system,
            max_tokens=1,  # Dummy value, not used by conversion
        )
        litellm_request_dict = convert_anthropic_to_litellm(simulated_msg_request)
        litellm_messages_for_count = litellm_request_dict.get("messages", [])
        system_prompt_text = litellm_request_dict.get("system_prompt")

        vertex_history = convert_litellm_messages_to_vertex_content(litellm_messages_for_count)
        vertex_system_instruction = Part.from_text(system_prompt_text) if system_prompt_text else None

        # --- Call SDK count_tokens ---
        # SDK is already initialized by the credential manager, no need to re-initialize
        model = GenerativeModel(
            actual_gemini_model_id, system_instruction=vertex_system_instruction  # Pass system prompt here too
        )
        logger.info(f"[{request_id}] Calling Vertex AI count_tokens_async")
        count_response = await model.count_tokens_async(contents=vertex_history)
        token_count = count_response.total_tokens
        logger.info(f"[{request_id}] Vertex SDK token count successful: {token_count}")

        response = TokenCountResponse(input_tokens=token_count)
        response_headers = {"X-Mapped-Model": actual_gemini_model_id, "X-Request-ID": request_id}
        log_request_beautifully(
            raw_request.method,
            raw_request.url.path,
            original_model_name,
            actual_gemini_model_id,
            len(vertex_history),
            0,
            status_code,
        )
        return JSONResponse(content=response.dict(), headers=response_headers)

    # --- Fallback Estimation (Only if SDK call fails unexpectedly) ---
    except google.api_core.exceptions.GoogleAPICallError as e:
        logger.error(f"[{request_id}] Vertex SDK count_tokens failed: {e}", exc_info=True)
        status_code = getattr(e, "code", 502)
        # Don't fallback here, report the upstream error
        log_request_beautifully(
            raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, 0, 0, status_code
        )
        raise HTTPException(
            status_code=status_code, detail=f"[{request_id}] Upstream count_tokens error: {e.message or str(e)}"
        )
    except HTTPException as e:
        # Log and re-raise if auth/init failed
        log_request_beautifully(
            raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, 0, 0, e.status_code
        )
        # Add request_id to exception detail for traceability
        if not str(e.detail).startswith(f"[{request_id}]"):
            e.detail = f"[{request_id}] {e.detail}"
        raise e
    except Exception as e:  # Fallback to basic estimation for truly unexpected errors
        # TODO: This fallback character-based token estimation will be removed in Task 5
        # and replaced with proper error propagation
        logger.error(
            f"[{request_id}] Unexpected error during token counting, falling back to estimation: {e}", exc_info=True
        )
        prompt_text = ""
        if request_data.system:
            system_text_fallback = (
                request_data.system
                if isinstance(request_data.system, str)
                else "\n".join([b.text for b in request_data.system if b.type == "text"])
            )
            prompt_text += system_text_fallback + "\n"
        for msg in request_data.messages:
            msg_content_fallback = ""
            if isinstance(msg.content, str):
                msg_content_fallback = msg.content
            elif isinstance(msg.content, list):
                msg_content_fallback = "\n".join([b.text for b in msg.content if b.type == "text"])
            prompt_text += msg_content_fallback + "\n"

        token_estimate = len(prompt_text) // 4  # Rough estimate
        logger.warning(f"[{request_id}] Using char/4 estimation: {token_estimate}")
        status_code = 200  # Return 200 but with an estimate
        log_request_beautifully(
            raw_request.method,
            raw_request.url.path,
            original_model_name,
            actual_gemini_model_id,
            len(request_data.messages),
            0,
            status_code,
        )
        # Note: This entire token estimation fallback will be removed in Task 5
        # and replaced with proper error propagation
        return JSONResponse(
            content={
                "input_tokens": token_estimate,
                "note": f"[{request_id}] Token count is an estimate due to upstream error",
            },
            headers={
                "X-Mapped-Model": actual_gemini_model_id,
                "X-Request-ID": request_id,
                "X-Token-Estimation": "true",
            },
        )
    finally:
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Token count request completed in {processing_time:.3f}s")


@app.get("/", include_in_schema=False)
async def root():
    """
    Provide basic service information and health check.

    This endpoint returns general information about the proxy service including:
    - Service status
    - Target Gemini model configurations
    - Library versions in use
    - Credential manager status

    Returns:
        dict: Service information dictionary with status and configuration details
    """
    # Check if credential manager has been initialized
    cred_status = "initialized" if credential_manager._initialized_event.is_set() else "initializing"
    cred_expiry = "unknown"

    # Get expiry time if available
    if credential_manager._expiry_time:
        expiry_time = credential_manager._expiry_time
        now = time.time()
        if expiry_time > now:
            cred_expiry = f"valid for {int((expiry_time - now) / 60)} minutes"
        else:
            cred_expiry = "expired (refresh pending)"

    return {
        "message": "Anthropic API Compatible Proxy using Native Vertex AI SDK with Custom Gemini Auth",
        "status": "running",
        "target_gemini_models": {"BIG": GEMINI_BIG_MODEL, "SMALL": GEMINI_SMALL_MODEL},
        "litellm_version": getattr(litellm, "__version__", "unknown"),
        "vertexai_version": getattr(vertexai, "__version__", "unknown"),
        "credential_manager": {
            "status": cred_status,
            "token_expiry": cred_expiry,
            "centralized_auth": True,
        },
    }


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"

    # Get request ID from header or generate a new one with more descriptive prefix
    path = request.url.path
    prefix = "req_"
    if "/count_tokens" in path:
        prefix = "tok_"
    elif "/" == path:  # Root endpoint
        prefix = "inf_"  # info request

    request_id = request.headers.get("X-Request-ID") or f"{prefix}{uuid.uuid4().hex[:12]}"

    # Log incoming request with ID
    logger.info(f"[{request_id}] Incoming request: {request.method} {request.url.path} from {client_host}")
    logger.debug(f"[{request_id}] Request Headers: {dict(request.headers)}")

    # Store request ID in request state for access in endpoints
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        process_time = time.time() - start_time

        # Ensure request ID is in response header
        if "X-Request-ID" not in response.headers:
            response.headers["X-Request-ID"] = request_id

        # Log outgoing response with ID
        response_log_detail = f"status={response.status_code}, time={process_time:.3f}s"
        mapped_model = response.headers.get("X-Mapped-Model")
        if mapped_model:
            response_log_detail += f", mapped_model={mapped_model}"
        logger.info(f"[{request_id}] Response completed: {request.method} {request.url.path} ({response_log_detail})")
        logger.debug(f"[{request_id}] Response Headers: {dict(response.headers)}")
        return response

    except Exception as e:
        # Log unhandled exceptions caught by middleware (should be rare with endpoint handlers)
        process_time = time.time() - start_time
        logger.critical(
            f"[{request_id}] Unhandled exception in middleware/endpoint: {type(e).__name__} after {process_time:.3f}s",
            exc_info=True,
        )
        # Return a generic 500 response
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_server_error", "message": f"[{request_id}] Internal Server Error"}},
            headers={"X-Request-ID": request_id},
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", f"ex_{uuid.uuid4().hex[:12]}")
    logger.error(
        f"[{request_id}] HTTPException: Status={exc.status_code}, Detail={exc.detail}",
        exc_info=(exc.status_code >= 500),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"type": "api_error", "message": f"[{request_id}] {exc.detail}"}},
        headers={"X-Request-ID": request_id},  # Ensure ID is in error response
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", f"ex_{uuid.uuid4().hex[:12]}")
    logger.critical(f"[{request_id}] Unhandled Exception Handler: {type(exc).__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": {"type": "internal_server_error", "message": f"[{request_id}] Internal Server Error: {str(exc)}"}
        },
        headers={"X-Request-ID": request_id},
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8082))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload_flag = "--reload" in sys.argv or os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    print("--- Starting Anthropic Proxy (Native Vertex SDK) ---")
    print(f" Listening on: {host}:{port}")
    print(f" Log Level:    {log_level}")
    print(f" Auto-Reload:  {reload_flag}")
    print(f" Target Models: BIG='{GEMINI_BIG_MODEL}', SMALL='{GEMINI_SMALL_MODEL}'")
    print(f" Tool Temp Override: {TOOL_CALL_TEMPERATURE_OVERRIDE}")
    print(f" LiteLLM Ver:  {getattr(litellm, '__version__', 'unknown')}")
    print(f" VertexAI Ver: {getattr(vertexai, '__version__', 'unknown')}")
    if not os.getenv("WORKSPACE_ID") or not os.getenv("AUTH_URL"):
        print(
            f"{Colors.BOLD}{Colors.RED}!!! WARNING: WORKSPACE_ID or AUTH_URL environment variables not set! Authentication will fail. !!!{Colors.RESET}"
        )
    print("----------------------------------------------------")

    # Use uvicorn.run to start the server
    uvicorn.run(
        "server:app",  # Points to the FastAPI app instance in this file
        host=host,
        port=port,
        log_level=log_level,
        reload=reload_flag,
    )
