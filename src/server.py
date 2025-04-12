import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

import google.api_core.exceptions  # To catch API call errors
import google.auth  # To catch auth errors during vertexai.init
import litellm  # Still used for format definitions, fallback token counting
import requests
import uvicorn
import vertexai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.oauth2.credentials import Credentials as OAuth2Credentials
# Import specific types from vertexai for clarity in conversion/auth
from pydantic import BaseModel, field_validator
from vertexai.generative_models import (
    Content, Part, FinishReason, FunctionDeclaration,
    GenerativeModel, Tool, GenerationConfig, GenerationResponse, FunctionCall  # <-- Added FunctionCall
)

# --- Configuration ---

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG for detailed logs, change to INFO for less noise
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("AnthropicGeminiProxy") # Specific logger name


# Configure LiteLLM logging (optional: reduce verbosity)
# litellm.set_verbose=False
litellm.success_callback = [] # Disable default success logs
litellm.failure_callback = [] # Disable default failure logs

# Configure uvicorn and other libraries to be quieter
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google.auth.compute_engine._metadata").setLevel(logging.WARNING)
logging.getLogger("google.api_core.bidi").setLevel(logging.WARNING)
logging.getLogger("google.cloud.aiplatform").setLevel(logging.WARNING) # Quieten Vertex SDK logs if needed


# --- Model Configuration ---
# Get specific Gemini model names from environment or use defaults
GEMINI_BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-1.5-pro-latest")
GEMINI_SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-1.5-flash-latest")

# Define ANSI color codes for terminal output
class Colors:
    CYAN = "\033[96m"; BLUE = "\033[94m"; GREEN = "\033[92m"; YELLOW = "\033[93m"
    RED = "\033[91m"; MAGENTA = "\033[95m"; RESET = "\033[0m"; BOLD = "\033[1m"
    UNDERLINE = "\033[4m"; DIM = "\033[2m"

# --- FastAPI App Initialization ---
app = FastAPI(title="Anthropic to Custom Gemini Proxy (Native SDK Call)")


# --- Custom Exceptions ---
class AuthenticationError(Exception):
    """Custom exception for authentication failures."""
    pass


# --- Custom Authentication (Keep Your Function) ---
def get_gemini_credentials():
    """
    Authenticates with the custom endpoint and returns Vertex AI credentials.
    Returns: tuple (project_id, location, OAuth2Credentials)
    Raises: AuthenticationError
    """
    workspace_id = os.getenv("WORKSPACE_ID")
    model_name_for_auth = os.getenv("MODEL_NAME", GEMINI_BIG_MODEL) # Use configured BIG model for auth by default
    auth_url = os.getenv("AUTH_URL")

    if not all([workspace_id, auth_url]):
        logger.error("Missing required environment variables: WORKSPACE_ID and/or AUTH_URL")
        raise AuthenticationError("Missing required environment variables: WORKSPACE_ID, AUTH_URL")

    logger.info(f"Attempting custom authentication for workspace '{workspace_id}' (using model '{model_name_for_auth}' for auth)")
    logger.debug(f"Authentication URL: {auth_url}")

    payload = {"workspace_id": workspace_id, "model_name": model_name_for_auth}
    logger.info(f"Requesting temporary token from {auth_url}")

    try:
        resp = requests.post(auth_url, headers=None, json=payload, timeout=30)
        if resp.status_code != 200:
            error_msg = f"Authentication request failed: Status {resp.status_code}"
            logger.error(f"{error_msg}. Response: {resp.text[:500]}...")
            raise AuthenticationError(f"{error_msg}. Check auth server and credentials.")

        try: credentials_data = resp.json()
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON response from auth server: {e}"
            logger.error(f"{error_msg}. Raw Response: {resp.text[:500]}...")
            raise AuthenticationError(error_msg)

        token = credentials_data.get("token")
        project_id = credentials_data.get("project_id")
        location = credentials_data.get("region")
        expires_on_str = credentials_data.get("expires_on", "N/A")

        if not all([token, project_id, location]):
            missing = [k for k in ["token", "project_id", "region"] if not credentials_data.get(k)]
            logger.error(f"Authentication failed: Missing {missing} in response. Data: {credentials_data}")
            raise AuthenticationError(f"Missing required fields ({missing}) in authentication response.")

        logger.info(f"Successfully fetched custom Gemini credentials. Project: {project_id}, Location: {location}, Valid until: {expires_on_str}")
        temp_creds = OAuth2Credentials(token) # This is the object LiteLLM choked on
        return project_id, location, temp_creds

    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Failed to connect to authentication server: {e}. Check network and URL.")
    except requests.exceptions.Timeout as e:
        logger.error(f"Timeout during authentication to {auth_url}: {e}", exc_info=True)
        raise AuthenticationError(f"Authentication request timed out: {e}. Auth server might be down or overloaded.")
    except requests.exceptions.RequestException as e:
        response_text = getattr(e.response, 'text', '(No response text available)')
        logger.error(f"Network error during authentication to {auth_url}: {e}. Response: {response_text}", exc_info=True)
        raise AuthenticationError(f"Network error connecting to auth endpoint: {e}")
    except AuthenticationError: raise # Re-raise specific auth errors
    except Exception as e:
        logger.error(f"Unexpected error during custom authentication: {e}", exc_info=True)
        raise AuthenticationError(f"Unexpected error during authentication: {e}")


# --- Helper Functions ---

def clean_gemini_schema(schema: Any) -> Any:
    """ Recursively removes fields unsupported by Gemini from a JSON schema dict. """
    if isinstance(schema, dict):
        # Remove fields known to cause issues with Gemini's schema validation
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        # Gemini might not support all string formats, remove if not explicitly supported
        if schema.get("type") == "string" and "format" in schema:
            # Keep only formats known to be generally safe or potentially useful
            if schema["format"] not in {"enum", "date-time"}: # Adjust this list if needed
                logger.debug(f"Removing potentially unsupported string format '{schema['format']}'")
                schema.pop("format")
        # Recurse into nested structures
        for key, value in list(schema.items()):
            if key in ["properties", "items"] or isinstance(value, (dict, list)):
                 schema[key] = clean_gemini_schema(value)
            # Remove null values as they can sometimes cause issues
            elif value is None:
                schema.pop(key)
    elif isinstance(schema, list):
        return [clean_gemini_schema(item) for item in schema]
    return schema

def map_model_name(anthropic_model_name: str) -> str:
    """ Maps Anthropic model names to specific Gemini models. Returns the Gemini model ID. """
    original_model = anthropic_model_name
    mapped_gemini_model = GEMINI_BIG_MODEL # Default

    logger.debug(f"Attempting to map model: '{original_model}' -> Target Gemini BIG='{GEMINI_BIG_MODEL}', SMALL='{GEMINI_SMALL_MODEL}'")

    clean_name = anthropic_model_name.lower().split('@')[0]
    if clean_name.startswith("anthropic/"): clean_name = clean_name[10:]
    elif clean_name.startswith("gemini/"): clean_name = clean_name[7:] # Allow direct gemini model names like 'gemini/gemini-1.5-pro-latest'

    if "haiku" in clean_name:
        mapped_gemini_model = GEMINI_SMALL_MODEL
        logger.info(f"Mapping '{original_model}' (Haiku) -> Target Gemini SMALL '{mapped_gemini_model}'")
    elif "sonnet" in clean_name or "opus" in clean_name:
        mapped_gemini_model = GEMINI_BIG_MODEL
        logger.info(f"Mapping '{original_model}' (Sonnet/Opus) -> Target Gemini BIG '{mapped_gemini_model}'")
    elif clean_name == GEMINI_BIG_MODEL.lower() or clean_name == GEMINI_SMALL_MODEL.lower():
        mapped_gemini_model = clean_name # Use the directly specified Gemini model
        logger.info(f"Using directly specified target Gemini model: '{mapped_gemini_model}'")
    else:
        logger.warning(f"Unrecognized Anthropic model name '{original_model}'. Defaulting to BIG model '{mapped_gemini_model}'.")

    # Return just the Gemini model ID (e.g., "gemini-1.5-pro-latest") for the SDK
    return mapped_gemini_model


def log_request_beautifully(method, path, original_model, mapped_model, num_messages, num_tools, status_code):
    """ Log requests in a colorized format. """
    try:
        original_display = f"{Colors.CYAN}{original_model}{Colors.RESET}"
        endpoint = path.split("?")[0]
        # mapped_model here is the Gemini ID
        mapped_display_name = mapped_model
        mapped_color = Colors.GREEN # Always green for Gemini target
        mapped_display = f"{mapped_color}{mapped_display_name}{Colors.RESET}"
        tools_str = f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}" if num_tools > 0 else f"{Colors.DIM}{num_tools} tools{Colors.RESET}"
        messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
        status_color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
        status_symbol = "✓" if 200 <= status_code < 300 else "✗"
        status_str = f"{status_color}{status_symbol} {status_code}{Colors.RESET}"
        log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
        model_line = f"  {original_display} → {mapped_display} ({messages_str}, {tools_str})"
        print(log_line); print(model_line); sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error during beautiful logging: {e}")
        # Fallback plain log
        print(f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools")


# --- Pydantic Models (Anthropic Format - Keep As Is) ---
class ContentBlockText(BaseModel): type: Literal["text"]; text: str
class ContentBlockImageSource(BaseModel): type: Literal["base64"]; media_type: str; data: str
class ContentBlockImage(BaseModel): type: Literal["image"]; source: ContentBlockImageSource
class ContentBlockToolUse(BaseModel): type: Literal["tool_use"]; id: str; name: str; input: Dict[str, Any]
class ContentBlockToolResult(BaseModel): type: Literal["tool_result"]; tool_use_id: str; content: Union[str, List[Dict[str, Any]]]; is_error: Optional[bool] = False
# Use Field alias for Pydantic v2 compatibility if needed, though Union should work
ContentBlock = Union[ContentBlockText, ContentBlockImage, ContentBlockToolUse, ContentBlockToolResult]

class SystemContent(BaseModel): type: Literal["text"]; text: str
class Message(BaseModel): role: Literal["user", "assistant"]; content: Union[str, List[ContentBlock]]
class ToolInputSchema(BaseModel): type: Literal["object"] = "object"; properties: Dict[str, Any]; required: Optional[List[str]] = None
class ToolDefinition(BaseModel): name: str; description: Optional[str] = None; input_schema: ToolInputSchema

class MessagesRequest(BaseModel):
    model: str # This will hold the *mapped* Gemini model ID after validation
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
    original_model_name: Optional[str] = None # Internal field to store original name pre-mapping

    @field_validator("model")
    def validate_and_map_model(cls, v, info):
        # The validator now just returns the mapped Gemini model ID
        return map_model_name(v)

class TokenCountRequest(BaseModel):
    model: str # Mapped Gemini model ID
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    original_model_name: Optional[str] = None # Internal field

    @field_validator("model")
    def validate_and_map_model_token_count(cls, v, info):
        return map_model_name(v)

class TokenCountResponse(BaseModel): input_tokens: int
class Usage(BaseModel): input_tokens: int; output_tokens: int
class MessagesResponse(BaseModel):
    id: str; type: Literal["message"] = "message"; role: Literal["assistant"] = "assistant"
    model: str # Original Anthropic model name
    content: List[ContentBlock]
    stop_reason: Optional[Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "content_filtered"]] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# --- Conversion Functions ---

# >>> Keep: convert_anthropic_to_litellm <<<
# This is still useful to get a standardized intermediate format (OpenAI-like)
# before converting to the native Vertex SDK format.
def convert_anthropic_to_litellm(request: MessagesRequest) -> Dict[str, Any]:
    """Converts Anthropic MessagesRequest to LiteLLM input dict (OpenAI format)."""
    litellm_messages = []

    # Handle System Prompt -> Becomes a separate parameter for Gemini SDK
    system_text = None
    if request.system:
        system_text = request.system if isinstance(request.system, str) else "\n".join([b.text for b in request.system if b.type == "text"])
        if system_text:
            # Store it separately, don't add to messages list for Vertex conversion later
             logger.debug("System prompt extracted for Vertex SDK.")
        else: system_text = None # Ensure it's None if empty

    # Handle Messages -> Convert to OpenAI message format first
    for msg in request.messages:
        is_tool_response_message = False
        content_list = [] # For multimodal content within a single message
        tool_calls_list = [] # For assistant requesting tools

        if isinstance(msg.content, str):
            content_list.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            for block in msg.content:
                if block.type == "text":
                    content_list.append({"type": "text", "text": block.text})
                elif block.type == "image" and msg.role == "user": # Images only supported for user role
                    content_list.append({"type": "image_url", "image_url": {"url": f"data:{block.source.media_type};base64,{block.source.data}"}})
                    logger.debug("Image block added to intermediate format.")
                elif block.type == "tool_use" and msg.role == "assistant":
                    # Convert Anthropic tool_use to OpenAI tool_calls format
                    tool_calls_list.append({
                        "id": block.id,
                        "type": "function",
                        "function": {"name": block.name, "arguments": json.dumps(block.input)} # Arguments must be JSON string
                    })
                    logger.debug(f"Assistant tool_use '{block.name}' converted to intermediate tool_calls.")
                elif block.type == "tool_result" and msg.role == "user":
                    # If previous user text exists, send it first
                    if content_list:
                        litellm_messages.append({"role": "user", "content": content_list})
                        content_list = [] # Reset content list

                    # Convert Anthropic tool_result to OpenAI tool message format
                    tool_content = block.content
                    # Ensure content is a string (JSON if possible) for OpenAI format
                    if not isinstance(tool_content, str):
                        try: tool_content = json.dumps(tool_content)
                        except Exception: tool_content = str(tool_content) # Fallback to string representation

                    litellm_messages.append({
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_content
                    })
                    logger.debug(f"User tool_result for '{block.tool_use_id}' converted to intermediate tool message.")
                    is_tool_response_message = True
                    break # Process only the tool result block for this message

        # Add the assembled message if it wasn't a tool response handled above
        if not is_tool_response_message:
             litellm_msg = {"role": msg.role}
             # Simplify content if only text
             if len(content_list) == 1 and content_list[0]["type"] == "text":
                 litellm_msg["content"] = content_list[0]["text"]
             elif content_list: # Keep as list for multimodal
                 litellm_msg["content"] = content_list
             else:
                 litellm_msg["content"] = None # Or empty string ""? Let's use None for clarity

             # Add tool calls if any (for assistant messages)
             if tool_calls_list:
                 litellm_msg["tool_calls"] = tool_calls_list

             # Only add message if it has content or tool calls
             if litellm_msg.get("content") is not None or litellm_msg.get("tool_calls"):
                 litellm_messages.append(litellm_msg)
             elif msg.role == "assistant" and not litellm_msg.get("content") and not litellm_msg.get("tool_calls"):
                  # Handle case where assistant message might be empty (e.g., after tool call)
                  # OpenAI format expects content: null or content: ""
                  litellm_msg["content"] = ""
                  litellm_messages.append(litellm_msg)


    # --- Assemble LiteLLM/OpenAI Request Dictionary ---
    # Note: request.model already contains the *mapped* Gemini ID from the validator
    litellm_request = {
        "model": request.model, # Mapped Gemini ID
        "messages": litellm_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False
    }
    # Add optional parameters
    if request.temperature is not None: litellm_request["temperature"] = request.temperature
    if request.top_p is not None: litellm_request["top_p"] = request.top_p
    if request.top_k is not None: litellm_request["top_k"] = request.top_k
    if request.stop_sequences: litellm_request["stop"] = request.stop_sequences # For GenerationConfig later
    if request.metadata: litellm_request["metadata"] = request.metadata # Keep metadata if needed downstream

    # Store system text separately in the dict for easy access later
    if system_text:
        litellm_request["system_prompt"] = system_text

    # Convert Anthropic Tools to OpenAI Tool Format (and clean schema)
    if request.tools:
        openai_tools = []
        for tool in request.tools:
            input_schema = tool.input_schema.dict(exclude_unset=True)
            logger.debug(f"Cleaning schema for intermediate tool format: {tool.name}")
            # Clean the schema *before* putting it into the intermediate format
            cleaned_schema = clean_gemini_schema(input_schema)

            # Ensure basic structure expected by Vertex SDK later
            if 'properties' in cleaned_schema and 'type' not in cleaned_schema:
                cleaned_schema['type'] = 'object'
            if cleaned_schema.get('type') == 'object' and 'properties' not in cleaned_schema:
                cleaned_schema['properties'] = {} # Ensure properties key exists

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": cleaned_schema # Use the cleaned schema
                }
            })
        if openai_tools:
            litellm_request["tools"] = openai_tools
            logger.debug(f"Converted {len(openai_tools)} tools to intermediate OpenAI format.")

    # Convert Anthropic Tool Choice to OpenAI Tool Choice Format
    # Note: Vertex has a different `tool_config`, this mapping might be approximate
    if request.tool_choice:
        choice_type = request.tool_choice.get("type")
        if choice_type == "any" or choice_type == "auto":
            litellm_request["tool_choice"] = "auto" # Map to OpenAI 'auto'
        elif choice_type == "tool" and "name" in request.tool_choice:
            # Map to OpenAI specific function choice
            litellm_request["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice["name"]}}
        else: # Includes 'none' or other types
            litellm_request["tool_choice"] = "none" # Map to OpenAI 'none'
        logger.debug(f"Converted tool_choice '{choice_type}' to intermediate format '{litellm_request['tool_choice']}'.")

    logger.debug(f"Intermediate LiteLLM/OpenAI Request Prepared: {json.dumps(litellm_request, indent=2, default=lambda o: '<not serializable>')}")
    return litellm_request


# +++ NEW: Conversion from LiteLLM/OpenAI format to Vertex AI SDK format +++

def convert_litellm_tools_to_vertex_tools(litellm_tools: Optional[List[Dict]]) -> Optional[List[Tool]]:
    """
    Converts LiteLLM/OpenAI tools list (from intermediate format) to a list
    containing a SINGLE Vertex AI SDK Tool object, which in turn contains
    all function declarations. Returns None if no valid tools are found.
    """
    if not litellm_tools:
        return None

    all_function_declarations: List[FunctionDeclaration] = [] # Collect declarations here

    for tool in litellm_tools:
        if tool.get("type") == "function":
            func_data = tool.get("function", {})
            name = func_data.get("name")
            description = func_data.get("description")
            parameters_schema = func_data.get("parameters") # Schema should already be cleaned

            if name and parameters_schema is not None: # Allow empty schema {}
                try:
                    # Create the FunctionDeclaration
                    func_decl = FunctionDeclaration(
                        name=name,
                        description=description or "",
                        parameters=parameters_schema
                    )
                    # Add the declaration to our list
                    all_function_declarations.append(func_decl)
                    logger.debug(f"Collected FunctionDeclaration for tool '{name}'.")
                except Exception as e:
                    # Log detailed error if schema validation fails at SDK level
                    logger.error(f"Failed to create Vertex FunctionDeclaration for tool '{name}': {e}. Schema: {parameters_schema}", exc_info=True)
            else:
                logger.warning(f"Skipping tool conversion to Vertex format due to missing name or parameters: {tool}")
        else:
             logger.warning(f"Skipping non-function tool type during Vertex tool conversion: {tool.get('type')}")

    # If we collected any declarations, wrap them in a SINGLE Tool object
    if all_function_declarations:
        vertex_tool_wrapper = Tool(function_declarations=all_function_declarations)
        logger.info(f"Created a single Vertex Tool containing {len(all_function_declarations)} function declarations.")
        # Return a list containing just this one Tool object
        return [vertex_tool_wrapper]
    else:
        # No valid function declarations found
        logger.info("No valid function declarations found to create a Vertex Tool.")
        return None

def convert_litellm_messages_to_vertex_content(litellm_messages: List[Dict]) -> List[Content]:
    """
    Converts LiteLLM/OpenAI message list (intermediate format) to Vertex AI SDK Content list.
    Includes fix (v4) for creating Parts with FunctionCalls using from_dict
    and refined merge logic (v2).
    """
    vertex_content_list: List[Content] = []
    request_id_for_logging = "conv_test" # Placeholder for logging context

    for i, msg in enumerate(litellm_messages):
        role = msg.get("role")
        intermediate_content = msg.get("content") # Content from OpenAI-like format
        tool_calls = msg.get("tool_calls") # OpenAI format tool_calls
        tool_call_id = msg.get("tool_call_id") # OpenAI format tool_call_id (for role='tool')

        vertex_role: Optional[Literal["user", "model"]] = None
        is_tool_result_message = False
        is_assistant_call_message = False

        # --- Determine Vertex Role ---
        if role == "user":
            vertex_role = "user"
        elif role == "assistant":
            vertex_role = "model"
            if tool_calls:
                is_assistant_call_message = True
        elif role == "tool":
            vertex_role = "user" # Tool results ARE a user turn
            is_tool_result_message = True
        elif role == "system":
            logger.warning(f"[{request_id_for_logging}] System message found in list; should have been handled separately. Skipping.")
            continue
        else:
            logger.warning(f"[{request_id_for_logging}] Unrecognized role '{role}' in intermediate message, skipping.")
            continue

        # --- Create Parts based on Content and Tool Calls/Results ---
        current_turn_parts: List[Part] = []

        # 1. Handle Tool Result (is_tool_result_message == True)
        # (Logic remains the same as v2/v3)
        if is_tool_result_message:
            if hasattr(Part, "from_function_response") and tool_call_id and intermediate_content is not None:
                 tool_response_dict = {"output": None}
                 try:
                     if isinstance(intermediate_content, str):
                         parsed_content = json.loads(intermediate_content)
                         tool_response_dict["output"] = parsed_content
                         logger.debug(f"[{request_id_for_logging}] Tool result content for id {tool_call_id} parsed as JSON.")
                     else:
                         tool_response_dict["output"] = intermediate_content
                         logger.warning(f"[{request_id_for_logging}] Tool result content for id {tool_call_id} was not a string, using as is: {type(intermediate_content)}")
                 except (json.JSONDecodeError, TypeError):
                     tool_response_dict["output"] = intermediate_content
                     logger.debug(f"[{request_id_for_logging}] Tool result content for id {tool_call_id} is not JSON, sending as string under 'output' key.")
                 except Exception as e:
                      logger.error(f"[{request_id_for_logging}] Error processing tool result content for {tool_call_id}: {e}. Content: {str(intermediate_content)[:100]}...")
                      tool_response_dict["output"] = f"Error processing content: {e}"

                 original_func_name = "unknown_function"
                 for j in range(i - 1, -1, -1):
                     prev_msg = litellm_messages[j]
                     if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                         for tc in prev_msg["tool_calls"]:
                             if tc.get("id") == tool_call_id:
                                 original_func_name = tc.get("function", {}).get("name", original_func_name)
                                 break
                         if original_func_name != "unknown_function": break

                 if original_func_name == "unknown_function":
                      logger.warning(f"[{request_id_for_logging}] Could not find original function name for tool_call_id '{tool_call_id}'. Using placeholder name.")

                 try:
                     function_response_part = Part.from_function_response(
                         name=original_func_name,
                         response=tool_response_dict
                     )
                     current_turn_parts.append(function_response_part)
                     logger.debug(f"[{request_id_for_logging}] Created Part.from_function_response for tool '{original_func_name}' (id: {tool_call_id}).")
                 except Exception as e:
                     logger.error(f"[{request_id_for_logging}] Failed to create Part.from_function_response for {original_func_name}: {e}", exc_info=True)
                     current_turn_parts.append(Part.from_text(f"[Error creating tool response part for {original_func_name}: {e}]"))

            elif not hasattr(Part, "from_function_response"):
                 logger.error(f"[{request_id_for_logging}] Skipping tool result conversion: Part.from_function_response not found in SDK.")
                 continue
            else:
                 logger.warning(f"[{request_id_for_logging}] Skipping tool message conversion due to missing tool_call_id or content: {msg}")
                 continue

        # 2. Handle Assistant asking for Tool Call (is_assistant_call_message == True)
        elif is_assistant_call_message:
             if isinstance(intermediate_content, str) and intermediate_content.strip():
                  current_turn_parts.append(Part.from_text(intermediate_content))
                  logger.debug(f"[{request_id_for_logging}] Added accompanying text part to assistant tool call message.")

             for tc in tool_calls:
                 if tc.get("type") == "function":
                     func_data = tc.get("function", {})
                     func_name = func_data.get("name")
                     func_args_str = func_data.get("arguments", "{}")
                     try:
                         func_args = json.loads(func_args_str)
                         if func_name:
                             # Create the FunctionCall object first
                             vertex_function_call = FunctionCall(name=func_name, args=func_args)

                             # --- !!! CORE FIX APPLIED HERE (v4) !!! ---
                             # Create the Part using from_dict, providing the function_call structure
                             try:
                                 part_for_call = Part.from_dict({
                                     "function_call": {
                                         "name": vertex_function_call.name,
                                         "args": vertex_function_call.args
                                     }
                                 })
                                 current_turn_parts.append(part_for_call)
                                 logger.debug(f"[{request_id_for_logging}] Created Part containing FunctionCall via from_dict: {func_name}({func_args})")
                             except Exception as e_dict:
                                 logger.error(f"[{request_id_for_logging}] Failed to create Part via from_dict: {e_dict}", exc_info=True)
                                 # If from_dict fails, we have no reliable way to create this part based on previous errors.
                                 # Adding an error text part might be better than skipping the turn.
                                 current_turn_parts.append(Part.from_text(f"[ERROR: Failed to construct function_call part for {func_name}]"))
                             # --- !!! END FIX (v4) !!! ---
                         else:
                              logger.warning(f"[{request_id_for_logging}] Skipping assistant tool call part due to missing function name: {tc}")
                     except json.JSONDecodeError:
                          logger.error(f"[{request_id_for_logging}] Failed to parse function arguments JSON for assistant tool call '{func_name}': {func_args_str}", exc_info=True)
                     except Exception as e: # Catch potential errors during FunctionCall creation itself
                          logger.error(f"[{request_id_for_logging}] Failed to create FunctionCall object for {func_name}: {e}", exc_info=True)
                 else:
                     logger.warning(f"[{request_id_for_logging}] Skipping non-function tool call part in assistant message: {tc}")

        # 3. Handle Regular Text/Image Content (Non-tool related parts)
        # (Logic remains the same as v2/v3)
        elif intermediate_content:
            if isinstance(intermediate_content, str):
                current_turn_parts.append(Part.from_text(intermediate_content))
            elif isinstance(intermediate_content, list):
                if role != "user":
                    logger.warning(f"[{request_id_for_logging}] Multimodal content (list) found for role '{role}', but only 'user' role typically supports images. Processing anyway.")
                for item in intermediate_content:
                    item_type = item.get("type")
                    if item_type == "text":
                        current_turn_parts.append(Part.from_text(item.get("text", "")))
                    elif item_type == "image_url" and role == "user":
                        image_url_data = item.get("image_url", {}).get("url", "")
                        if image_url_data.startswith("data:"):
                            try:
                                header, b64data = image_url_data.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_data = base64.b64decode(b64data)
                                current_turn_parts.append(Part.from_data(data=image_data, mime_type=mime_type))
                                logger.debug(f"[{request_id_for_logging}] Added image part with mime_type: {mime_type}")
                            except Exception as e:
                                logger.error(f"[{request_id_for_logging}] Failed to process base64 image data URL: {e}", exc_info=True)
                                current_turn_parts.append(Part.from_text(f"[Error processing image: {e}]"))
                        else:
                            logger.warning(f"[{request_id_for_logging}] Skipping image URL that is not a data URL: {image_url_data[:100]}...")
                            current_turn_parts.append(Part.from_text("[Skipped non-data image URL]"))
                    elif item_type == "image_url" and role != "user":
                         logger.warning(f"[{request_id_for_logging}] Skipping image part for non-user role '{role}'.")
                         current_turn_parts.append(Part.from_text("[Skipped image for non-user role]"))
                    else:
                         logger.warning(f"[{request_id_for_logging}] Unsupported item type in multimodal content list: {item_type}")
            else:
                 logger.warning(f"[{request_id_for_logging}] Unsupported content type for role '{role}': {type(intermediate_content)}")


        # --- Append Parts to Vertex Content List ---
        if current_turn_parts:
             valid_parts = [p for p in current_turn_parts if isinstance(p, Part)]
             if len(valid_parts) != len(current_turn_parts):
                 logger.warning(f"[{request_id_for_logging}] Some generated items were not valid Part objects and were filtered out.")
             if not valid_parts:
                  logger.warning(f"[{request_id_for_logging}] Message with role '{role}' resulted in no valid parts after processing. Skipping message.")
                  continue

             # --- Refined Merge Logic (v2 - unchanged) ---
             should_merge = False
             if vertex_content_list:
                 last_content = vertex_content_list[-1]
                 if last_content.role == vertex_role:
                     is_last_content_a_tool_response_turn = any(
                         hasattr(p, 'function_response') and p.function_response for p in last_content.parts
                     )
                     if is_tool_result_message:
                         if is_last_content_a_tool_response_turn:
                             should_merge = True
                             logger.debug(f"[{request_id_for_logging}] Merging tool result part onto existing user tool response turn.")
                     else:
                         if not is_last_content_a_tool_response_turn:
                             should_merge = True
                             logger.debug(f"[{request_id_for_logging}] Merging standard parts (non-tool-result).")
             # --- End Refined Merge Logic (v2) ---

             if should_merge:
                  logger.debug(f"[{request_id_for_logging}] Merging {len(valid_parts)} parts into previous Content object (role: {vertex_role})")
                  vertex_content_list[-1].parts.extend(valid_parts)
             else:
                  if vertex_role:
                      vertex_content_list.append(Content(parts=valid_parts, role=vertex_role))
                      logger.debug(f"[{request_id_for_logging}] Created new Content object (role: {vertex_role}) with {len(valid_parts)} part(s): {[type(p).__name__ for p in valid_parts]}")
                  else:
                       logger.error(f"[{request_id_for_logging}] Cannot create Content object without a valid Vertex role (was '{role}'). Skipping parts.")

        elif role != "system":
             logger.warning(f"[{request_id_for_logging}] Message with role '{role}' resulted in no parts, skipping.")

    # --- Final Logging ---
    # (Logging remains the same as v2/v3)
    logger.info(f"[{request_id_for_logging}] Converted {len(litellm_messages)} intermediate messages -> {len(vertex_content_list)} Vertex Content objects.")
    try:
        final_history_repr = []
        for content_idx, content in enumerate(vertex_content_list):
             part_reprs = []
             for p_idx, p in enumerate(content.parts):
                  part_type = "Unknown"
                  part_detail = ""
                  if hasattr(p, 'function_call') and getattr(p, 'function_call', None):
                      part_type = 'FunctionCall'
                      part_detail = f"({p.function_call.name})"
                  elif hasattr(p, 'function_response') and getattr(p, 'function_response', None):
                      part_type = 'FunctionResponse'
                      part_detail = f"({p.function_response.name})"
                  elif hasattr(p, 'text') and getattr(p, 'text', None) is not None:
                      part_type = 'Text'
                      part_detail = f"({len(p.text)} chars)"
                  elif hasattr(p, 'inline_data') and getattr(p, 'inline_data', None):
                      part_type = 'Data'
                      part_detail = f"({p.inline_data.mime_type})"
                  else:
                       part_type = type(p).__name__
                  part_reprs.append(f"{part_type}{part_detail}")
             final_history_repr.append({
                 "index": content_idx,
                 "role": content.role,
                 "part_types": part_reprs
             })
        logger.debug(f"[{request_id_for_logging}] Final Vertex history structure: {json.dumps(final_history_repr, indent=2)}")
    except Exception as e:
         logger.debug(f"[{request_id_for_logging}] Could not serialize final Vertex history structure for logging: {e}")

    return vertex_content_list


# >>> Keep: convert_litellm_to_anthropic <<<
# Converts the *adapted* LiteLLM/OpenAI format (from Vertex response) back to Anthropic Non-Streaming Response
def convert_litellm_to_anthropic(response_chunk: Union[Dict, Any], original_model_name: str) -> Optional[MessagesResponse]:
    """ Converts non-streaming LiteLLM/OpenAI format response (dict or object) to Anthropic MessagesResponse. """
    request_id = response_chunk.get("request_id", "unknown") # Get request ID if passed through
    logger.info(f"[{request_id}] Converting adapted LiteLLM/OpenAI response to Anthropic MessagesResponse format.")
    try:
        # Ensure input is a dictionary
        resp_dict = {}
        if isinstance(response_chunk, dict):
            resp_dict = response_chunk
        elif hasattr(response_chunk, 'model_dump'): # Pydantic v2
            resp_dict = response_chunk.model_dump()
        elif hasattr(response_chunk, 'dict'): # Pydantic v1
            resp_dict = response_chunk.dict()
        else:
            try: resp_dict = vars(response_chunk) # Fallback for simple objects
            except TypeError:
                 logger.error(f"[{request_id}] Cannot convert response_chunk of type {type(response_chunk)} to dict.")
                 raise ValueError("Input response_chunk is not convertible to dict.")

        # Extract data using .get for safety
        resp_id = resp_dict.get("id", f"msg_{uuid.uuid4().hex[:24]}")
        choices = resp_dict.get("choices", [])
        usage_data = resp_dict.get("usage", {}) or {} # Ensure usage is a dict

        anthropic_content: List[ContentBlock] = []
        # Map OpenAI finish reasons to Anthropic stop reasons
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filtered",
            # Add mappings for any other potential finish reasons if needed
        }
        openai_finish_reason = "stop" # Default

        if choices:
            choice = choices[0] # Assume only one choice
            openai_finish_reason = choice.get("finish_reason", "stop")
            message = choice.get("message", {}) or {} # Ensure message is a dict

            text_content = message.get("content")
            tool_calls = message.get("tool_calls") # List of tool calls made by the assistant

            # 1. Add text content block if present
            if text_content and isinstance(text_content, str):
                anthropic_content.append(ContentBlockText(type="text", text=text_content))
                logger.debug(f"[{request_id}] Added text content block.")

            # 2. Add tool_use content blocks if present
            if tool_calls and isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict) and tc.get("type") == "function":
                        func = tc.get("function", {})
                        args_str = func.get("arguments", "{}")
                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}") # Use provided ID or generate one
                        tool_name = func.get("name", "unknown_tool")

                        # Parse arguments JSON string back into a dict for Anthropic input
                        try:
                            args_input = json.loads(args_str)
                        except json.JSONDecodeError:
                            logger.warning(f"[{request_id}] Non-streaming: Failed to parse tool arguments JSON: {args_str}. Sending raw string.")
                            args_input = {"raw_arguments": args_str}
                        except Exception as e:
                             logger.error(f"[{request_id}] Non-streaming: Error parsing tool arguments: {e}. Args: {args_str}")
                             args_input = {"error_parsing_arguments": str(e), "raw_arguments": args_str}


                        anthropic_content.append(ContentBlockToolUse(
                            type="tool_use",
                            id=tool_id,
                            name=tool_name,
                            input=args_input
                        ))
                        logger.debug(f"[{request_id}] Added tool_use content block: id={tool_id}, name={tool_name}")
                    else:
                        logger.warning(f"[{request_id}] Skipping conversion of non-function tool_call in response: {tc}")

        # Ensure there's always at least one content block (even if empty text)
        # Anthropic requires content to be a non-empty list.
        if not anthropic_content:
            logger.warning(f"[{request_id}] No content generated, adding empty text block.")
            anthropic_content.append(ContentBlockText(type="text", text=""))

        # Map the finish reason
        anthropic_stop_reason = stop_reason_map.get(openai_finish_reason, "end_turn")
        logger.debug(f"[{request_id}] Mapped finish_reason '{openai_finish_reason}' to stop_reason '{anthropic_stop_reason}'.")

        # Create the final Anthropic response object
        return MessagesResponse(
            id=resp_id,
            model=original_model_name, # Use the original model name requested by the client
            type="message",
            role="assistant",
            content=anthropic_content,
            stop_reason=anthropic_stop_reason,
            stop_sequence=None, # OpenAI format doesn't typically return the sequence matched
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0),
                output_tokens=usage_data.get("completion_tokens", 0)
            )
        )
    except Exception as e:
         # Log detailed error during conversion
         logger.error(f"[{request_id}] Failed to convert adapted LiteLLM/OpenAI response to Anthropic format: {e}", exc_info=True)
         # Return a minimal error response in Anthropic format
         return MessagesResponse(
             id=f"error_{uuid.uuid4().hex[:24]}",
             model=original_model_name,
             type="message",
             role="assistant",
             content=[ContentBlockText(type="text", text=f"Error processing model response: {str(e)}")],
             stop_reason="end_turn", # Or maybe a custom error reason?
             usage=Usage(input_tokens=0, output_tokens=0)
         )


# >>> Keep: handle_streaming <<<
# Converts the *adapted* LiteLLM/OpenAI stream (from Vertex stream) to Anthropic SSE Stream
async def handle_streaming(response_generator: AsyncGenerator[Dict[str, Any], None], request: MessagesRequest, request_id: str):
    """ Converts adapted LiteLLM/OpenAI format async generator to Anthropic SSE stream. """
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    # Use the original model name provided by the client for Anthropic events
    response_model_name = request.original_model_name or request.model # Fallback to mapped ID if original is missing
    logger.info(f"[{request_id}] Starting Anthropic SSE stream conversion (message {message_id}, model: {response_model_name})")

    # --- Stream Initialization ---
    # 1. Send message_start event
    start_event_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": response_model_name,
            "content": [], # Content starts empty
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0} # Initial usage
        }
    }
    yield f"event: message_start\ndata: {json.dumps(start_event_data)}\n\n"
    logger.debug(f"[{request_id}] Sent message_start")

    # 2. Send initial ping event
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    logger.debug(f"[{request_id}] Sent initial ping")

    # --- Stream Processing ---
    content_block_index = -1 # Track the index of the current content block (text or tool_use)
    current_block_type: Optional[Literal["text", "tool_use"]] = None
    text_started = False # Flag to track if the current text block has been started
    tool_calls_buffer = {} # Buffer to assemble tool call arguments {openai_tc_index: {id: str, name: str, args: str, block_idx: int}}
    final_usage = {"input_tokens": 0, "output_tokens": 0} # Accumulate usage
    final_stop_reason: Optional[str] = None # Store the final stop reason

    # Mapping from OpenAI finish reasons to Anthropic stop reasons
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "content_filtered",
    }

    try:
        async for chunk in response_generator:
            logger.debug(f"[{request_id}] Processing adapted LiteLLM/OpenAI Chunk: {chunk}")

            # Safety check for chunk format
            if not isinstance(chunk, dict):
                 logger.warning(f"[{request_id}] Skipping invalid chunk format: {type(chunk)}")
                 continue

            choices = chunk.get("choices", [])
            if not choices or not isinstance(choices, list):
                 logger.warning(f"[{request_id}] Skipping chunk with missing or invalid 'choices': {chunk}")
                 continue

            choice = choices[0]
            delta = choice.get("delta", {}) or {} # Ensure delta is a dict
            finish_reason = choice.get("finish_reason") # OpenAI finish reason

            # --- Accumulate Usage from final chunk ---
            # Usage info might appear in the last chunk along with finish_reason
            chunk_usage = chunk.get("usage")
            if chunk_usage and isinstance(chunk_usage, dict):
                 # Only update if values are present and > 0, prefer existing values if chunk has 0
                 final_usage["input_tokens"] = chunk_usage.get("prompt_tokens") or final_usage["input_tokens"]
                 final_usage["output_tokens"] = chunk_usage.get("completion_tokens") or final_usage["output_tokens"]
                 logger.debug(f"[{request_id}] Updated usage from chunk: {final_usage}")
                 # Send ping after receiving usage (often in final chunk)
                 yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"; logger.debug(f"[{request_id}] Sent ping after usage update")


            # --- Process Delta Content ---
            text_delta = delta.get("content")
            tool_calls_delta = delta.get("tool_calls") # List of tool call deltas

            # 1. Handle Text Delta
            if text_delta and isinstance(text_delta, str):
                # If currently in a tool_use block, stop it first
                if current_block_type == "tool_use":
                    last_tool_block_idx = tool_calls_buffer[max(tool_calls_buffer.keys())]['block_idx']
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': last_tool_block_idx})}\n\n"
                    logger.debug(f"[{request_id}] Stopped tool block {last_tool_block_idx} due to incoming text.")
                    current_block_type = None

                # Start a new text block if not already started
                if not text_started:
                    content_block_index += 1
                    current_block_type = "text"
                    text_started = True
                    start_event = {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""} # Start with empty text
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                    logger.debug(f"[{request_id}] Started text block {content_block_index}")

                # Send the text delta
                delta_event = {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": text_delta}
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                logger.debug(f"[{request_id}] Sent text delta: '{text_delta[:50]}...'")

            # 2. Handle Tool Calls Delta
            if tool_calls_delta and isinstance(tool_calls_delta, list):
                # If currently in a text block, stop it first
                if current_block_type == "text" and text_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
                    logger.debug(f"[{request_id}] Stopped text block {content_block_index} due to incoming tool call.")
                    current_block_type = None
                    text_started = False

                # Process each tool call delta in the list
                for tc_delta in tool_calls_delta:
                    if not isinstance(tc_delta, dict): continue # Skip invalid format

                    # OpenAI tool index (usually 0 for the first tool, 1 for second, etc.)
                    # We rely on this index to aggregate arguments for the *same* tool call.
                    tc_openai_index = tc_delta.get("index", 0)
                    tc_id = tc_delta.get("id") # ID for the specific tool call instance
                    func_delta = tc_delta.get("function", {}) or {}
                    func_name = func_delta.get("name")
                    args_delta = func_delta.get("arguments") # Argument JSON string fragment

                    # --- Start a new tool_use block if necessary ---
                    if tc_openai_index not in tool_calls_buffer:
                        # Need ID and Name to start the Anthropic block
                        if tc_id and func_name:
                            content_block_index += 1
                            current_block_type = "tool_use"
                            tool_calls_buffer[tc_openai_index] = {
                                "id": tc_id,
                                "name": func_name,
                                "args": "", # Initialize empty args string
                                "block_idx": content_block_index # Store the Anthropic block index
                            }
                            start_event = {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {"type": "tool_use", "id": tc_id, "name": func_name, "input": {}} # Input starts empty
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                            logger.debug(f"[{request_id}] Started tool_use block {content_block_index} (id: {tc_id}, name: {func_name})")
                        # Handle case where ID might come first, then name in a later chunk (less common now)
                        elif tc_id and not func_name:
                             tool_calls_buffer[tc_openai_index] = {"id": tc_id, "name": None, "args": "", "block_idx": None}
                             logger.debug(f"[{request_id}] Received tool ID {tc_id} first for index {tc_openai_index}, waiting for name.")
                        else:
                             logger.warning(f"[{request_id}] Cannot start tool block for index {tc_openai_index} without ID and/or Name. Delta: {tc_delta}")
                             continue # Cannot start block yet

                    # --- If name arrives later for an existing ID ---
                    elif tc_openai_index in tool_calls_buffer and func_name and tool_calls_buffer[tc_openai_index]["name"] is None:
                         tool_info = tool_calls_buffer[tc_openai_index]
                         if tool_info["id"] == tc_id: # Ensure ID matches if provided again
                             content_block_index += 1
                             current_block_type = "tool_use"
                             tool_info["name"] = func_name
                             tool_info["block_idx"] = content_block_index
                             start_event = {
                                 "type": "content_block_start",
                                 "index": content_block_index,
                                 "content_block": {"type": "tool_use", "id": tool_info["id"], "name": func_name, "input": {}}
                             }
                             yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                             logger.debug(f"[{request_id}] Started tool_use block {content_block_index} for index {tc_openai_index} after receiving name ({func_name})")
                         else:
                             logger.warning(f"[{request_id}] Received name '{func_name}' for index {tc_openai_index}, but ID mismatch (expected {tool_info['id']}, got {tc_id}). Skipping.")


                    # --- Append argument fragments if block has started ---
                    if tc_openai_index in tool_calls_buffer and args_delta and tool_calls_buffer[tc_openai_index]["block_idx"] is not None:
                        tool_info = tool_calls_buffer[tc_openai_index]
                        tool_info["args"] += args_delta # Append the JSON fragment
                        # Send Anthropic input_json_delta
                        delta_event = {
                            "type": "content_block_delta",
                            "index": tool_info["block_idx"],
                            "delta": {"type": "input_json_delta", "partial_json": args_delta}
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                        logger.debug(f"[{request_id}] Sent tool args delta for block {tool_info['block_idx']}: '{args_delta[:50]}...'")

            # --- Process Finish Reason ---
            if finish_reason:
                # Map OpenAI finish reason to Anthropic stop reason
                final_stop_reason = stop_reason_map.get(finish_reason, "end_turn")
                logger.info(f"[{request_id}] Received final finish_reason: '{finish_reason}' -> Mapped to stop_reason: '{final_stop_reason}'")
                # The loop will break after processing this chunk's content (if any)
                break # Exit loop after processing the chunk containing the finish reason


        # --- End of Stream ---
        # 1. Stop the last active content block (if any)
        if current_block_type == "text" and text_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
            logger.debug(f"[{request_id}] Stopped final text block {content_block_index}")
        elif current_block_type == "tool_use":
            # Find the index of the last tool block started
            if tool_calls_buffer:
                 last_tool_block_idx = tool_calls_buffer[max(tool_calls_buffer.keys())]['block_idx']
                 if last_tool_block_idx is not None:
                     yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': last_tool_block_idx})}\n\n"
                     logger.debug(f"[{request_id}] Stopped final tool_use block {last_tool_block_idx}")
            else:
                 logger.warning(f"[{request_id}] Current block type is tool_use, but buffer is empty. Cannot stop block.")


        # 2. Send final message_delta event with stop reason and accumulated output tokens
        if final_stop_reason is None:
             logger.warning(f"[{request_id}] Stream finished without receiving a finish_reason. Defaulting to 'end_turn'.")
             final_stop_reason = "end_turn"

        final_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": None # Not typically provided by OpenAI stream
            },
            "usage": {"output_tokens": final_usage.get("output_tokens", 0)} # Send accumulated output tokens
        }
        yield f"event: message_delta\ndata: {json.dumps(final_delta_event)}\n\n"
        logger.debug(f"[{request_id}] Sent final message_delta (stop_reason: {final_stop_reason}, usage: {final_delta_event['usage']})")

        # 3. Send message_stop event
        yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
        logger.debug(f"[{request_id}] Sent message_stop")

    except Exception as e:
         # Log error during stream processing
         logger.error(f"[{request_id}] Error during Anthropic SSE stream conversion: {e}", exc_info=True)
         try:
             # Try to send an error event to the client
             error_payload = {
                 "type": "error",
                 "error": {"type": "internal_server_error", "message": f"Stream processing error: {str(e)}"}
             }
             yield f"event: error\ndata: {json.dumps(error_payload)}\n\n"
             # Always send message_stop after an error
             yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
             logger.debug(f"[{request_id}] Sent error event and message_stop after exception.")
         except Exception as e2:
             # Log if sending the error fails
             logger.error(f"[{request_id}] Failed to send error event to client: {e2}")
    finally:
        logger.info(f"[{request_id}] Anthropic SSE stream conversion finished.")
        # Optional: Send a custom [DONE] marker if needed, but Anthropic spec uses message_stop
        # yield "data: [DONE]\n\n"


# +++ NEW: Adapter for Vertex AI stream to LiteLLM/OpenAI stream format +++

async def adapt_vertex_stream_to_litellm(
    vertex_stream: AsyncGenerator[GenerationResponse, None],
    request_id: str,
    model_id_for_chunk: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Adapts the native Vertex AI SDK stream chunks (GenerationResponse)
    to the OpenAI streaming chunk format expected by handle_streaming.
    Handles parts containing either text or function calls safely.
    """
    logger.info(f"[{request_id}] Starting Vertex AI stream adaptation...")
    current_tool_calls = {}
    openai_tool_index_counter = 0
    tool_call_emitted_in_stream = False # Flag to track if any tool call was processed

    try:
        async for chunk in vertex_stream:
            logger.debug(f"[{request_id}] Raw Vertex Chunk: {chunk}")

            delta_content = None
            delta_tool_calls = []
            vertex_finish_reason_enum = None
            usage_metadata = None

            # --- Check for Usage Metadata ---
            if chunk.usage_metadata:
                 usage_metadata = {
                     "prompt_tokens": chunk.usage_metadata.prompt_token_count,
                     "completion_tokens": chunk.usage_metadata.candidates_token_count,
                     "total_tokens": chunk.usage_metadata.total_token_count,
                 }
                 logger.info(f"[{request_id}] Received Vertex usage metadata: {usage_metadata}")

            # --- Process Candidate Parts ---
            if chunk.candidates:
                candidate = chunk.candidates[0]

                # Store the raw Vertex finish reason if present
                if candidate.finish_reason and candidate.finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED:
                    vertex_finish_reason_enum = candidate.finish_reason
                    logger.info(f"[{request_id}] Received Vertex finish reason enum: {vertex_finish_reason_enum.name}")

                # Process Parts for Content and Tool Calls
                if candidate.content and candidate.content.parts:
                    for part_index, part in enumerate(candidate.content.parts):
                        # --- Safely check for text or function_call ---
                        part_processed = False
                        try:
                            # Attempt to access text. If it exists, process it.
                            current_text = part.text
                            delta_content = current_text
                            logger.debug(f"[{request_id}] Vertex text delta: '{delta_content[:50]}...'")
                            part_processed = True
                        except AttributeError:
                            # .text failed, so it's not a simple text part.
                            # Now check if it's a function call part.
                            try:
                                current_function_call = part.function_call
                                # Process the function call
                                tool_call_emitted_in_stream = True # Set the flag
                                fc = current_function_call
                                tool_call_id = f"toolu_{uuid.uuid4().hex[:12]}"
                                try:
                                    args_str = json.dumps(fc.args) if fc.args else "{}"
                                except Exception as e:
                                     logger.error(f"[{request_id}] Failed to dump streaming function call args to JSON: {e}. Args: {fc.args}")
                                     args_str = json.dumps({"error": f"Failed to serialize args: {e}"})

                                openai_tool_call = {
                                    "index": openai_tool_index_counter,
                                    "id": tool_call_id, "type": "function",
                                    "function": {"name": fc.name, "arguments": args_str}
                                }
                                delta_tool_calls.append(openai_tool_call)
                                current_tool_calls[part_index] = {"id": tool_call_id, "name": fc.name, "args": args_str, "openai_index": openai_tool_index_counter}
                                logger.debug(f"[{request_id}] Vertex tool call delta: index={openai_tool_index_counter}, id={tool_call_id}, name={fc.name}, args='{args_str[:50]}...'")
                                openai_tool_index_counter += 1
                                part_processed = True
                            except AttributeError:
                                # .function_call also failed. Log an unknown part type.
                                logger.warning(f"[{request_id}] Unknown or unexpected part type encountered in Vertex stream chunk (not text or function_call): {part}")
                        except Exception as e:
                             # Catch any other unexpected error during part processing
                             logger.error(f"[{request_id}] Unexpected error processing part {part_index}: {e}", exc_info=True)

            # --- Construct and Yield the LiteLLM/OpenAI Delta Chunk ---
            # (This logic remains the same as before)
            openai_delta_for_choice = {}
            if delta_content is not None: openai_delta_for_choice["content"] = delta_content
            if delta_tool_calls: openai_delta_for_choice["tool_calls"] = delta_tool_calls
            if openai_delta_for_choice: openai_delta_for_choice["role"] = "assistant"

            if openai_delta_for_choice:
                litellm_chunk = {
                    "id": f"chatcmpl-adap-{request_id}", "object": "chat.completion.chunk", "created": int(time.time()),
                    "model": model_id_for_chunk,
                    "choices": [{"index": 0, "delta": openai_delta_for_choice, "finish_reason": None, "logprobs": None}],
                    "usage": None
                }
                logger.debug(f"[{request_id}] Yielding adapted LiteLLM chunk: {litellm_chunk}")
                yield litellm_chunk

            # --- Yield Final Chunk if Vertex Finish Reason Received ---
            # (This logic remains the same as before)
            if vertex_finish_reason_enum:
                 final_openai_finish_reason = "stop" # Default
                 if tool_call_emitted_in_stream:
                      final_openai_finish_reason = "tool_calls"
                      logger.info(f"[{request_id}] Setting final finish_reason to 'tool_calls' because tool calls were emitted.")
                 else:
                      finish_reason_map = {
                          FinishReason.STOP.name: "stop", FinishReason.MAX_TOKENS.name: "length",
                          FinishReason.SAFETY.name: "content_filter", FinishReason.RECITATION.name: "content_filter",
                          FinishReason.OTHER.name: "stop",
                      }
                      final_openai_finish_reason = finish_reason_map.get(vertex_finish_reason_enum.name, "stop")
                      logger.info(f"[{request_id}] Mapping final Vertex finish_reason {vertex_finish_reason_enum.name} -> {final_openai_finish_reason}")

                 final_litellm_chunk = {
                     "id": f"chatcmpl-adap-{request_id}", "object": "chat.completion.chunk", "created": int(time.time()),
                     "model": model_id_for_chunk,
                     "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": final_openai_finish_reason, "logprobs": None}],
                     "usage": usage_metadata
                 }
                 logger.debug(f"[{request_id}] Yielding final adapted LiteLLM chunk with finish_reason: {final_litellm_chunk}")
                 yield final_litellm_chunk
                 break # Stop iteration

    except Exception as e:
        # (Error handling remains the same)
        logger.error(f"[{request_id}] Error during Vertex stream adaptation: {e}", exc_info=True)
        yield {
             "id": f"chatcmpl-adap-{request_id}", "object": "chat.completion.chunk", "created": int(time.time()),
             "model": model_id_for_chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}],
             "usage": None, "_error": f"Error adapting Vertex stream: {e}"
        }
    finally:
        # (Finally block remains the same)
        logger.info(f"[{request_id}] Vertex AI stream adaptation finished.")


# +++ NEW: Conversion from Vertex AI non-streaming response to LiteLLM/OpenAI dict +++
def convert_vertex_response_to_litellm(
    response: GenerationResponse,
    model_id: str,
    request_id: str
) -> Dict[str, Any]:
    """Converts a non-streaming Vertex AI GenerationResponse to a LiteLLM/OpenAI-like dictionary."""
    logger.info(f"[{request_id}] Converting Vertex non-streaming response to LiteLLM/OpenAI format.")
    output_text = ""
    tool_calls = [] # List for OpenAI tool_calls structure
    finish_reason_str = "stop" # Default
    has_function_call = False # Flag to track if tool calls are present

    # --- Process Candidate ---
    if response.candidates:
        candidate = response.candidates[0] # Assume first candidate

        # Map finish reason (using .name for robustness) - EXCLUDE TOOL_CALL
        finish_reason_map = {
            FinishReason.STOP.name: "stop",
            FinishReason.MAX_TOKENS.name: "length",
            FinishReason.SAFETY.name: "content_filter",
            FinishReason.RECITATION.name: "content_filter",
            # FinishReason.TOOL_CALL.name: "tool_calls", # REMOVED - Does not exist
            FinishReason.OTHER.name: "stop",
        }
        # Get the actual finish reason from Vertex
        vertex_finish_reason = candidate.finish_reason

        # Extract content (text and tool calls) from parts
        if candidate.content and candidate.content.parts:
            openai_tool_index_counter = 0
            for part in candidate.content.parts:
                if part.text:
                    output_text += part.text # Concatenate text parts
                elif part.function_call:
                    has_function_call = True # Mark that a tool call was found
                    fc = part.function_call
                    tool_call_id = f"toolu_{uuid.uuid4().hex[:12]}" # Generate ID
                    try: args_str = json.dumps(fc.args) if fc.args else "{}"
                    except Exception as e:
                         logger.error(f"[{request_id}] Failed to dump non-streaming function call args to JSON: {e}. Args: {fc.args}")
                         args_str = json.dumps({"error": f"Failed to serialize args: {e}"})

                    tool_calls.append({
                        "index": openai_tool_index_counter, # Include index for consistency
                        "id": tool_call_id,
                        "type": "function",
                        "function": {"name": fc.name, "arguments": args_str}
                    })
                    logger.debug(f"[{request_id}] Converted non-streaming tool call: index={openai_tool_index_counter}, id={tool_call_id}, name={fc.name}")
                    openai_tool_index_counter += 1
                elif part.function_response:
                     logger.warning(f"[{request_id}] Unexpected FunctionResponse part found in non-streaming Vertex response: {part.function_response}")

        # --- Determine Final OpenAI Finish Reason ---
        if has_function_call:
            # If function calls are present, the reason is 'tool_calls' for OpenAI
            finish_reason_str = "tool_calls"
            logger.debug(f"[{request_id}] Setting finish_reason to 'tool_calls' due to presence of function calls.")
        elif vertex_finish_reason and vertex_finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED:
            # Otherwise, use the mapped Vertex finish reason
            finish_reason_str = finish_reason_map.get(vertex_finish_reason.name, "stop")
            logger.debug(f"[{request_id}] Mapped non-streaming finish_reason: {vertex_finish_reason.name} -> {finish_reason_str}")
        else:
            # Default to 'stop' if no specific reason provided or mapped
            logger.debug(f"[{request_id}] Using default finish_reason 'stop'.")


    # --- Extract Usage ---
    prompt_tokens = 0; completion_tokens = 0; total_tokens = 0
    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        logger.info(f"[{request_id}] Extracted non-streaming usage: In={prompt_tokens}, Out={completion_tokens}, Total={total_tokens}")

    # --- Construct the LiteLLM/OpenAI-like Message ---
    message_content = {"role": "assistant"}
    if output_text: message_content["content"] = output_text
    else: message_content["content"] = None
    if tool_calls: message_content["tool_calls"] = tool_calls
    if message_content["content"] is None and not message_content.get("tool_calls"):
         logger.warning(f"[{request_id}] Non-streaming response has no text and no tool calls. Message content set to null.")

    # --- Construct the Final LiteLLM/OpenAI-like Response Dict ---
    litellm_response = {
        "id": f"chatcmpl-vert-{request_id}", "object": "chat.completion", "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "message": message_content, "finish_reason": finish_reason_str, "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
        "request_id": request_id
    }
    logger.debug(f"[{request_id}] Converted non-streaming Vertex response to LiteLLM/OpenAI format: {litellm_response}")
    return litellm_response


# --- API Endpoints ---

@app.post("/v1/messages", response_model=None) # response_model=None for StreamingResponse
async def create_message(request_data: MessagesRequest, raw_request: Request):
    """ Handles Anthropic /v1/messages endpoint using Native Vertex AI SDK with Custom Auth. """
    request_id = f"req_{uuid.uuid4().hex[:12]}" # Unique ID for this request
    start_time = time.time()

    try:
        # --- Request Parsing and Model Mapping ---
        # Extract original model name BEFORE validation potentially modifies request_data.model
        try:
            raw_body = await raw_request.body()
            original_model_name = json.loads(raw_body.decode()).get("model", "unknown-request-body")
        except Exception:
            # Fallback if raw body parsing fails (shouldn't happen with Pydantic validation)
            original_model_name = request_data.original_model_name or "unknown-fallback"
            logger.warning(f"[{request_id}] Failed to parse raw request body for original model name. Using fallback: {original_model_name}")

        # Store original name, request_data.model now holds the *mapped* Gemini ID
        request_data.original_model_name = original_model_name
        actual_gemini_model_id = request_data.model # Contains mapped ID like "gemini-1.5-pro-latest"

        logger.info(f"[{request_id}] Processing '/v1/messages': Original='{original_model_name}', Target SDK Model='{actual_gemini_model_id}', Stream={request_data.stream}")

        # --- Custom Authentication ---
        project_id, location, temp_creds = None, None, None
        try:
            # Run synchronous auth function in a thread pool
            project_id, location, temp_creds = await asyncio.to_thread(get_gemini_credentials)
            logger.info(f"[{request_id}] Custom authentication successful. Project: {project_id}, Location: {location}")
        except AuthenticationError as e:
            logger.error(f"[{request_id}] Custom Authentication Failed: {e}")
            # Use 503 Service Unavailable, as it's an issue connecting to a dependency (auth service)
            raise HTTPException(status_code=503, detail=f"Authentication Service Error: {e}")
        except Exception as e:
            logger.error(f"[{request_id}] Unexpected error during authentication thread: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Unexpected authentication error")

        # --- Initialize Vertex AI SDK (Per-Request) ---
        try:
            # vertexai.init() is generally lightweight setup, but run in thread just in case
            await asyncio.to_thread(vertexai.init, project=project_id, location=location, credentials=temp_creds)
            logger.info(f"[{request_id}] Vertex AI SDK initialized successfully for this request.")
        except google.auth.exceptions.GoogleAuthError as e:
             logger.error(f"[{request_id}] Vertex AI SDK Initialization Failed (Auth Error): {e}", exc_info=True)
             # This likely means the token from custom auth was invalid/expired
             raise HTTPException(status_code=401, detail=f"Vertex SDK Auth Init Error (Invalid Credentials?): {e}")
        except Exception as e:
            logger.error(f"[{request_id}] Vertex AI SDK Initialization Failed (Unexpected): {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Vertex SDK Init Error: {e}")


        # --- Convert Anthropic Request -> Intermediate LiteLLM/OpenAI Format ---
        litellm_request_dict = convert_anthropic_to_litellm(request_data)
        litellm_messages = litellm_request_dict.get("messages", [])
        litellm_tools = litellm_request_dict.get("tools") # Tools in OpenAI format
        system_prompt_text = litellm_request_dict.get("system_prompt") # Extracted system prompt

        # --- Convert Intermediate Format -> Vertex AI SDK Format ---
        vertex_history = convert_litellm_messages_to_vertex_content(litellm_messages)
        vertex_tools = convert_litellm_tools_to_vertex_tools(litellm_tools)
        vertex_system_instruction = Part.from_text(system_prompt_text) if system_prompt_text else None

        # --- Prepare Generation Config for Vertex AI ---
        generation_config = GenerationConfig(
            max_output_tokens=request_data.max_tokens,
            temperature=request_data.temperature,
            top_p=request_data.top_p,
            top_k=request_data.top_k,
            stop_sequences=request_data.stop_sequences if request_data.stop_sequences else None,
            # candidate_count=1 # Default is 1
        )
        logger.debug(f"[{request_id}] Vertex GenerationConfig: {generation_config}")

        # --- Safety settings (Using Vertex defaults for now) ---
        safety_settings = None
        # Example: Block Thresholds (adjust as needed)
        # safety_settings = {
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        # }

        # --- Tool Config (Handling Tool Choice - Basic Mapping) ---
        # Vertex AI ToolConfig is different from OpenAI tool_choice.
        # We can map 'none' and 'auto'. Forcing a specific tool is more complex.
        vertex_tool_config = None
        intermediate_tool_choice = litellm_request_dict.get("tool_choice")
        if intermediate_tool_choice == "none":
             # TODO: Check how to explicitly disable tools in Vertex SDK if needed.
             # For now, we just don't pass tools if choice is none? Or pass empty tool_config?
             # Let's try omitting tools list if choice is none.
             if vertex_tools:
                  logger.warning(f"[{request_id}] Tool choice is 'none', but tools were provided. Ignoring tools.")
                  vertex_tools = None # Effectively disable tools
        elif intermediate_tool_choice == "auto":
             # This is the default behavior if tools are provided. No specific config needed.
             pass
        elif isinstance(intermediate_tool_choice, dict) and intermediate_tool_choice.get("type") == "function":
             # Forcing a specific function call - Vertex SDK might require different structure.
             # For now, log a warning and proceed with 'auto' behavior.
             forced_tool_name = intermediate_tool_choice.get("function", {}).get("name")
             logger.warning(f"[{request_id}] Forcing specific tool '{forced_tool_name}' not fully implemented for Vertex SDK. Proceeding with auto tool choice.")
             # vertex_tool_config = ToolConfig(...) # Add specific Vertex config if available/needed

        # Log request details before calling API
        num_vertex_content = len(vertex_history) if vertex_history else 0
        num_vertex_tools = len(vertex_tools) if vertex_tools else 0
        # Log with mapped model ID for clarity on what's being called
        log_request_beautifully(raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, num_vertex_content, num_vertex_tools, 200) # Log success assumption


        # --- Instantiate Vertex AI Model ---
        model = GenerativeModel(
            actual_gemini_model_id,
            system_instruction=vertex_system_instruction
        )

        # --- Call Native Vertex AI SDK ---
        if request_data.stream:
            # --- Streaming Call ---
            logger.info(f"[{request_id}] Calling Vertex AI generate_content_async (streaming)")
            try:
                vertex_stream_generator = await model.generate_content_async(
                    contents=vertex_history,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    tools=vertex_tools,
                    # tool_config=vertex_tool_config, # Add if implemented
                    stream=True,
                )
            except google.api_core.exceptions.InvalidArgument as e:
                 logger.error(f"[{request_id}] Vertex API Invalid Argument Error (Check Request Structure/Schema): {e}", exc_info=True)
                 raise HTTPException(status_code=400, detail=f"Upstream API Invalid Argument: {e.message or str(e)}")
            except google.api_core.exceptions.GoogleAPICallError as e:
                 logger.error(f"[{request_id}] Vertex API Call Error (Streaming): {e}", exc_info=True)
                 http_status = getattr(e, 'code', 502) # Default to 502 Bad Gateway
                 raise HTTPException(status_code=http_status, detail=f"Upstream API Error (Streaming): {e.message or str(e)}")

            # Adapt the Vertex stream to LiteLLM/OpenAI stream format
            adapted_stream = adapt_vertex_stream_to_litellm(vertex_stream_generator, request_id, actual_gemini_model_id)

            # Use the existing handle_streaming function with the adapted stream
            headers = {
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Mapped-Model": actual_gemini_model_id, # Header shows the actual model used
                "X-Request-ID": request_id # Pass request ID back
            }
            # Pass request_data (with original_model_name) and request_id to handle_streaming
            return StreamingResponse(
                 handle_streaming(adapted_stream, request_data, request_id),
                 media_type="text/event-stream",
                 headers=headers
            )

        else:
            # --- Non-Streaming Call ---
            logger.info(f"[{request_id}] Calling Vertex AI generate_content_async (non-streaming)")
            try:
                vertex_response = await model.generate_content_async(
                    contents=vertex_history,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    tools=vertex_tools,
                    # tool_config=vertex_tool_config,
                    stream=False,
                )
                logger.debug(f"[{request_id}] Raw Vertex AI Non-Streaming Response: {vertex_response}")

            except google.api_core.exceptions.InvalidArgument as e:
                 logger.error(f"[{request_id}] Vertex API Invalid Argument Error (Check Request Structure/Schema): {e}", exc_info=True)
                 raise HTTPException(status_code=400, detail=f"Upstream API Invalid Argument: {e.message or str(e)}")
            except google.api_core.exceptions.GoogleAPICallError as e:
                 logger.error(f"[{request_id}] Vertex API Call Error (Non-Streaming): {e}", exc_info=True)
                 http_status = getattr(e, 'code', 502) # Default to 502 Bad Gateway
                 raise HTTPException(status_code=http_status, detail=f"Upstream API Error (Non-Streaming): {e.message or str(e)}")


            # --- Convert Vertex AI Response -> Intermediate LiteLLM/OpenAI Format ---
            litellm_like_response = convert_vertex_response_to_litellm(vertex_response, actual_gemini_model_id, request_id)

            # --- Convert Intermediate Format -> Final Anthropic Format ---
            anthropic_response = convert_litellm_to_anthropic(litellm_like_response, original_model_name) # Use original name here

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
        # Log and re-raise FastAPI HTTP exceptions, potentially adding request_id
        logger.error(f"[{request_id}] HTTPException during '/v1/messages': Status={e.status_code}, Detail={e.detail}", exc_info=(e.status_code >= 500))
        # Modify detail to include request ID?
        # e.detail = f"[{request_id}] {e.detail}" # Be careful modifying exception details
        raise e
    except Exception as e:
        # Catch-all for unexpected errors
        logger.critical(f"[{request_id}] Unhandled Exception during '/v1/messages': {type(e).__name__}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: [{request_id}] {str(e)}")


# --- Token Counting Endpoint (Using Native SDK) ---
@app.post("/v1/messages/count_tokens", response_model=TokenCountResponse)
async def count_tokens(request_data: TokenCountRequest, raw_request: Request):
    """ Estimates token count using the Native Vertex AI SDK's count_tokens method. """
    request_id = f"tok_{uuid.uuid4().hex[:12]}"
    start_time = time.time()
    token_count = 0 # Default
    status_code = 200 # Assume success initially

    # --- Request Parsing and Model Mapping ---
    try:
        raw_body = await raw_request.body()
        original_model_name = json.loads(raw_body.decode()).get("model", "unknown-request-body")
    except Exception:
        original_model_name = request_data.original_model_name or "unknown-fallback"
    request_data.original_model_name = original_model_name
    actual_gemini_model_id = request_data.model # Contains mapped ID

    logger.info(f"[{request_id}] Processing '/v1/messages/count_tokens': Original='{original_model_name}', Target SDK='{actual_gemini_model_id}'")

    try:
        # --- Custom Auth + Vertex Init (Required for SDK count_tokens) ---
        project_id, location, temp_creds = None, None, None
        try:
            project_id, location, temp_creds = await asyncio.to_thread(get_gemini_credentials)
            logger.info(f"[{request_id}] Auth successful for token count.")
            await asyncio.to_thread(vertexai.init, project=project_id, location=location, credentials=temp_creds)
            logger.info(f"[{request_id}] Vertex AI SDK initialized for token count.")
        except AuthenticationError as e:
            logger.error(f"[{request_id}] Auth Failed for token count: {e}")
            status_code = 503; raise HTTPException(status_code=status_code, detail=f"Authentication Service Error: {e}")
        except google.auth.exceptions.GoogleAuthError as e:
             logger.error(f"[{request_id}] Vertex SDK Init Failed for token count (Auth Error): {e}", exc_info=True)
             status_code = 401; raise HTTPException(status_code=status_code, detail=f"Vertex SDK Auth Init Error: {e}")
        except Exception as e:
            logger.error(f"[{request_id}] Error during Auth/Init for token count: {e}", exc_info=True)
            status_code = 500; raise HTTPException(status_code=status_code, detail=f"Auth/Init Error: {e}")

        # --- Convert messages for counting (using intermediate format) ---
        # Need to simulate a MessagesRequest for the conversion function
        simulated_msg_request = MessagesRequest(
             model=actual_gemini_model_id, # Pass the mapped name
             messages=request_data.messages,
             system=request_data.system,
             max_tokens=1 # Dummy value, not used by conversion
        )
        litellm_request_dict = convert_anthropic_to_litellm(simulated_msg_request)
        litellm_messages_for_count = litellm_request_dict.get("messages", [])
        system_prompt_text = litellm_request_dict.get("system_prompt")

        vertex_history = convert_litellm_messages_to_vertex_content(litellm_messages_for_count)
        vertex_system_instruction = Part.from_text(system_prompt_text) if system_prompt_text else None

        # --- Call SDK count_tokens ---
        model = GenerativeModel(
            actual_gemini_model_id,
            system_instruction=vertex_system_instruction # Pass system prompt here too
        )
        logger.info(f"[{request_id}] Calling Vertex AI count_tokens_async")
        count_response = await model.count_tokens_async(contents=vertex_history)
        token_count = count_response.total_tokens
        logger.info(f"[{request_id}] Vertex SDK token count successful: {token_count}")

        response = TokenCountResponse(input_tokens=token_count)
        response_headers = {"X-Mapped-Model": actual_gemini_model_id, "X-Request-ID": request_id}
        log_request_beautifully(raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, len(vertex_history), 0, status_code)
        return JSONResponse(content=response.dict(), headers=response_headers)

    # --- Fallback Estimation (Only if SDK call fails unexpectedly) ---
    except google.api_core.exceptions.GoogleAPICallError as e:
         logger.error(f"[{request_id}] Vertex SDK count_tokens failed: {e}", exc_info=True)
         status_code = getattr(e, 'code', 502)
         # Don't fallback here, report the upstream error
         log_request_beautifully(raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, 0, 0, status_code)
         raise HTTPException(status_code=status_code, detail=f"Upstream count_tokens error: {e.message or str(e)}")
    except HTTPException as e:
         # Log and re-raise if auth/init failed
         log_request_beautifully(raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, 0, 0, e.status_code)
         raise e
    except Exception as e: # Fallback to basic estimation for truly unexpected errors
        logger.error(f"[{request_id}] Unexpected error during token counting, falling back to estimation: {e}", exc_info=True)
        prompt_text = ""
        if request_data.system:
            system_text_fallback = request_data.system if isinstance(request_data.system, str) else "\n".join([b.text for b in request_data.system if hasattr(b, 'type') and b.type == "text"])
            prompt_text += system_text_fallback + "\n"
        for msg in request_data.messages:
            msg_content_fallback = ""
            if isinstance(msg.content, str):
                 msg_content_fallback = msg.content
            elif isinstance(msg.content, list):
                 msg_content_fallback = "\n".join([b.text for b in msg.content if hasattr(b, 'type') and b.type == "text"])
            prompt_text += msg_content_fallback + "\n"

        token_estimate = len(prompt_text) // 4 # Rough estimate
        logger.warning(f"[{request_id}] Using char/4 estimation: {token_estimate}")
        status_code = 200 # Return 200 but with an estimate
        log_request_beautifully(raw_request.method, raw_request.url.path, original_model_name, actual_gemini_model_id, len(request_data.messages), 0, status_code)
        return JSONResponse(
            content={"input_tokens": token_estimate},
            headers={"X-Mapped-Model": actual_gemini_model_id, "X-Request-ID": request_id, "X-Token-Estimation": "true"}
        )
    finally:
         processing_time = time.time() - start_time
         logger.info(f"[{request_id}] Token count request completed in {processing_time:.3f}s")


# --- Root Endpoint ---
@app.get("/", include_in_schema=False)
async def root():
    """ Root endpoint providing basic service info. """
    return {
        "message": "Anthropic API Compatible Proxy using Native Vertex AI SDK with Custom Gemini Auth",
        "status": "running",
        "target_gemini_models": {"BIG": GEMINI_BIG_MODEL, "SMALL": GEMINI_SMALL_MODEL},
        "litellm_version": getattr(litellm, '__version__', 'unknown'), # Still useful info
        "vertexai_version": getattr(vertexai, '__version__', 'unknown')
    }


# --- Middleware & Exception Handlers (Keep As Is) ---
@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    request_id = request.headers.get("X-Request-ID") or f"mw_{uuid.uuid4().hex[:12]}" # Get or generate ID

    # Log incoming request with ID
    logger.info(f"[{request_id}] Incoming request: {request.method} {request.url.path} from {client_host}")
    logger.debug(f"[{request_id}] Request Headers: {dict(request.headers)}")

    # Add request ID to request state for access in endpoints if needed
    request.state.request_id = request_id

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # Add request ID to response header
        response.headers["X-Request-ID"] = request_id

        # Log outgoing response with ID
        response_log_detail = f"status={response.status_code}, time={process_time:.3f}s"
        mapped_model = response.headers.get("X-Mapped-Model")
        if mapped_model: response_log_detail += f", mapped_model={mapped_model}"
        logger.info(f"[{request_id}] Response completed: {request.method} {request.url.path} ({response_log_detail})")
        logger.debug(f"[{request_id}] Response Headers: {dict(response.headers)}")
        return response

    except Exception as e:
        # Log unhandled exceptions caught by middleware (should be rare with endpoint handlers)
        process_time = time.time() - start_time
        logger.critical(f"[{request_id}] Unhandled exception in middleware/endpoint: {type(e).__name__} after {process_time:.3f}s", exc_info=True)
        # Return a generic 500 response
        return JSONResponse(
            status_code=500,
            content={"error": {"type": "internal_server_error", "message": f"[{request_id}] Internal Server Error"}},
            headers={"X-Request-ID": request_id}
        )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", f"ex_{uuid.uuid4().hex[:12]}")
    logger.error(f"[{request_id}] HTTPException: Status={exc.status_code}, Detail={exc.detail}", exc_info=(exc.status_code >= 500))
    return JSONResponse(
        status_code=exc.status_code,
        # Include type and message, potentially add request ID to message
        content={"error": {"type": "api_error", "message": f"[{request_id}] {exc.detail}"}},
        headers={"X-Request-ID": request_id} # Ensure ID is in error response
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", f"ex_{uuid.uuid4().hex[:12]}")
    logger.critical(f"[{request_id}] Unhandled Exception Handler: {type(exc).__name__}: {exc}", exc_info=True)
    # Basic error structure
    return JSONResponse(
        status_code=500,
        content={"error": {"type": "internal_server_error", "message": f"[{request_id}] Internal Server Error: {str(exc)}"}},
        headers={"X-Request-ID": request_id}
    )


# --- Server Startup ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8082))
    host = os.getenv("HOST", "0.0.0.0")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    # Simple reload detection based on env var or argument
    reload_flag = "--reload" in sys.argv or os.getenv("UVICORN_RELOAD", "false").lower() == "true"

    print(f"--- Starting Anthropic Proxy (Native Vertex SDK) ---")
    print(f" Listening on: {host}:{port}")
    print(f" Log Level:    {log_level}")
    print(f" Auto-Reload:  {reload_flag}")
    print(f" Target Models: BIG='{GEMINI_BIG_MODEL}', SMALL='{GEMINI_SMALL_MODEL}'")
    print(f" LiteLLM Ver:  {getattr(litellm, '__version__', 'unknown')}")
    print(f" VertexAI Ver: {getattr(vertexai, '__version__', 'unknown')}")
    if not os.getenv("WORKSPACE_ID") or not os.getenv("AUTH_URL"):
        print(f"{Colors.BOLD}{Colors.RED}!!! WARNING: WORKSPACE_ID or AUTH_URL environment variables not set! Authentication will fail. !!!{Colors.RESET}")
    print(f"----------------------------------------------------")

    # Use uvicorn.run to start the server
    uvicorn.run(
        "server:app", # Points to the FastAPI app instance in this file
        host=host,
        port=port,
        log_level=log_level,
        reload=reload_flag,
        # Consider adding access_log=False if logs are too noisy and handled by middleware/endpoint logs
        # access_log=False
    )