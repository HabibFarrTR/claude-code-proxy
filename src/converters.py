import base64
import json
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from vertexai.generative_models import (  # <-- Added FunctionCall
    Content,
    FinishReason,
    FunctionCall,
    FunctionDeclaration,
    GenerationResponse,
    Part,
    Tool,
)

from src.models import (
    ContentBlock,
    ContentBlockText,
    ContentBlockToolUse,
    MessagesRequest,
    MessagesResponse,
    Usage,
)
from src.utils import get_logger

logger = get_logger()


# This is still useful to get a standardized intermediate format (OpenAI-like)
# before converting to the native Vertex SDK format.
def convert_anthropic_to_litellm(request: MessagesRequest) -> Dict[str, Any]:
    """Converts Anthropic MessagesRequest to LiteLLM input dict (OpenAI format)."""
    litellm_messages = []

    # Handle System Prompt -> Becomes a separate parameter for Gemini SDK
    system_text = None
    if request.system:
        system_text = (
            request.system
            if isinstance(request.system, str)
            else "\n".join([b.text for b in request.system if b.type == "text"])
        )
        if system_text:
            # Store it separately, don't add to messages list for Vertex conversion later
            logger.debug("System prompt extracted for Vertex SDK.")
        else:
            system_text = None  # Ensure it's None if empty

    # Handle Messages -> Convert to OpenAI message format first
    for msg in request.messages:
        is_tool_response_message = False
        content_list = []  # For multimodal content within a single message
        tool_calls_list = []  # For assistant requesting tools

        if isinstance(msg.content, str):
            content_list.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            for block in msg.content:
                if block.type == "text":
                    content_list.append({"type": "text", "text": block.text})
                elif block.type == "image" and msg.role == "user":  # Images only supported for user role
                    content_list.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{block.source.media_type};base64,{block.source.data}"},
                        }
                    )
                    logger.debug("Image block added to intermediate format.")
                elif block.type == "tool_use" and msg.role == "assistant":
                    # Convert Anthropic tool_use to OpenAI tool_calls format
                    tool_calls_list.append(
                        {
                            "id": block.id,
                            "type": "function",
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.input),
                            },  # Arguments must be JSON string
                        }
                    )
                    logger.debug(f"Assistant tool_use '{block.name}' converted to intermediate tool_calls.")
                elif block.type == "tool_result" and msg.role == "user":
                    # If previous user text exists, send it first
                    if content_list:
                        litellm_messages.append({"role": "user", "content": content_list})
                        content_list = []  # Reset content list

                    # Convert Anthropic tool_result to OpenAI tool message format
                    tool_content = block.content
                    # Ensure content is a string (JSON if possible) for OpenAI format
                    if not isinstance(tool_content, str):
                        try:
                            tool_content = json.dumps(tool_content)
                        except Exception:
                            tool_content = str(tool_content)  # Fallback to string representation

                    litellm_messages.append(
                        {"role": "tool", "tool_call_id": block.tool_use_id, "content": tool_content}
                    )
                    logger.debug(f"User tool_result for '{block.tool_use_id}' converted to intermediate tool message.")
                    is_tool_response_message = True
                    break  # Process only the tool result block for this message

        # Add the assembled message if it wasn't a tool response handled above
        if not is_tool_response_message:
            litellm_msg = {"role": msg.role}
            # Simplify content if only text
            if len(content_list) == 1 and content_list[0]["type"] == "text":
                litellm_msg["content"] = content_list[0]["text"]
            elif content_list:  # Keep as list for multimodal
                litellm_msg["content"] = content_list
            else:
                litellm_msg["content"] = None  # Or empty string ""? Let's use None for clarity

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
        "model": request.model,  # Mapped Gemini ID
        "messages": litellm_messages,
        "max_tokens": request.max_tokens,
        "stream": request.stream or False,
    }
    # Add optional parameters
    if request.temperature is not None:
        litellm_request["temperature"] = request.temperature
    if request.top_p is not None:
        litellm_request["top_p"] = request.top_p
    if request.top_k is not None:
        litellm_request["top_k"] = request.top_k
    if request.stop_sequences:
        litellm_request["stop"] = request.stop_sequences  # For GenerationConfig later
    if request.metadata:
        litellm_request["metadata"] = request.metadata  # Keep metadata if needed downstream

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
            if "properties" in cleaned_schema and "type" not in cleaned_schema:
                cleaned_schema["type"] = "object"
            if cleaned_schema.get("type") == "object" and "properties" not in cleaned_schema:
                cleaned_schema["properties"] = {}  # Ensure properties key exists

            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": cleaned_schema,  # Use the cleaned schema
                    },
                }
            )
        if openai_tools:
            litellm_request["tools"] = openai_tools
            logger.debug(f"Converted {len(openai_tools)} tools to intermediate OpenAI format.")

    # Convert Anthropic Tool Choice to OpenAI Tool Choice Format
    # Note: Vertex has a different `tool_config`, this mapping might be approximate
    if request.tool_choice:
        choice_type = request.tool_choice.get("type")
        if choice_type == "any" or choice_type == "auto":
            litellm_request["tool_choice"] = "auto"  # Map to OpenAI 'auto'
        elif choice_type == "tool" and "name" in request.tool_choice:
            # Map to OpenAI specific function choice
            litellm_request["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice["name"]}}
        else:  # Includes 'none' or other types
            litellm_request["tool_choice"] = "none"  # Map to OpenAI 'none'
        logger.debug(
            f"Converted tool_choice '{choice_type}' to intermediate format '{litellm_request['tool_choice']}'."
        )

    logger.debug(
        f"Intermediate LiteLLM/OpenAI Request Prepared: {json.dumps(litellm_request, indent=2, default=lambda o: '<not serializable>')}"
    )
    return litellm_request


def clean_gemini_schema(schema: Any) -> Any:
    """Recursively removes fields unsupported by Gemini from a JSON schema dict."""
    if isinstance(schema, dict):
        # Remove fields known to cause issues with Gemini's schema validation
        schema.pop("additionalProperties", None)
        schema.pop("default", None)
        # Gemini might not support all string formats, remove if not explicitly supported
        if schema.get("type") == "string" and "format" in schema:
            # Keep only formats known to be generally safe or potentially useful
            # Removed 'enum' as it's not a standard format, handled by 'enum' keyword directly
            if schema["format"] not in {"date-time", "uri"}:  # Keep uri? Check Gemini docs. Let's keep for now.
                logger.debug(f"Removing potentially unsupported string format '{schema['format']}'")
                schema.pop("format")
        # Recurse into nested structures
        for key, value in list(schema.items()):  # Use list() for safe iteration while potentially popping
            if key in ["properties", "items"] or isinstance(value, (dict, list)):
                schema[key] = clean_gemini_schema(value)
            # Remove null values as they can sometimes cause issues
            elif value is None:
                # Check if the key is 'required' list, nulls are invalid there anyway
                if key == "required" and isinstance(schema[key], list):
                    schema[key] = [item for item in schema[key] if item is not None]
                    if not schema[key]:  # Remove empty required list
                        schema.pop(key)
                elif key != "required":  # Don't remove required itself if it becomes null, but remove other nulls
                    logger.debug(f"Removing null value for key '{key}'")
                    schema.pop(key)

    elif isinstance(schema, list):
        # Also clean items within lists (e.g., in 'required' list or 'enum' list)
        return [clean_gemini_schema(item) for item in schema if item is not None]  # Remove None items from lists too
    return schema


# Conversion from LiteLLM/OpenAI format to Vertex AI SDK format +++
def convert_litellm_tools_to_vertex_tools(litellm_tools: Optional[List[Dict]]) -> Optional[List[Tool]]:
    """
    Converts LiteLLM/OpenAI tools list (from intermediate format) to a list
    containing a SINGLE Vertex AI SDK Tool object, which in turn contains
    all function declarations. Returns None if no valid tools are found.
    """
    if not litellm_tools:
        return None

    all_function_declarations: List[FunctionDeclaration] = []  # Collect declarations here

    for tool in litellm_tools:
        if tool.get("type") == "function":
            func_data = tool.get("function", {})
            name = func_data.get("name")
            description = func_data.get("description")
            parameters_schema = func_data.get("parameters")  # Schema should already be cleaned

            if name and parameters_schema is not None:  # Allow empty schema {}
                try:
                    # Create the FunctionDeclaration
                    func_decl = FunctionDeclaration(
                        name=name, description=description or "", parameters=parameters_schema
                    )
                    # Add the declaration to our list
                    all_function_declarations.append(func_decl)
                    logger.debug(f"Collected FunctionDeclaration for tool '{name}'.")
                except Exception as e:
                    # Log detailed error if schema validation fails at SDK level
                    logger.error(
                        f"Failed to create Vertex FunctionDeclaration for tool '{name}': {e}. Schema: {parameters_schema}",
                        exc_info=True,
                    )
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
    request_id_for_logging = "conv_test"  # Placeholder for logging context

    for i, msg in enumerate(litellm_messages):
        role = msg.get("role")
        intermediate_content = msg.get("content")  # Content from OpenAI-like format
        tool_calls = msg.get("tool_calls")  # OpenAI format tool_calls
        tool_call_id = msg.get("tool_call_id")  # OpenAI format tool_call_id (for role='tool')

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
            vertex_role = "user"  # Tool results ARE a user turn
            is_tool_result_message = True
        elif role == "system":
            logger.warning(
                f"[{request_id_for_logging}] System message found in list; should have been handled separately. Skipping."
            )
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
                        logger.debug(
                            f"[{request_id_for_logging}] Tool result content for id {tool_call_id} parsed as JSON."
                        )
                    else:
                        tool_response_dict["output"] = intermediate_content
                        logger.warning(
                            f"[{request_id_for_logging}] Tool result content for id {tool_call_id} was not a string, using as is: {type(intermediate_content)}"
                        )
                except (json.JSONDecodeError, TypeError):
                    tool_response_dict["output"] = intermediate_content
                    logger.debug(
                        f"[{request_id_for_logging}] Tool result content for id {tool_call_id} is not JSON, sending as string under 'output' key."
                    )
                except Exception as e:
                    logger.error(
                        f"[{request_id_for_logging}] Error processing tool result content for {tool_call_id}: {e}. Content: {str(intermediate_content)[:100]}..."
                    )
                    tool_response_dict["output"] = f"Error processing content: {e}"

                original_func_name = "unknown_function"
                for j in range(i - 1, -1, -1):
                    prev_msg = litellm_messages[j]
                    if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc.get("id") == tool_call_id:
                                original_func_name = tc.get("function", {}).get("name", original_func_name)
                                break
                        if original_func_name != "unknown_function":
                            break

                if original_func_name == "unknown_function":
                    logger.warning(
                        f"[{request_id_for_logging}] Could not find original function name for tool_call_id '{tool_call_id}'. Using placeholder name."
                    )

                try:
                    function_response_part = Part.from_function_response(
                        name=original_func_name, response=tool_response_dict
                    )
                    current_turn_parts.append(function_response_part)
                    logger.debug(
                        f"[{request_id_for_logging}] Created Part.from_function_response for tool '{original_func_name}' (id: {tool_call_id})."
                    )
                except Exception as e:
                    logger.error(
                        f"[{request_id_for_logging}] Failed to create Part.from_function_response for {original_func_name}: {e}",
                        exc_info=True,
                    )
                    current_turn_parts.append(
                        Part.from_text(f"[Error creating tool response part for {original_func_name}: {e}]")
                    )

            elif not hasattr(Part, "from_function_response"):
                logger.error(
                    f"[{request_id_for_logging}] Skipping tool result conversion: Part.from_function_response not found in SDK."
                )
                continue
            else:
                logger.warning(
                    f"[{request_id_for_logging}] Skipping tool message conversion due to missing tool_call_id or content: {msg}"
                )
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
                                part_for_call = Part.from_dict(
                                    {
                                        "function_call": {
                                            "name": vertex_function_call.name,
                                            "args": vertex_function_call.args,
                                        }
                                    }
                                )
                                current_turn_parts.append(part_for_call)
                                logger.debug(
                                    f"[{request_id_for_logging}] Created Part containing FunctionCall via from_dict: {func_name}({func_args})"
                                )
                            except Exception as e_dict:
                                logger.error(
                                    f"[{request_id_for_logging}] Failed to create Part via from_dict: {e_dict}",
                                    exc_info=True,
                                )
                                # If from_dict fails, we have no reliable way to create this part based on previous errors.
                                # Adding an error text part might be better than skipping the turn.
                                current_turn_parts.append(
                                    Part.from_text(f"[ERROR: Failed to construct function_call part for {func_name}]")
                                )
                            # --- !!! END FIX (v4) !!! ---
                        else:
                            logger.warning(
                                f"[{request_id_for_logging}] Skipping assistant tool call part due to missing function name: {tc}"
                            )
                    except json.JSONDecodeError:
                        logger.error(
                            f"[{request_id_for_logging}] Failed to parse function arguments JSON for assistant tool call '{func_name}': {func_args_str}",
                            exc_info=True,
                        )
                    except Exception as e:  # Catch potential errors during FunctionCall creation itself
                        logger.error(
                            f"[{request_id_for_logging}] Failed to create FunctionCall object for {func_name}: {e}",
                            exc_info=True,
                        )
                else:
                    logger.warning(
                        f"[{request_id_for_logging}] Skipping non-function tool call part in assistant message: {tc}"
                    )

        # 3. Handle Regular Text/Image Content (Non-tool related parts)
        # (Logic remains the same as v2/v3)
        elif intermediate_content:
            if isinstance(intermediate_content, str):
                current_turn_parts.append(Part.from_text(intermediate_content))
            elif isinstance(intermediate_content, list):
                if role != "user":
                    logger.warning(
                        f"[{request_id_for_logging}] Multimodal content (list) found for role '{role}', but only 'user' role typically supports images. Processing anyway."
                    )
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
                                logger.error(
                                    f"[{request_id_for_logging}] Failed to process base64 image data URL: {e}",
                                    exc_info=True,
                                )
                                current_turn_parts.append(Part.from_text(f"[Error processing image: {e}]"))
                        else:
                            logger.warning(
                                f"[{request_id_for_logging}] Skipping image URL that is not a data URL: {image_url_data[:100]}..."
                            )
                            current_turn_parts.append(Part.from_text("[Skipped non-data image URL]"))
                    elif item_type == "image_url" and role != "user":
                        logger.warning(f"[{request_id_for_logging}] Skipping image part for non-user role '{role}'.")
                        current_turn_parts.append(Part.from_text("[Skipped image for non-user role]"))
                    else:
                        logger.warning(
                            f"[{request_id_for_logging}] Unsupported item type in multimodal content list: {item_type}"
                        )
            else:
                logger.warning(
                    f"[{request_id_for_logging}] Unsupported content type for role '{role}': {type(intermediate_content)}"
                )

        # --- Append Parts to Vertex Content List ---
        if current_turn_parts:
            valid_parts = [p for p in current_turn_parts if isinstance(p, Part)]
            if len(valid_parts) != len(current_turn_parts):
                logger.warning(
                    f"[{request_id_for_logging}] Some generated items were not valid Part objects and were filtered out."
                )
            if not valid_parts:
                logger.warning(
                    f"[{request_id_for_logging}] Message with role '{role}' resulted in no valid parts after processing. Skipping message."
                )
                continue

            # --- Refined Merge Logic (v2 - unchanged) ---
            should_merge = False
            if vertex_content_list:
                last_content = vertex_content_list[-1]
                if last_content.role == vertex_role:
                    is_last_content_a_tool_response_turn = any(
                        hasattr(p, "function_response") and p.function_response for p in last_content.parts
                    )
                    if is_tool_result_message:
                        if is_last_content_a_tool_response_turn:
                            should_merge = True
                            logger.debug(
                                f"[{request_id_for_logging}] Merging tool result part onto existing user tool response turn."
                            )
                    else:
                        if not is_last_content_a_tool_response_turn:
                            should_merge = True
                            logger.debug(f"[{request_id_for_logging}] Merging standard parts (non-tool-result).")
            # --- End Refined Merge Logic (v2) ---

            if should_merge:
                logger.debug(
                    f"[{request_id_for_logging}] Merging {len(valid_parts)} parts into previous Content object (role: {vertex_role})"
                )
                vertex_content_list[-1].parts.extend(valid_parts)
            else:
                if vertex_role:
                    vertex_content_list.append(Content(parts=valid_parts, role=vertex_role))
                    logger.debug(
                        f"[{request_id_for_logging}] Created new Content object (role: {vertex_role}) with {len(valid_parts)} part(s): {[type(p).__name__ for p in valid_parts]}"
                    )
                else:
                    logger.error(
                        f"[{request_id_for_logging}] Cannot create Content object without a valid Vertex role (was '{role}'). Skipping parts."
                    )

        elif role != "system":
            logger.warning(f"[{request_id_for_logging}] Message with role '{role}' resulted in no parts, skipping.")

    # --- Final Logging ---
    # (Logging remains the same as v2/v3)
    logger.info(
        f"[{request_id_for_logging}] Converted {len(litellm_messages)} intermediate messages -> {len(vertex_content_list)} Vertex Content objects."
    )
    try:
        final_history_repr = []
        for content_idx, content in enumerate(vertex_content_list):
            part_reprs = []
            for p_idx, p in enumerate(content.parts):
                part_type = "Unknown"
                part_detail = ""
                if hasattr(p, "function_call") and getattr(p, "function_call", None):
                    part_type = "FunctionCall"
                    part_detail = f"({p.function_call.name})"
                elif hasattr(p, "function_response") and getattr(p, "function_response", None):
                    part_type = "FunctionResponse"
                    part_detail = f"({p.function_response.name})"
                elif hasattr(p, "text") and getattr(p, "text", None) is not None:
                    part_type = "Text"
                    part_detail = f"({len(p.text)} chars)"
                elif hasattr(p, "inline_data") and getattr(p, "inline_data", None):
                    part_type = "Data"
                    part_detail = f"({p.inline_data.mime_type})"
                else:
                    part_type = type(p).__name__
                part_reprs.append(f"{part_type}{part_detail}")
            final_history_repr.append({"index": content_idx, "role": content.role, "part_types": part_reprs})
        logger.debug(
            f"[{request_id_for_logging}] Final Vertex history structure: {json.dumps(final_history_repr, indent=2)}"
        )
    except Exception as e:
        logger.debug(f"[{request_id_for_logging}] Could not serialize final Vertex history structure for logging: {e}")

    return vertex_content_list


# Converts the *adapted* LiteLLM/OpenAI format (from Vertex response) back to Anthropic Non-Streaming Response
def convert_litellm_to_anthropic(
    response_chunk: Union[Dict, Any], original_model_name: str
) -> Optional[MessagesResponse]:
    """Converts non-streaming LiteLLM/OpenAI format response (dict or object) to Anthropic MessagesResponse."""
    request_id = response_chunk.get("request_id", "unknown")  # Get request ID if passed through
    logger.info(f"[{request_id}] Converting adapted LiteLLM/OpenAI response to Anthropic MessagesResponse format.")
    try:
        # Ensure input is a dictionary
        resp_dict = {}
        if isinstance(response_chunk, dict):
            resp_dict = response_chunk
        elif hasattr(response_chunk, "model_dump"):  # Pydantic v2
            resp_dict = response_chunk.model_dump()
        elif hasattr(response_chunk, "dict"):  # Pydantic v1
            resp_dict = response_chunk.dict()
        else:
            try:
                resp_dict = vars(response_chunk)  # Fallback for simple objects
            except TypeError:
                logger.error(f"[{request_id}] Cannot convert response_chunk of type {type(response_chunk)} to dict.")
                raise ValueError("Input response_chunk is not convertible to dict.")

        # Extract data using .get for safety
        resp_id = resp_dict.get("id", f"msg_{uuid.uuid4().hex[:24]}")
        choices = resp_dict.get("choices", [])
        usage_data = resp_dict.get("usage", {}) or {}  # Ensure usage is a dict

        anthropic_content: List[ContentBlock] = []
        # Map OpenAI finish reasons to Anthropic stop reasons
        stop_reason_map = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "content_filtered",
            # Add mappings for any other potential finish reasons if needed
        }
        openai_finish_reason = "stop"  # Default

        if choices:
            choice = choices[0]  # Assume only one choice
            openai_finish_reason = choice.get("finish_reason", "stop")
            message = choice.get("message", {}) or {}  # Ensure message is a dict

            text_content = message.get("content")
            tool_calls = message.get("tool_calls")  # List of tool calls made by the assistant

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
                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}")  # Use provided ID or generate one
                        tool_name = func.get("name", "unknown_tool")

                        # Parse arguments JSON string back into a dict for Anthropic input
                        try:
                            args_input = json.loads(args_str)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{request_id}] Non-streaming: Failed to parse tool arguments JSON: {args_str}. Sending raw string."
                            )
                            args_input = {"raw_arguments": args_str}
                        except Exception as e:
                            logger.error(
                                f"[{request_id}] Non-streaming: Error parsing tool arguments: {e}. Args: {args_str}"
                            )
                            args_input = {"error_parsing_arguments": str(e), "raw_arguments": args_str}

                        anthropic_content.append(
                            ContentBlockToolUse(type="tool_use", id=tool_id, name=tool_name, input=args_input)
                        )
                        logger.debug(f"[{request_id}] Added tool_use content block: id={tool_id}, name={tool_name}")
                    else:
                        logger.warning(
                            f"[{request_id}] Skipping conversion of non-function tool_call in response: {tc}"
                        )

        # Ensure there's always at least one content block (even if empty text)
        # Anthropic requires content to be a non-empty list.
        if not anthropic_content:
            logger.warning(f"[{request_id}] No content generated, adding empty text block.")
            anthropic_content.append(ContentBlockText(type="text", text=""))

        # Map the finish reason
        anthropic_stop_reason = stop_reason_map.get(openai_finish_reason, "end_turn")
        logger.debug(
            f"[{request_id}] Mapped finish_reason '{openai_finish_reason}' to stop_reason '{anthropic_stop_reason}'."
        )

        # Create the final Anthropic response object
        return MessagesResponse(
            id=resp_id,
            model=original_model_name,  # Use the original model name requested by the client
            type="message",
            role="assistant",
            content=anthropic_content,
            stop_reason=anthropic_stop_reason,
            stop_sequence=None,  # OpenAI format doesn't typically return the sequence matched
            usage=Usage(
                input_tokens=usage_data.get("prompt_tokens", 0), output_tokens=usage_data.get("completion_tokens", 0)
            ),
        )
    except Exception as e:
        # Log detailed error during conversion
        logger.error(
            f"[{request_id}] Failed to convert adapted LiteLLM/OpenAI response to Anthropic format: {e}", exc_info=True
        )
        # Return a minimal error response in Anthropic format
        return MessagesResponse(
            id=f"error_{uuid.uuid4().hex[:24]}",
            model=original_model_name,
            type="message",
            role="assistant",
            content=[ContentBlockText(type="text", text=f"Error processing model response: {str(e)}")],
            stop_reason="end_turn",  # Or maybe a custom error reason?
            usage=Usage(input_tokens=0, output_tokens=0),
        )


# Converts the *adapted* LiteLLM/OpenAI stream (from Vertex stream) to Anthropic SSE Stream
async def convert_litellm_to_anthropic_sse(
    response_generator: AsyncGenerator[Dict[str, Any], None], request: MessagesRequest, request_id: str
):
    """Converts adapted LiteLLM/OpenAI format async generator to Anthropic SSE stream."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    # Use the original model name provided by the client for Anthropic events
    response_model_name = request.original_model_name or request.model  # Fallback to mapped ID if original is missing
    logger.info(
        f"[{request_id}] Starting Anthropic SSE stream conversion (message {message_id}, model: {response_model_name})"
    )

    # --- Stream Initialization ---
    # 1. Send message_start event
    start_event_data = {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": response_model_name,
            "content": [],  # Content starts empty
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},  # Initial usage
        },
    }
    yield f"event: message_start\ndata: {json.dumps(start_event_data)}\n\n"
    logger.debug(f"[{request_id}] Sent message_start")

    # 2. Send initial ping event
    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
    logger.debug(f"[{request_id}] Sent initial ping")

    # --- Stream Processing ---
    content_block_index = -1  # Track the index of the current content block (text or tool_use)
    current_block_type: Optional[Literal["text", "tool_use"]] = None
    text_started = False  # Flag to track if the current text block has been started
    tool_calls_buffer = (
        {}
    )  # Buffer to assemble tool call arguments {openai_tc_index: {id: str, name: str, args: str, block_idx: int}}
    final_usage = {"input_tokens": 0, "output_tokens": 0}  # Accumulate usage
    final_stop_reason: Optional[str] = None  # Store the final stop reason

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
            delta = choice.get("delta", {}) or {}  # Ensure delta is a dict
            finish_reason = choice.get("finish_reason")  # OpenAI finish reason

            # --- Accumulate Usage from final chunk ---
            # Usage info might appear in the last chunk along with finish_reason
            chunk_usage = chunk.get("usage")
            if chunk_usage and isinstance(chunk_usage, dict):
                # Only update if values are present and > 0, prefer existing values if chunk has 0
                final_usage["input_tokens"] = chunk_usage.get("prompt_tokens") or final_usage["input_tokens"]
                final_usage["output_tokens"] = chunk_usage.get("completion_tokens") or final_usage["output_tokens"]
                logger.debug(f"[{request_id}] Updated usage from chunk: {final_usage}")
                # Send ping after receiving usage (often in final chunk)
                yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"
                logger.debug(f"[{request_id}] Sent ping after usage update")

            # --- Process Delta Content ---
            text_delta = delta.get("content")
            tool_calls_delta = delta.get("tool_calls")  # List of tool call deltas

            # 1. Handle Text Delta
            if text_delta and isinstance(text_delta, str):
                # If currently in a tool_use block, stop it first
                if current_block_type == "tool_use":
                    last_tool_block_idx = tool_calls_buffer[max(tool_calls_buffer.keys())]["block_idx"]
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
                        "content_block": {"type": "text", "text": ""},  # Start with empty text
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                    logger.debug(f"[{request_id}] Started text block {content_block_index}")

                # Send the text delta
                delta_event = {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": text_delta},
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
                    if not isinstance(tc_delta, dict):
                        continue  # Skip invalid format

                    # OpenAI tool index (usually 0 for the first tool, 1 for second, etc.)
                    # We rely on this index to aggregate arguments for the *same* tool call.
                    tc_openai_index = tc_delta.get("index", 0)
                    tc_id = tc_delta.get("id")  # ID for the specific tool call instance
                    func_delta = tc_delta.get("function", {}) or {}
                    func_name = func_delta.get("name")
                    args_delta = func_delta.get("arguments")  # Argument JSON string fragment

                    # --- Start a new tool_use block if necessary ---
                    if tc_openai_index not in tool_calls_buffer:
                        # Need ID and Name to start the Anthropic block
                        if tc_id and func_name:
                            content_block_index += 1
                            current_block_type = "tool_use"
                            tool_calls_buffer[tc_openai_index] = {
                                "id": tc_id,
                                "name": func_name,
                                "args": "",  # Initialize empty args string
                                "block_idx": content_block_index,  # Store the Anthropic block index
                            }
                            start_event = {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tc_id,
                                    "name": func_name,
                                    "input": {},
                                },  # Input starts empty
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                            logger.debug(
                                f"[{request_id}] Started tool_use block {content_block_index} (id: {tc_id}, name: {func_name})"
                            )
                        # Handle case where ID might come first, then name in a later chunk (less common now)
                        elif tc_id and not func_name:
                            tool_calls_buffer[tc_openai_index] = {
                                "id": tc_id,
                                "name": None,
                                "args": "",
                                "block_idx": None,
                            }
                            logger.debug(
                                f"[{request_id}] Received tool ID {tc_id} first for index {tc_openai_index}, waiting for name."
                            )
                        else:
                            logger.warning(
                                f"[{request_id}] Cannot start tool block for index {tc_openai_index} without ID and/or Name. Delta: {tc_delta}"
                            )
                            continue  # Cannot start block yet

                    # --- If name arrives later for an existing ID ---
                    elif (
                        tc_openai_index in tool_calls_buffer
                        and func_name
                        and tool_calls_buffer[tc_openai_index]["name"] is None
                    ):
                        tool_info = tool_calls_buffer[tc_openai_index]
                        if tool_info["id"] == tc_id:  # Ensure ID matches if provided again
                            content_block_index += 1
                            current_block_type = "tool_use"
                            tool_info["name"] = func_name
                            tool_info["block_idx"] = content_block_index
                            start_event = {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_info["id"],
                                    "name": func_name,
                                    "input": {},
                                },
                            }
                            yield f"event: content_block_start\ndata: {json.dumps(start_event)}\n\n"
                            logger.debug(
                                f"[{request_id}] Started tool_use block {content_block_index} for index {tc_openai_index} after receiving name ({func_name})"
                            )
                        else:
                            logger.warning(
                                f"[{request_id}] Received name '{func_name}' for index {tc_openai_index}, but ID mismatch (expected {tool_info['id']}, got {tc_id}). Skipping."
                            )

                    # --- Append argument fragments if block has started ---
                    if (
                        tc_openai_index in tool_calls_buffer
                        and args_delta
                        and tool_calls_buffer[tc_openai_index]["block_idx"] is not None
                    ):
                        tool_info = tool_calls_buffer[tc_openai_index]
                        tool_info["args"] += args_delta  # Append the JSON fragment
                        # Send Anthropic input_json_delta
                        delta_event = {
                            "type": "content_block_delta",
                            "index": tool_info["block_idx"],
                            "delta": {"type": "input_json_delta", "partial_json": args_delta},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"
                        logger.debug(
                            f"[{request_id}] Sent tool args delta for block {tool_info['block_idx']}: '{args_delta[:50]}...'"
                        )

            # --- Process Finish Reason ---
            if finish_reason:
                # Map OpenAI finish reason to Anthropic stop reason
                final_stop_reason = stop_reason_map.get(finish_reason, "end_turn")
                logger.info(
                    f"[{request_id}] Received final finish_reason: '{finish_reason}' -> Mapped to stop_reason: '{final_stop_reason}'"
                )
                # The loop will break after processing this chunk's content (if any)
                break  # Exit loop after processing the chunk containing the finish reason

        # --- End of Stream ---
        # 1. Stop the last active content block (if any)
        if current_block_type == "text" and text_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_index})}\n\n"
            logger.debug(f"[{request_id}] Stopped final text block {content_block_index}")
        elif current_block_type == "tool_use":
            # Find the index of the last tool block started
            if tool_calls_buffer:
                last_tool_block_idx = tool_calls_buffer[max(tool_calls_buffer.keys())]["block_idx"]
                if last_tool_block_idx is not None:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': last_tool_block_idx})}\n\n"
                    logger.debug(f"[{request_id}] Stopped final tool_use block {last_tool_block_idx}")
            else:
                logger.warning(
                    f"[{request_id}] Current block type is tool_use, but buffer is empty. Cannot stop block."
                )

        # 2. Send final message_delta event with stop reason and accumulated output tokens
        if final_stop_reason is None:
            logger.warning(
                f"[{request_id}] Stream finished without receiving a finish_reason. Defaulting to 'end_turn'."
            )
            final_stop_reason = "end_turn"

        final_delta_event = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": None,  # Not typically provided by OpenAI stream
            },
            "usage": {"output_tokens": final_usage.get("output_tokens", 0)},  # Send accumulated output tokens
        }
        yield f"event: message_delta\ndata: {json.dumps(final_delta_event)}\n\n"
        logger.debug(
            f"[{request_id}] Sent final message_delta (stop_reason: {final_stop_reason}, usage: {final_delta_event['usage']})"
        )

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
                "error": {"type": "internal_server_error", "message": f"Stream processing error: {str(e)}"},
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


# Adapter for Vertex AI stream to LiteLLM/OpenAI stream format +++
async def adapt_vertex_stream_to_litellm(
    vertex_stream: AsyncGenerator[GenerationResponse, None], request_id: str, model_id_for_chunk: str
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Adapts the native Vertex AI SDK stream chunks (GenerationResponse)
    to the OpenAI streaming chunk format expected by handle_streaming.
    Handles parts containing either text or function calls safely.
    Injects an error message for MALFORMED_FUNCTION_CALL finish reason.
    """
    logger.info(f"[{request_id}] Starting Vertex AI stream adaptation...")
    current_tool_calls = {}
    openai_tool_index_counter = 0
    tool_call_emitted_in_stream = False  # Flag to track if any tool call was processed
    malformed_call_detected = False
    malformed_call_message = None

    try:
        async for chunk in vertex_stream:
            logger.debug(f"[{request_id}] Raw Vertex Chunk: {chunk}")

            delta_content = None
            delta_tool_calls = []
            vertex_finish_reason_enum = None
            usage_metadata = None
            chunk_finish_message = None  # Store finish message if present

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

                # Store the raw Vertex finish reason and message if present
                if candidate.finish_reason and candidate.finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED:
                    vertex_finish_reason_enum = candidate.finish_reason
                    chunk_finish_message = getattr(candidate, "finish_message", None)  # Get finish message safely
                    logger.info(
                        f"[{request_id}] Received Vertex finish reason enum: {vertex_finish_reason_enum.name}, Message: {chunk_finish_message}"
                    )

                    # Detect Malformed Function Call ***
                    if vertex_finish_reason_enum == FinishReason.MALFORMED_FUNCTION_CALL:
                        malformed_call_detected = True
                        malformed_call_message = chunk_finish_message
                        logger.error(
                            f"[{request_id}] MALFORMED_FUNCTION_CALL detected by Vertex API. Message: {malformed_call_message}"
                        )
                        # Do not process parts in this chunk if it's the final error chunk
                        # The API likely doesn't include valid parts alongside this error.
                        # We will handle yielding an error message later, before the final chunk.

                # Process Parts for Content and Tool Calls ONLY IF no malformed call detected in *this chunk*
                # (It's possible to get content chunks *before* the final error chunk)
                if not malformed_call_detected and candidate.content and candidate.content.parts:
                    for part_index, part in enumerate(candidate.content.parts):
                        # --- Safely check for text or function_call ---
                        try:
                            # Attempt to access text. If it exists, process it.
                            current_text = part.text
                            delta_content = current_text
                            logger.debug(f"[{request_id}] Vertex text delta: '{delta_content[:50]}...'")
                        except AttributeError:
                            # .text failed, so it's not a simple text part.
                            # Now check if it's a function call part.
                            try:
                                current_function_call = part.function_call
                                # Process the function call
                                tool_call_emitted_in_stream = True  # Set the flag
                                fc = current_function_call
                                tool_call_id = f"toolu_{uuid.uuid4().hex[:12]}"
                                try:
                                    # Ensure args is serializable, default to empty dict string if None/empty
                                    args_str = json.dumps(fc.args) if fc.args else "{}"
                                except Exception as e:
                                    logger.error(
                                        f"[{request_id}] Failed to dump streaming function call args to JSON: {e}. Args: {fc.args}"
                                    )
                                    args_str = json.dumps({"error": f"Failed to serialize args: {e}"})

                                openai_tool_call = {
                                    "index": openai_tool_index_counter,
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {"name": fc.name, "arguments": args_str},
                                }
                                delta_tool_calls.append(openai_tool_call)
                                # Store info needed for potential future reference if needed (though not currently used downstream)
                                current_tool_calls[part_index] = {
                                    "id": tool_call_id,
                                    "name": fc.name,
                                    "args": args_str,
                                    "openai_index": openai_tool_index_counter,
                                }
                                logger.debug(
                                    f"[{request_id}] Vertex tool call delta: index={openai_tool_index_counter}, id={tool_call_id}, name={fc.name}, args='{args_str[:50]}...'"
                                )
                                openai_tool_index_counter += 1
                            except AttributeError:
                                # .function_call also failed. Log an unknown part type.
                                logger.warning(
                                    f"[{request_id}] Unknown or unexpected part type encountered in Vertex stream chunk (not text or function_call): {part}"
                                )
                            except Exception as e:
                                # Catch any other unexpected error during part processing
                                logger.error(
                                    f"[{request_id}] Unexpected error processing function call part {part_index}: {e}",
                                    exc_info=True,
                                )
                        except Exception as e:
                            # Catch any other unexpected error during part processing
                            logger.error(
                                f"[{request_id}] Unexpected error processing part {part_index}: {e}", exc_info=True
                            )

            # --- Construct and Yield the LiteLLM/OpenAI Delta Chunk ---
            # Only yield if we have content/tool calls OR if it's the final chunk with usage/finish reason
            openai_delta_for_choice = {}
            if delta_content is not None:
                openai_delta_for_choice["content"] = delta_content
            if delta_tool_calls:
                openai_delta_for_choice["tool_calls"] = delta_tool_calls
            # Add role only if there's content or tool calls in this delta
            if openai_delta_for_choice:
                openai_delta_for_choice["role"] = "assistant"

            # Yield chunk if it contains actual delta content or tool calls
            if openai_delta_for_choice:
                litellm_chunk = {
                    "id": f"chatcmpl-adap-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id_for_chunk,
                    "choices": [
                        {"index": 0, "delta": openai_delta_for_choice, "finish_reason": None, "logprobs": None}
                    ],
                    "usage": None,  # Usage usually comes in the final chunk
                }
                logger.debug(f"[{request_id}] Yielding adapted LiteLLM chunk: {litellm_chunk}")
                yield litellm_chunk

            # --- Check if this chunk signals the end (either normally or via malformed call) ---
            if vertex_finish_reason_enum:
                # Yield Error Message BEFORE Final Chunk if Malformed Call ***
                if malformed_call_detected:
                    error_message = "[Proxy Error: The model generated an invalid tool call and could not complete the request. Please try rephrasing.]"
                    # Optionally include part of the raw error if useful for debugging, but keep it concise for the user
                    # if malformed_call_message:
                    #    error_message += f" (Details: {malformed_call_message[:100]}...)"

                    error_delta = {"role": "assistant", "content": error_message}
                    error_chunk = {
                        "id": f"chatcmpl-adap-{request_id}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id_for_chunk,
                        "choices": [{"index": 0, "delta": error_delta, "finish_reason": None, "logprobs": None}],
                        "usage": None,
                    }
                    logger.warning(
                        f"[{request_id}] Yielding injected error message chunk due to MALFORMED_FUNCTION_CALL."
                    )
                    yield error_chunk

                # --- Determine and Yield Final Chunk ---
                final_openai_finish_reason = "stop"  # Default

                # IMPORTANT: Check if the *reason* indicates tool calls should have happened,
                # even if the call itself was malformed. Gemini might use OTHER or STOP even
                # when attempting a call that fails validation.
                # Let's prioritize the malformed detection.
                if malformed_call_detected:
                    # Map malformed call to 'stop' for OpenAI format, but the text message above explains the real issue.
                    final_openai_finish_reason = "stop"
                    logger.info(
                        f"[{request_id}] Mapping final Vertex finish_reason MALFORMED_FUNCTION_CALL -> 'stop' (error message injected separately)"
                    )
                elif tool_call_emitted_in_stream:
                    # If valid tool calls were successfully emitted earlier in the stream, AND the finish reason isn't something overriding like MAX_TOKENS or SAFETY
                    # It's *possible* the final finish reason from Vertex could still be STOP even after a successful tool call part.
                    # Let's make the finish reason 'tool_calls' if we emitted any, unless it's explicitly MAX_TOKENS or SAFETY.
                    if vertex_finish_reason_enum not in [
                        FinishReason.MAX_TOKENS,
                        FinishReason.SAFETY,
                        FinishReason.RECITATION,
                    ]:
                        final_openai_finish_reason = "tool_calls"
                        logger.info(
                            f"[{request_id}] Setting final finish_reason to 'tool_calls' because valid tool calls were emitted."
                        )
                    else:
                        # Map MAX_TOKENS/SAFETY/RECITATION appropriately
                        finish_reason_map = {
                            FinishReason.MAX_TOKENS.name: "length",
                            FinishReason.SAFETY.name: "content_filter",
                            FinishReason.RECITATION.name: "content_filter",
                        }
                        final_openai_finish_reason = finish_reason_map.get(vertex_finish_reason_enum.name, "stop")
                        logger.info(
                            f"[{request_id}] Mapping final Vertex finish_reason {vertex_finish_reason_enum.name} -> {final_openai_finish_reason} (despite prior tool calls)"
                        )
                else:
                    # No tool calls emitted, map normal finish reasons
                    finish_reason_map = {
                        FinishReason.STOP.name: "stop",
                        FinishReason.MAX_TOKENS.name: "length",
                        FinishReason.SAFETY.name: "content_filter",
                        FinishReason.RECITATION.name: "content_filter",
                        FinishReason.OTHER.name: "stop",  # Map OTHER to stop
                        # MALFORMED_FUNCTION_CALL is handled above
                    }
                    final_openai_finish_reason = finish_reason_map.get(vertex_finish_reason_enum.name, "stop")
                    logger.info(
                        f"[{request_id}] Mapping final Vertex finish_reason {vertex_finish_reason_enum.name} -> {final_openai_finish_reason}"
                    )

                # Construct the final chunk
                final_litellm_chunk = {
                    "id": f"chatcmpl-adap-{request_id}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id_for_chunk,
                    "choices": [
                        {
                            "index": 0,
                            # Delta can be empty here, role is optional in final chunk if no content
                            "delta": {},  # Or {"role": "assistant"} - doesn't strictly matter for OpenAI final chunk
                            "finish_reason": final_openai_finish_reason,
                            "logprobs": None,
                        }
                    ],
                    "usage": usage_metadata,  # Include usage if available
                }
                logger.debug(
                    f"[{request_id}] Yielding final adapted LiteLLM chunk with finish_reason: {final_litellm_chunk}"
                )
                yield final_litellm_chunk
                break  # Stop iteration after the final chunk

    except Exception as e:
        logger.error(f"[{request_id}] Error during Vertex stream adaptation: {e}", exc_info=True)
        # Yield an error chunk in OpenAI format
        yield {
            "id": f"chatcmpl-adap-{request_id}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_id_for_chunk,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": f"[Proxy Error adapting stream: {e}]"},
                    "finish_reason": "stop",
                }
            ],  # Use 'stop' as finish reason for errors in stream
            "usage": None,
            "_error": f"Error adapting Vertex stream: {e}",
        }
    finally:
        logger.info(f"[{request_id}] Vertex AI stream adaptation finished.")


# Conversion from Vertex AI non-streaming response to LiteLLM/OpenAI dict +++
def convert_vertex_response_to_litellm(response: GenerationResponse, model_id: str, request_id: str) -> Dict[str, Any]:
    """Converts a non-streaming Vertex AI GenerationResponse to a LiteLLM/OpenAI-like dictionary."""
    logger.info(f"[{request_id}] Converting Vertex non-streaming response to LiteLLM/OpenAI format.")
    output_text = ""
    tool_calls = []  # List for OpenAI tool_calls structure
    finish_reason_str = "stop"  # Default
    has_function_call = False  # Flag to track if tool calls are present

    # --- Process Candidate ---
    if response.candidates:
        candidate = response.candidates[0]  # Assume first candidate

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
                    output_text += part.text  # Concatenate text parts
                elif part.function_call:
                    has_function_call = True  # Mark that a tool call was found
                    fc = part.function_call
                    tool_call_id = f"toolu_{uuid.uuid4().hex[:12]}"  # Generate ID
                    try:
                        args_str = json.dumps(fc.args) if fc.args else "{}"
                    except Exception as e:
                        logger.error(
                            f"[{request_id}] Failed to dump non-streaming function call args to JSON: {e}. Args: {fc.args}"
                        )
                        args_str = json.dumps({"error": f"Failed to serialize args: {e}"})

                    tool_calls.append(
                        {
                            "index": openai_tool_index_counter,  # Include index for consistency
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": fc.name, "arguments": args_str},
                        }
                    )
                    logger.debug(
                        f"[{request_id}] Converted non-streaming tool call: index={openai_tool_index_counter}, id={tool_call_id}, name={fc.name}"
                    )
                    openai_tool_index_counter += 1
                elif part.function_response:
                    logger.warning(
                        f"[{request_id}] Unexpected FunctionResponse part found in non-streaming Vertex response: {part.function_response}"
                    )

        # --- Determine Final OpenAI Finish Reason ---
        if has_function_call:
            # If function calls are present, the reason is 'tool_calls' for OpenAI
            finish_reason_str = "tool_calls"
            logger.debug(f"[{request_id}] Setting finish_reason to 'tool_calls' due to presence of function calls.")
        elif vertex_finish_reason and vertex_finish_reason != FinishReason.FINISH_REASON_UNSPECIFIED:
            # Otherwise, use the mapped Vertex finish reason
            finish_reason_str = finish_reason_map.get(vertex_finish_reason.name, "stop")
            logger.debug(
                f"[{request_id}] Mapped non-streaming finish_reason: {vertex_finish_reason.name} -> {finish_reason_str}"
            )
        else:
            # Default to 'stop' if no specific reason provided or mapped
            logger.debug(f"[{request_id}] Using default finish_reason 'stop'.")

    # --- Extract Usage ---
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    if response.usage_metadata:
        prompt_tokens = response.usage_metadata.prompt_token_count
        completion_tokens = response.usage_metadata.candidates_token_count
        total_tokens = response.usage_metadata.total_token_count
        logger.info(
            f"[{request_id}] Extracted non-streaming usage: In={prompt_tokens}, Out={completion_tokens}, Total={total_tokens}"
        )

    # --- Construct the LiteLLM/OpenAI-like Message ---
    message_content = {"role": "assistant"}
    if output_text:
        message_content["content"] = output_text
    else:
        message_content["content"] = None
    if tool_calls:
        message_content["tool_calls"] = tool_calls
    if message_content["content"] is None and not message_content.get("tool_calls"):
        logger.warning(
            f"[{request_id}] Non-streaming response has no text and no tool calls. Message content set to null."
        )

    # --- Construct the Final LiteLLM/OpenAI-like Response Dict ---
    litellm_response = {
        "id": f"chatcmpl-vert-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [{"index": 0, "message": message_content, "finish_reason": finish_reason_str, "logprobs": None}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "total_tokens": total_tokens},
        "request_id": request_id,
    }
    logger.debug(f"[{request_id}] Converted non-streaming Vertex response to LiteLLM/OpenAI format: {litellm_response}")
    return litellm_response
