"""
Utility functions for the AI Platform proxy.
"""

import json
import logging
import sys
from typing import Any, Dict

logger = logging.getLogger(__name__)

# List of supported AI Platform models
AIPLATFORM_MODELS = [
    "gemini-2.5-pro-preview-03-25",  # Main model for high-quality outputs
    "gemini-2.0-flash",  # Faster, smaller model for simpler queries
]


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
