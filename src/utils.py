import json
import logging
import os
import sys
import uuid
from typing import Dict, List, Optional, Any, Literal

logger = logging.getLogger("AnthropicGeminiProxy")

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

# Get specific Gemini model names from environment or use defaults
GEMINI_BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-1.5-pro-latest")
GEMINI_SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-1.5-flash-latest")

# --- Helper Functions ---
def map_model_name(anthropic_model_name: str) -> str:
    """Maps Anthropic model names to specific Gemini models. Returns the Gemini model ID."""
    original_model = anthropic_model_name
    mapped_gemini_model = GEMINI_BIG_MODEL  # Default

    logger.debug(
        f"Attempting to map model: '{original_model}' -> Target Gemini BIG='{GEMINI_BIG_MODEL}', SMALL='{GEMINI_SMALL_MODEL}'"
    )

    clean_name = anthropic_model_name.lower().split("@")[0]
    if clean_name.startswith("anthropic/"):
        clean_name = clean_name[10:]
    elif clean_name.startswith("gemini/"):
        clean_name = clean_name[7:]  # Allow direct gemini model names like 'gemini/gemini-1.5-pro-latest'

    if "haiku" in clean_name:
        mapped_gemini_model = GEMINI_SMALL_MODEL
        logger.info(f"Mapping '{original_model}' (Haiku) -> Target Gemini SMALL '{mapped_gemini_model}'")
    elif "sonnet" in clean_name or "opus" in clean_name:
        mapped_gemini_model = GEMINI_BIG_MODEL
        logger.info(f"Mapping '{original_model}' (Sonnet/Opus) -> Target Gemini BIG '{mapped_gemini_model}'")
    elif clean_name == GEMINI_BIG_MODEL.lower() or clean_name == GEMINI_SMALL_MODEL.lower():
        mapped_gemini_model = clean_name  # Use the directly specified Gemini model
        logger.info(f"Using directly specified target Gemini model: '{mapped_gemini_model}'")
    else:
        logger.warning(
            f"Unrecognized Anthropic model name '{original_model}'. Defaulting to BIG model '{mapped_gemini_model}'."
        )

    # Return just the Gemini model ID (e.g., "gemini-1.5-pro-latest") for the SDK
    return mapped_gemini_model


def log_request_beautifully(method, path, original_model, mapped_model, num_messages, num_tools, status_code):
    """Log requests in a colorized format."""
    try:
        original_display = f"{Colors.CYAN}{original_model}{Colors.RESET}"
        endpoint = path.split("?")[0]
        # mapped_model here is the Gemini ID
        mapped_display_name = mapped_model
        mapped_color = Colors.GREEN  # Always green for Gemini target
        mapped_display = f"{mapped_color}{mapped_display_name}{Colors.RESET}"
        tools_str = (
            f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
            if num_tools > 0
            else f"{Colors.DIM}{num_tools} tools{Colors.RESET}"
        )
        messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"
        status_color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
        status_symbol = "✓" if 200 <= status_code < 300 else "✗"
        status_str = f"{status_color}{status_symbol} {status_code}{Colors.RESET}"
        log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
        model_line = f"  {original_display} -> {mapped_display} ({messages_str}, {tools_str})"
        print(log_line)
        print(model_line)
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error during beautiful logging: {e}")
        # Fallback plain log
        print(
            f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools"
        )

# Function to clean schema for Gemini compatibility
def clean_gemini_schema(schema: Dict) -> Dict:
    """Clean JSON schema for Gemini compatibility"""
    # Implement schema cleaning logic here
    # This is a placeholder - the actual implementation would need to be extracted from server.py
    return schema