"""Configuration settings for the proxy server.

Loads environment variables and defines configuration constants used throughout the application.
"""

import os

from dotenv import load_dotenv

from src.utils import get_logger

logger = get_logger()

load_dotenv()

# Model configuration for mapping between Claude and Gemini models
GEMINI_BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-2.5-pro")
GEMINI_SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-2.5-flash")
logger.info(f"Using BIG model: {GEMINI_BIG_MODEL}, SMALL model: {GEMINI_SMALL_MODEL}")

# Temperature override settings for better tool calling reliability
OVERRIDE_TEMPERATURE = os.environ.get("OVERRIDE_TEMPERATURE", "false").lower() == "true"
TEMPERATURE_OVERRIDE = float(os.environ.get("TEMPERATURE", "0.7"))
TOOL_CALL_TEMPERATURE_OVERRIDE = float(os.environ.get("TOOL_CALL_TEMPERATURE", "0.5"))

if OVERRIDE_TEMPERATURE:
    logger.info(f"Temperature overrides enabled: {TEMPERATURE_OVERRIDE} (tool call: {TOOL_CALL_TEMPERATURE_OVERRIDE})")
