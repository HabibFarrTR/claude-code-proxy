import os

from dotenv import load_dotenv

from src.utils import get_logger

logger = get_logger()

# Load environment variables from .env file
load_dotenv()


GEMINI_BIG_MODEL = os.environ.get("BIG_MODEL", "gemini-1.5-pro-latest")
GEMINI_SMALL_MODEL = os.environ.get("SMALL_MODEL", "gemini-1.5-flash-latest")
logger.info(f"Using BIG model: {GEMINI_BIG_MODEL}, SMALL model: {GEMINI_SMALL_MODEL}")
OVERRIDE_TEMPERATURE = os.environ.get("OVERRIDE_TEMPERATURE", "false").lower() == "true"
if OVERRIDE_TEMPERATURE:
    TEMPERATURE_OVERRIDE = float(os.environ.get("TEMPERATURE", "0.7"))
    TOOL_CALL_TEMPERATURE_OVERRIDE = float(os.environ.get("TOOL_CALL_TEMPERATURE", "0.5"))
    logger.info(f"Temperature overrides enabled: {TEMPERATURE_OVERRIDE} (tool call: {TOOL_CALL_TEMPERATURE_OVERRIDE})")
