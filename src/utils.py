"""Utility functions for logging and formatting.

Provides central logging configuration, color formatting, request visualization,
and specialized tool usage event logging to JSON Lines file.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

from loguru import logger
from rich.pretty import pretty_repr


class Colors:
    """ANSI color and formatting codes for terminal output styling."""

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


class LoggerService:
    """Centralized logging service implementing a singleton pattern using Loguru.

    Provides consistent logging configuration across the application with:
    - File logging for all levels (DEBUG and up) with rotation
    - Console logging with higher levels (INFO or DEBUG)
    - Timestamped log files for better tracking across runs
    - Standardized formatting with process, thread, and line information
    """

    _instance = None
    _initialized = False
    _console_handler_id = None

    @classmethod
    def get_instance(cls) -> "LoggerService":
        """Get or create the singleton logger instance.

        Returns:
            LoggerService: The singleton instance
        """
        if cls._instance is None:
            cls._instance = LoggerService()
        return cls._instance

    def __init__(self):
        """Initialize the logger configuration.

        Only runs once due to singleton pattern. Uses loguru for advanced logging features.
        """
        if LoggerService._initialized:
            return

        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Create timestamped log filename for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_pattern = str(logs_dir / f"{timestamp}_proxy.log")

        # Remove default loguru handler
        logger.remove()

        # Add file handler with rotation and retention policies
        logger.add(
            log_file_pattern,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {process}:{thread.name} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate when file reaches 10 MB
            retention=10,  # Keep up to 10 rotated logs
            compression="zip",  # Compress rotated logs
            enqueue=True,  # Use a separate thread for logging (non-blocking)
        )

        # Add console handler
        self._console_handler_id = logger.add(
            sys.stderr,
            level="INFO",  # Default to INFO level
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}:{function}:{line}</cyan> - {message}",
            colorize=True,
        )

        # Mark as initialized
        LoggerService._initialized = True

        logger.info("Logger initialized with file: {}", log_file_pattern)

    def get_logger(self):
        """Get the loguru logger instance.

        Returns:
            loguru.logger: The configured logger
        """
        return logger

    def enable_debug_logging(self):
        """Enable debug level logging to console.

        Removes the current console handler and adds a new one with DEBUG level.

        Returns:
            loguru.logger: The reconfigured logger with debug level enabled
        """
        # Remove the current console handler
        if self._console_handler_id is not None:
            logger.remove(self._console_handler_id)

        # Add a new console handler with DEBUG level
        self._console_handler_id = logger.add(
            sys.stderr,
            level="DEBUG",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | <cyan>{name}:{function}:{line}</cyan> - {message}",
            colorize=True,
        )
        logger.debug("Debug console logging enabled")
        return logger


def get_logger():
    """Get the application-wide configured logger instance.

    Returns:
        loguru.logger: The centrally configured logger
    """
    return LoggerService.get_instance().get_logger()


def enable_debug_logging():
    """Enable debug level logging to console.

    Returns:
        loguru.logger: The reconfigured logger with debug level enabled
    """
    return LoggerService.get_instance().enable_debug_logging()


def log_request_beautifully(method, path, original_model, mapped_model, num_messages, num_tools, status_code):
    """Log API requests in a colorized, human-readable format.

    Creates a visually distinctive terminal output for request monitoring with color-coded
    status indicators, model mapping information, and request details.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request endpoint path
        original_model: Source model requested (Claude model name)
        mapped_model: Target model used (Gemini model name)
        num_messages: Number of messages in the request
        num_tools: Number of tools in the request
        status_code: HTTP status code of the response
    """
    try:
        original_display = f"{Colors.CYAN}{original_model}{Colors.RESET}"
        endpoint = path.split("?")[0]
        mapped_display_name = mapped_model
        mapped_color = Colors.GREEN  # Green indicates target Gemini model
        mapped_display = f"{mapped_color}{mapped_display_name}{Colors.RESET}"

        # Highlight tool presence with magenta if tools exist, dim if none
        tools_str = (
            f"{Colors.MAGENTA}{num_tools} tools{Colors.RESET}"
            if num_tools > 0
            else f"{Colors.DIM}{num_tools} tools{Colors.RESET}"
        )
        messages_str = f"{Colors.BLUE}{num_messages} messages{Colors.RESET}"

        # Visual indicator for success/failure
        status_color = Colors.GREEN if 200 <= status_code < 300 else Colors.RED
        status_symbol = "✓" if 200 <= status_code < 300 else "✗"
        status_str = f"{status_color}{status_symbol} {status_code}{Colors.RESET}"

        log_line = f"{Colors.BOLD}{method} {endpoint}{Colors.RESET} {status_str}"
        model_line = f"  {original_display} → {mapped_display} ({messages_str}, {tools_str})"

        # Print to console
        print(log_line)
        print(model_line)
        sys.stdout.flush()

        # Also log structured information to loguru logger
        logger.info(
            "Request processed: {method} {endpoint} - {status_code}",
            method=method,
            endpoint=endpoint,
            status_code=status_code,
            original_model=original_model,
            mapped_model=mapped_model,
            num_messages=num_messages,
            num_tools=num_tools,
        )
    except Exception as e:
        logger.error("Error during beautiful logging: {error}", error=str(e))
        # Fallback to plain log format if colorized version fails
        print(
            f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools"
        )


def smart_format_str(obj, max_string=500, max_length=100, indent=2) -> str:
    """Format an object to a string with rich formatting."""
    return pretty_repr(obj, max_string=max_string, max_length=max_length, indent_size=indent)


def smart_format_proto_str(obj, max_string=500, max_length=100, indent=2) -> str:
    """Format a proto object to a string with rich formatting."""
    # Convert proto to dict and format
    formatted_obj = proto_to_dict(obj)
    return smart_format_str(formatted_obj, max_string, max_length, indent)


def proto_to_dict(obj) -> Union[Dict, List[Dict]]:
    """Convert proto objects to dictionaries recursively."""
    # If object has to_dict method (proto), use it
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        return obj.to_dict()

    # Handle lists/tuples containing protos
    elif isinstance(obj, (list, tuple)):
        return [proto_to_dict(item) for item in obj]

    # Handle dictionaries containing protos
    elif isinstance(obj, dict):
        return {k: proto_to_dict(v) for k, v in obj.items()}

    # Return other types unchanged
    else:
        return obj


# Tool Events Logger for JSONL file
# Create an asyncio Lock to ensure thread-safe writing to the JSONL file
_tool_events_lock = asyncio.Lock()


async def log_tool_event(
    request_id: str,
    tool_name: Optional[str],
    status: Literal["attempt", "success", "failure"],
    stage: Literal["gemini_request", "gemini_response", "client_response", "client_execution_report"],
    details: Optional[Dict] = None,
) -> None:
    """Log tool usage events to a separate JSON Lines file for analysis.

    This function captures structured data about tool usage events at different
    stages of the request/response cycle, writing events to a timestamped tool_events.jsonl
    file in a thread-safe manner.

    Args:
        request_id: The unique identifier for the request
        tool_name: The name of the tool being used (or None for general events)
        status: Whether this is an attempt, success, or failure
        stage: Which part of the process (request to Gemini, response from Gemini, or response to client)
        details: Optional additional information about the event
    """
    try:
        # Ensure logs directory exists
        logs_dir = Path(__file__).parent.parent / "logs" / "tool_events"
        logs_dir.mkdir(exist_ok=True)

        # Create a datestamp for daily tool event files
        datestamp = datetime.utcnow().strftime("%Y%m%d")
        jsonl_path = logs_dir / f"{datestamp}_proxy.jsonl"

        # Construct the event object
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "tool_name": tool_name,
            "status": status,
            "stage": stage,
        }

        # Add details if provided
        if details:
            event["details"] = details

        # Acquire lock for thread-safe file access
        async with _tool_events_lock:
            # Open in append mode and write the JSON object
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(event) + "\n")

        # Use loguru's structured logging
        logger.debug(
            "Tool event logged: {status} {stage} for {tool}",
            status=status,
            stage=stage,
            tool=tool_name or "unknown",
            request_id=request_id,
        )
    except Exception as e:
        # Log error but don't fail the request
        logger.error("Failed to log tool event: {error}", error=str(e), request_id=request_id, exc_info=True)
