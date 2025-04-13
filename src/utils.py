"""Utility functions for logging and formatting.

Provides central logging configuration, color formatting, and request visualization.
"""

import logging
import sys
from pathlib import Path


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
    """Centralized logging service implementing a singleton pattern.
    
    Provides consistent logging configuration across the application with:
    - File logging for all levels (DEBUG and up)
    - Console logging for higher levels (INFO and up)
    - Synchronized access to prevent configuration conflicts
    """

    _instance = None
    _initialized = False

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
        
        Only runs once due to singleton pattern. Subsequent instantiations
        return without performing initialization again.
        """
        if LoggerService._initialized:
            return

        # Create logs directory if it doesn't exist
        logs_dir = Path(__file__).parent.parent.parent.parent / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Configure root logger
        self.logger = logging.getLogger("neuro_symbolic")
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers (important for streamlit hot-reloading)
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create file handler for all logs
        file_handler = logging.FileHandler(logs_dir / "application.log")
        file_handler.setLevel(logging.DEBUG)

        # Create console handler with higher level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatters and add to handlers
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        )
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Mark as initialized
        LoggerService._initialized = True

        self.logger.info("Logger initialized")

    def get_logger(self) -> logging.Logger:
        """Get the configured application logger.
        
        Returns:
            logging.Logger: The configured logger with file and console handlers
        """
        return self.logger


def get_logger() -> logging.Logger:
    """Get the application-wide configured logger instance.
    
    Returns:
        logging.Logger: The centrally configured logger
    """
    return LoggerService.get_instance().get_logger()


logger = get_logger()


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
        print(log_line)
        print(model_line)
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error during beautiful logging: {e}")
        # Fallback to plain log format if colorized version fails
        print(
            f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools"
        )
