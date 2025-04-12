import logging
import sys
from pathlib import Path


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


class LoggerService:
    """Centralized logging service for the application"""

    _instance = None
    _initialized = False

    @classmethod
    def get_instance(cls) -> "LoggerService":
        """Get or create the singleton logger instance"""
        if cls._instance is None:
            cls._instance = LoggerService()
        return cls._instance

    def __init__(self):
        """Initialize the logger - only runs once due to singleton pattern"""
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
        """Get the configured logger"""
        return self.logger


# Create a module-level function to get the logger
def get_logger() -> logging.Logger:
    """Get the application logger"""
    return LoggerService.get_instance().get_logger()


logger = get_logger()


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
        model_line = f"  {original_display} → {mapped_display} ({messages_str}, {tools_str})"
        print(log_line)
        print(model_line)
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Error during beautiful logging: {e}")
        # Fallback plain log
        print(
            f"{method} {path} {status_code} | {original_model} -> {mapped_model} | {num_messages} msgs, {num_tools} tools"
        )
