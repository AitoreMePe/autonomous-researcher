import logging
from rich.console import Console
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme

# Custom theme for the console
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "error": "bold red",
    "success": "bold green",
    "thought": "italic cyan",
    "code": "bold yellow",
    "result": "white"
})

console = Console(theme=custom_theme)

def setup_logging():
    """Sets up logging to both file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("agent.log", encoding='utf-8'),
            # We don't add RichHandler here because we want manual control over console output
            # to keep it "elegant" and not just a stream of logs.
        ]
    )
    # Create a separate logger for the file that doesn't propagate to root
    file_logger = logging.getLogger("agent_file")
    file_logger.setLevel(logging.DEBUG)
    # Use UTF-8 encoding for file handler
    handler = logging.FileHandler("agent_file.log", encoding='utf-8')
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    file_logger.addHandler(handler)
    return file_logger

# Global file logger instance
logger = setup_logging()

def log_step(step_name, status="INFO"):
    """Logs a step to the file."""
    # Replace problematic Unicode characters with ASCII equivalents
    safe_status = status.replace('≥', '>=').replace('≤', '<=').replace('≠', '!=').replace('→', '->').replace('←', '<-')
    try:
        logger.info(f"[{step_name}] {safe_status}")
    except UnicodeEncodeError:
        # Fallback: encode to ASCII with replacement
        logger.info(f"[{step_name}] {safe_status.encode('ascii', 'replace').decode('ascii')}")

def print_panel(content, title, style="info"):
    """Prints a rich panel to the console."""
    from rich.markup import escape
    # Escape Rich markup characters in content to prevent MarkupError with tags like [RUN_EXPERIMENT]
    safe_content = escape(content)
    console.print(Panel(safe_content, title=title, border_style=style, expand=False))

def print_status(message, style="info"):
    """Prints a status message."""
    from rich.markup import escape
    # Escape Rich markup characters in message to prevent MarkupError with tags like [RUN_EXPERIMENT]
    safe_message = escape(message)
    console.print(f"[{style}]{safe_message}[/{style}]")
