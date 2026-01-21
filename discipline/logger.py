"""
Logging module for the Discipline system.

Logs feeding events, violations, and system status.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


class DisciplineLogger:
    """
    Logger for the Discipline cat monitoring system.
    """

    def __init__(
        self,
        log_file: str = "logs/discipline.log",
        level: str = "INFO",
        max_size_mb: int = 10,
        backup_count: int = 5,
    ):
        """
        Initialize the logger.

        Args:
            log_file: Path to log file
            level: Logging level (DEBUG, INFO, WARNING, ERROR)
            max_size_mb: Maximum log file size before rotation
            backup_count: Number of backup files to keep
        """
        self.log_file = Path(log_file)
        self.level = level

        # Ensure log directory exists
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Configure loguru
        logger.remove()  # Remove default handler

        # Console handler
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        )

        # File handler with rotation
        logger.add(
            str(self.log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            rotation=f"{max_size_mb} MB",
            retention=backup_count,
            compression="zip",
        )

        self._logger = logger

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._logger.info(self._format_message(message, kwargs))

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._logger.debug(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._logger.error(self._format_message(message, kwargs))

    def _format_message(self, message: str, data: dict) -> str:
        """Format a log message with optional data."""
        if data:
            return f"{message} | {json.dumps(data)}"
        return message

    def log_detection(
        self,
        cat_name: str,
        confidence: float,
        position: tuple[int, int],
        bowl: Optional[str] = None,
    ) -> None:
        """Log a cat detection event."""
        self.debug(
            "Cat detected",
            cat=cat_name,
            confidence=round(confidence, 3),
            position=position,
            bowl=bowl,
        )

    def log_violation(
        self,
        cat_name: str,
        bowl_name: str,
        owner_present: bool,
        sprayed: bool,
    ) -> None:
        """Log a violation event."""
        self.warning(
            "VIOLATION: Wrong cat at bowl",
            cat=cat_name,
            bowl=bowl_name,
            owner_present=owner_present,
            sprayed=sprayed,
        )

    def log_spray(self, cat_name: str, bowl_name: str, duration_ms: int) -> None:
        """Log a spray event."""
        self.info(
            "SPRAY ACTIVATED",
            target_cat=cat_name,
            bowl=bowl_name,
            duration_ms=duration_ms,
        )

    def log_feeding_start(self, cat_name: str, bowl_name: str) -> None:
        """Log when a cat starts eating."""
        self.info(
            "Feeding started",
            cat=cat_name,
            bowl=bowl_name,
        )

    def log_feeding_end(self, cat_name: str, bowl_name: str, duration_s: float) -> None:
        """Log when a cat stops eating."""
        self.info(
            "Feeding ended",
            cat=cat_name,
            bowl=bowl_name,
            duration_s=round(duration_s, 1),
        )

    def log_system_status(self, status: dict) -> None:
        """Log system status."""
        self.debug("System status", **status)

    def log_startup(self, config: dict) -> None:
        """Log system startup."""
        self.info(
            "Discipline system starting",
            spray_enabled=config.get("spray", {}).get("enabled", True),
            cats=list(config.get("cats", {}).keys()),
        )

    def log_shutdown(self, stats: dict) -> None:
        """Log system shutdown."""
        self.info(
            "Discipline system shutting down",
            **stats,
        )
