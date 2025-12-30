#!/usr/bin/env python3
"""
Configuration Settings Module (V0).

This module centralizes all configuration parameters for the WayFinder
CLI application.

Example:
    >>> from config.settings import settings
    >>> print(settings.LOG_FORMAT)

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 0.1
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Settings:
    """
    Application configuration settings.

    Attributes:
        DATA_YEARS: List of years available for search.
        API_TIMEOUT: Timeout in seconds for API requests.
        LOG_LEVEL: Default logging level.
        LOG_FORMAT: Format string for log messages.
        LOG_DATE_FORMAT: Date format for log timestamps.
    """

    # =========================================================================
    # DATA SETTINGS
    # =========================================================================

    DATA_YEARS: List[int] = None
    """Years of Parcoursup data available."""

    API_TIMEOUT: int = 30
    """Timeout in seconds for API requests."""

    # =========================================================================
    # LOGGING SETTINGS
    # =========================================================================

    LOG_LEVEL: str = "INFO"
    """Default logging level."""

    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Format string for log messages."""

    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    """Date format for log timestamps."""

    def __post_init__(self) -> None:
        """Post-initialization to set default values."""
        if self.DATA_YEARS is None:
            self.DATA_YEARS = [2024, 2023, 2022, 2021]


# Global settings instance
settings = Settings()
