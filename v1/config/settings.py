#!/usr/bin/env python3
"""
Configuration Settings Module.

This module centralizes all configuration parameters for the WayFinder
application, including data paths, API endpoints, ML parameters, and
logging settings.

Example:
    >>> from config.settings import settings
    >>> print(settings.DATA_PATH)
    'data/parcoursup_data.csv'

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 1.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List

# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """
    Application configuration settings.

    This dataclass centralizes all configuration parameters used throughout
    the WayFinder application. Values can be overridden via environment
    variables prefixed with 'WAYFINDER_'.

    Attributes:
        DATA_YEARS: List of years to download from Parcoursup API.
        DATA_DIR: Directory for storing downloaded data files.
        DATA_PATH: Path to the main Parcoursup CSV data file.
        MODEL_DIR: Directory for storing trained ML models.
        MODEL_PATH: Path to the trained clustering model.
        API_BASE_URL: Base URL for the Parcoursup API.
        API_TIMEOUT: Timeout in seconds for API requests.
        N_CLUSTERS: Number of clusters for K-Means clustering.
        LOG_LEVEL: Default logging level.
        LOG_FORMAT: Format string for log messages.

    Example:
        >>> settings = Settings()
        >>> settings.ensure_directories()
        >>> print(settings.DATA_PATH)
    """

    # =========================================================================
    # DATA SETTINGS
    # =========================================================================

    DATA_YEARS: List[int] = field(default_factory=lambda: [
        2024, 2023, 2022, 2021
    ])
    """Years of Parcoursup data to download and analyze."""

    DATA_DIR: str = "data"
    """Directory for storing downloaded data files."""

    DATA_PATH: str = "data/parcoursup_data.csv"
    """Path to the main Parcoursup CSV data file."""

    MODEL_DIR: str = "models"
    """Directory for storing trained ML models."""

    MODEL_PATH: str = "models/parcoursup_model_v3.pkl"
    """Path to the trained clustering model."""

    # =========================================================================
    # API SETTINGS
    # =========================================================================

    API_BASE_URL: str = "https://data.enseignementsup-recherche.gouv.fr"
    """Base URL for the Parcoursup API."""

    API_TIMEOUT: int = 180
    """Timeout in seconds for API requests."""

    DATASET_IDS: Dict[int, str] = field(default_factory=lambda: {
        2024: "fr-esr-parcoursup",
        2023: "fr-esr-parcoursup_2023",
        2022: "fr-esr-parcoursup_2022",
        2021: "fr-esr-parcoursup_2021",
    })
    """Mapping of years to dataset IDs on the API."""

    # =========================================================================
    # ML SETTINGS
    # =========================================================================

    N_CLUSTERS: int = 5
    """Number of clusters for K-Means clustering."""

    RANDOM_STATE: int = 42
    """Random seed for reproducibility."""

    # =========================================================================
    # LOGGING SETTINGS
    # =========================================================================

    LOG_LEVEL: str = "INFO"
    """Default logging level."""

    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """Format string for log messages."""

    LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
    """Date format for log timestamps."""

    # =========================================================================
    # UI SETTINGS
    # =========================================================================

    APP_TITLE: str = "WayFinder - Parcoursup"
    """Application title displayed in browser tab."""

    MAX_COMPARE_FORMATIONS: int = 4
    """Maximum number of formations that can be compared."""

    MAX_SEARCH_RESULTS: int = 50
    """Maximum number of search results to display."""

    # =========================================================================
    # METHODS
    # =========================================================================

    def __post_init__(self) -> None:
        """
        Post-initialization hook to load environment variable overrides.

        Environment variables take precedence over default values.
        All environment variables should be prefixed with 'WAYFINDER_'.
        """
        # Override with environment variables if present
        self.DATA_PATH = os.environ.get(
            'WAYFINDER_DATA_PATH',
            self.DATA_PATH
        )
        self.MODEL_PATH = os.environ.get(
            'WAYFINDER_MODEL_PATH',
            self.MODEL_PATH
        )
        self.LOG_LEVEL = os.environ.get(
            'WAYFINDER_LOG_LEVEL',
            self.LOG_LEVEL
        )

        logger.debug("Settings initialized with DATA_PATH=%s", self.DATA_PATH)

    def ensure_directories(self) -> None:
        """
        Create necessary directories if they don't exist.

        Creates the data and models directories specified in the settings.
        Logs the creation of each directory.

        Example:
            >>> settings = Settings()
            >>> settings.ensure_directories()
        """
        for directory in [self.DATA_DIR, self.MODEL_DIR]:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.info("Created directory: %s", directory)


# Global settings instance for easy import
settings = Settings()
