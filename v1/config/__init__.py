"""
Configuration Package for WayFinder Application.

This package contains all configuration settings and constants
used throughout the application.

Modules:
    settings: Main configuration settings dataclass.

Example:
    >>> from config import settings
    >>> print(settings.DATA_PATH)
"""

from .settings import Settings, settings

__all__ = ['Settings', 'settings']
