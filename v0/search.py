#!/usr/bin/env python3
"""
Search Module - Formation Search and Data Extraction (V0).

This module provides functionality for searching Parcoursup formations
via the API and extracting formation statistics.

Features:
    - Real-time formation search via the Parcoursup API
    - Formation statistics extraction and normalization
    - Cached API requests for performance

Constants:
    API_BASE: Base URL for the Parcoursup API.
    DATASETS: Mapping of years to dataset identifiers.

Example:
    >>> from search import search_formations, extract_formation_stats
    >>> results = search_formations("informatique", year=2024)
    >>> if results:
    ...     stats = extract_formation_stats(results[0])
    ...     print(stats['taux_acces'])

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 0.1
"""

import logging
from functools import lru_cache
from typing import Dict, List, Optional

import requests

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - API CONFIGURATION
# =============================================================================

API_BASE: str = (
    "https://data.enseignementsup-recherche.gouv.fr"
    "/api/explore/v2.1/catalog/datasets"
)
"""Base URL for the Parcoursup API endpoints."""

DATASETS: Dict[int, str] = {
    2024: "fr-esr-parcoursup",
    2023: "fr-esr-parcoursup_2023",
    2022: "fr-esr-parcoursup_2022",
    2021: "fr-esr-parcoursup_2021",
}
"""Mapping of years to Parcoursup dataset identifiers."""

API_TIMEOUT: int = 30
"""Timeout in seconds for API requests."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _safe_int(value) -> int:
    """
    Safely convert a value to integer.

    Args:
        value: Value to convert (can be None, str, int, float).

    Returns:
        int: Converted integer value, or 0 if conversion fails.
    """
    try:
        return int(value) if value else 0
    except (ValueError, TypeError):
        return 0


def _safe_float(value) -> float:
    """
    Safely convert a value to float.

    Args:
        value: Value to convert (can be None, str, int, float).

    Returns:
        float: Converted float value, or 0.0 if conversion fails.
    """
    try:
        return float(value) if value else 0.0
    except (ValueError, TypeError):
        return 0.0


# =============================================================================
# API FUNCTIONS
# =============================================================================

@lru_cache(maxsize=128)
def _fetch_api_cached(
    dataset: str,
    where: str,
    limit: int,
    order_by: str
) -> Optional[str]:
    """
    Cached API fetch (internal function).

    Uses lru_cache for caching. Returns JSON string for hashability.

    Args:
        dataset: The dataset identifier.
        where: OData filter expression.
        limit: Maximum number of records.
        order_by: Field to sort by.

    Returns:
        Optional[str]: JSON response as string, or None if failed.
    """
    url = f"{API_BASE}/{dataset}/records"
    params = {"limit": limit}

    if where:
        params["where"] = where
    if order_by:
        params["order_by"] = order_by

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        logger.debug("API request successful: %s", dataset)
        return response.text
    except requests.RequestException as error:
        logger.warning("API request failed for %s: %s", dataset, error)
        return None


def fetch_api(
    dataset: str,
    where: str = "",
    limit: int = 100,
    order_by: str = ""
) -> Optional[Dict]:
    """
    Fetch data from the Parcoursup API.

    Makes a GET request to the OpenDataSoft API with optional filtering
    and ordering parameters. Results are cached for performance.

    Args:
        dataset: The dataset identifier (e.g., "fr-esr-parcoursup").
        where: Optional OData filter expression for querying.
        limit: Maximum number of records to return (default: 100).
        order_by: Optional field to sort results by.

    Returns:
        Optional[Dict]: JSON response as dictionary, or None if request failed.

    Example:
        >>> data = fetch_api("fr-esr-parcoursup", limit=10)
        >>> if data:
        ...     print(len(data.get("results", [])))
    """
    import json

    result = _fetch_api_cached(dataset, where, limit, order_by)
    if result:
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            logger.error("Failed to parse API response")
            return None
    return None


def search_formations(
    term: str,
    year: int = 2024,
    limit: int = 50
) -> List[Dict]:
    """
    Search for formations matching a search term.

    Searches both establishment names and formation names for the given term.
    Results are ordered by total number of wishes (popularity).

    Args:
        term: Search term to match against formation/establishment names.
        year: Year of data to search (default: 2024).
        limit: Maximum number of results to return (default: 50).

    Returns:
        List[Dict]: List of matching formation records.

    Example:
        >>> results = search_formations("informatique", year=2024)
        >>> for result in results[:3]:
        ...     print(result.get("lib_for_voe_ins"))
    """
    dataset = DATASETS.get(year)
    if not dataset:
        logger.warning("No dataset found for year %d", year)
        return []

    # Sanitize search term to prevent injection
    term_clean = term.replace("'", "''").replace('"', '').strip()

    # Build search query for both establishment and formation names
    where_clause = (
        f"search(g_ea_lib_vx, '{term_clean}') OR "
        f"search(lib_for_voe_ins, '{term_clean}')"
    )

    data = fetch_api(
        dataset,
        where=where_clause,
        limit=limit,
        order_by="voe_tot DESC"
    )

    results = data.get("results", []) if data else []
    logger.info("Found %d results for '%s' (%d)", len(results), term, year)
    return results


# =============================================================================
# DATA EXTRACTION
# =============================================================================

def extract_formation_stats(formation_data: Dict) -> Dict:
    """
    Extract and normalize statistics from a formation record.

    Converts raw API data into a standardized format with calculated
    percentages and cleaned values. Handles missing data gracefully.

    Args:
        formation_data: Raw formation record from the API.

    Returns:
        Dict: Normalized formation statistics including:
            - nom, etablissement, academie
            - taux_acces, voeux_total, admis_total
            - pct_boursiers, pct_mention_tb, pct_bac_general, etc.

    Example:
        >>> stats = extract_formation_stats(api_record)
        >>> print(f"Taux d'accès: {stats['taux_acces']}%")
    """
    # Extract basic counts
    admis_total = _safe_int(formation_data.get("acc_tot", 0))
    admis_boursiers = _safe_int(formation_data.get("acc_brs", 0))
    voeux_total = _safe_int(formation_data.get("voe_tot", 0))
    prop_total = _safe_int(formation_data.get("prop_tot", 0))

    # Calculate boursier percentage
    pct_boursiers = 0.0
    if admis_total > 0 and admis_boursiers > 0:
        pct_boursiers = (admis_boursiers / admis_total) * 100
    elif formation_data.get("pct_brs"):
        pct_boursiers = _safe_float(formation_data.get("pct_brs"))

    # Calculate taux d'accès (access rate)
    # Different columns depending on year:
    # - 2022+: taux_acces
    # - 2021: taux_acces_ens
    taux_acces = 0.0
    if formation_data.get("taux_acces"):
        taux_acces = _safe_float(formation_data.get("taux_acces"))
    elif formation_data.get("taux_acces_ens"):
        taux_acces = _safe_float(formation_data.get("taux_acces_ens"))
    elif voeux_total > 0:
        # Calculate from proposals or admissions
        if prop_total > 0:
            taux_acces = (prop_total / voeux_total) * 100
        elif admis_total > 0:
            taux_acces = (admis_total / voeux_total) * 100
        # Clip to valid range
        taux_acces = min(100, max(0, taux_acces))

    # Build stats dictionary
    stats = {
        "nom": formation_data.get("lib_for_voe_ins", ""),
        "etablissement": formation_data.get("g_ea_lib_vx", ""),
        "filiere": formation_data.get("fili", ""),
        "filiere_detail": formation_data.get("fil_lib_voe_acc", ""),
        "academie": formation_data.get("acad_mies", ""),
        "cod_aff": formation_data.get("cod_aff_form", ""),
        "capacite": _safe_int(
            formation_data.get("capa_fin", formation_data.get("capa", 0))
        ),
        "voeux_total": voeux_total,
        "admis_total": admis_total,
        "taux_acces": taux_acces,
        "admis_bg": _safe_int(formation_data.get("acc_bg", 0)),
        "admis_bt": _safe_int(formation_data.get("acc_bt", 0)),
        "admis_bp": _safe_int(formation_data.get("acc_bp", 0)),
        "admis_sans_mention": _safe_int(formation_data.get("acc_sansmention", 0)),
        "admis_ab": _safe_int(formation_data.get("acc_ab", 0)),
        "admis_b": _safe_int(formation_data.get("acc_b", 0)),
        "admis_tb": _safe_int(formation_data.get("acc_tb", 0)),
        "admis_tbf": _safe_int(formation_data.get("acc_tbf", 0)),
        "admis_boursiers": admis_boursiers,
        "pct_boursiers": pct_boursiers,
        "admis_meme_academie": _safe_int(formation_data.get("acc_aca_orig", 0)),
        "pct_meme_academie": _safe_float(formation_data.get("pct_aca_orig", 0)),
    }

    # Calculate bac type percentages
    total_bac = stats["admis_bg"] + stats["admis_bt"] + stats["admis_bp"]
    if total_bac > 0:
        stats["pct_admis_bg"] = (stats["admis_bg"] / total_bac) * 100
        stats["pct_admis_bt"] = (stats["admis_bt"] / total_bac) * 100
        stats["pct_admis_bp"] = (stats["admis_bp"] / total_bac) * 100
    else:
        stats["pct_admis_bg"] = 0
        stats["pct_admis_bt"] = 0
        stats["pct_admis_bp"] = 0

    # Calculate mention percentages
    total_mentions = (
        stats["admis_sans_mention"]
        + stats["admis_ab"]
        + stats["admis_b"]
        + stats["admis_tb"]
        + stats["admis_tbf"]
    )
    if total_mentions > 0:
        stats["pct_sans_mention"] = (stats["admis_sans_mention"] / total_mentions) * 100
        stats["pct_ab"] = (stats["admis_ab"] / total_mentions) * 100
        stats["pct_b"] = (stats["admis_b"] / total_mentions) * 100
        stats["pct_tb"] = (stats["admis_tb"] / total_mentions) * 100
        stats["pct_tbf"] = (stats["admis_tbf"] / total_mentions) * 100
    else:
        stats["pct_sans_mention"] = 0
        stats["pct_ab"] = 0
        stats["pct_b"] = 0
        stats["pct_tb"] = 0
        stats["pct_tbf"] = 0

    return stats
