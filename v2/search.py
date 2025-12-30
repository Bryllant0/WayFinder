#!/usr/bin/env python3
"""
Search Module - Formation Search and Admission Probability.

This module provides functionality for searching Parcoursup formations,
calculating admission probabilities based on student profiles, and
recommending specialty combinations (doublettes).

Main Features:
    - Real-time formation search via the Parcoursup API
    - Admission probability calculation based on student profile
    - Specialty pair (doublette) recommendations by formation type
    - Historical data aggregation for trend analysis

Constants:
    SPECIALITES: Mapping of specialty names to short codes.
    MENTIONS: Mapping of grade mentions to point values.
    TYPES_BAC: List of baccalauréat types.
    ACADEMIES: List of French academic regions.
    DOUBLETTES_PAR_TYPE: Recommended specialty pairs by formation type.

Example:
    >>> from search import search_formations, calculate_admission_probability
    >>> results = search_formations("informatique", limit=10)
    >>> print(len(results))
    10

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 2.0
"""

import logging
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

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
    2020: "fr-esr-parcoursup_2020",
    2019: "fr-esr-parcoursup-2019",
    2018: "fr-esr-parcoursup-2018",
}
"""Mapping of years to Parcoursup dataset identifiers."""

# =============================================================================
# CONSTANTS - STUDENT PROFILE OPTIONS
# =============================================================================

SPECIALITES: Dict[str, str] = {
    "Histoire-géo, géopolitique, sciences po": "hggsp",
    "Humanités, littérature, philosophie": "hlp",
    "Langues, littératures étrangères": "llcer",
    "Mathématiques": "mat",
    "Numérique et sciences informatiques": "nsi",
    "Physique-chimie": "pc",
    "Sciences de la vie et de la Terre": "svt",
    "Sciences de l'ingénieur": "si",
    "Sciences économiques et sociales": "ses",
}
"""Mapping of specialty full names to short codes."""

MENTIONS: Dict[str, int] = {
    "Sans mention (< 12)": 0,
    "Assez bien (12-14)": 1,
    "Bien (14-16)": 2,
    "Très bien (16-18)": 3,
    "Très bien félicitations (18+)": 4,
}
"""Mapping of grade mentions to point values for probability calculation."""

TYPES_BAC: List[str] = ["Général", "Technologique", "Professionnel"]
"""List of baccalauréat types available in France."""

ACADEMIES: List[str] = [
    "Aix-Marseille", "Amiens", "Besançon", "Bordeaux", "Clermont-Ferrand",
    "Créteil", "Dijon", "Grenoble", "Lille", "Limoges", "Lyon",
    "Montpellier", "Nancy-Metz", "Nantes", "Nice", "Normandie",
    "Orléans-Tours", "Paris", "Poitiers", "Reims", "Rennes",
    "Strasbourg", "Toulouse", "Versailles",
]
"""List of French academic regions (académies)."""

# =============================================================================
# CONSTANTS - SPECIALTY RECOMMENDATIONS BY FORMATION TYPE
# =============================================================================

DOUBLETTES_PAR_TYPE: Dict[str, List[Tuple[str, int]]] = {
    "CPGE_scientifique": [
        ("Maths & Physique-Chimie", 95),
        ("Maths & NSI", 88),
        ("Maths & SI", 82),
        ("Maths & SVT", 75),
        ("Physique-Chimie & SVT", 68),
        ("Physique-Chimie & SI", 62),
        ("Maths & SES", 55),
        ("NSI & Physique-Chimie", 48),
        ("SVT & SI", 42),
        ("Maths & HGGSP", 35),
    ],
    "CPGE_commerce": [
        ("Maths & SES", 95),
        ("Maths & HGGSP", 88),
        ("SES & HGGSP", 82),
        ("Maths & Langues", 75),
        ("SES & Langues", 68),
        ("HGGSP & Langues", 62),
        ("Maths & Humanités", 55),
        ("SES & Humanités", 48),
        ("HGGSP & Humanités", 42),
        ("Langues & Humanités", 35),
    ],
    "PASS_medecine": [
        ("Physique-Chimie & SVT", 95),
        ("Maths & Physique-Chimie", 88),
        ("Maths & SVT", 82),
        ("Physique-Chimie & Maths", 75),
        ("SVT & Maths", 68),
        ("Physique-Chimie & SI", 62),
        ("SVT & SI", 55),
        ("Maths & NSI", 48),
        ("Physique-Chimie & NSI", 42),
        ("SVT & SES", 35),
    ],
    "informatique": [
        ("Maths & NSI", 95),
        ("Maths & Physique-Chimie", 88),
        ("Maths & SI", 82),
        ("NSI & Physique-Chimie", 75),
        ("NSI & SI", 68),
        ("Maths & SES", 62),
        ("NSI & SVT", 55),
        ("Physique-Chimie & SI", 48),
        ("Maths & HGGSP", 42),
        ("NSI & SES", 35),
    ],
    "ingenieur": [
        ("Maths & Physique-Chimie", 95),
        ("Maths & SI", 88),
        ("Maths & NSI", 82),
        ("Physique-Chimie & SI", 75),
        ("Maths & SVT", 68),
        ("Physique-Chimie & NSI", 62),
        ("SI & NSI", 55),
        ("Physique-Chimie & SVT", 48),
        ("Maths & SES", 42),
        ("SI & SVT", 35),
    ],
    "droit": [
        ("HGGSP & SES", 95),
        ("HGGSP & Humanités", 88),
        ("SES & Langues", 82),
        ("HGGSP & Langues", 75),
        ("Humanités & Langues", 68),
        ("SES & Humanités", 62),
        ("HGGSP & Maths", 55),
        ("SES & Maths", 48),
        ("Langues & Maths", 42),
        ("Humanités & Maths", 35),
    ],
    "lettres": [
        ("Humanités & Langues", 95),
        ("HGGSP & Humanités", 88),
        ("Humanités & Arts", 82),
        ("HGGSP & Langues", 75),
        ("Langues & Langues", 68),
        ("HGGSP & Arts", 62),
        ("SES & Humanités", 55),
        ("Humanités & Maths", 48),
        ("Langues & SES", 42),
        ("HGGSP & SES", 35),
    ],
    "economie": [
        ("Maths & SES", 95),
        ("SES & HGGSP", 88),
        ("Maths & HGGSP", 82),
        ("SES & Langues", 75),
        ("Maths & Langues", 68),
        ("HGGSP & Langues", 62),
        ("SES & Humanités", 55),
        ("Maths & Humanités", 48),
        ("HGGSP & Humanités", 42),
        ("SES & NSI", 35),
    ],
    "general": [
        ("Maths & Physique-Chimie", 95),
        ("Maths & SES", 88),
        ("Maths & NSI", 82),
        ("Maths & SVT", 75),
        ("Maths & HGGSP", 68),
        ("Physique-Chimie & SVT", 62),
        ("SES & HGGSP", 55),
        ("Maths & Langues", 48),
        ("SES & Langues", 42),
        ("HGGSP & Langues", 35),
    ],
}
"""Recommended specialty pairs (doublettes) by formation type with scores."""


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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_api(
    dataset: str,
    where: str = "",
    limit: int = 100,
    order_by: str = ""
) -> Optional[Dict]:
    """
    Fetch data from the Parcoursup API.

    Makes a GET request to the OpenDataSoft API with optional filtering
    and ordering parameters. Results are cached for 1 hour.

    Args:
        dataset: The dataset identifier (e.g., "fr-esr-parcoursup").
        where: Optional OData filter expression for querying.
        limit: Maximum number of records to return (default: 100).
        order_by: Optional field to sort results by.

    Returns:
        Optional[Dict]: JSON response as dictionary, or None if request failed.

    Example:
        >>> data = fetch_api("fr-esr-parcoursup", where="session=2024", limit=10)
        >>> if data:
        ...     print(len(data.get("results", [])))
    """
    url = f"{API_BASE}/{dataset}/records"
    params = {"limit": limit}

    if where:
        params["where"] = where
    if order_by:
        params["order_by"] = order_by

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        logger.debug("API request successful: %s", dataset)
        return response.json()
    except requests.RequestException as error:
        logger.warning("API request failed for %s: %s", dataset, error)
        return None


@st.cache_data(ttl=3600, show_spinner=False)
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

    return data.get("results", []) if data else []


# Alias for backward compatibility
search_formations_suptracker = search_formations


@st.cache_data(ttl=3600, show_spinner=False)
def get_formation_history(
    etablissement: str,
    formation: str
) -> Dict[int, Dict]:
    """
    Retrieve historical data for a specific formation across all years.

    Searches for the same formation at the same establishment across
    all available years (2018-2024) to build a historical trend.

    Args:
        etablissement: Exact name of the establishment.
        formation: Name of the formation to search for (partial match).

    Returns:
        Dict[int, Dict]: Dictionary mapping years to formation records.

    Example:
        >>> history = get_formation_history(
        ...     "Université Paris-Saclay",
        ...     "Licence Informatique"
        ... )
        >>> for year, data in history.items():
        ...     print(f"{year}: {data.get('taux_acces')}%")
    """
    history = {}

    for year, dataset in DATASETS.items():
        # Sanitize establishment name
        etab_clean = etablissement.replace("'", "''")
        where_clause = f"g_ea_lib_vx = '{etab_clean}'"

        data = fetch_api(dataset, where=where_clause, limit=100)

        if data and "results" in data:
            # Find matching formation by partial name match
            for record in data["results"]:
                record_formation = record.get("lib_for_voe_ins", "").lower()
                if formation.lower() in record_formation:
                    history[year] = record
                    break

    logger.debug("Found history for %d years", len(history))
    return history


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
    # - 2020-2021: taux_acces_ens
    # - 2018-2019: must be calculated
    taux_acces = 0.0
    if formation_data.get("taux_acces"):
        taux_acces = _safe_float(formation_data.get("taux_acces"))
    elif formation_data.get("taux_acces_ens"):
        taux_acces = _safe_float(formation_data.get("taux_acces_ens"))
    elif voeux_total > 0:
        # Calculate for 2018-2019 data
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


# =============================================================================
# FORMATION TYPE DETECTION
# =============================================================================

def get_formation_type(stats: Dict) -> str:
    """
    Determine the formation type for recommending appropriate specialties.

    Analyzes the formation name and details to categorize it into one of
    the predefined formation types used for doublette recommendations.

    Args:
        stats: Formation statistics dictionary with 'nom', 'filiere',
               and 'filiere_detail' keys.

    Returns:
        str: Formation type identifier. One of:
            - "CPGE_scientifique"
            - "CPGE_commerce"
            - "PASS_medecine"
            - "informatique"
            - "ingenieur"
            - "droit"
            - "lettres"
            - "economie"
            - "general" (default)

    Example:
        >>> stats = {"nom": "CPGE MPSI", "filiere_detail": ""}
        >>> get_formation_type(stats)
        'CPGE_scientifique'
    """
    # Combine all text fields for keyword matching
    nom = (
        stats.get("nom", "")
        + " " + stats.get("filiere", "")
        + " " + stats.get("filiere_detail", "")
    ).lower()

    # CPGE scientifiques
    cpge_sci_keywords = [
        "mpsi", "pcsi", "mp2i", "bcpst", "ptsi", "tsi", "tb", "ats"
    ]
    if any(keyword in nom for keyword in cpge_sci_keywords):
        return "CPGE_scientifique"
    if "cpge" in nom and any(
        k in nom for k in ["scientifique", "science", "math", "physique", "chimie", "biologie"]
    ):
        return "CPGE_scientifique"
    if "classe préparatoire" in nom and any(
        k in nom for k in ["scientifique", "mpsi", "pcsi", "bcpst"]
    ):
        return "CPGE_scientifique"

    # CPGE commerce / économie
    cpge_eco_keywords = ["ecg", "ecs", "ece", "d1", "d2"]
    if any(keyword in nom for keyword in cpge_eco_keywords):
        return "CPGE_commerce"
    if "cpge" in nom and any(
        k in nom for k in ["commerce", "économique", "economique", "gestion"]
    ):
        return "CPGE_commerce"
    if "classe préparatoire" in nom and any(
        k in nom for k in ["commerce", "économique", "economique"]
    ):
        return "CPGE_commerce"

    # CPGE littéraires
    cpge_lit_keywords = ["khâgne", "hypokhâgne", "a/l", "b/l", "chartes"]
    if any(keyword in nom for keyword in cpge_lit_keywords):
        return "lettres"
    if "cpge" in nom and any(
        k in nom for k in ["littéraire", "litteraire", "lettres"]
    ):
        return "lettres"

    # Médecine / Santé
    medecine_keywords = [
        "pass", "l.as", "las ", "médecine", "medecine", "santé", "sante",
        "maïeutique", "pharmacie", "odontologie", "kinésithérapie",
        "infirmier", "ifsi"
    ]
    if any(keyword in nom for keyword in medecine_keywords):
        return "PASS_medecine"

    # Informatique / Numérique
    info_keywords = [
        "informatique", "numérique", "numerique", "data", "cyber",
        "nsi", "but info", "dut info"
    ]
    if any(keyword in nom for keyword in info_keywords):
        return "informatique"

    # Ingénieur
    ingenieur_keywords = [
        "ingénieur", "ingenieur", "école d'ingénieur", "ecole d'ingenieur",
        "insa", "inpg", "polytech", "enseeiht", "ensimag"
    ]
    if any(keyword in nom for keyword in ingenieur_keywords):
        return "ingenieur"

    # Droit / Sciences Po
    droit_keywords = [
        "droit", "science po", "sciences po", "iep", "juridique",
        "sciences politiques"
    ]
    if any(keyword in nom for keyword in droit_keywords):
        return "droit"

    # Économie / Commerce / Gestion
    eco_keywords = [
        "économie", "economie", "gestion", "commerce", "management",
        "aes", "éco-gestion", "eco-gestion"
    ]
    if any(keyword in nom for keyword in eco_keywords):
        return "economie"

    # BTS / BUT scientifiques
    if ("bts" in nom or "but" in nom) and any(
        k in nom for k in [
            "électronique", "electronique", "électrotechnique",
            "mécanique", "mecanique", "maintenance", "productique"
        ]
    ):
        return "ingenieur"

    # BTS / BUT tertiaires
    if ("bts" in nom or "but" in nom) and any(
        k in nom for k in [
            "comptabilité", "comptabilite", "commerce", "gestion",
            "banque", "assurance", "muc", "ndrc"
        ]
    ):
        return "economie"

    # Licences par domaine
    if "licence" in nom:
        if any(
            k in nom for k in [
                "mathématiques", "mathematiques", "physique", "chimie",
                "sciences de la vie", "biologie"
            ]
        ):
            return "PASS_medecine"  # Scientific profile
        if any(k in nom for k in ["informatique", "info"]):
            return "informatique"
        if "droit" in nom:
            return "droit"
        if any(k in nom for k in ["économie", "economie", "gestion", "aes"]):
            return "economie"

    # Default to general
    return "general"


# =============================================================================
# PROBABILITY CALCULATION
# =============================================================================

def calculate_doublette_advantage(
    specialites: List[str],
    formation_type: str
) -> Tuple[float, str, int]:
    """
    Calculate the advantage factor for a specialty pair (doublette).

    Evaluates how well a student's chosen specialties match the typical
    profile of admitted students for a given formation type.

    Args:
        specialites: List of the student's chosen specialties.
        formation_type: Type of formation (from get_formation_type).

    Returns:
        Tuple containing:
            - float: Advantage multiplier (e.g., 1.4 for optimal match)
            - str: Human-readable explanation of the match quality
            - int: Rank from 1 (best) to 5 (worst)

    Example:
        >>> advantage, explanation, rank = calculate_doublette_advantage(
        ...     ["Mathématiques", "Physique-chimie"],
        ...     "CPGE_scientifique"
        ... )
        >>> print(f"Advantage: {advantage}, {explanation}")
        Advantage: 1.4, Doublette optimale
    """
    # Handle missing or incomplete specialties
    if not specialites or len(specialites) < 2:
        return 1.0, "Spécialités non renseignées", 3

    # Check for presence of each specialty
    has_maths = "Mathématiques" in specialites
    has_pc = "Physique-chimie" in specialites
    has_svt = "Sciences de la vie et de la Terre" in specialites
    has_nsi = "Numérique et sciences informatiques" in specialites
    has_ses = "Sciences économiques et sociales" in specialites
    has_hggsp = "Histoire-géo, géopolitique, sciences po" in specialites
    has_si = "Sciences de l'ingénieur" in specialites

    # Evaluate based on formation type
    if formation_type == "CPGE_scientifique":
        if has_maths and has_pc:
            return 1.4, "Doublette optimale", 1
        elif has_maths and (has_nsi or has_si):
            return 1.3, "Excellente doublette", 1
        elif has_maths:
            return 0.9, "Avec Maths", 3
        else:
            return 0.3, "Sans Maths - très rare", 5

    elif formation_type == "CPGE_commerce":
        if has_maths and has_ses:
            return 1.35, "Doublette optimale ECG", 1
        elif has_maths and has_hggsp:
            return 1.3, "Excellente doublette", 1
        elif has_maths:
            return 1.0, "Avec Maths", 3
        else:
            return 0.7, "Sans Maths", 4

    elif formation_type == "PASS_medecine":
        if has_pc and has_svt:
            return 1.4, "Doublette optimale PASS", 1
        elif has_maths and has_pc:
            return 1.3, "Excellente doublette", 1
        elif has_pc or has_svt:
            return 0.9, "Une spé scientifique", 3
        else:
            return 0.4, "Profil atypique", 5

    elif formation_type == "informatique":
        if has_maths and has_nsi:
            return 1.5, "Doublette parfaite", 1
        elif has_maths:
            return 1.1, "Avec Maths", 2
        else:
            return 0.4, "Sans Maths", 5

    elif formation_type == "ingenieur":
        if has_maths and (has_pc or has_si):
            return 1.4, "Doublette optimale", 1
        elif has_maths:
            return 0.9, "Avec Maths", 3
        else:
            return 0.35, "Sans Maths", 5

    elif formation_type == "droit":
        if has_hggsp and has_ses:
            return 1.25, "Doublette optimale", 1
        elif has_hggsp or has_ses:
            return 1.1, "Bonne doublette", 2
        else:
            return 1.0, "Neutre", 3

    elif formation_type == "lettres":
        has_humanites = "Humanités, littérature, philosophie" in specialites
        has_langues = any("Langue" in s or "langue" in s for s in specialites)
        if has_humanites and has_langues:
            return 1.3, "Doublette optimale", 1
        elif has_humanites or has_hggsp:
            return 1.15, "Bonne doublette", 2
        elif has_langues:
            return 1.1, "Avec Langues", 2
        else:
            return 0.9, "Profil atypique", 3

    elif formation_type == "economie":
        if has_maths and has_ses:
            return 1.3, "Doublette optimale", 1
        elif has_maths or has_ses:
            return 1.1, "Bonne doublette", 2
        else:
            return 0.9, "Neutre", 3

    # Default case
    return 1.0, "Peu d'impact", 3


def calculate_admission_probability(
    stats: Dict,
    type_bac: str,
    mention: str,
    specialites: List[str],
    boursier: bool,
    moyenne: float,
    academie: str
) -> Dict:
    """
    Calculate personalized admission probability for a formation.

    Combines multiple factors including bac type, mention, specialties,
    scholarship status, and geographic origin to estimate admission chances.

    Args:
        stats: Formation statistics (from extract_formation_stats).
        type_bac: Student's bac type ("Général", "Technologique", "Professionnel").
        mention: Expected or obtained mention.
        specialites: List of chosen specialties (for Bac Général).
        boursier: Whether the student is a scholarship recipient.
        moyenne: Student's average grade.
        academie: Student's academic region.

    Returns:
        Dict containing:
            - probability: Estimated admission percentage (0-95)
            - confidence: Confidence level ("Élevée" or "Moyenne")
            - blocking_factors: List of blocking issues
            - warnings: List of warning messages
            - positive_factors: List of positive factors
            - conseils: List of personalized recommendations
            - details: Detailed calculation factors

    Example:
        >>> result = calculate_admission_probability(
        ...     stats, "Général", "Bien (14-16)",
        ...     ["Mathématiques", "NSI"], True, 15.5, "Paris"
        ... )
        >>> print(f"Probability: {result['probability']}%")
    """
    # Initialize result structure
    result = {
        "probability": 0,
        "confidence": "Faible",
        "blocking_factors": [],
        "warnings": [],
        "positive_factors": [],
        "conseils": [],
        "details": {},
    }

    # Check for valid data
    if not stats or stats.get("admis_total", 0) == 0:
        result["blocking_factors"].append("Aucune donnée disponible")
        return result

    formation_type = get_formation_type(stats)
    taux_acces = stats.get("taux_acces", 50)

    # Calculate bac type factor
    pct_bac = {
        "Général": stats.get("pct_admis_bg", 0),
        "Technologique": stats.get("pct_admis_bt", 0),
        "Professionnel": stats.get("pct_admis_bp", 0),
    }
    result["details"]["pct_admis_par_bac"] = pct_bac

    bac_factor = pct_bac[type_bac] / 100
    if pct_bac[type_bac] == 0:
        result["probability"] = 0
        result["blocking_factors"].append(f"Aucun bachelier {type_bac} admis")
        return result
    elif pct_bac[type_bac] < 5:
        result["warnings"].append(
            f"Très peu de {type_bac} admis ({pct_bac[type_bac]:.1f}%)"
        )

    # Calculate mention factor
    mention_idx = MENTIONS.get(mention, 0)
    mention_factors = [0.5, 0.75, 1.0, 1.4, 1.8]
    mention_factor = mention_factors[mention_idx]

    # Analyze mention requirements
    pct_tb_tbf = stats.get("pct_tb", 0) + stats.get("pct_tbf", 0)
    pct_b_plus = pct_tb_tbf + stats.get("pct_b", 0)

    if pct_tb_tbf > 60:
        result["conseils"].append(
            f"Mentions des admis : {pct_tb_tbf:.0f}% ont TB ou TB+. "
            "Une mention TB est quasi-indispensable."
        )
        if mention_idx < 3:
            result["warnings"].append("Votre mention peut être insuffisante")
            mention_factor *= 0.7
    elif pct_b_plus > 70:
        result["conseils"].append(
            f"Mentions des admis : {pct_b_plus:.0f}% ont Bien ou mieux. "
            "Visez au moins Bien."
        )

    # Calculate specialty factor (for Bac Général only)
    spe_factor = 1.0
    spe_rank = 3
    if type_bac == "Général" and len(specialites) >= 2:
        spe_factor, spe_expl, spe_rank = calculate_doublette_advantage(
            specialites, formation_type
        )
        if spe_rank <= 2:
            result["positive_factors"].append(spe_expl)
        elif spe_rank >= 4:
            result["warnings"].append(spe_expl)

        # Add doublette recommendations
        doublettes = DOUBLETTES_PAR_TYPE.get(
            formation_type, DOUBLETTES_PAR_TYPE["general"]
        )
        top3 = ", ".join([d[0] for d in doublettes[:3]])
        result["conseils"].append(f"Doublettes populaires : {top3}")

    # Calculate scholarship factor
    boursier_factor = 1.0
    pct_boursiers = stats.get("pct_boursiers", 0)

    if boursier:
        if pct_boursiers > 0:
            if pct_boursiers <= 8:
                boursier_factor = 1.3
                result["positive_factors"].append(
                    f"Boursier très valorisé (quota proche du min : {pct_boursiers:.0f}%)"
                )
            elif pct_boursiers <= 15:
                boursier_factor = 1.15
                result["positive_factors"].append(
                    f"Statut boursier valorisé ({pct_boursiers:.0f}% de boursiers)"
                )
            else:
                boursier_factor = 1.05
                result["positive_factors"].append("Boursier (légèrement valorisé)")
        result["conseils"].append(
            f"Quota boursiers : {pct_boursiers:.0f}% des admis sont boursiers"
        )

    # Calculate geographic factor
    academie_factor = 1.0
    pct_meme_acad = stats.get("pct_meme_academie", 50)
    formation_academie = stats.get("academie", "")

    if formation_academie:
        meme_academie = (
            academie.lower() in formation_academie.lower()
            or formation_academie.lower() in academie.lower()
        )

        if pct_meme_acad > 70:
            if meme_academie:
                academie_factor = 1.2
                result["positive_factors"].append(
                    f"Même académie (formation locale : {pct_meme_acad:.0f}%)"
                )
            else:
                academie_factor = 0.8
                result["warnings"].append(
                    f"Hors académie (formation locale : {pct_meme_acad:.0f}%)"
                )
        elif pct_meme_acad < 30:
            result["conseils"].append(
                f"Formation à recrutement national ({pct_meme_acad:.0f}% locaux)"
            )

    # Calculate average grade factor
    moyenne_factor = 1.0
    if moyenne >= 18:
        moyenne_factor = 1.2
    elif moyenne >= 16:
        moyenne_factor = 1.1
    elif moyenne < 12:
        moyenne_factor = 0.8

    # Calculate final probability
    if taux_acces < 30:
        # For selective formations, specialty matters more
        prob = (
            taux_acces
            * bac_factor
            * mention_factor
            * (spe_factor ** 1.5)
            * boursier_factor
            * academie_factor
            * moyenne_factor
        )
    else:
        prob = (
            taux_acces
            * bac_factor
            * mention_factor
            * spe_factor
            * boursier_factor
            * academie_factor
            * moyenne_factor
        )

    # Clip probability to valid range
    prob = max(1, min(95, round(prob)))

    # Set result values
    result["probability"] = prob
    result["confidence"] = "Élevée" if stats["admis_total"] > 100 else "Moyenne"
    result["details"]["taux_acces"] = taux_acces
    result["details"]["spe_factor"] = spe_factor
    result["details"]["boursier_factor"] = boursier_factor
    result["details"]["academie_factor"] = academie_factor
    result["details"]["formation_type"] = formation_type

    return result


# =============================================================================
# UI COMPONENTS
# =============================================================================

def display_stat_box(value: str, label: str, col) -> None:
    """
    Display a styled statistics box in a Streamlit column.

    Args:
        value: The main value to display (e.g., "85%").
        label: The label below the value (e.g., "Taux d'accès").
        col: Streamlit column object to render in.
    """
    col.markdown(
        f"""
        <div class="stat-box">
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_probability_result(result: Dict) -> None:
    """
    Display the admission probability result with styling.

    Renders a colored probability display box along with conseils,
    blocking factors, warnings, and positive factors.

    Args:
        result: Result dictionary from calculate_admission_probability.
    """
    prob = result["probability"]

    # Determine CSS class based on probability
    if prob == 0:
        css_class = "prob-zero"
    elif prob < 20:
        css_class = "prob-low"
    elif prob < 50:
        css_class = "prob-medium"
    else:
        css_class = "prob-high"

    # Display main probability box
    st.markdown(
        f"""
        <div class="prob-display {css_class}">
            <div style="color: #ffffff !important; font-size: 4rem; font-weight: 700;">
                {prob}%
            </div>
            <div style="font-size: 1.2rem; margin-top: 10px; color: #ffffff !important;">
                Estimation d'admission
            </div>
            <div style="font-size: 0.9rem; opacity: 0.9; color: #ffffff !important;">
                Confiance : {result['confidence']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Display personalized recommendations
    if result.get("conseils"):
        st.markdown("### Conseils personnalisés")
        for conseil in result["conseils"]:
            st.markdown(
                f"""
                <div class="conseil-box">
                    <div class="conseil-text">{conseil}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Display factors in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Facteurs bloquants**")
        if result.get("blocking_factors"):
            for factor in result["blocking_factors"]:
                st.error(factor)
        else:
            st.success("Aucun")

    with col2:
        st.markdown("**Points d'attention**")
        if result.get("warnings"):
            for warning in result["warnings"]:
                st.warning(warning)
        else:
            st.success("Aucun")

    with col3:
        st.markdown("**Points forts**")
        if result.get("positive_factors"):
            for positive in result["positive_factors"]:
                st.success(positive)
        else:
            st.info("Aucun")


def get_top_doublettes_for_formations(
    formations: List[Dict]
) -> List[Tuple[str, int]]:
    """
    Get the top recommended specialty pairs for a list of formations.

    Aggregates doublette scores across all formations and returns
    the top 3 most recommended specialty combinations.

    Args:
        formations: List of formation statistics dictionaries.

    Returns:
        List of tuples (doublette_name, total_score) sorted by score.

    Example:
        >>> top_doublettes = get_top_doublettes_for_formations(formations_list)
        >>> for doub, score in top_doublettes:
        ...     print(f"{doub}: {score}")
    """
    doublettes_count: Dict[str, int] = {}

    for formation in formations:
        formation_type = get_formation_type(formation)
        doublettes = DOUBLETTES_PAR_TYPE.get(
            formation_type, DOUBLETTES_PAR_TYPE["general"]
        )

        # Add scores for top 3 doublettes of each formation
        for doub, score in doublettes[:3]:
            if doub not in doublettes_count:
                doublettes_count[doub] = 0
            doublettes_count[doub] += score

    # Sort by score descending and return top 3
    sorted_doublettes = sorted(
        doublettes_count.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_doublettes[:3]
