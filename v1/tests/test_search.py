#!/usr/bin/env python3
"""
Unit Tests for search.py Module.

This module contains tests for the formation search, probability calculation,
and specialty recommendation functions used in the WayFinder application.

Test Categories:
    - Formation type detection
    - Admission probability calculation
    - Doublette (specialty pair) recommendations
    - Formation statistics extraction

Usage:
    python -m pytest tests/test_search.py -v

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
"""

import sys
import os

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import (
    DOUBLETTES_PAR_TYPE,
    MENTIONS,
    SPECIALITES,
    TYPES_BAC,
    calculate_doublette_advantage,
    get_formation_type,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_formation_cpge():
    """Create a sample CPGE formation."""
    return {
        "nom": "CPGE MPSI",
        "etablissement": "Lycée Louis-le-Grand",
        "filiere_detail": "Classe préparatoire scientifique",
        "taux_acces": 5.0,
        "voeux_total": 5000,
    }


@pytest.fixture
def sample_formation_licence():
    """Create a sample Licence formation."""
    return {
        "nom": "Licence Informatique",
        "etablissement": "Université Paris-Saclay",
        "filiere_detail": "Informatique",
        "taux_acces": 45.0,
        "voeux_total": 2000,
    }


# =============================================================================
# TEST: get_formation_type
# =============================================================================

class TestGetFormationType:
    """Test suite for get_formation_type function."""

    def test_cpge_scientifique_mpsi(self):
        """Test detection of CPGE scientifique MPSI."""
        formation = {"nom": "CPGE MPSI", "filiere_detail": ""}
        assert get_formation_type(formation) == "CPGE_scientifique"

    def test_cpge_scientifique_pcsi(self):
        """Test detection of CPGE scientifique PCSI."""
        formation = {"nom": "CPGE PCSI", "filiere_detail": ""}
        assert get_formation_type(formation) == "CPGE_scientifique"

    def test_cpge_scientifique_bcpst(self):
        """Test detection of CPGE scientifique BCPST."""
        formation = {"nom": "CPGE BCPST", "filiere_detail": "Biologie"}
        assert get_formation_type(formation) == "CPGE_scientifique"

    def test_cpge_commerce_ecg(self):
        """Test detection of CPGE commerce ECG."""
        formation = {"nom": "CPGE ECG", "filiere_detail": "Économie"}
        assert get_formation_type(formation) == "CPGE_commerce"

    def test_cpge_lettres_khagne(self):
        """Test detection of CPGE littéraire Khâgne."""
        formation = {"nom": "CPGE Khâgne A/L", "filiere_detail": "Lettres"}
        assert get_formation_type(formation) == "lettres"

    def test_cpge_lettres_hypokhagne(self):
        """Test detection of CPGE littéraire Hypokhâgne."""
        formation = {"nom": "CPGE Hypokhâgne", "filiere_detail": ""}
        assert get_formation_type(formation) == "lettres"

    def test_medecine_pass(self):
        """Test detection of médecine PASS."""
        formation = {"nom": "PASS - Parcours Accès Santé", "filiere_detail": ""}
        assert get_formation_type(formation) == "PASS_medecine"

    def test_medecine_las(self):
        """Test detection of médecine L.AS."""
        formation = {"nom": "L.AS Droit option santé", "filiere_detail": ""}
        assert get_formation_type(formation) == "PASS_medecine"

    def test_informatique(self):
        """Test detection of informatique formations."""
        formation = {"nom": "Licence Informatique", "filiere_detail": ""}
        assert get_formation_type(formation) == "informatique"

    def test_informatique_data(self):
        """Test detection of data science formations."""
        formation = {"nom": "BUT Data Science", "filiere_detail": "Numérique"}
        assert get_formation_type(formation) == "informatique"

    def test_ingenieur(self):
        """Test detection of engineering schools."""
        formation = {"nom": "École d'ingénieur", "filiere_detail": ""}
        assert get_formation_type(formation) == "ingenieur"

    def test_ingenieur_insa(self):
        """Test detection of INSA."""
        formation = {"nom": "INSA Lyon", "filiere_detail": ""}
        assert get_formation_type(formation) == "ingenieur"

    def test_droit(self):
        """Test detection of law formations."""
        formation = {"nom": "Licence Droit", "filiere_detail": "Juridique"}
        assert get_formation_type(formation) == "droit"

    def test_science_po(self):
        """Test detection of Sciences Po."""
        formation = {"nom": "Sciences Po Paris", "filiere_detail": ""}
        assert get_formation_type(formation) == "droit"

    def test_economie(self):
        """Test detection of economics formations."""
        formation = {"nom": "Licence Économie Gestion", "filiere_detail": ""}
        assert get_formation_type(formation) == "economie"

    def test_unknown_formation(self):
        """Test that unknown formations return 'general'."""
        formation = {"nom": "Formation Mystère", "filiere_detail": "Inconnu"}
        assert get_formation_type(formation) == "general"

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        formation = {"nom": "LICENCE INFORMATIQUE", "filiere_detail": ""}
        assert get_formation_type(formation) == "informatique"

    def test_empty_formation(self):
        """Test handling of empty formation data."""
        formation = {"nom": "", "filiere_detail": ""}
        assert get_formation_type(formation) == "general"

    def test_missing_fields(self):
        """Test handling of missing fields."""
        formation = {"nom": "Licence Droit"}
        assert get_formation_type(formation) == "droit"


# =============================================================================
# TEST: calculate_doublette_advantage
# =============================================================================

class TestCalculateDoubletteAdvantage:
    """Test suite for calculate_doublette_advantage function."""

    def test_maths_physique_for_cpge_scientifique(self):
        """Test that Maths + Physique is advantageous for CPGE scientifique."""
        advantage, _, _ = calculate_doublette_advantage(
            ["Mathématiques", "Physique-chimie"],
            "CPGE_scientifique"
        )
        assert advantage > 1.0

    def test_maths_ses_for_cpge_commerce(self):
        """Test that Maths + SES is advantageous for CPGE commerce."""
        advantage, _, _ = calculate_doublette_advantage(
            ["Mathématiques", "Sciences économiques et sociales"],
            "CPGE_commerce"
        )
        assert advantage > 1.0

    def test_empty_specialites(self):
        """Test with empty specialites list."""
        advantage, _, _ = calculate_doublette_advantage([], "CPGE_scientifique")
        # Returns base score of 1.0 when no specialites
        assert advantage >= 0

    def test_single_specialite(self):
        """Test with single specialite."""
        advantage, _, _ = calculate_doublette_advantage(
            ["Mathématiques"],
            "CPGE_scientifique"
        )
        assert isinstance(advantage, (int, float))

    def test_unknown_formation_type(self):
        """Test handling of unknown formation type."""
        advantage, _, _ = calculate_doublette_advantage(
            ["Mathématiques", "Physique-chimie"],
            "unknown_type"
        )
        assert isinstance(advantage, (int, float))


# =============================================================================
# TEST: Constants
# =============================================================================

class TestConstants:
    """Test suite for module constants."""

    def test_specialites_not_empty(self):
        """Test that SPECIALITES dictionary is not empty."""
        assert len(SPECIALITES) > 0

    def test_specialites_have_codes(self):
        """Test that all specialties have short codes."""
        for name, code in SPECIALITES.items():
            assert len(code) > 0
            assert len(code) <= 5

    def test_mentions_have_points(self):
        """Test that all mentions have associated point values."""
        for mention, points in MENTIONS.items():
            assert isinstance(points, (int, float))
            assert points >= 0

    def test_mentions_ordered(self):
        """Test that mentions are ordered by value."""
        values = list(MENTIONS.values())
        assert values[0] <= values[-1]

    def test_types_bac_includes_general(self):
        """Test that bac types include Général."""
        assert "Général" in TYPES_BAC

    def test_doublettes_par_type_coverage(self):
        """Test that main formation types have doublette recommendations."""
        required_types = [
            "CPGE_scientifique",
            "CPGE_commerce",
            "PASS_medecine",
            "ingenieur",
        ]
        for formation_type in required_types:
            assert formation_type in DOUBLETTES_PAR_TYPE
            assert len(DOUBLETTES_PAR_TYPE[formation_type]) > 0


# =============================================================================
# TEST: Integration
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_formation_to_doublette_pipeline(self):
        """Test the complete pipeline from formation to doublette recommendation."""
        formation = {"nom": "CPGE MPSI", "filiere_detail": ""}

        formation_type = get_formation_type(formation)
        assert formation_type == "CPGE_scientifique"

        assert formation_type in DOUBLETTES_PAR_TYPE
        doublettes = DOUBLETTES_PAR_TYPE[formation_type]
        assert len(doublettes) > 0

        top_doublette = doublettes[0]
        assert "Maths" in top_doublette[0] or "Physique" in top_doublette[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
