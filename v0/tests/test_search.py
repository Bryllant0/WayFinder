#!/usr/bin/env python3
"""
Unit Tests for search.py Module (V0).

This module contains tests for the formation search and
data extraction functions.

Test Categories:
    - Helper functions (_safe_int, _safe_float)
    - API functions (fetch_api, search_formations)
    - Data extraction (extract_formation_stats)

Usage:
    python -m pytest tests/test_search.py -v

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
"""

import sys
import os

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search import (
    _safe_int,
    _safe_float,
    extract_formation_stats,
    search_formations,
    DATASETS,
    API_BASE,
)


# =============================================================================
# TEST: Helper Functions
# =============================================================================

class TestSafeInt:
    """Test suite for _safe_int helper function."""

    def test_with_integer(self):
        """Test conversion of integer value."""
        assert _safe_int(42) == 42

    def test_with_string_number(self):
        """Test conversion of string number."""
        assert _safe_int("123") == 123

    def test_with_float(self):
        """Test conversion of float value."""
        assert _safe_int(3.7) == 3

    def test_with_none(self):
        """Test handling of None value."""
        assert _safe_int(None) == 0

    def test_with_empty_string(self):
        """Test handling of empty string."""
        assert _safe_int("") == 0

    def test_with_invalid_string(self):
        """Test handling of non-numeric string."""
        assert _safe_int("abc") == 0


class TestSafeFloat:
    """Test suite for _safe_float helper function."""

    def test_with_float(self):
        """Test conversion of float value."""
        assert _safe_float(3.14) == 3.14

    def test_with_integer(self):
        """Test conversion of integer value."""
        assert _safe_float(42) == 42.0

    def test_with_string_number(self):
        """Test conversion of string number."""
        assert _safe_float("12.5") == 12.5

    def test_with_none(self):
        """Test handling of None value."""
        assert _safe_float(None) == 0.0

    def test_with_empty_string(self):
        """Test handling of empty string."""
        assert _safe_float("") == 0.0

    def test_with_invalid_string(self):
        """Test handling of non-numeric string."""
        assert _safe_float("xyz") == 0.0


# =============================================================================
# TEST: Constants
# =============================================================================

class TestConstants:
    """Test suite for module constants."""

    def test_datasets_years_available(self):
        """Test that expected years are in DATASETS."""
        expected_years = [2024, 2023, 2022, 2021]
        for year in expected_years:
            assert year in DATASETS

    def test_datasets_values_are_strings(self):
        """Test that all dataset values are strings."""
        for year, dataset in DATASETS.items():
            assert isinstance(dataset, str)
            assert len(dataset) > 0

    def test_api_base_is_valid_url(self):
        """Test that API_BASE is a valid URL."""
        assert API_BASE.startswith("https://")
        assert "enseignementsup" in API_BASE


# =============================================================================
# TEST: extract_formation_stats
# =============================================================================

class TestExtractFormationStats:
    """Test suite for extract_formation_stats function."""

    @pytest.fixture
    def sample_formation(self):
        """Provide sample formation data for testing."""
        return {
            "lib_for_voe_ins": "Licence - Informatique",
            "g_ea_lib_vx": "Université Paris-Saclay",
            "acad_mies": "Versailles",
            "fili": "Licence",
            "voe_tot": 5000,
            "acc_tot": 500,
            "capa_fin": 600,
            "taux_acces": 45.5,
            "acc_bg": 400,
            "acc_bt": 80,
            "acc_bp": 20,
            "acc_sansmention": 50,
            "acc_ab": 100,
            "acc_b": 150,
            "acc_tb": 150,
            "acc_tbf": 50,
            "acc_brs": 100,
            "pct_aca_orig": 60.0,
        }

    def test_extracts_basic_info(self, sample_formation):
        """Test extraction of basic formation info."""
        stats = extract_formation_stats(sample_formation)
        assert stats["nom"] == "Licence - Informatique"
        assert stats["etablissement"] == "Université Paris-Saclay"
        assert stats["academie"] == "Versailles"

    def test_extracts_numeric_values(self, sample_formation):
        """Test extraction of numeric statistics."""
        stats = extract_formation_stats(sample_formation)
        assert stats["voeux_total"] == 5000
        assert stats["admis_total"] == 500
        assert stats["capacite"] == 600
        assert stats["taux_acces"] == 45.5

    def test_calculates_bac_percentages(self, sample_formation):
        """Test calculation of bac type percentages."""
        stats = extract_formation_stats(sample_formation)
        # Total bac = 400 + 80 + 20 = 500
        assert stats["pct_admis_bg"] == 80.0  # 400/500 * 100
        assert stats["pct_admis_bt"] == 16.0  # 80/500 * 100
        assert stats["pct_admis_bp"] == 4.0   # 20/500 * 100

    def test_calculates_mention_percentages(self, sample_formation):
        """Test calculation of mention percentages."""
        stats = extract_formation_stats(sample_formation)
        # Total mentions = 50 + 100 + 150 + 150 + 50 = 500
        assert stats["pct_sans_mention"] == 10.0  # 50/500 * 100
        assert stats["pct_ab"] == 20.0  # 100/500 * 100
        assert stats["pct_b"] == 30.0   # 150/500 * 100
        assert stats["pct_tb"] == 30.0  # 150/500 * 100
        assert stats["pct_tbf"] == 10.0 # 50/500 * 100

    def test_handles_missing_data(self):
        """Test handling of incomplete formation data."""
        minimal_data = {
            "lib_for_voe_ins": "Formation Test",
        }
        stats = extract_formation_stats(minimal_data)
        assert stats["nom"] == "Formation Test"
        assert stats["voeux_total"] == 0
        assert stats["taux_acces"] == 0.0

    def test_handles_empty_dict(self):
        """Test handling of empty dictionary."""
        stats = extract_formation_stats({})
        assert stats["nom"] == ""
        assert stats["etablissement"] == ""
        assert stats["taux_acces"] == 0.0

    def test_returns_dict(self, sample_formation):
        """Test that function returns a dictionary."""
        stats = extract_formation_stats(sample_formation)
        assert isinstance(stats, dict)

    def test_boursier_percentage(self, sample_formation):
        """Test calculation of boursier percentage."""
        stats = extract_formation_stats(sample_formation)
        # 100 boursiers / 500 admis * 100 = 20%
        assert stats["pct_boursiers"] == 20.0


# =============================================================================
# TEST: search_formations
# =============================================================================

class TestSearchFormations:
    """Test suite for search_formations function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        results = search_formations("test", year=2024)
        assert isinstance(results, list)

    def test_invalid_year_returns_empty(self):
        """Test that invalid year returns empty list."""
        results = search_formations("informatique", year=1900)
        assert results == []

    def test_empty_search_term(self):
        """Test behavior with empty search term."""
        results = search_formations("", year=2024)
        assert isinstance(results, list)

    def test_short_search_term(self):
        """Test that short search terms still work."""
        results = search_formations("IF", year=2024)
        assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
