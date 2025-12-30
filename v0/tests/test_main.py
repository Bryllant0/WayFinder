#!/usr/bin/env python3
"""
Unit Tests for main.py Module (V0).

This module contains tests for the CLI functions including
argument parsing and display utilities.

Test Categories:
    - Display functions
    - Argument parsing

Usage:
    python -m pytest tests/test_main.py -v

Author: Bryan Boislève - Mizaan-Abbas Katchera - Nawfel Bouazza
"""

import sys
import os
from io import StringIO
from unittest.mock import patch

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    print_header,
    print_divider,
    print_formation_card,
    print_stats,
    parse_arguments,
)


# =============================================================================
# TEST: Display Functions
# =============================================================================

class TestPrintHeader:
    """Test suite for print_header function."""

    def test_prints_title(self, capsys):
        """Test that header contains the title."""
        print_header("Test Title")
        captured = capsys.readouterr()
        assert "TEST TITLE" in captured.out

    def test_prints_separators(self, capsys):
        """Test that header has separator lines."""
        print_header("Test")
        captured = capsys.readouterr()
        assert "=" in captured.out


class TestPrintDivider:
    """Test suite for print_divider function."""

    def test_prints_dashes(self, capsys):
        """Test that divider prints dashes."""
        print_divider()
        captured = capsys.readouterr()
        assert "-" in captured.out

    def test_prints_60_dashes(self, capsys):
        """Test that divider is 60 characters."""
        print_divider()
        captured = capsys.readouterr()
        assert "-" * 60 in captured.out


class TestPrintFormationCard:
    """Test suite for print_formation_card function."""

    @pytest.fixture
    def sample_formation(self):
        """Sample formation data."""
        return {
            "lib_for_voe_ins": "Licence - Informatique",
            "g_ea_lib_vx": "Université Test",
            "voe_tot": 1000,
            "taux_acces": 50.0,
        }

    def test_prints_formation_name(self, sample_formation, capsys):
        """Test that formation name is displayed."""
        print_formation_card(sample_formation, 1)
        captured = capsys.readouterr()
        assert "Licence - Informatique" in captured.out

    def test_prints_index(self, sample_formation, capsys):
        """Test that index is displayed."""
        print_formation_card(sample_formation, 5)
        captured = capsys.readouterr()
        assert "[5]" in captured.out

    def test_prints_establishment(self, sample_formation, capsys):
        """Test that establishment is displayed."""
        print_formation_card(sample_formation, 1)
        captured = capsys.readouterr()
        assert "Université Test" in captured.out

    def test_handles_missing_data(self, capsys):
        """Test handling of missing formation data."""
        incomplete = {}
        print_formation_card(incomplete, 1)
        captured = capsys.readouterr()
        assert "Formation inconnue" in captured.out


class TestPrintStats:
    """Test suite for print_stats function."""

    @pytest.fixture
    def sample_stats(self):
        """Sample formation statistics."""
        return {
            "nom": "Test Formation",
            "etablissement": "Test University",
            "academie": "Test Academy",
            "taux_acces": 45.5,
            "voeux_total": 1000,
            "admis_total": 200,
            "capacite": 250,
            "pct_admis_bg": 70.0,
            "pct_admis_bt": 20.0,
            "pct_admis_bp": 10.0,
            "pct_sans_mention": 10.0,
            "pct_ab": 20.0,
            "pct_b": 30.0,
            "pct_tb": 30.0,
            "pct_tbf": 10.0,
            "pct_boursiers": 25.0,
            "pct_meme_academie": 60.0,
        }

    def test_prints_formation_name(self, sample_stats, capsys):
        """Test that formation name is in header (uppercase)."""
        print_stats(sample_stats)
        captured = capsys.readouterr()
        # print_header converts to uppercase
        assert "TEST FORMATION" in captured.out

    def test_prints_taux_acces(self, sample_stats, capsys):
        """Test that access rate is displayed."""
        print_stats(sample_stats)
        captured = capsys.readouterr()
        assert "45.5%" in captured.out

    def test_prints_bac_percentages(self, sample_stats, capsys):
        """Test that bac percentages are displayed."""
        print_stats(sample_stats)
        captured = capsys.readouterr()
        assert "70.0%" in captured.out  # Bac Général
        assert "20.0%" in captured.out  # Bac Techno

    def test_prints_sections(self, sample_stats, capsys):
        """Test that all sections are present."""
        print_stats(sample_stats)
        captured = capsys.readouterr()
        assert "INDICATEURS PRINCIPAUX" in captured.out
        assert "RÉPARTITION PAR TYPE DE BAC" in captured.out
        assert "RÉPARTITION PAR MENTION" in captured.out


# =============================================================================
# TEST: Argument Parsing
# =============================================================================

class TestParseArguments:
    """Test suite for parse_arguments function."""

    def test_no_args_returns_no_command(self):
        """Test that no arguments returns no command."""
        with patch.object(sys, 'argv', ['main.py']):
            args = parse_arguments()
            assert args.command is None

    def test_search_command_parsed(self):
        """Test parsing of search command."""
        with patch.object(sys, 'argv', ['main.py', 'search', 'informatique']):
            args = parse_arguments()
            assert args.command == 'search'
            assert args.query == 'informatique'

    def test_search_with_year(self):
        """Test search command with year option."""
        with patch.object(sys, 'argv', ['main.py', 'search', 'droit', '-y', '2023']):
            args = parse_arguments()
            assert args.command == 'search'
            assert args.query == 'droit'
            assert args.year == 2023

    def test_search_with_limit(self):
        """Test search command with limit option."""
        with patch.object(sys, 'argv', ['main.py', 'search', 'test', '-n', '20']):
            args = parse_arguments()
            assert args.limit == 20

    def test_verbose_flag(self):
        """Test verbose flag parsing."""
        with patch.object(sys, 'argv', ['main.py', '-v']):
            args = parse_arguments()
            assert args.verbose is True

    def test_default_year_is_2024(self):
        """Test that default year is 2024."""
        with patch.object(sys, 'argv', ['main.py', 'search', 'test']):
            args = parse_arguments()
            assert args.year == 2024

    def test_default_limit_is_10(self):
        """Test that default limit is 10."""
        with patch.object(sys, 'argv', ['main.py', 'search', 'test']):
            args = parse_arguments()
            assert args.limit == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
