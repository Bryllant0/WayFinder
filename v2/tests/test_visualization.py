#!/usr/bin/env python3
"""
Unit Tests for visualization.py Module.

This module contains tests for the data analysis and visualization
functions used in the WayFinder application.

Test Categories:
    - Taux d'accès computation
    - Selectivity labeling
    - Chart styling
    - Data loading

Usage:
    python -m pytest tests/test_visualization.py -v

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
"""

import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import apply_chart_style, compute_taux_acces, get_selectivity_label


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_parcoursup_data():
    """Create sample Parcoursup data for testing."""
    return pd.DataFrame({
        'session': [2024, 2023, 2022, 2021, 2019, 2018],
        'taux_acces': [45.0, 50.0, np.nan, np.nan, np.nan, np.nan],
        'taux_acces_ens': [np.nan, np.nan, 55.0, 60.0, np.nan, np.nan],
        'voe_tot': [1000, 1200, 1100, 900, 800, 750],
        'prop_tot': [450, 600, 605, 540, 400, 375],
        'acc_tot': [400, 550, 580, 500, 380, 350],
    })


# =============================================================================
# TEST: compute_taux_acces
# =============================================================================

class TestComputeTauxAcces:
    """Test suite for compute_taux_acces function."""

    def test_with_existing_taux_acces(self, sample_parcoursup_data):
        """Test that existing taux_acces values are preserved."""
        result = compute_taux_acces(sample_parcoursup_data)
        assert result.loc[0, 'taux_acces'] == 45.0

    def test_with_taux_acces_ens(self, sample_parcoursup_data):
        """Test that taux_acces_ens is used when taux_acces is missing."""
        result = compute_taux_acces(sample_parcoursup_data)
        assert result.loc[2, 'taux_acces'] == 55.0

    def test_calculation_from_prop_tot(self, sample_parcoursup_data):
        """Test calculation from prop_tot when both columns are NaN."""
        result = compute_taux_acces(sample_parcoursup_data)
        # For 2019: 400/800 * 100 = 50%
        assert result.loc[4, 'taux_acces'] == pytest.approx(50.0, rel=0.1)

    def test_calculation_with_nan_values(self):
        """Test handling of NaN values in calculation."""
        df = pd.DataFrame({
            'voe_tot': [100, np.nan, 200],
            'prop_tot': [50, 100, np.nan],
        })
        result = compute_taux_acces(df)
        assert 'taux_acces' in result.columns

    def test_clipping_to_valid_range(self):
        """Test that calculated values are clipped to 0-100."""
        df = pd.DataFrame({
            'voe_tot': [100, 50],
            'prop_tot': [150, 100],  # Would give >100%
        })
        result = compute_taux_acces(df)
        assert result['taux_acces'].max() <= 100

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled."""
        df = pd.DataFrame({
            'voe_tot': [0, 100],
            'prop_tot': [50, 50],
        })
        result = compute_taux_acces(df)
        assert not result['taux_acces'].isna().all()

    def test_returns_copy(self, sample_parcoursup_data):
        """Test that original DataFrame is not modified."""
        original_id = id(sample_parcoursup_data)
        result = compute_taux_acces(sample_parcoursup_data)
        assert id(result) != original_id


# =============================================================================
# TEST: get_selectivity_label
# =============================================================================

class TestGetSelectivityLabel:
    """Test suite for get_selectivity_label function."""

    def test_very_selective(self):
        """Test label for very selective formations (<20%)."""
        assert get_selectivity_label(10) == "Très sélectif"
        assert get_selectivity_label(19) == "Très sélectif"

    def test_selective(self):
        """Test label for selective formations (20-40%)."""
        assert get_selectivity_label(20) == "Sélectif"
        assert get_selectivity_label(39) == "Sélectif"

    def test_moderately_selective(self):
        """Test label for moderately selective formations (40-60%)."""
        assert get_selectivity_label(40) == "Modérément sélectif"
        assert get_selectivity_label(59) == "Modérément sélectif"

    def test_accessible(self):
        """Test label for accessible formations (60-80%)."""
        assert get_selectivity_label(60) == "Accessible"
        assert get_selectivity_label(79) == "Accessible"

    def test_very_accessible(self):
        """Test label for very accessible formations (>=80%)."""
        assert get_selectivity_label(80) == "Très accessible"
        assert get_selectivity_label(100) == "Très accessible"


# =============================================================================
# TEST: apply_chart_style
# =============================================================================

class TestApplyChartStyle:
    """Test suite for apply_chart_style function."""

    def test_transparent_background(self):
        """Test that chart has transparent background."""
        fig = go.Figure()
        styled_fig = apply_chart_style(fig)
        assert styled_fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'
        assert styled_fig.layout.plot_bgcolor == 'rgba(0,0,0,0)'

    def test_returns_figure(self):
        """Test that function returns a Figure object."""
        fig = go.Figure()
        result = apply_chart_style(fig)
        assert isinstance(result, go.Figure)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
