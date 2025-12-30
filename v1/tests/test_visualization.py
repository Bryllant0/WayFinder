#!/usr/bin/env python3
"""
Unit Tests for visualization.py Module (V1).

This module contains tests for the chart styling functions
used in the WayFinder application.

Test Categories:
    - Chart styling

Usage:
    python -m pytest tests/test_visualization.py -v

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
"""

import sys
import os

import plotly.graph_objects as go
import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import apply_chart_style


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

    def test_font_color_set(self):
        """Test that font color is set correctly."""
        fig = go.Figure()
        styled_fig = apply_chart_style(fig)
        assert styled_fig.layout.font.color == '#334155'

    def test_modifies_in_place(self):
        """Test that the function modifies the figure in place."""
        fig = go.Figure()
        result = apply_chart_style(fig)
        # Should be the same object
        assert result is fig

    def test_with_scatter_plot(self):
        """Test styling with an actual scatter plot."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
        styled_fig = apply_chart_style(fig)
        assert styled_fig.layout.paper_bgcolor == 'rgba(0,0,0,0)'

    def test_with_bar_chart(self):
        """Test styling with a bar chart."""
        fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 2, 3]))
        styled_fig = apply_chart_style(fig)
        assert styled_fig.layout.plot_bgcolor == 'rgba(0,0,0,0)'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
