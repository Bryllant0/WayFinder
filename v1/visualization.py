#!/usr/bin/env python3
"""
Visualization Module - Chart Styling Utilities (V1).

This module provides chart styling utilities for the WayFinder
Parcoursup analytics application.

Functions:
    apply_chart_style: Apply consistent transparent styling to Plotly figures.

Example:
    >>> from visualization import apply_chart_style
    >>> fig = px.scatter(df, x='x', y='y')
    >>> fig = apply_chart_style(fig)
    >>> st.plotly_chart(fig)

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 1.0
"""

import logging

import plotly.graph_objects as go

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# CHART STYLING UTILITIES
# =============================================================================

def apply_chart_style(fig: go.Figure) -> go.Figure:
    """
    Apply consistent transparent styling to a Plotly figure.

    Sets transparent background, consistent font colors, and subtle grid lines
    to match the application's light theme.

    Args:
        fig: Plotly figure object to style.

    Returns:
        go.Figure: The styled figure (modified in place and returned).

    Example:
        >>> fig = px.scatter(df, x='x', y='y')
        >>> fig = apply_chart_style(fig)
        >>> st.plotly_chart(fig)
    """
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#334155'),
    )
    fig.update_xaxes(
        gridcolor='rgba(203,213,225,0.3)',
        zerolinecolor='rgba(203,213,225,0.3)',
    )
    fig.update_yaxes(
        gridcolor='rgba(203,213,225,0.3)',
        zerolinecolor='rgba(203,213,225,0.3)',
    )

    logger.debug("Chart style applied")
    return fig
