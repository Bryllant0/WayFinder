#!/usr/bin/env python3
"""
Visualization Module - Charts and Analysis Components.

This module provides Streamlit UI components and visualization functions
for the WayFinder Parcoursup analytics application. It handles data loading,
chart generation, and rendering of analysis tabs.

Main Components:
    - Chart styling utilities for consistent Plotly visualizations
    - Taux d'accès computation across different data years
    - Tab rendering functions for Streamlit interface
    - Data loading with caching for performance

Functions:
    apply_chart_style: Apply consistent transparent styling to Plotly figures.
    compute_taux_acces: Calculate/harmonize access rates across years.
    load_analyzer: Load data and ML model with Streamlit caching.
    render_tab_*: Render various analysis tabs.

Example:
    >>> from visualization import load_analyzer, render_tab_tendances
    >>> analyzer, df, df_latest = load_analyzer()
    >>> render_tab_tendances(analyzer, df, df_latest)

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 2.0
"""

import logging
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
    return fig


# =============================================================================
# DATA PROCESSING UTILITIES
# =============================================================================

def compute_taux_acces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate and harmonize the access rate (taux d'accès) for all years.

    Different Parcoursup data years use different column names:
        - 2018-2019: No taux_acces column, must be calculated
        - 2020-2021: Uses taux_acces_ens column
        - 2022+: Uses taux_acces column

    Formula: taux_acces = (propositions or admis / voeux) * 100

    Args:
        df: DataFrame containing Parcoursup data.

    Returns:
        pd.DataFrame: DataFrame with normalized 'taux_acces' column.

    Note:
        Returns a copy of the DataFrame to avoid modifying the original.

    Example:
        >>> df = pd.read_csv('parcoursup_data.csv')
        >>> df = compute_taux_acces(df)
        >>> print(df['taux_acces'].mean())
    """
    df = df.copy()

    # Step 1: Merge taux_acces and taux_acces_ens if both exist
    if 'taux_acces' in df.columns and 'taux_acces_ens' in df.columns:
        df['taux_acces'] = df['taux_acces'].combine_first(df['taux_acces_ens'])
    elif 'taux_acces' not in df.columns and 'taux_acces_ens' in df.columns:
        df['taux_acces'] = df['taux_acces_ens']

    # Step 2: Create column if it doesn't exist
    if 'taux_acces' not in df.columns:
        df['taux_acces'] = np.nan

    # Step 3: Calculate for rows where taux is NaN (2018-2019 data)
    mask_missing = df['taux_acces'].isna()
    if mask_missing.any():
        # Find the voeux (wishes) column
        voeux_col = None
        for col in ['voe_tot', 'nb_voe_pp', 'nb_voe_tot']:
            if col in df.columns:
                voeux_col = col
                break

        # Find the propositions/admis column
        prop_col = None
        for col in ['prop_tot', 'nb_prop', 'acc_tot', 'nb_acc_tot']:
            if col in df.columns:
                prop_col = col
                break

        # Calculate taux d'accès
        if voeux_col and prop_col:
            calculated = (
                df.loc[mask_missing, prop_col]
                / df.loc[mask_missing, voeux_col].replace(0, 1)
            ) * 100
            df.loc[mask_missing, 'taux_acces'] = calculated.clip(0, 100)

    return df


def get_selectivity_label(taux: float) -> str:
    """
    Get a human-readable selectivity label based on access rate.

    Args:
        taux: Access rate percentage (0-100).

    Returns:
        str: Selectivity label in French.

    Example:
        >>> get_selectivity_label(15)
        'Très sélectif'
        >>> get_selectivity_label(85)
        'Très accessible'
    """
    if taux < 20:
        return "Très sélectif"
    elif taux < 40:
        return "Sélectif"
    elif taux < 60:
        return "Modérément sélectif"
    elif taux < 80:
        return "Accessible"
    else:
        return "Très accessible"


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_resource
def load_analyzer():
    """
    Load the Parcoursup analyzer with data and ML models.

    Attempts to load pre-trained models if available, otherwise trains
    new models from the data. Results are cached for performance.

    Returns:
        Tuple containing:
            - analyzer: ParcoursupAnalyzer instance (or None if unavailable)
            - df: Full DataFrame with all years
            - df_latest: DataFrame with only the latest year

    Example:
        >>> analyzer, df, df_latest = load_analyzer()
        >>> if analyzer:
        ...     print(f"Loaded {len(df)} records")
    """
    try:
        from data_loader import ParcoursupAnalyzer
    except ImportError:
        logger.warning("Could not import ParcoursupAnalyzer")
        return None, None, None

    analyzer = ParcoursupAnalyzer()

    model_path = "models/parcoursup_model_v3.pkl"
    data_path = "data/parcoursup_data.csv"

    # Try to load existing model or train new one
    if os.path.exists(model_path):
        logger.info("Loading existing model from %s", model_path)
        analyzer.load_model(model_path)
    elif os.path.exists(data_path):
        logger.info("Training new model from %s", data_path)
        analyzer.load_data(data_path)
        analyzer.prepare_features()
        analyzer.train_clustering()
        os.makedirs("models", exist_ok=True)
        analyzer.save_model(model_path)
    else:
        logger.warning("No data or model found")
        return None, None, None

    df = analyzer.df
    df_latest = analyzer.df_latest

    # Compute/harmonize taux d'accès for all years
    df = compute_taux_acces(df)
    df_latest = compute_taux_acces(df_latest)
    analyzer.df = df
    analyzer.df_latest = df_latest

    logger.info("Analyzer loaded with %d total records", len(df))
    return analyzer, df, df_latest


# =============================================================================
# TAB: SIMILAR FORMATIONS
# =============================================================================

def render_tab_recherche_etablissement(analyzer, df, df_latest) -> None:
    """
    Render the 'Similar Formations' tab.

    Allows users to find formations similar to a selected one using
    K-Nearest Neighbors algorithm.

    Args:
        analyzer: ParcoursupAnalyzer instance with trained KNN models.
        df: Full DataFrame with all years of data.
        df_latest: DataFrame with only the latest year.
    """
    st.header("Trouvez des formations similaires")

    # Get selected formation from session state
    selected_from_search = st.session_state.get("selected_formation", None)

    # Initialize display variables
    selected_formation = None
    formation_row = None
    taux = 0
    tension = 0
    n_similar = 10
    selected_etab = None

    col_search, col_results = st.columns([1, 2])

    with col_search:
        st.subheader("1. Choisissez un établissement")
        etablissements = sorted(df_latest['g_ea_lib_vx'].dropna().unique())

        # Pre-fill with establishment from selected formation
        default_etab_search = ""
        if selected_from_search:
            default_etab_search = selected_from_search.get("etablissement", "")[:20]

        etab_search = st.text_input(
            "Recherche",
            value=default_etab_search,
            placeholder="Ex: Louis-le-Grand, Sorbonne...",
            key="analyzer_etab_search"
        )

        # Filter establishments based on search
        if etab_search:
            filtered_etabs = [
                e for e in etablissements
                if etab_search.lower() in e.lower()
            ]
        else:
            filtered_etabs = etablissements[:100]

        # Find pre-selected establishment index
        default_etab_idx = 0
        if selected_from_search and filtered_etabs:
            etab_name = selected_from_search.get("etablissement", "")
            for i, e in enumerate(filtered_etabs):
                if etab_name.lower() in e.lower() or e.lower() in etab_name.lower():
                    default_etab_idx = i
                    break

        if filtered_etabs:
            selected_etab = st.selectbox(
                "Établissement",
                options=filtered_etabs,
                index=default_etab_idx,
                key="analyzer_etab_select"
            )

        # Formation selection within establishment
        if selected_etab:
            etab_formations = df_latest[df_latest['g_ea_lib_vx'] == selected_etab]
            st.subheader("2. Choisissez une formation")
            formation_options = etab_formations['form_lib_voe_acc'].tolist()

            # Find pre-selected formation index
            default_form_idx = 0
            if selected_from_search and formation_options:
                form_name = selected_from_search.get("nom", "")
                for i, f in enumerate(formation_options):
                    if (form_name.lower()[:30] in f.lower()
                            or f.lower() in form_name.lower()):
                        default_form_idx = i
                        break

            if formation_options:
                selected_formation = st.selectbox(
                    "Formation",
                    options=formation_options,
                    index=default_form_idx,
                    key="analyzer_formation_select"
                )

            # Display selected formation info
            if selected_formation:
                formation_row = etab_formations[
                    etab_formations['form_lib_voe_acc'] == selected_formation
                ].iloc[0]

                st.subheader("Formation sélectionnée")
                taux = formation_row.get(
                    'taux_acces',
                    formation_row.get('taux_acces_ens', 0)
                )
                tension = formation_row.get('tension', 0)

                st.markdown(f"**{selected_formation}**")
                st.markdown(
                    f"{formation_row.get('acad_mies', 'N/A')} "
                    f"({formation_row.get('region_etab_aff', 'N/A')})"
                )
                st.markdown(f"**{get_selectivity_label(taux)}**")

                col_m1, col_m2 = st.columns(2)
                col_m1.metric("Taux d'accès", f"{taux:.0f}%")
                col_m2.metric("Tension", f"{tension:.1f}")

                n_similar = st.slider(
                    "Nombre d'alternatives",
                    5, 20, 10,
                    key="analyzer_n_similar"
                )

    # Display similar formations
    with col_results:
        if selected_formation and formation_row is not None:
            st.subheader("Formations similaires")
            target_idx = formation_row.name
            formation_name = selected_formation

            if formation_name in analyzer.knn_models:
                knn = analyzer.knn_models[formation_name]
                formation_indices = analyzer.knn_indices[formation_name]
                target_local_idx = analyzer.df_latest.index.get_loc(target_idx)
                target_features = analyzer.feature_matrix[
                    target_local_idx
                ].reshape(1, -1)

                n_results = min(n_similar + 1, len(formation_indices))
                distances, knn_indices = knn.kneighbors(
                    target_features,
                    n_neighbors=n_results
                )

                # Build results
                results_data = []
                for knn_idx, dist in zip(knn_indices[0], distances[0]):
                    global_idx = formation_indices[knn_idx]
                    if global_idx == target_idx:
                        continue
                    row = df_latest.loc[global_idx]
                    similarity = 1 / (1 + dist)
                    results_data.append({
                        'Établissement': row.get('g_ea_lib_vx', 'N/A'),
                        'Académie': row.get('acad_mies', 'N/A'),
                        'Région': row.get('region_etab_aff', 'N/A'),
                        "Taux d'accès": row.get(
                            'taux_acces',
                            row.get('taux_acces_ens', 0)
                        ),
                        'Tension': row.get('tension', 0),
                        '% Mention TB': row.get('pct_mention_tb', 0),
                        'Similarité': similarity,
                    })

                if results_data:
                    results_df = pd.DataFrame(results_data)
                    results_df = results_df.sort_values(
                        'Similarité',
                        ascending=False
                    ).head(n_similar)

                    st.markdown(
                        f"**{len(results_df)} établissements** "
                        "proposent cette formation"
                    )

                    # Scatter plot of alternatives
                    fig = px.scatter(
                        results_df,
                        x="Taux d'accès",
                        y="Tension",
                        size="Similarité",
                        color="Taux d'accès",
                        hover_name="Établissement",
                        hover_data=["Académie", "% Mention TB"],
                        color_continuous_scale="RdYlGn",
                        title="Comparaison des alternatives"
                    )

                    # Add target marker
                    fig.add_trace(go.Scatter(
                        x=[taux],
                        y=[tension],
                        mode='markers',
                        marker=dict(size=20, color='blue', symbol='star'),
                        name='Votre cible',
                        hovertext=selected_etab
                    ))

                    fig = apply_chart_style(fig)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width="stretch")

                    # Format display dataframe
                    display_df = results_df.copy()
                    display_df['Similarité'] = display_df['Similarité'].apply(
                        lambda x: f"{x:.0%}"
                    )
                    display_df["Taux d'accès"] = display_df["Taux d'accès"].apply(
                        lambda x: f"{x:.0f}%"
                    )
                    display_df['Tension'] = display_df['Tension'].apply(
                        lambda x: f"{x:.1f}"
                    )
                    display_df['% Mention TB'] = display_df['% Mention TB'].apply(
                        lambda x: f"{x:.0f}%"
                    )
                    st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.warning("Pas assez de données pour trouver des alternatives.")


# =============================================================================
# TAB: TREND ANALYSIS
# =============================================================================

def render_tab_tendances(analyzer, df, df_latest) -> None:
    """
    Render the 'Trends Analysis' tab.

    Shows historical trends for a selected formation type with
    linear regression predictions.

    Args:
        analyzer: ParcoursupAnalyzer instance.
        df: Full DataFrame with all years of data.
        df_latest: DataFrame with only the latest year.
    """
    st.header("Analyse des tendances d'une formation")

    # Search and selection
    col_search, col_select = st.columns([1, 1])
    formations_types = df['form_lib_voe_acc'].value_counts().head(100).index.tolist()

    with col_search:
        trend_search = st.text_input(
            "Recherchez une formation",
            placeholder="Ex: Licence, CPGE, droit...",
            key="analyzer_trend_search"
        )

    if trend_search:
        filtered_formations = [
            f for f in formations_types
            if trend_search.lower() in f.lower()
        ]
    else:
        filtered_formations = formations_types

    with col_select:
        selected_trend_formation = st.selectbox(
            "Type de formation",
            options=filtered_formations[:50],
            index=0 if filtered_formations else None,
            key="analyzer_trend_select"
        )

    st.divider()

    # Display results
    if selected_trend_formation:
        mask = df['form_lib_voe_acc'].str.contains(
            selected_trend_formation,
            case=False,
            na=False
        )
        df_filtered = df[mask].copy()

        if len(df_filtered) == 0:
            st.error("Aucune donnée trouvée pour cette formation.")
            return

        # Calculate taux d'accès for all years
        df_filtered = compute_taux_acces(df_filtered)
        df_filtered['taux_acces_combined'] = df_filtered['taux_acces']

        # Aggregate by year
        trends = df_filtered.groupby('session').agg({
            'taux_acces_combined': 'mean',
            'tension': 'mean',
            'voe_tot': 'sum',
            'form_lib_voe_acc': 'count'
        }).round(2)
        trends.columns = ["Taux d'accès", "Tension", "Total vœux", "Nb formations"]
        trends = trends.dropna(subset=["Taux d'accès"])

        if len(trends) >= 3:
            # Import sklearn for regression
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score

            X = trends.index.values.reshape(-1, 1).astype(float)
            y = trends["Taux d'accès"].values

            # Fit linear regression
            model = LinearRegression()
            model.fit(X, y)
            pente = model.coef_[0]
            y_pred = model.predict(X)
            r2 = r2_score(y, y_pred)

            # Project to 2025
            projection_2025 = model.predict([[2025]])[0]

            # Display trend interpretation
            if pente < -1:
                st.error("FORTE BAISSE - Formation de plus en plus sélective")
            elif pente < -0.3:
                st.warning("BAISSE - Devient plus sélective")
            elif pente > 1:
                st.success("FORTE HAUSSE - Formation de plus en plus accessible")
            elif pente > 0.3:
                st.info("HAUSSE - Devient plus accessible")
            else:
                st.info("STABLE - Pas de changement significatif")

            # Display metrics
            col_t1, col_t2, col_t3 = st.columns(3)
            col_t1.metric("Pente", f"{pente:+.2f}%/an")
            col_t2.metric("Fiabilité (R²)", f"{r2:.0%}")
            col_t3.metric("Projection 2025", f"{projection_2025:.1f}%")

            # Prepare chart data
            years_list = [int(y) for y in trends.index.tolist()]
            years_with_2025 = years_list + [2025]
            y_pred_with_2025 = list(y_pred) + [projection_2025]

            # Create subplot figure
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Évolution du taux d'accès", "Évolution de la tension")
            )

            # Access rate chart
            fig.add_trace(
                go.Scatter(
                    x=years_list,
                    y=trends["Taux d'accès"],
                    mode='lines+markers',
                    name="Taux d'accès réel",
                    line=dict(color='#3b82f6', width=3)
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=years_with_2025,
                    y=y_pred_with_2025,
                    mode='lines',
                    name="Régression linéaire",
                    line=dict(color='#ef4444', dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[2025],
                    y=[projection_2025],
                    mode='markers',
                    name="Prédiction 2025",
                    marker=dict(size=12, color='#ef4444', symbol='star')
                ),
                row=1, col=1
            )

            # Tension chart
            fig.add_trace(
                go.Scatter(
                    x=years_list,
                    y=trends["Tension"],
                    mode='lines+markers',
                    name="Tension",
                    line=dict(color='#f59e0b', width=3)
                ),
                row=1, col=2
            )

            fig = apply_chart_style(fig)
            fig.update_layout(height=400, showlegend=True)
            fig.update_xaxes(
                tickmode='array',
                tickvals=years_with_2025,
                ticktext=[str(y) for y in years_with_2025]
            )
            st.plotly_chart(fig, width="stretch")

            # Data table
            st.subheader("Données par année")
            display_trends = trends.copy()
            display_trends['Prédit'] = y_pred.round(1)
            display_trends['Écart'] = (trends["Taux d'accès"] - y_pred).round(1)
            st.dataframe(display_trends, width="stretch")
        else:
            st.warning(f"Seulement {len(trends)} année(s) de données. Minimum 3 requis.")


# =============================================================================
# TAB: OVERVIEW
# =============================================================================

def render_tab_vue_ensemble(analyzer, df, df_latest) -> None:
    """
    Render the 'Overview' tab with global statistics and ML analysis.

    Includes scatter plots, distribution charts, clustering visualization,
    and detailed statistics tables.

    Args:
        analyzer: ParcoursupAnalyzer instance.
        df: Full DataFrame with all years of data.
        df_latest: DataFrame with only the latest year.
    """
    st.header("Vue d'ensemble des formations")

    col_filters, col_viz = st.columns([1, 3])

    # Filters
    with col_filters:
        st.subheader("Filtres")

        regions = ['Toutes'] + sorted(
            df_latest['region_etab_aff'].dropna().unique().tolist()
        )
        selected_region = st.selectbox("Région", regions, key="analyzer_region")

        types_formation = ['Tous'] + sorted(
            df_latest['form_lib_voe_acc'].value_counts().head(30).index.tolist()
        )
        selected_type = st.selectbox(
            "Type de formation",
            types_formation,
            key="analyzer_type"
        )

        taux_range = st.slider(
            "Taux d'accès (%)",
            0, 100, (0, 100),
            key="analyzer_taux_range"
        )

        # Apply filters
        df_filtered_viz = df_latest.copy()
        if selected_region != 'Toutes':
            df_filtered_viz = df_filtered_viz[
                df_filtered_viz['region_etab_aff'] == selected_region
            ]
        if selected_type != 'Tous':
            df_filtered_viz = df_filtered_viz[
                df_filtered_viz['form_lib_voe_acc'] == selected_type
            ]
        df_filtered_viz = df_filtered_viz[
            (df_filtered_viz['taux_acces'] >= taux_range[0])
            & (df_filtered_viz['taux_acces'] <= taux_range[1])
        ]

        st.metric("Formations filtrées", len(df_filtered_viz))

    # Main visualization
    with col_viz:
        if len(df_filtered_viz) > 0:
            cluster_colors = ['#1e40af', '#dc2626', '#059669', '#d97706', '#7c3aed']
            plot_df = df_filtered_viz.head(500).copy()

            if 'cluster' in plot_df.columns:
                plot_df['cluster_str'] = plot_df['cluster'].astype(str)
                fig = px.scatter(
                    plot_df,
                    x='taux_acces',
                    y='tension',
                    color='cluster_str',
                    color_discrete_sequence=cluster_colors,
                    size='capa_fin' if 'capa_fin' in plot_df.columns else None,
                    hover_name='g_ea_lib_vx',
                    hover_data=['form_lib_voe_acc', 'acad_mies'],
                    title="Sélectivité vs Tension",
                    labels={
                        'taux_acces': "Taux d'accès (%)",
                        'tension': 'Tension (vœux/places)',
                        'cluster_str': 'Groupe'
                    }
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x='taux_acces',
                    y='tension',
                    color='taux_acces',
                    size='capa_fin' if 'capa_fin' in plot_df.columns else None,
                    hover_name='g_ea_lib_vx',
                    hover_data=['form_lib_voe_acc', 'acad_mies'],
                    title="Sélectivité vs Tension",
                    labels={
                        'taux_acces': "Taux d'accès (%)",
                        'tension': 'Tension (vœux/places)'
                    },
                    color_continuous_scale='RdYlGn'
                )

            fig = apply_chart_style(fig)
            fig.update_layout(height=500)
            fig.update_traces(marker=dict(opacity=0.7))
            st.plotly_chart(fig, width="stretch")

    # Distribution charts
    if len(df_filtered_viz) > 0:
        st.divider()
        col_d1, col_d2 = st.columns(2)

        with col_d1:
            fig_region = px.pie(
                df_filtered_viz['region_etab_aff'].value_counts().head(10).reset_index(),
                values='count',
                names='region_etab_aff',
                title="Répartition par région"
            )
            fig_region = apply_chart_style(fig_region)
            fig_region.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50))
            fig_region.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_region, width="stretch")

        with col_d2:
            fig_taux = px.histogram(
                df_filtered_viz,
                x='taux_acces',
                nbins=20,
                title="Distribution du taux d'accès",
                labels={
                    'taux_acces': "Taux d'accès (%)",
                    'count': 'Nombre de formations'
                }
            )
            fig_taux = apply_chart_style(fig_taux)
            fig_taux.update_layout(height=450, margin=dict(t=50, b=50, l=50, r=50))
            fig_taux.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig_taux, width="stretch")

    # Machine Learning section
    st.divider()
    st.header("Analyse par Machine Learning")

    col_ml1, col_ml2 = st.columns(2)

    with col_ml1:
        st.subheader("K-Means Clustering")
        if 'cluster' in df_latest.columns:
            if 'pca_x' in df_latest.columns and 'pca_y' in df_latest.columns:
                cluster_colors = ['#1e40af', '#dc2626', '#059669', '#d97706', '#7c3aed']
                sample_df = df_latest.sample(min(2000, len(df_latest))).copy()
                sample_df['cluster_str'] = sample_df['cluster'].astype(str)

                fig_pca = px.scatter(
                    sample_df,
                    x='pca_x',
                    y='pca_y',
                    color='cluster_str',
                    color_discrete_sequence=cluster_colors,
                    hover_name='g_ea_lib_vx',
                    hover_data=['form_lib_voe_acc', 'taux_acces'],
                    title="Visualisation PCA des clusters",
                    labels={
                        'pca_x': 'Composante 1 (variance maximale)',
                        'pca_y': 'Composante 2',
                        'cluster_str': 'Groupe'
                    }
                )
                fig_pca = apply_chart_style(fig_pca)
                fig_pca.update_traces(marker=dict(size=8, opacity=0.7))
                st.plotly_chart(fig_pca, width="stretch")

    with col_ml2:
        st.subheader("Modèles KNN")
        knn_stats = []
        for formation, knn in analyzer.knn_models.items():
            n_etabs = len(analyzer.knn_indices[formation])
            knn_stats.append({
                'Formation': formation[:50],
                'Nb établissements': n_etabs,
                'K voisins': knn.n_neighbors
            })

        knn_df = pd.DataFrame(knn_stats).sort_values(
            'Nb établissements',
            ascending=False
        ).head(20)

        fig_knn = px.bar(
            knn_df,
            x='Nb établissements',
            y='Formation',
            orientation='h',
            title="Top 20 formations par nombre d'établissements"
        )
        fig_knn = apply_chart_style(fig_knn)
        fig_knn.update_layout(height=500)
        st.plotly_chart(fig_knn, width="stretch")

    # Cluster characteristics table
    if 'cluster' in df_latest.columns:
        st.subheader("Caractéristiques des clusters")
        cluster_stats = df_latest.groupby('cluster').agg({
            'taux_acces': 'mean',
            'tension': 'mean',
            'pct_mention_tb': 'mean',
            'form_lib_voe_acc': 'count'
        }).round(1)
        cluster_stats.columns = [
            "Taux d'accès moyen",
            "Tension moyenne",
            "% Mention TB",
            "Nb formations"
        ]
        st.dataframe(cluster_stats, width="stretch")

    # Detailed statistics section
    st.divider()
    st.header("Statistiques détaillées")

    # Top 10 most selective
    st.subheader("Top 10 - Plus sélectifs")
    df_selectif_filtered = df_latest[df_latest['acc_tot'] > 0].copy()
    top_10_selectif = df_selectif_filtered.nsmallest(10, 'taux_acces')[
        ['g_ea_lib_vx', 'form_lib_voe_acc', 'taux_acces', 'tension']
    ]
    top_10_selectif.columns = ['Établissement', 'Formation', "Taux d'accès", 'Tension']
    st.dataframe(top_10_selectif, width="stretch", hide_index=True)

    # Top 10 highest tension
    st.subheader("Top 10 - Plus de tension")
    df_tension_filtered = df_latest[
        (df_latest['tension'] > 0) & (df_latest['tension'] < 1000)
    ].copy()
    top_10_tension = df_tension_filtered.nlargest(10, 'tension')[
        ['g_ea_lib_vx', 'form_lib_voe_acc', 'taux_acces', 'tension']
    ]
    top_10_tension.columns = ['Établissement', 'Formation', "Taux d'accès", 'Tension']
    st.dataframe(top_10_tension, width="stretch", hide_index=True)

    # Top 10 most accessible
    st.subheader("Top 10 - Plus accessibles")
    df_accessible_filtered = df_latest[
        (df_latest['voe_tot'] >= 50) & (df_latest['taux_acces'] <= 100)
    ].copy()
    top_10_accessible = df_accessible_filtered.nlargest(10, 'taux_acces')[
        ['g_ea_lib_vx', 'form_lib_voe_acc', 'taux_acces', 'tension']
    ]
    top_10_accessible.columns = ['Établissement', 'Formation', "Taux d'accès", 'Tension']
    st.dataframe(top_10_accessible, width="stretch", hide_index=True)

    # Yearly evolution
    st.divider()
    st.subheader("Évolution globale par année")

    df_with_taux = compute_taux_acces(df)
    yearly_stats = df_with_taux.groupby('session').agg({
        'taux_acces': 'mean',
        'tension': 'mean',
        'voe_tot': 'sum',
        'form_lib_voe_acc': 'count'
    }).round(2)
    yearly_stats.columns = [
        "Taux d'accès moyen",
        "Tension moyenne",
        "Total vœux",
        "Nb formations"
    ]

    years_list = [int(y) for y in yearly_stats.index.tolist()]

    col_y1, col_y2 = st.columns(2)

    with col_y1:
        fig_yearly = px.line(
            yearly_stats.reset_index(),
            x='session',
            y=["Taux d'accès moyen", "Tension moyenne"],
            title="Évolution des indicateurs",
            labels={
                'session': 'Année',
                'value': 'Valeur',
                'variable': 'Indicateur'
            }
        )
        fig_yearly = apply_chart_style(fig_yearly)
        fig_yearly.update_xaxes(
            tickmode='array',
            tickvals=years_list,
            ticktext=[str(y) for y in years_list]
        )
        st.plotly_chart(fig_yearly, width="stretch")

    with col_y2:
        fig_voeux = px.bar(
            yearly_stats.reset_index(),
            x='session',
            y='Total vœux',
            title="Évolution du nombre total de vœux",
            labels={'session': 'Année', 'Total vœux': 'Nombre de vœux'}
        )
        fig_voeux = apply_chart_style(fig_voeux)
        fig_voeux.update_xaxes(
            tickmode='array',
            tickvals=years_list,
            ticktext=[str(y) for y in years_list]
        )
        st.plotly_chart(fig_voeux, width="stretch")

    st.dataframe(yearly_stats, width="stretch")
