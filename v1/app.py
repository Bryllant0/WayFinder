#!/usr/bin/env python3
"""
WayFinder - Parcoursup Analytics Application (V1).

This is the main entry point for the WayFinder Streamlit application.
It provides an interface for analyzing Parcoursup data, including
formation search, admission probability estimation, and comparison tools.

Application Features:
    - Formation search with real-time API queries
    - Personal profile-based admission probability calculation
    - Formation comparison (up to 4 formations)
    - Detailed statistics for selected formations

Tabs:
    1. Rechercher: Search and explore formations
    2. Statistiques: Detailed statistics for selected formation
    3. Mes Chances: Personalized admission probability
    4. Comparatif: Compare multiple formations

Usage:
    streamlit run app.py

Configuration:
    The application uses .streamlit/config.toml for theme settings.

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Course: Introduction to Python - CentraleSupélec
Version: 1.0
"""

import logging

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from search import (
    ACADEMIES,
    DOUBLETTES_PAR_TYPE,
    MENTIONS,
    SPECIALITES,
    TYPES_BAC,
    calculate_admission_probability,
    display_probability_result,
    display_stat_box,
    extract_formation_stats,
    get_formation_type,
    get_top_doublettes_for_formations,
    search_formations,
)
from visualization import apply_chart_style

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="WayFinder - Parcoursup",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS STYLES
# =============================================================================

st.markdown("""
<style>
    .stApp { background-color: #f5f7fa !important; }
    .main .block-container { background-color: #ffffff; padding: 2rem; border-radius: 12px; }
    
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%) !important; }
    [data-testid="stSidebar"] * { color: #1e3a5f !important; }
    
    .main .stSelectbox > div > div, .main [data-baseweb="select"] > div, [data-baseweb="select"], [data-baseweb="popover"] {
        background-color: #ffffff !important; color: #1e293b !important; border: 1px solid #e2e8f0 !important;
    }
    [data-baseweb="menu"] { background-color: #ffffff !important; }
    [data-baseweb="menu"] li { background-color: #ffffff !important; color: #1e293b !important; }
    [data-baseweb="menu"] li:hover { background-color: #f1f5f9 !important; }
    
    .stTextInput input { background-color: #ffffff !important; color: #1e293b !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; }
    
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1e3a5f !important; }
    .sub-header { color: #64748b !important; font-size: 1.1rem; margin-bottom: 2rem; }
    
    .stTabs [data-baseweb="tab-list"] { width: 100%; display: flex; gap: 0; background-color: #f1f5f9; border-radius: 12px; padding: 4px; flex-wrap: wrap; }
    .stTabs [data-baseweb="tab"] { flex: 1; justify-content: center; background-color: transparent !important; color: #64748b !important; border-radius: 10px; padding: 10px 6px; font-weight: 500; font-size: 0.85rem; min-width: fit-content; }
    .stTabs [aria-selected="true"] { background-color: #ffffff !important; color: #1e3a5f !important; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    
    .stat-box { background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%); padding: 24px 16px; border-radius: 12px; text-align: center; border: 1px solid #e2e8f0; }
    .stat-value { font-size: 2rem; font-weight: 700; color: #1e3a5f !important; }
    .stat-label { font-size: 0.85rem; color: #64748b !important; margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
    
    .stButton > button { background-color: #ffffff !important; color: #1e3a5f !important; border: 1px solid #e2e8f0 !important; border-radius: 8px !important; font-weight: 500 !important; }
    .stButton > button:hover { background-color: #f1f5f9 !important; border-color: #3b82f6 !important; color: #3b82f6 !important; }
    
    .formation-card { background: #ffffff; padding: 16px 20px; border-radius: 10px; border: 1px solid #e2e8f0; margin: 12px 0; transition: all 0.2s; }
    .formation-card:hover { background: #f8fafc; border-color: #3b82f6; }
    .formation-title { font-size: 1rem; font-weight: 600; color: #1e293b !important; margin-bottom: 4px; }
    .formation-subtitle { font-size: 0.85rem; color: #64748b !important; }
    
    .prob-display { font-size: 4rem; font-weight: 700; text-align: center; padding: 40px; border-radius: 16px; margin: 20px 0; color: #ffffff !important; }
    .prob-display * { color: #ffffff !important; }
    .prob-high { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .prob-medium { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .prob-low { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    .prob-zero { background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); }
    
    .section-header { font-size: 1.1rem; font-weight: 600; color: #1e3a5f !important; margin: 1.5rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 2px solid #e2e8f0; }
    
    .main h1, .main h2, .main h3 { color: #1e3a5f !important; }
    .main p, .main span, .main label { color: #334155 !important; }
    
    .conseil-box { background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); padding: 16px 20px; border-radius: 10px; border-left: 4px solid #3b82f6; margin: 10px 0; }
    .conseil-box * { color: #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HEADER
# =============================================================================

st.markdown('<h1 class="main-header">WayFinder</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Introduction to Python - CentraleSupélec</p>',
    unsafe_allow_html=True
)

# =============================================================================
# SIDEBAR - USER PROFILE
# =============================================================================

with st.sidebar:
    st.markdown("## Mon Profil")

    type_bac = st.selectbox("Type de Bac", TYPES_BAC, key="type_bac")
    mention = st.selectbox("Mention visée", list(MENTIONS.keys()), index=2, key="mention")

    # Specialty selection (for Bac Général only)
    if type_bac == "Général":
        specialites = st.multiselect(
            "Spécialités (2 en terminale)",
            list(SPECIALITES.keys()),
            max_selections=2,
            key="specialites"
        )
    else:
        specialites = []

    boursier = st.checkbox("Boursier", key="boursier")
    moyenne = st.slider("Moyenne générale", 8.0, 20.0, 14.0, 0.5, key="moyenne")
    academie = st.selectbox("Académie", ACADEMIES, key="academie")

    st.markdown("---")
    st.markdown("## Ma sélection")

    # Comparison list display
    comparatif = st.session_state.get("formations_comparatif", [])
    if comparatif:
        st.markdown(f"**{len(comparatif)}/4** formations")
        for f in comparatif:
            st.caption(f"• {f['nom'][:40]}...")
    else:
        st.caption("Aucune formation")

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

if "selected_formation" not in st.session_state:
    st.session_state.selected_formation = None
if "formations_comparatif" not in st.session_state:
    st.session_state.formations_comparatif = []

comparatif = st.session_state.formations_comparatif

# =============================================================================
# MAIN TABS
# =============================================================================

tabs = st.tabs(["Rechercher", "Statistiques", "Mes Chances", "Comparatif"])

# =============================================================================
# TAB 1: SEARCH
# =============================================================================

with tabs[0]:
    col1, col2 = st.columns([5, 1])
    with col1:
        search_term = st.text_input(
            "Formation ou établissement",
            placeholder="Ex: ESSEC, CPGE MPSI, Licence Droit...",
            key="search_input"
        )
    with col2:
        year = st.selectbox("Année", [2024, 2023, 2022], key="search_year")

    if search_term and len(search_term) >= 3:
        results = search_formations(search_term, year)
        if results:
            total_results = len(results)
            st.success(f"{total_results} résultat(s) trouvé(s)")

            # Display up to 50 results
            nb_display = min(total_results, 50)

            for idx, f in enumerate(results[:nb_display]):
                # Handle None values
                lib_for = f.get('lib_for_voe_ins') or 'Formation inconnue'
                etab = f.get('g_ea_lib_vx') or 'Établissement inconnu'
                voeux = f.get('voe_tot') or 0
                taux = f.get('taux_acces') or f.get('taux_acces_ens') or 0

                st.markdown(
                    f"""
                    <div class="formation-card">
                        <div class="formation-title">{lib_for}</div>
                        <div class="formation-subtitle">
                            {etab} - {voeux:,} vœux - Taux : {taux:.0f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button(
                        "Charger les statistiques",
                        key=f"voir_{idx}",
                        width="stretch"
                    ):
                        st.session_state.selected_formation = extract_formation_stats(f)
                        logger.info("Formation selected: %s", lib_for)
                        st.rerun()
                with c2:
                    if st.button(
                        "Ajouter au comparatif",
                        key=f"add_{idx}",
                        width="stretch"
                    ):
                        if len(comparatif) < 4:
                            comparatif.append(extract_formation_stats(f))
                            st.session_state.formations_comparatif = comparatif
                            logger.info("Formation added to comparison: %s", lib_for)
                            st.rerun()

            if total_results > nb_display:
                st.info(
                    f"{total_results - nb_display} resultats supplementaires "
                    "non affichés."
                )
        else:
            st.warning("Aucun résultat")

# =============================================================================
# TAB 2: STATISTICS
# =============================================================================

with tabs[1]:
    if st.session_state.selected_formation:
        s = st.session_state.selected_formation

        st.markdown(
            f'<h2 style="color: #1e3a5f !important;">{s["nom"]}</h2>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="color: #64748b !important;">{s["etablissement"]} • '
            f'{s["academie"]}</p>',
            unsafe_allow_html=True
        )

        # Main statistics row
        col1, col2, col3, col4 = st.columns(4)
        display_stat_box(f'{s["taux_acces"]:.0f}%', "Taux d'accès", col1)
        display_stat_box(f'{s["voeux_total"]:,}', "Vœux totaux", col2)
        display_stat_box(f'{s["admis_total"]:,}', "Admis", col3)
        display_stat_box(f'{s["capacite"]:,}', "Capacité", col4)

        st.markdown("---")

        # Bac type distribution chart
        st.markdown(
            '<p class="section-header">Répartition par type de bac</p>',
            unsafe_allow_html=True
        )

        bac_data = {
            "Type": ["Général", "Technologique", "Professionnel"],
            "Pourcentage": [
                s["pct_admis_bg"],
                s["pct_admis_bt"],
                s["pct_admis_bp"]
            ]
        }
        fig = px.bar(
            bac_data,
            x="Type",
            y="Pourcentage",
            color="Type",
            color_discrete_sequence=["#3b82f6", "#f59e0b", "#10b981"]
        )
        fig = apply_chart_style(fig)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width='stretch')

        # Mention distribution chart
        st.markdown(
            '<p class="section-header">Répartition par mention</p>',
            unsafe_allow_html=True
        )

        mention_data = {
            "Mention": ["Sans", "AB", "B", "TB", "TB+"],
            "Pourcentage": [
                s["pct_sans_mention"],
                s["pct_ab"],
                s["pct_b"],
                s["pct_tb"],
                s["pct_tbf"]
            ]
        }
        fig = px.bar(
            mention_data,
            x="Mention",
            y="Pourcentage",
            color="Mention",
            color_discrete_sequence=[
                "#ef4444", "#f59e0b", "#eab308", "#22c55e", "#10b981"
            ]
        )
        fig = apply_chart_style(fig)
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, width='stretch')

        # Additional statistics
        st.markdown(
            '<p class="section-header">Autres indicateurs</p>',
            unsafe_allow_html=True
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("% Boursiers", f'{s["pct_boursiers"]:.1f}%')
            st.metric("% Même académie", f'{s["pct_meme_academie"]:.1f}%')
        with col_b:
            formation_type = get_formation_type(s)
            st.metric("Type détecté", formation_type)

            # Show recommended specialties
            doublettes = DOUBLETTES_PAR_TYPE.get(
                formation_type,
                DOUBLETTES_PAR_TYPE["general"]
            )
            top3 = [d[0] for d in doublettes[:3]]
            st.markdown("**Doublettes recommandées:**")
            for d in top3:
                st.caption(f"• {d}")
    else:
        st.info("Sélectionnez une formation dans l'onglet Rechercher")

# =============================================================================
# TAB 3: MY CHANCES
# =============================================================================

with tabs[2]:
    if st.session_state.selected_formation:
        s = st.session_state.selected_formation

        st.markdown(
            f'<h2 style="color: #1e3a5f !important;">{s["nom"]}</h2>',
            unsafe_allow_html=True
        )

        # Calculate and display admission probability
        result = calculate_admission_probability(
            s, type_bac, mention, specialites, boursier, moyenne, academie
        )
        display_probability_result(result)
    else:
        st.info("Sélectionnez une formation dans l'onglet Rechercher")

# =============================================================================
# TAB 4: COMPARISON
# =============================================================================

with tabs[3]:
    if not comparatif:
        st.info("Ajoutez des formations depuis l'onglet Rechercher (max 4)")
    else:
        st.markdown(
            f'<h2 style="color: #1e3a5f !important;">'
            f'Comparatif ({len(comparatif)} formations)</h2>',
            unsafe_allow_html=True
        )

        # Show top doublettes for all selected formations
        st.markdown(
            '<p class="section-header">Top 3 doublettes pour ces formations</p>',
            unsafe_allow_html=True
        )

        top_doub = get_top_doublettes_for_formations(comparatif)
        for i, (doub, score) in enumerate(top_doub, 1):
            st.markdown(f"{i}. **{doub}**")

        st.markdown("---")

        # Display formations in columns
        cols = st.columns(len(comparatif))
        for idx, (col, f) in enumerate(zip(cols, comparatif)):
            with col:
                st.markdown(
                    f'<p style="font-weight: 600; color: #1e3a5f !important; '
                    f'font-size: 0.95rem;">{f["nom"]}</p>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<p style="color: #64748b !important; font-size: 0.85rem;">'
                    f'{f["etablissement"]}</p>',
                    unsafe_allow_html=True
                )
                st.metric("Taux d'accès", f"{f['taux_acces']:.0f}%")
                st.metric("Vœux", f"{f['voeux_total']:,}")

                # Calculate probability for this formation
                prob = calculate_admission_probability(
                    f, type_bac, mention, specialites, boursier, moyenne, academie
                )
                st.metric("Mes chances", f"{prob['probability']}%")

                # Remove button
                if st.button("Retirer", key=f"rm_{idx}"):
                    comparatif.pop(idx)
                    st.session_state.formations_comparatif = comparatif
                    logger.info("Formation removed from comparison")
                    st.rerun()

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown(
    '<div style="text-align: center; color: #6B7280; font-size: 0.85rem; '
    'padding: 10px;">Les données proviennent de sources publiques '
    '(data.gouv.fr) et sont fournies à titre indicatif. '
    'Vérifiez toujours les informations officielles sur Parcoursup.</div>',
    unsafe_allow_html=True
)
