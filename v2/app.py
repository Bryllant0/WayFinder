#!/usr/bin/env python3
"""
WayFinder - Parcoursup Analytics Application.

This is the main entry point for the WayFinder Streamlit application.
It provides a comprehensive interface for analyzing Parcoursup data,
including formation search, admission probability estimation, and
historical trend analysis.

Application Features:
    - Formation search with real-time API queries
    - Personal profile-based admission probability calculation
    - Formation comparison (up to 4 formations)
    - Historical trend analysis with ML predictions
    - Clustering visualization of similar formations

Tabs:
    1. Rechercher: Search and explore formations
    2. Statistiques: Detailed statistics for selected formation
    3. Mes Chances: Personalized admission probability
    4. Comparatif: Compare multiple formations
    5. Formations similaires: Find similar programs
    6. Tendances: Analyze historical trends
    7. Vue d'ensemble: Global statistics and clustering

Usage:
    streamlit run app.py

Configuration:
    The application uses .streamlit/config.toml for theme settings.
    Data is loaded from data/parcoursup_data.csv.
    ML model is loaded from models/parcoursup_model_v3.pkl.

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Course: Introduction to Python - CentraleSupélec
Version: 2.0
"""

import logging

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

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
from visualization import (
    apply_chart_style,
    compute_taux_acces,
    load_analyzer,
    render_tab_recherche_etablissement,
    render_tab_tendances,
    render_tab_vue_ensemble,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="WayFinder - Parcoursup",
    page_icon="W",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CSS
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
st.markdown('<p class="sub-header">Introduction to Python - CentraleSupélec</p>', unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown('<h3 style="color: #1e3a5f !important; margin-bottom: 1rem;">Mon Profil</h3>', unsafe_allow_html=True)
    
    type_bac = st.selectbox("Type de bac", TYPES_BAC, key="sb_bac")
    specialites = []
    if type_bac == "Général":
        specialites = st.multiselect("Spécialités (2)", list(SPECIALITES.keys()), max_selections=2, key="sb_spe")
    mention = st.selectbox("Mention", list(MENTIONS.keys()), key="sb_mention")
    moyenne = st.slider("Moyenne", 8.0, 20.0, 14.0, 0.5, key="sb_moy")
    academie = st.selectbox("Académie", ACADEMIES, key="sb_acad")
    boursier = st.checkbox("Je suis boursier", key="sb_bours")
    
    st.markdown("---")
    comparatif = st.session_state.get("formations_comparatif", [])
    if comparatif:
        st.markdown(f'<p style="color: #1e3a5f !important;"><strong>Comparatif : {len(comparatif)}/4</strong></p>', unsafe_allow_html=True)
        if st.button("Vider le comparatif", key="sb_vider"):
            st.session_state.formations_comparatif = []
            st.rerun()

# =============================================================================
# STATE
# =============================================================================

if "selected_formation" not in st.session_state: st.session_state.selected_formation = None
if "formations_comparatif" not in st.session_state: st.session_state.formations_comparatif = []

# =============================================================================
# CHARGEMENT ANALYZER
# =============================================================================

analyzer, df_analyzer, df_latest = load_analyzer()
ANALYZER_AVAILABLE = analyzer is not None

# =============================================================================
# ONGLETS
# =============================================================================

if ANALYZER_AVAILABLE:
    tabs = st.tabs(["Rechercher", "Statistiques", "Mes Chances", "Comparatif", "Formations similaires", "Tendances", "Vue d'ensemble"])
else:
    tabs = st.tabs(["Rechercher", "Statistiques", "Mes Chances", "Comparatif"])
    st.sidebar.warning("Module Analyzer non disponible. Vérifiez que parcoursup_analyzer_v3.py et les données sont présents.")

# =============================================================================
# TAB 1: RECHERCHE
# =============================================================================

with tabs[0]:
    col1, col2 = st.columns([5, 1])
    with col1:
        search_term = st.text_input("Formation ou établissement", placeholder="Ex: ESSEC, CPGE MPSI, Licence Droit...", key="search_input")
    with col2:
        year = st.selectbox("Année", [2024, 2023, 2022], key="search_year")
    
    if search_term and len(search_term) >= 3:
        results = search_formations(search_term, year)
        if results:
            total_results = len(results)
            st.success(f"{total_results} résultat(s) trouvé(s)")
            
            # Afficher jusqu'à 50 résultats
            nb_display = min(total_results, 50)
            
            for idx, f in enumerate(results[:nb_display]):
                # Gérer les valeurs None
                lib_for = f.get('lib_for_voe_ins') or 'Formation inconnue'
                etab = f.get('g_ea_lib_vx') or 'Établissement inconnu'
                voeux = f.get('voe_tot') or 0
                taux = f.get('taux_acces') or f.get('taux_acces_ens') or 0
                
                st.markdown(f"""
                <div class="formation-card">
                    <div class="formation-title">{lib_for}</div>
                    <div class="formation-subtitle">{etab} - {voeux:,} vœux - Taux : {taux:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns([1, 1])
                with c1:
                    if st.button("Charger les statistiques", key=f"voir_{idx}", width='stretch'):
                        st.session_state.selected_formation = extract_formation_stats(f)
                        st.rerun()
                with c2:
                    if st.button("Ajouter au comparatif", key=f"add_{idx}", width='stretch'):
                        if len(comparatif) < 4:
                            comparatif.append(extract_formation_stats(f))
                            st.session_state.formations_comparatif = comparatif
                            st.rerun()
            
            if total_results > nb_display:
                st.info(f"{total_results - nb_display} résultats supplémentaires non affichés. Utilisez le slider ci-dessus pour en voir plus.")
        else:
            st.warning("Aucun résultat")

# =============================================================================
# TAB 2: STATISTIQUES
# =============================================================================

with tabs[1]:
    if st.session_state.selected_formation:
        s = st.session_state.selected_formation
        
        st.markdown(f'<h2 style="color: #1e3a5f !important; margin-bottom: 0.5rem;">{s["nom"]}</h2>', unsafe_allow_html=True)
        st.markdown(f'<p style="color: #64748b !important; font-size: 1rem;">{s["etablissement"]} - {s["academie"]}</p>', unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        display_stat_box(f"{s['capacite']:,}", "Capacité", c1)
        display_stat_box(f"{s['voeux_total']:,}", "Vœux", c2)
        display_stat_box(f"{s['admis_total']:,}", "Admis", c3)
        display_stat_box(f"{s['taux_acces']:.0f}%", "Taux d'accès", c4)
        pct_b = s['pct_boursiers']
        display_stat_box(f"{pct_b:.0f}%" if pct_b > 0 else "<5%", "Boursiers", c5)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<p class="section-header">Répartition par Bac</p>', unsafe_allow_html=True)
            fig = px.pie(values=[s['admis_bg'], s['admis_bt'], s['admis_bp']], names=['Général', 'Techno', 'Pro'], color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
            fig.update_traces(textinfo='percent', textposition='auto', textfont=dict(color='#1e293b', size=13))
            fig = apply_chart_style(fig)
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), margin=dict(t=20, b=100, l=20, r=20), height=380)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.markdown('<p class="section-header">Répartition par Mention</p>', unsafe_allow_html=True)
            fig = px.pie(values=[s['admis_sans_mention'], s['admis_ab'], s['admis_b'], s['admis_tb'], s['admis_tbf']], names=['Sans mention', 'Assez Bien', 'Bien', 'Très Bien', 'TB Félicitations'], color_discrete_sequence=['#ef4444', '#f59e0b', '#eab308', '#22c55e', '#10b981'])
            fig.update_traces(textinfo='percent', textposition='auto', textfont=dict(color='#1e293b', size=13))
            fig = apply_chart_style(fig)
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), margin=dict(t=20, b=100, l=20, r=20), height=380)
            st.plotly_chart(fig, width='stretch')
        
        with col3:
            st.markdown('<p class="section-header">Origine géographique</p>', unsafe_allow_html=True)
            autres = max(0, s['admis_total'] - s['admis_meme_academie'])
            fig = px.pie(values=[s['admis_meme_academie'], autres], names=['Même académie', 'Autre académie'], color_discrete_sequence=['#3b82f6', '#94a3b8'])
            fig.update_traces(textinfo='percent', textposition='auto', textfont=dict(color='#1e293b', size=13))
            fig = apply_chart_style(fig)
            fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5), margin=dict(t=20, b=100, l=20, r=20), height=380)
            st.plotly_chart(fig, width='stretch')
        
        st.markdown("---")
        st.markdown('<p class="section-header">Doublettes les plus adaptées pour cette formation</p>', unsafe_allow_html=True)
        formation_type = get_formation_type(s)
        doublettes = DOUBLETTES_PAR_TYPE.get(formation_type, DOUBLETTES_PAR_TYPE["general"])
        
        # 10 doublettes avec dégradé vert foncé -> vert clair -> orange clair
        colors_gradient = ['#059669', '#10b981', '#34d399', '#6ee7b7', '#a7f3d0', '#d1fae5', '#fef3c7', '#fde68a', '#fcd34d', '#fbbf24']
        
        df_doub = pd.DataFrame(doublettes[:10], columns=["Doublette", "Popularité"])
        df_doub = df_doub.iloc[::-1]  # Inverser pour que la meilleure soit en haut
        
        fig = go.Figure()
        for i, row in enumerate(df_doub.itertuples()):
            color_idx = 9 - i  # Inverser l'index pour les couleurs
            fig.add_trace(go.Bar(
                y=[row.Doublette],
                x=[row.Popularité],
                orientation='h',
                marker_color=colors_gradient[color_idx],
                text=f"{row.Popularité}%",
                textposition='inside',
                textfont=dict(color='white' if color_idx < 6 else '#1e293b', size=14),
                showlegend=False
            ))
        
        fig = apply_chart_style(fig)
        fig.update_layout(
            barmode='stack',
            yaxis=dict(title=''),
            xaxis=dict(title=dict(text='Score de popularité')),
            height=350,
            margin=dict(t=20, b=50, l=200, r=20)
        )
        st.plotly_chart(fig, width='stretch')
        
        # =================================================================
        # SECTION STATS AVANCÉES (intégrée)
        # =================================================================
        
        st.markdown("---")
        st.markdown('<p class="section-header">Évolution historique et prédictions</p>', unsafe_allow_html=True)
        
        if ANALYZER_AVAILABLE:
            # Rechercher toutes les données historiques
            nom_formation = s.get("nom", "")
            etablissement = s.get("etablissement", "")
            cod_aff = s.get("cod_aff", "")  # Code unique de la formation
            
            df_filtered = df_analyzer.copy()
            
            # ================================================================
            # MÉTHODE 1: Recherche par code formation (le plus fiable)
            # ================================================================
            if cod_aff and 'cod_aff_form' in df_filtered.columns:
                df_by_code = df_filtered[df_filtered['cod_aff_form'] == cod_aff]
                if len(df_by_code) > 0:
                    df_filtered = df_by_code
            
            # ================================================================
            # MÉTHODE 2: Recherche EXACTE par nom (si pas de résultat par code)
            # ================================================================
            if len(df_filtered) == len(df_analyzer):
                # D'abord essayer un match EXACT sur établissement
                if 'g_ea_lib_vx' in df_filtered.columns and etablissement:
                    df_exact_etab = df_filtered[df_filtered['g_ea_lib_vx'] == etablissement]
                    if len(df_exact_etab) > 0:
                        df_filtered = df_exact_etab
                
                # Ensuite essayer un match EXACT sur formation
                if 'lib_for_voe_ins' in df_filtered.columns and nom_formation and len(df_filtered) < len(df_analyzer):
                    df_exact_form = df_filtered[df_filtered['lib_for_voe_ins'] == nom_formation]
                    if len(df_exact_form) > 0:
                        df_filtered = df_exact_form
            
            # ================================================================
            # MÉTHODE 3: Recherche SOUPLE (fallback si match exact échoue)
            # ================================================================
            if len(df_filtered) == len(df_analyzer):
                # Nettoyer les noms pour une recherche plus souple
                etab_words = [w for w in etablissement.lower().split() if len(w) > 2][:5]
                form_words = [w for w in nom_formation.lower().split() if len(w) > 2][:3]
                
                # Exclure les mots trop communs pour l'établissement
                common_words = {'lycée', 'lycee', 'college', 'collège', 'université', 'universite', 
                               'ecole', 'école', 'institut', 'iut', 'campus', 'site'}
                etab_words_filtered = [w for w in etab_words if w not in common_words]
                
                # Si tous les mots sont communs, garder les originaux
                if not etab_words_filtered:
                    etab_words_filtered = etab_words
                
                # Trouver la colonne établissement
                etab_col = None
                for col in ['g_ea_lib_vx', 'g_ea_lib', 'etablissement', 'etab']:
                    if col in df_filtered.columns:
                        etab_col = col
                        break
                
                # Filtre établissement
                if etab_words_filtered and etab_col:
                    etab_text = df_filtered[etab_col].str.lower().fillna('')
                    match_count = sum(etab_text.str.contains(word, regex=False) for word in etab_words_filtered)
                    df_filtered = df_filtered[match_count >= len(etab_words_filtered)]
                
                # Filtre formation - chercher dans lib_for_voe_ins en priorité
                if form_words and len(df_filtered) > 0 and 'lib_for_voe_ins' in df_filtered.columns:
                    form_text = df_filtered['lib_for_voe_ins'].str.lower().fillna('')
                    match_count = sum(form_text.str.contains(word, regex=False) for word in form_words)
                    df_filtered_form = df_filtered[match_count >= len(form_words)]  # TOUS les mots doivent matcher
                    
                    if len(df_filtered_form) > 0:
                        df_filtered = df_filtered_form
            
            if len(df_filtered) == 0:
                st.warning("Aucune donnée historique trouvée pour cette formation.")
            elif len(df_filtered) > 0:
                # Calculer le taux d'accès pour toutes les années (y compris 2018-2019)
                df_filtered = compute_taux_acces(df_filtered)
                df_filtered['taux_acces_combined'] = df_filtered['taux_acces']
                
                # Agréger par année
                agg_dict = {'taux_acces_combined': 'mean', 'voe_tot': 'sum'}
                if 'acc_tot' in df_filtered.columns:
                    agg_dict['acc_tot'] = 'sum'
                
                trends = df_filtered.groupby('session').agg(agg_dict).round(2)
                
                if 'acc_tot' in trends.columns:
                    trends.columns = ["Taux d'accès", "Vœux", "Admis"]
                else:
                    trends.columns = ["Taux d'accès", "Vœux"]
                
                trends = trends.dropna(subset=["Taux d'accès"])
                
                # Afficher les années trouvées
                st.caption(f"Données disponibles : {len(trends)} années ({int(trends.index.min())} - {int(trends.index.max())})")
                
                if len(trends) >= 2:
                    from sklearn.linear_model import LinearRegression
                    from sklearn.metrics import r2_score
                    
                    X = trends.index.values.reshape(-1, 1).astype(float)
                    y_taux = trends["Taux d'accès"].values
                    y_voeux = trends["Vœux"].values
                    
                    # Régression taux d'accès
                    model_taux = LinearRegression()
                    model_taux.fit(X, y_taux)
                    pente_taux = model_taux.coef_[0]
                    y_pred_taux = model_taux.predict(X)
                    r2_taux = r2_score(y_taux, y_pred_taux) if len(y_taux) > 2 else 0
                    projection_taux_2025 = model_taux.predict([[2025]])[0]
                    
                    # Régression vœux
                    model_voeux = LinearRegression()
                    model_voeux.fit(X, y_voeux)
                    pente_voeux = model_voeux.coef_[0]
                    y_pred_voeux = model_voeux.predict(X)
                    projection_voeux_2025 = model_voeux.predict([[2025]])[0]
                    
                    # Interprétation
                    if pente_taux < -1:
                        st.error("FORTE BAISSE du taux d'accès - Formation de plus en plus sélective")
                    elif pente_taux < -0.3:
                        st.warning("BAISSE du taux d'accès - Devient plus sélective")
                    elif pente_taux > 1:
                        st.success("FORTE HAUSSE du taux d'accès - Devient plus accessible")
                    elif pente_taux > 0.3:
                        st.info("HAUSSE du taux d'accès - Devient plus accessible")
                    else:
                        st.info("STABLE - Pas de changement significatif")
                    
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Pente taux", f"{pente_taux:+.2f}%/an")
                    col_m2.metric("Projection 2025", f"{projection_taux_2025:.1f}%")
                    col_m3.metric("Pente vœux", f"{pente_voeux:+.0f}/an")
                    col_m4.metric("Vœux prévu 2025", f"{projection_voeux_2025:,.0f}")
                    
                    # Graphiques
                    years_list = [int(y) for y in trends.index.tolist()]
                    years_with_2025 = years_list + [2025]
                    y_pred_taux_with_2025 = list(y_pred_taux) + [projection_taux_2025]
                    y_pred_voeux_with_2025 = list(y_pred_voeux) + [projection_voeux_2025]
                    
                    fig = make_subplots(rows=1, cols=2, subplot_titles=("Évolution du taux d'accès", "Évolution des vœux"))
                    
                    # Taux d'accès
                    fig.add_trace(go.Scatter(x=years_list, y=trends["Taux d'accès"], mode='lines+markers', name="Taux réel", line=dict(color='#3b82f6', width=3), marker=dict(size=10)), row=1, col=1)
                    fig.add_trace(go.Scatter(x=years_with_2025, y=y_pred_taux_with_2025, mode='lines', name="Tendance", line=dict(color='#ef4444', dash='dash')), row=1, col=1)
                    fig.add_trace(go.Scatter(x=[2025], y=[projection_taux_2025], mode='markers', name="Prédiction 2025", marker=dict(size=14, color='#ef4444', symbol='star')), row=1, col=1)
                    
                    # Vœux
                    fig.add_trace(go.Scatter(x=years_list, y=trends["Vœux"], mode='lines+markers', name="Vœux réels", line=dict(color='#10b981', width=3), marker=dict(size=10)), row=1, col=2)
                    fig.add_trace(go.Scatter(x=years_with_2025, y=y_pred_voeux_with_2025, mode='lines', name="Tendance vœux", line=dict(color='#f59e0b', dash='dash')), row=1, col=2)
                    fig.add_trace(go.Scatter(x=[2025], y=[projection_voeux_2025], mode='markers', name="Prédiction vœux 2025", marker=dict(size=14, color='#f59e0b', symbol='star')), row=1, col=2)
                    
                    fig = apply_chart_style(fig)
                    fig.update_layout(height=450, showlegend=True)
                    fig.update_xaxes(tickmode='array', tickvals=years_with_2025, ticktext=[str(y) for y in years_with_2025])
                    st.plotly_chart(fig, width='stretch')
                    
                    st.subheader("Données historiques")
                    st.dataframe(trends, width='stretch')
                else:
                    st.warning("Pas assez de données pour l'analyse de tendance (minimum 2 années requises).")
        else:
            st.info("Module Analyzer non disponible pour les prédictions.")
    else:
        st.info("Sélectionnez une formation dans l'onglet Rechercher")

# =============================================================================
# TAB 3: MES CHANCES
# =============================================================================

with tabs[2]:
    if st.session_state.selected_formation:
        s = st.session_state.selected_formation
        st.markdown(f'<h2 style="color: #1e3a5f !important;">{s["nom"]}</h2>', unsafe_allow_html=True)
        result = calculate_admission_probability(s, type_bac, mention, specialites, boursier, moyenne, academie)
        display_probability_result(result)
    else:
        st.info("Sélectionnez une formation")

# =============================================================================
# TAB 4: COMPARATIF
# =============================================================================

with tabs[3]:
    if not comparatif:
        st.info("Ajoutez des formations depuis l'onglet Rechercher (max 4)")
    else:
        st.markdown(f'<h2 style="color: #1e3a5f !important;">Comparatif ({len(comparatif)} formations)</h2>', unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Top 3 doublettes pour ces formations</p>', unsafe_allow_html=True)
        top_doub = get_top_doublettes_for_formations(comparatif)
        for i, (doub, score) in enumerate(top_doub, 1):
            st.markdown(f"{i}. **{doub}**")
        
        st.markdown("---")
        cols = st.columns(len(comparatif))
        for idx, (col, f) in enumerate(zip(cols, comparatif)):
            with col:
                st.markdown(f'<p style="font-weight: 600; color: #1e3a5f !important; font-size: 0.95rem;">{f["nom"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color: #64748b !important; font-size: 0.85rem;">{f["etablissement"]}</p>', unsafe_allow_html=True)
                st.metric("Taux d'accès", f"{f['taux_acces']:.0f}%")
                st.metric("Vœux", f"{f['voeux_total']:,}")
                prob = calculate_admission_probability(f, type_bac, mention, specialites, boursier, moyenne, academie)
                st.metric("Mes chances", f"{prob['probability']}%")
                if st.button("Retirer", key=f"rm_{idx}"):
                    comparatif.pop(idx)
                    st.session_state.formations_comparatif = comparatif
                    st.rerun()

# =============================================================================
# ONGLETS ANALYZER
# =============================================================================

if ANALYZER_AVAILABLE:
    with tabs[4]: render_tab_recherche_etablissement(analyzer, df_analyzer, df_latest)
    with tabs[5]: render_tab_tendances(analyzer, df_analyzer, df_latest)
    with tabs[6]: render_tab_vue_ensemble(analyzer, df_analyzer, df_latest)

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.markdown('<div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 10px;">Les données proviennent de sources publiques (data.gouv.fr) et sont fournies à titre indicatif. Les projections statistiques ne constituent pas une garantie d\'admission. Vérifiez toujours les informations officielles sur Parcoursup.</div>', unsafe_allow_html=True)
