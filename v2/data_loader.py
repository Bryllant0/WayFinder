#!/usr/bin/env python3
"""
Data Loader Module - Parcoursup Data Download and ML Analysis.

This module provides functionality for downloading, processing, and analyzing
Parcoursup data from the French Ministry of Higher Education's open data API.
It includes machine learning capabilities for clustering formations and finding
similar programs.

Main Components:
    ParcoursupDownloader: Downloads data from the OpenDataSoft API.
    ParcoursupAnalyzer: Processes data, trains ML models, and provides analysis.

Features:
    - Multi-year data download (2018-2024)
    - Automatic column harmonization across different dataset versions
    - K-Means clustering for formation grouping
    - Nearest neighbors for finding similar formations
    - PCA for dimensionality reduction and visualization

Example:
    >>> from data_loader import ParcoursupAnalyzer
    >>> analyzer = ParcoursupAnalyzer(verbose=True)
    >>> analyzer.download_data(years=[2024, 2023])
    >>> analyzer.prepare_features()
    >>> analyzer.train_clustering()
    >>> similar = analyzer.find_similar_formations("Licence Informatique", n=5)

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 2.0
"""

import logging
import os
import pickle
import time
import warnings
from io import StringIO
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

CLUSTERING_FEATURES: List[str] = [
    'taux_acces',
    'tension',
    'pct_mention_tb',
]
"""Features used for K-Means clustering."""

COLUMN_NAMES_FR: Dict[str, str] = {
    'session': 'Année',
    'cod_uai': 'Code établissement',
    'g_ea_lib_vx': 'Établissement',
    'region_etab_aff': 'Région',
    'acad_mies': 'Académie',
    'dep': 'Département',
    'fili': 'Type de formation',
    'form_lib_voe_acc': 'Formation',
    'fil_lib_voe_acc': 'Filière détaillée',
    'selectivite': 'Sélectivité',
    'capa_fin': 'Capacité (places)',
    'voe_tot': 'Nombre de vœux',
    'tension': 'Tension (vœux/places)',
    'acc_tot': 'Admis total',
    'taux_acces': "Taux d'accès (%)",
    'pct_mention_tb': '% Mention TB',
    'cluster': 'Groupe',
}
"""French translations for column names."""


# =============================================================================
# DOWNLOADER CLASS
# =============================================================================

class ParcoursupDownloader:
    """
    Downloads Parcoursup data from the OpenDataSoft API.

    This class handles API communication, including automatic fallback
    to alternative endpoints and batch downloading via records API
    if CSV export fails.

    Attributes:
        verbose: Whether to print progress messages.
        working_base_url: The currently working API base URL.

    Example:
        >>> downloader = ParcoursupDownloader(verbose=True)
        >>> df = downloader.download_year(2024)
        >>> print(len(df))
    """

    DATASETS: Dict[int, str] = {
        2024: "fr-esr-parcoursup",
        2023: "fr-esr-parcoursup_2023",
        2022: "fr-esr-parcoursup_2022",
        2021: "fr-esr-parcoursup_2021",
        2020: "fr-esr-parcoursup_2020",
        2019: "fr-esr-parcoursup-2019",
        2018: "fr-esr-parcoursup-2018",
    }
    """Mapping of years to dataset identifiers."""

    BASE_URLS: List[str] = [
        "https://data.enseignementsup-recherche.gouv.fr",
        "https://data.education.gouv.fr",
    ]
    """List of API base URLs to try."""

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the downloader.

        Args:
            verbose: If True, print progress messages during download.
        """
        self.verbose = verbose
        self.working_base_url: Optional[str] = None

    def log(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.

        Args:
            message: Message to log.
        """
        if self.verbose:
            print(message)
        logger.info(message)

    def test_api_access(self) -> bool:
        """
        Test access to the API and find a working base URL.

        Returns:
            bool: True if an accessible API was found, False otherwise.
        """
        for base_url in self.BASE_URLS:
            try:
                response = requests.get(
                    f"{base_url}/api/explore/v2.1/catalog/datasets?limit=1",
                    timeout=10
                )
                if response.status_code == 200:
                    self.working_base_url = base_url
                    self.log(f"API accessible: {base_url}")
                    return True
            except requests.RequestException as error:
                self.log(f"{base_url}: {type(error).__name__}")
                continue
        return False

    def download_year(
        self,
        year: int,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a specific year.

        Args:
            year: The year to download (2018-2024).
            limit: Optional maximum number of records to download.

        Returns:
            Optional[pd.DataFrame]: Downloaded data or None if failed.

        Raises:
            ValueError: If the year is not available.
        """
        if year not in self.DATASETS:
            self.log(f"Year {year} not available")
            return None

        dataset_id = self.DATASETS[year]

        if not self.working_base_url and not self.test_api_access():
            return None

        self.log(f"Downloading {year} ({dataset_id})...")

        # Try CSV export first (faster)
        url = (
            f"{self.working_base_url}/api/explore/v2.1/"
            f"catalog/datasets/{dataset_id}/exports/csv"
        )
        params = {"delimiter": ";", "use_labels": "false"}
        if limit:
            params["limit"] = limit

        try:
            response = requests.get(url, params=params, timeout=180)

            if response.status_code == 200 and len(response.text) > 100:
                df = pd.read_csv(StringIO(response.text), sep=";", low_memory=False)

                if 'session' not in df.columns:
                    df['session'] = year

                self.log(f"   {len(df)} formations downloaded")
                return df
            else:
                self.log(f"   Invalid response (status={response.status_code})")
                return self._download_via_records(dataset_id, year, limit)

        except requests.RequestException as error:
            self.log(f"   CSV export error: {error}")
            return self._download_via_records(dataset_id, year, limit)

    def _download_via_records(
        self,
        dataset_id: str,
        year: int,
        limit: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fallback download method using the records API.

        Args:
            dataset_id: The dataset identifier.
            year: The year being downloaded.
            limit: Optional maximum number of records.

        Returns:
            Optional[pd.DataFrame]: Downloaded data or None if failed.
        """
        self.log("   Trying records API...")

        all_records: List[Dict] = []
        offset = 0
        batch_size = 100
        max_records = limit or 15000

        while offset < max_records:
            url = (
                f"{self.working_base_url}/api/explore/v2.1/"
                f"catalog/datasets/{dataset_id}/records"
            )
            params = {"limit": batch_size, "offset": offset}

            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    self.log(f"   API error: {response.status_code}")
                    break

                data = response.json()
                records = data.get("results", [])
                if not records:
                    break

                for record in records:
                    fields = record.get("record", {}).get("fields", {})
                    if not fields:
                        fields = record.get("fields", record)
                    all_records.append(fields)

                offset += batch_size
                time.sleep(0.2)  # Rate limiting

            except requests.RequestException as error:
                self.log(f"   Error: {error}")
                break

        if all_records:
            df = pd.DataFrame(all_records)
            if 'session' not in df.columns:
                df['session'] = year
            self.log(f"   {len(df)} formations (via records API)")
            return df

        self.log(f"   Failed for {year}")
        return None

    def download_all(
        self,
        years: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download data for multiple years.

        Args:
            years: List of years to download. If None, downloads all available.

        Returns:
            Optional[pd.DataFrame]: Combined data from all years.
        """
        if years is None:
            years = sorted(self.DATASETS.keys(), reverse=True)

        if not self.test_api_access():
            return None

        dfs: List[pd.DataFrame] = []
        for year in years:
            df = self.download_year(year)
            if df is not None:
                dfs.append(df)
            time.sleep(1)  # Rate limiting between years

        if not dfs:
            return None

        combined = pd.concat(dfs, ignore_index=True)
        self.log(f"Total: {len(combined)} formations from {len(dfs)} years")
        return combined


# =============================================================================
# ANALYZER CLASS
# =============================================================================

class ParcoursupAnalyzer:
    """
    Parcoursup data analyzer with clustering and similarity search.

    This class provides functionality for loading Parcoursup data,
    training machine learning models for clustering and similarity
    search, and analyzing trends over time.

    Attributes:
        df: Full DataFrame with all years of data.
        df_latest: DataFrame with only the latest year.
        scaler: RobustScaler for feature normalization.
        kmeans: Trained K-Means clustering model.
        pca: Trained PCA model for visualization.
        knn_models: Dictionary of KNN models by formation type.
        knn_indices: Dictionary of formation indices by type.
        is_trained: Whether the models have been trained.

    Example:
        >>> analyzer = ParcoursupAnalyzer()
        >>> analyzer.load_data("data/parcoursup_data.csv")
        >>> analyzer.prepare_features()
        >>> analyzer.train_clustering()
        >>> results = analyzer.find_similar_formations("CPGE MPSI")
    """

    def __init__(self, verbose: bool = False) -> None:
        """
        Initialize the analyzer.

        Args:
            verbose: If True, print progress messages.
        """
        self.df: Optional[pd.DataFrame] = None
        self.df_latest: Optional[pd.DataFrame] = None
        self.scaler: Optional[RobustScaler] = None
        self.kmeans: Optional[KMeans] = None
        self.pca: Optional[PCA] = None
        self.nn_model: Optional[NearestNeighbors] = None
        self.feature_matrix: Optional[np.ndarray] = None
        self.is_trained: bool = False
        self.knn_models: Dict = {}
        self.knn_indices: Dict = {}
        self.verbose = verbose

    def log(self, message: str) -> None:
        """
        Log a message if verbose mode is enabled.

        Args:
            message: Message to log.
        """
        if self.verbose:
            print(message)
        logger.info(message)

    # =========================================================================
    # DATA LOADING
    # =========================================================================

    def download_data(
        self,
        years: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> 'ParcoursupAnalyzer':
        """
        Download data from the API.

        Args:
            years: List of years to download. If None, downloads all.
            save_path: Optional path to save the downloaded data.

        Returns:
            self: For method chaining.
        """
        downloader = ParcoursupDownloader(verbose=self.verbose)
        self.df = downloader.download_all(years=years)

        if self.df is not None:
            self._harmonize_columns()

            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                self.df.to_csv(save_path, index=False)
                self.log(f"Data saved to {save_path}")

            latest_year = self.df['session'].max()
            self.df_latest = self.df[self.df['session'] == latest_year].copy()

        return self

    def load_data(self, filepath: str) -> 'ParcoursupAnalyzer':
        """
        Load data from a local CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            self: For method chaining.
        """
        self.log(f"Loading data from {filepath}")
        self.df = pd.read_csv(filepath, low_memory=False)

        self._harmonize_columns()

        latest_year = self.df['session'].max()
        self.df_latest = self.df[self.df['session'] == latest_year].copy()

        self.log(f"Loaded {len(self.df)} records")
        return self

    def _harmonize_columns(self) -> None:
        """Harmonize column names across different data years."""
        renames = {
            'lib_for_voe_acc': 'form_lib_voe_acc',
            'g_ea_lib': 'g_ea_lib_vx',
            'region': 'region_etab_aff',
            'academie': 'acad_mies',
            'capacite': 'capa_fin',
            'nb_voeux': 'voe_tot',
            'nb_admis': 'acc_tot',
        }
        self.df = self.df.rename(columns=renames)

        if 'session' in self.df.columns:
            self.df['session'] = (
                pd.to_numeric(self.df['session'], errors='coerce')
                .fillna(0)
                .astype(int)
            )

        # Merge taux_acces columns (different years use different names)
        if 'taux_acces' in self.df.columns and 'taux_acces_ens' in self.df.columns:
            self.df['taux_acces'] = self.df['taux_acces'].combine_first(
                self.df['taux_acces_ens']
            )

        # Compute derived features
        self._compute_derived_features(self.df)

    # =========================================================================
    # FEATURE PREPARATION
    # =========================================================================

    def prepare_features(self) -> 'ParcoursupAnalyzer':
        """
        Prepare features for clustering.

        Returns:
            self: For method chaining.

        Raises:
            ValueError: If not enough features are available.
        """
        self._compute_derived_features(self.df)
        self._compute_derived_features(self.df_latest)

        # Map alternative column names
        feature_mapping = {
            'taux_acces_ens': 'taux_acces',
            'capa_fin': 'capacite',
            'pct_f': 'pct_femmes',
            'pct_bg': 'pct_bac_general',
            'pct_brs': 'pct_boursiers',
        }

        for old, new in feature_mapping.items():
            if old in self.df_latest.columns and new not in self.df_latest.columns:
                self.df_latest[new] = self.df_latest[old]
            if old in self.df.columns and new not in self.df.columns:
                self.df[new] = self.df[old]

        # Check available features
        available_features = [
            f for f in CLUSTERING_FEATURES
            if f in self.df_latest.columns
        ]

        if len(available_features) < 3:
            raise ValueError("Not enough features for clustering")

        # Prepare feature matrix
        X = self.df_latest[available_features].copy()
        X = X.fillna(X.median())

        self.scaler = RobustScaler()
        self.feature_matrix = self.scaler.fit_transform(X)

        self.log(f"Prepared {len(available_features)} features for clustering")
        return self

    def _compute_derived_features(self, df: pd.DataFrame) -> None:
        """
        Compute derived features like tension and percentages.

        Handles taux d'accès calculation for different years:
            - 2018-2019: No taux_acces column, must be calculated
            - 2020-2021: Uses taux_acces_ens column
            - 2022+: Uses taux_acces column

        Args:
            df: DataFrame to compute features for (modified in place).
        """
        # Calculate tension
        if 'voe_tot' in df.columns and 'capa_fin' in df.columns:
            df['tension'] = df['voe_tot'] / df['capa_fin'].replace(0, 1)

        # Merge taux_acces and taux_acces_ens
        if 'taux_acces' in df.columns and 'taux_acces_ens' in df.columns:
            df['taux_acces'] = df['taux_acces'].combine_first(df['taux_acces_ens'])
        elif 'taux_acces' not in df.columns and 'taux_acces_ens' in df.columns:
            df['taux_acces'] = df['taux_acces_ens']

        # Calculate taux_acces for missing values (2018-2019)
        if 'taux_acces' not in df.columns:
            df['taux_acces'] = np.nan

        mask_missing = df['taux_acces'].isna()
        if mask_missing.any():
            # Find voeux column
            voeux_col = None
            for col in ['voe_tot', 'nb_voe_pp', 'nb_voe_tot']:
                if col in df.columns:
                    voeux_col = col
                    break

            # Find propositions column
            prop_col = None
            for col in ['prop_tot', 'nb_prop', 'acc_tot', 'nb_acc_tot']:
                if col in df.columns:
                    prop_col = col
                    break

            if voeux_col and prop_col:
                calculated = (
                    df.loc[mask_missing, prop_col]
                    / df.loc[mask_missing, voeux_col].replace(0, 1)
                ) * 100
                df.loc[mask_missing, 'taux_acces'] = calculated.clip(0, 100)

        # Calculate other derived features
        if 'pct_mention_tb' not in df.columns:
            if 'acc_tb' in df.columns and 'acc_tot' in df.columns:
                df['pct_mention_tb'] = (
                    df['acc_tb'] / df['acc_tot'].replace(0, 1)
                ) * 100

        if 'pct_femmes' not in df.columns:
            if 'pct_f' in df.columns:
                df['pct_femmes'] = df['pct_f']
            elif 'acc_tot_f' in df.columns and 'acc_tot' in df.columns:
                df['pct_femmes'] = (
                    df['acc_tot_f'] / df['acc_tot'].replace(0, 1)
                ) * 100

        if 'pct_bac_general' not in df.columns:
            if 'pct_bg' in df.columns:
                df['pct_bac_general'] = df['pct_bg']
            elif 'acc_bg' in df.columns and 'acc_tot' in df.columns:
                df['pct_bac_general'] = (
                    df['acc_bg'] / df['acc_tot'].replace(0, 1)
                ) * 100

        if 'pct_boursiers' not in df.columns:
            if 'pct_brs' in df.columns:
                df['pct_boursiers'] = df['pct_brs']
            elif 'acc_brs' in df.columns and 'acc_tot' in df.columns:
                df['pct_boursiers'] = (
                    df['acc_brs'] / df['acc_tot'].replace(0, 1)
                ) * 100

        if 'capacite' not in df.columns and 'capa_fin' in df.columns:
            df['capacite'] = df['capa_fin']

    # =========================================================================
    # CLUSTERING AND KNN
    # =========================================================================

    def train_clustering(
        self,
        n_clusters: Optional[int] = None
    ) -> 'ParcoursupAnalyzer':
        """
        Train K-Means clustering and KNN models.

        Args:
            n_clusters: Number of clusters. If None, finds optimal.

        Returns:
            self: For method chaining.
        """
        if self.feature_matrix is None:
            self.prepare_features()

        if n_clusters is None:
            n_clusters = self._find_optimal_k()

        self.log(f"Training K-Means with {n_clusters} clusters...")

        # K-Means clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_latest['cluster'] = self.kmeans.fit_predict(self.feature_matrix)

        # PCA for visualization
        self.pca = PCA(n_components=2)
        coords = self.pca.fit_transform(self.feature_matrix)
        self.df_latest['pca_x'] = coords[:, 0]
        self.df_latest['pca_y'] = coords[:, 1]

        # KNN models by formation type
        self.knn_models = {}
        self.knn_indices = {}

        formation_types = self.df_latest['form_lib_voe_acc'].unique()

        for formation_type in formation_types:
            mask = self.df_latest['form_lib_voe_acc'] == formation_type
            indices = self.df_latest[mask].index.tolist()

            if len(indices) < 2:
                continue

            local_indices = [
                self.df_latest.index.get_loc(i)
                for i in indices
            ]
            X_type = self.feature_matrix[local_indices]

            n_neighbors = min(20, len(indices) - 1)
            knn = NearestNeighbors(
                n_neighbors=n_neighbors,
                metric='euclidean',
                algorithm='ball_tree'
            )
            knn.fit(X_type)

            self.knn_models[formation_type] = knn
            self.knn_indices[formation_type] = indices

        self.is_trained = True
        self.log(f"Training complete. {len(self.knn_models)} KNN models created.")
        return self

    def _find_optimal_k(self, max_k: int = 20) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Args:
            max_k: Maximum number of clusters to test.

        Returns:
            int: Optimal number of clusters.
        """
        best_k, best_score = 10, -1

        for k in range(5, min(max_k + 1, len(self.feature_matrix) // 20)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.feature_matrix)
            score = silhouette_score(self.feature_matrix, labels)

            if score > best_score:
                best_score = score
                best_k = k

        self.log(f"Optimal k={best_k} (silhouette={best_score:.3f})")
        return best_k

    def find_similar_formations(
        self,
        query: str,
        n: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Find formations similar to a query using KNN.

        Args:
            query: Search query for formation name.
            n: Number of similar formations to return.

        Returns:
            Optional[pd.DataFrame]: Similar formations or None if not found.

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_clustering() first.")

        mask = self.df_latest['form_lib_voe_acc'].str.contains(
            query, case=False, na=False
        )

        if mask.sum() == 0:
            return None

        idx = self.df_latest[mask].index[0]
        target = self.df_latest.loc[idx]
        target_formation = target['form_lib_voe_acc']

        return self._find_similar_from_index(idx, target_formation, n=n)

    def _find_similar_from_index(
        self,
        target_index: int,
        formation_name: str,
        n: int = 10
    ) -> Optional[pd.DataFrame]:
        """
        Find similar formations from a specific index.

        Args:
            target_index: Index of the target formation.
            formation_name: Name of the formation type.
            n: Number of results to return.

        Returns:
            Optional[pd.DataFrame]: Similar formations or None.
        """
        if formation_name not in self.knn_models:
            return None

        knn = self.knn_models[formation_name]
        formation_indices = self.knn_indices[formation_name]

        try:
            formation_indices.index(target_index)
        except ValueError:
            return None

        target_local_idx = self.df_latest.index.get_loc(target_index)
        target_features = self.feature_matrix[target_local_idx].reshape(1, -1)

        n_results = min(n + 1, len(formation_indices))
        distances, knn_indices = knn.kneighbors(
            target_features,
            n_neighbors=n_results
        )

        results_data = []
        for knn_idx, dist in zip(knn_indices[0], distances[0]):
            global_idx = formation_indices[knn_idx]

            if global_idx == target_index:
                continue

            row = self.df_latest.loc[global_idx]
            similarity = 1 / (1 + dist)

            results_data.append({
                'index': global_idx,
                'Établissement': row.get('g_ea_lib_vx', 'N/A'),
                'Académie': row.get('acad_mies', 'N/A'),
                'Région': row.get('region_etab_aff', 'N/A'),
                "Taux d'accès": row.get('taux_acces', 0),
                'Tension': row.get('tension', 0),
                '% Mention TB': row.get('pct_mention_tb', 0),
                'Similarité': similarity,
            })

        if not results_data:
            return None

        results_data = sorted(
            results_data,
            key=lambda x: x['Similarité'],
            reverse=True
        )[:n]
        result_indices = [r['index'] for r in results_data]

        return self.df_latest.loc[result_indices]

    # =========================================================================
    # MODEL PERSISTENCE
    # =========================================================================

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model.

        Raises:
            ValueError: If model is not trained.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        data = {
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'pca': self.pca,
            'knn_models': self.knn_models,
            'knn_indices': self.knn_indices,
            'feature_matrix': self.feature_matrix,
            'df_latest': self.df_latest,
            'df_full': self.df,
        }

        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        self.log(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> 'ParcoursupAnalyzer':
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the model file.

        Returns:
            self: For method chaining.
        """
        self.log(f"Loading model from {filepath}")

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.scaler = data['scaler']
        self.kmeans = data['kmeans']
        self.pca = data['pca']
        self.knn_models = data.get('knn_models', {})
        self.knn_indices = data.get('knn_indices', {})
        self.feature_matrix = data['feature_matrix']
        self.df_latest = data['df_latest']
        self.df = data['df_full']
        self.is_trained = True

        self.log(f"Model loaded. {len(self.knn_models)} KNN models available.")
        return self
