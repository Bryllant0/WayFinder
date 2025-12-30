#!/usr/bin/env python3
"""
Download Script - Parcoursup Data Download CLI.

This script provides a command-line interface for downloading Parcoursup data
from the French Ministry of Higher Education's open data API and training
machine learning models for the WayFinder application.

Features:
    - Download data for specific years (2018-2024)
    - Train and save ML clustering models
    - Verbose logging mode for debugging
    - Force re-download option

Usage:
    # Download all years with default settings
    python download_data.py

    # Download specific years
    python download_data.py --years 2024 2023 2022

    # Enable verbose logging
    python download_data.py --verbose

    # Force re-download even if data exists
    python download_data.py --force

    # Skip model training
    python download_data.py --skip-model

Author: Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza
Version: 2.0
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

# Check for required dependencies
try:
    import pandas as pd
except ImportError as error:
    print(f"Missing dependency: {error}")
    print("\nInstall dependencies with:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

from data_loader import ParcoursupAnalyzer
from config.settings import settings


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure and return the application logger.

    Sets up logging with appropriate format and level based on verbosity setting.

    Args:
        verbose: If True, sets logging level to DEBUG. Otherwise INFO.

    Returns:
        logging.Logger: Configured logger instance.

    Example:
        >>> logger = setup_logging(verbose=True)
        >>> logger.debug("Debug message")
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format=settings.LOG_FORMAT,
        datefmt=settings.LOG_DATE_FORMAT,
    )

    return logging.getLogger(__name__)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> args = parse_arguments()
        >>> print(args.years)
        [2024, 2023, 2022, 2021, 2020, 2019, 2018]
    """
    parser = argparse.ArgumentParser(
        description="Download Parcoursup data and train ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Download all years (2018-2024)
  %(prog)s --years 2024 2023 2022   Download specific years only
  %(prog)s --verbose                Enable debug logging
  %(prog)s --force                  Force re-download
  %(prog)s --skip-model             Download data only, skip ML training

For more information, see the README.md file.
        """,
    )

    parser.add_argument(
        '--years', '-y',
        type=int,
        nargs='+',
        default=settings.DATA_YEARS,
        help='Years to download (default: all 2018-2024)',
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=settings.DATA_PATH,
        help=f'Output CSV path (default: {settings.DATA_PATH})',
    )

    parser.add_argument(
        '--model-path', '-m',
        type=str,
        default=settings.MODEL_PATH,
        help=f'Model output path (default: {settings.MODEL_PATH})',
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) logging',
    )

    parser.add_argument(
        '--skip-model',
        action='store_true',
        help='Skip ML model training',
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if data exists',
    )

    return parser.parse_args()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_years(years: List[int], logger: logging.Logger) -> List[int]:
    """
    Validate the requested years against available datasets.

    Args:
        years: List of years requested by the user.
        logger: Logger instance for messages.

    Returns:
        List[int]: List of valid years that can be downloaded.

    Example:
        >>> valid_years = validate_years([2024, 2023, 2030], logger)
        >>> print(valid_years)
        [2024, 2023]
    """
    available_years = settings.DATA_YEARS
    valid_years = [y for y in years if y in available_years]
    invalid_years = [y for y in years if y not in available_years]

    if invalid_years:
        logger.warning(
            "Years not available: %s. Available: %s",
            invalid_years,
            available_years,
        )

    if not valid_years:
        logger.error("No valid years to download.")
        sys.exit(1)

    return valid_years


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_data(
    years: List[int],
    output_path: str,
    logger: logging.Logger,
    verbose: bool = False
) -> Optional[ParcoursupAnalyzer]:
    """
    Download Parcoursup data for the specified years.

    Args:
        years: List of years to download.
        output_path: Path to save the CSV data.
        logger: Logger instance for messages.
        verbose: Whether to enable verbose mode in the downloader.

    Returns:
        Optional[ParcoursupAnalyzer]: Analyzer instance with loaded data,
            or None if download failed.

    Example:
        >>> analyzer = download_data([2024, 2023], "data/test.csv", logger)
        >>> if analyzer:
        ...     print(f"Downloaded {len(analyzer.df)} records")
    """
    logger.info("=" * 60)
    logger.info("STARTING PARCOURSUP DATA DOWNLOAD")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Years to download: %s", years)
    logger.info("Output path: %s", output_path)
    logger.info("")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer and download
    analyzer = ParcoursupAnalyzer(verbose=verbose)

    try:
        analyzer.download_data(years=years, save_path=output_path)
    except Exception as error:
        logger.error("Download failed: %s", error)
        return None

    if analyzer.df is None or len(analyzer.df) == 0:
        logger.error("No data was downloaded.")
        return None

    logger.info("")
    logger.info("Download successful!")
    logger.info("  Total records: %d", len(analyzer.df))
    logger.info("  Years: %s", sorted(analyzer.df['session'].unique()))
    logger.info("  File saved: %s", output_path)
    logger.info("")

    return analyzer


def train_model(
    analyzer: ParcoursupAnalyzer,
    model_path: str,
    logger: logging.Logger
) -> bool:
    """
    Train and save ML clustering model.

    Args:
        analyzer: ParcoursupAnalyzer instance with loaded data.
        model_path: Path to save the trained model.
        logger: Logger instance for messages.

    Returns:
        bool: True if training succeeded, False otherwise.

    Example:
        >>> success = train_model(analyzer, "models/model.pkl", logger)
        >>> print("Training successful" if success else "Training failed")
    """
    logger.info("=" * 60)
    logger.info("TRAINING ML MODELS")
    logger.info("=" * 60)
    logger.info("")

    try:
        # Prepare features
        logger.info("Preparing features...")
        analyzer.prepare_features()

        # Train clustering
        logger.info("Training K-Means clustering...")
        analyzer.train_clustering()

        # Save model
        logger.info("Saving model to %s...", model_path)
        os.makedirs(os.path.dirname(model_path) or '.', exist_ok=True)
        analyzer.save_model(model_path)

        logger.info("")
        logger.info("Model training complete!")
        logger.info("  Clusters: %d", analyzer.kmeans.n_clusters)
        logger.info("  KNN models: %d", len(analyzer.knn_models))
        logger.info("  Model saved: %s", model_path)
        logger.info("")

        return True

    except Exception as error:
        logger.error("Model training failed: %s", error)
        return False


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main() -> int:
    """
    Main entry point for the download script.

    Parses arguments, downloads data, and optionally trains models.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    logger = setup_logging(verbose=args.verbose)

    # Check if data already exists
    if os.path.exists(args.output) and not args.force:
        logger.info("Data file already exists: %s", args.output)
        logger.info("Use --force to re-download.")

        # Load existing data for model training
        if not args.skip_model:
            analyzer = ParcoursupAnalyzer(verbose=args.verbose)
            analyzer.load_data(args.output)

            if not os.path.exists(args.model_path) or args.force:
                train_model(analyzer, args.model_path, logger)
            else:
                logger.info("Model already exists: %s", args.model_path)

        return 0

    # Validate years
    valid_years = validate_years(args.years, logger)

    # Download data
    analyzer = download_data(
        years=valid_years,
        output_path=args.output,
        logger=logger,
        verbose=args.verbose,
    )

    if analyzer is None:
        return 1

    # Train model
    if not args.skip_model:
        success = train_model(analyzer, args.model_path, logger)
        if not success:
            return 1

    # Final message
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("You can now launch the application:")
    logger.info("  streamlit run app.py")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
