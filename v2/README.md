# WayFinder - Parcoursup Analytics (V2)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

A comprehensive Streamlit web application for analyzing Parcoursup data, developed for the **Introduction to Python** course at CentraleSupelec.

## Features (V2)

- **Formation search**: Search among all Parcoursup formations
- **Detailed statistics**: View admission statistics with historical trends
- **Admission probability**: Personalized estimate based on your profile
- **Specialty recommendations**: Optimal combinations based on formation type
- **Formation comparison**: Compare up to 4 formations simultaneously
- **Similar formations**: Find similar programs using ML clustering
- **Trends analysis**: Historical evolution and predictions
- **Overview**: Global statistics, clustering visualization, and rankings

## Project Structure

```
wayfinder_v2/
├── app.py                    # Streamlit entry point
├── visualization.py          # Charts and UI components
├── search.py                 # API search and probability calculation
├── data_loader.py            # Data download and ML analysis
├── download_data.py          # CLI script with argparse
├── config/
│   ├── __init__.py
│   └── settings.py           # Centralized configuration
├── tests/
│   ├── __init__.py
│   ├── test_visualization.py
│   └── test_search.py
├── .streamlit/
│   └── config.toml           # Streamlit theme
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation

### 1. Clone the project

```bash
git clone <repository-url>
cd wayfinder_v2
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download data (strongly advised)

```bash
python download_data.py
```

### 5. Launch the application

```bash
streamlit run app.py
```

## Tests

### Run tests

```bash
# All tests
pytest tests/ -v

# Without warnings
pytest tests/ -v --disable-warnings
```

## Application Tabs

1. **Rechercher**: Search formations by name or establishment
2. **Statistiques**: Detailed statistics with historical evolution and predictions
3. **Mes Chances**: Personalized admission probability calculation
4. **Comparatif**: Compare up to 4 formations side by side
5. **Formations similaires**: Find similar programs based on ML clustering
6. **Tendances**: Historical trends analysis
7. **Vue d'ensemble**: Global statistics, rankings, and clustering visualization

## CLI Usage

```bash
# Download data for specific years
python download_data.py --years 2024 2023

# Verbose mode
python download_data.py --verbose

# Skip model training
python download_data.py --skip-model

# Force re-download
python download_data.py --force
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                              │
│                   (Streamlit Interface)                     │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
          ┌───────────▼───────────┐ ┌─────▼─────────────────┐
          │      search.py        │ │   visualization.py    │
          │  - API search         │ │  - Chart rendering    │
          │  - Probability calc   │ │  - Analysis tabs      │
          │  - Specialties        │ │  - UI components      │
          └───────────────────────┘ └───────────────────────┘
                      │
          ┌───────────▼───────────────────────────────────────┐
          │               data_loader.py                      │
          │  - Data download                                  │
          │  - ML clustering                                  │
          │  - Historical analysis                            │
          └───────────────────────────────────────────────────┘
```

## Data Sources

Data comes from the French Ministry of Higher Education OpenData API:
- **URL**: https://data.enseignementsup-recherche.gouv.fr
- **Datasets**: fr-esr-parcoursup
- **License**: Open License

## Authors

**Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza**  
Master Data Science & Business Analytics  
ESSEC Business School / CentraleSupelec

---

*Data comes from public sources and is provided for informational purposes only. Statistical projections do not guarantee admission. Always verify official information on Parcoursup.*
