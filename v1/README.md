# WayFinder - Parcoursup Analytics (V1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io/)

A Streamlit web application for analyzing Parcoursup data, developed for the **Introduction to Python** course at CentraleSupelec.

## Features (V1)

- **Formation search**: Search among all Parcoursup formations (2021-2024)
- **Detailed statistics**: View admission statistics for each formation
- **Admission probability**: Personalized estimate based on your profile
- **Specialty recommendations**: Optimal combinations based on formation type
- **Formation comparison**: Compare up to 4 formations simultaneously

## Project Structure

```
wayfinder_v1/
├── app.py                    # Streamlit entry point
├── visualization.py          # Chart utilities
├── search.py                 # API search and probability calculation
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
cd wayfinder_v1
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

### 4. Launch the application

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
2. **Statistiques**: View detailed statistics for a selected formation
3. **Mes Chances**: Calculate your personalized admission probability
4. **Comparatif**: Compare up to 4 formations side by side

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                              │
│                   (Streamlit Interface)                     │
└─────────────────────┬───────────────────┬───────────────────┘
                      │                   │
          ┌───────────▼───────────┐ ┌─────▼─────────────────┐
          │      search.py        │ │   visualization.py    │
          │  - API search         │ │  - Chart styling      │
          │  - Probability calc   │ │                       │
          │  - Specialties        │ │                       │
          └───────────────────────┘ └───────────────────────┘
```

## Data Sources

Data comes from the French Ministry of Higher Education OpenData API:
- **URL**: https://data.enseignementsup-recherche.gouv.fr
- **Datasets**: fr-esr-parcoursup (2021-2024)
- **License**: Open License

## Authors

**Bryan Boisleve - Mizaan-Abbas Katchera - Nawfel Bouazza**  
Master Data Science & Business Analytics  
ESSEC Business School / CentraleSupelec

---

*Data comes from public sources and is provided for informational purposes only.*
