# WayFinder - Parcoursup Analytics CLI (V0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A CLI application for analyzing Parcoursup data, developed for the **Introduction to Python** course at CentraleSupelec.

## Features (V0)

- **Formation search**: Search among all Parcoursup formations (2021-2024)
- **Detailed statistics**: Display admission statistics for each formation
- **Formation comparison**: Compare up to 4 formations side by side

## Project Structure

```
wayfinder_v0/
├── main.py                   # CLI entry point with argparse
├── search.py                 # API search and data extraction
├── config/
│   ├── __init__.py
│   └── settings.py           # Centralized configuration
├── tests/
│   ├── __init__.py
│   ├── test_search.py
│   └── test_main.py
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation

### 1. Clone the project

```bash
git clone <repository-url>
cd wayfinder_v0
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

## Usage

### Interactive mode (default)

```bash
python main.py
```

Launches an interactive menu with options:
1. Search for a formation
2. View statistics
3. Comparison
4. Change year
Q. Quit

### Command line mode

```bash
# Direct search
python main.py search "informatique"

# Search with specific year
python main.py search "CPGE MPSI" --year 2023

# Search with more results
python main.py search "droit" --limit 20

# Verbose mode (debug)
python main.py -v search "medecine"
```

### Help

```bash
python main.py --help
python main.py search --help
```

## Tests

### Run tests

```bash
# All tests
pytest tests/ -v

# Without warnings
pytest tests/ -v --disable-warnings

# Specific file
pytest tests/test_search.py -v
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                              │
│                   (CLI with argparse)                       │
│         - Interactive menu                                  │
│         - Commands: search                                  │
│         - Display functions                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────▼───────────────────────────────────────┐
          │                   search.py                       │
          │  - fetch_api(): Parcoursup API requests           │
          │  - search_formations(): Formation search          │
          │  - extract_formation_stats(): Stats extraction    │
          └───────────────────────────────────────────────────┘
                      │
          ┌───────────▼───────────────────────────────────────┐
          │              config/settings.py                   │
          │  - Centralized configuration                      │
          │  - Logging parameters                             │
          └───────────────────────────────────────────────────┘
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `-v, --verbose` | Debug mode with detailed logs | Off |
| `-y, --year` | Data year (2021-2024) | 2024 |
| `-n, --limit` | Maximum number of results | 10 |

## Interactive Menu Features

### Search (Option 1)
- Enter a search term (min 3 characters)
- Select a formation to view its statistics
- Results are sorted by popularity (number of wishes)

### Statistics (Option 2)
- Displays main indicators (access rate, wishes, admissions)
- Distribution by bac type (General, Techno, Pro)
- Distribution by honors (None, AB, B, TB, TB+)
- Other indicators (% scholarship holders, % same academy)

### Comparison (Option 3)
- Add up to 4 formations
- Compare indicators side by side
- Options to add/remove formations

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
