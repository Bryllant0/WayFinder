# WayFinder - Parcoursup Analytics Platform

A data-driven tool helping French students make informed decisions about their higher education choices by analyzing Parcoursup admission statistics.

## Authors

- Bryan Boislève
- Mizaan-Abbas Katchera
- Nawfel Bouazza

**Course:** Python  
**School:** ESSEC Business School / CentraleSupélec

## Project Overview

WayFinder addresses the information asymmetry faced by students navigating Parcoursup, the French national higher education admission platform. The application transforms raw statistical data from over 20,000 training programs into actionable insights, including personalized admission probability estimates, specialty recommendations, and historical trend analysis.

The project was developed incrementally across three versions, each building upon the previous one.

## Project Structure

```
GROUP-2/
├── README.md                    # This file
├── WayFinder Report.pdf         # Development report
├── v0/                          # CLI version (MVP)
├── v1/                          # Streamlit GUI version
└── v2/                          # Advanced version with ML features
```

## Version Overview

| Version | Interface | Key Features |
|---------|-----------|--------------|
| **V0** | CLI | Formation search, statistics display, basic comparison |
| **V1** | Streamlit GUI | User profiles, admission probability, specialty recommendations |
| **V2** | Streamlit GUI | ML clustering, similar formations, historical trends, predictions |

## Requirements

- Python 3.12 or higher
- Internet connection (for API access)

## Quick Start

### V0 - Command Line Interface

```bash
cd v0
pip install -r requirements.txt && pytest tests/ -v --disable-warnings && python main.py
```

### V1 - Web Interface

```bash
cd v1
pip install -r requirements.txt && pytest tests/ -v --disable-warnings && streamlit run app.py
```

### V2 - Web Interface with ML Features

For optimal performance, it is strongly advised to download the data first:

```bash
cd v2
pip install -r requirements.txt && pytest tests/ -v --disable-warnings && python download_data.py && streamlit run app.py
```

## Features by Version

### V0 (CLI)
- Search formations by name or establishment
- View detailed admission statistics
- Compare up to 4 formations
- Interactive menu system

### V1 (Streamlit GUI)
- All V0 features plus:
- User profile configuration (bac type, grades, specialties, academy)
- Personalized admission probability calculation
- Specialty (doublette) recommendations by formation type
- Interactive Plotly visualizations

### V2 (Advanced Analytics)
- All V1 features plus:
- K-Means clustering for formation grouping
- KNN-based similar formation recommendations
- Historical trend analysis (2021-2024)
- Linear regression predictions
- Global overview dashboard

## Data Sources

The application uses public data from the Parcoursup OpenData API, covering admission statistics from 2021 to 2024. Due to a major methodology change in 2021, formation-level analysis is restricted to post-2021 data to ensure consistency.

## Documentation

For detailed information about the project architecture, algorithms, and implementation, please refer to the **WayFinder Report.pdf** included in this folder.

## License

This project was developed for educational purposes as part of the Introduction to Python course at CentraleSupélec.
