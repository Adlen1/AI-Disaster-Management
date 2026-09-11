# Intelligent Pipeline for Wildfire Prediction in Algeria

## Overview

An explainable AI system for **next-day wildfire risk prediction in Algeria**, developed to support wildfire prevention and decision-making.

The system combines weather, satellite, environmental, geographical, and demographic data to estimate wildfire risk at the **commune level** and present the results through an interactive web dashboard.

## Features

*  **Next-day wildfire risk prediction** — LOW, MODERATE, or HIGH
*  **Commune-level risk mapping** across fire-prone areas of Algeria
*  **Explainable predictions** using SHAP-based feature analysis
*  **Risk prioritization** to highlight areas requiring greater attention
*  **Multi-source data integration** combining meteorological, satellite, environmental, and geographical data
*  **Interactive web dashboard** designed for decision support

## Academic Context

This project was developed as part of an **internship project at École Nationale Supérieure d'Informatique (ESI)**

### Developed by

**Alliouane Adlen**

### Supervised by

**M. Bouchama Nadir**

## Host Organization

**Centre de Recherche sur l'Information Scientifique et Technique (CERIST)**
Algiers, Algeria

CERIST is a research center under the Algerian Ministry of Higher Education and Scientific Research, working in areas related to scientific information, computer science, information and communication technologies, and digital systems.

## Results

The final Random Forest model was evaluated on unseen fire seasons from **2021–2025**, with an additional **2020 stress-test season**.

| Evaluation         |   Macro F1 |        AUC |
| ------------------ | ---------: | ---------: |
| Test (2021–2025)   | **0.5545** | **0.8318** |
| Stress Test (2020) | **0.5891** | **0.8253** |

A retrospective case study using operational predictions from August 2026 also showed elevated predicted risk in several communes affected by a subsequent wildfire outbreak.

## Report

The complete methodology, data preparation, modeling, evaluation, explainability analysis, and application are documented in the internship report:

**[📄 Read the Full Internship Report]()**

## Project Structure

```text
.
├── configs/                  # config.yaml — data paths, date ranges, thresholds
├── ingest/                   # one module per data source (gadm.py, dem.py, era5.py, firms.py, ...)
├── scripts/
│   ├── orchestrator.py       # runs the full pipeline: --mode train | operational
│   ├── integrate/
│   │   └── build_dataset.py  # merges curated sources into the commune × day dataset
│   ├── export_nb03_artifacts.py  # freezes train-only artifacts (fire rate, anomaly baselines, ERA5 mapping)
│   └── predict_today.py      # daily operational scoring + SHAP explanations + JSON/GeoJSON export
├── notebooks/
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_comparison.ipynb
│   ├── 05_model_tuning.ipynb
│   └── 06_xai.ipynb
├── backend/                  # FastAPI app serving predictions
├── frontend/                 # Next.js dashboard (simple + detailed views)
```
