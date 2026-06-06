> End-to-end daily Air Quality Index forecasting for Delhi using ensemble machine learning, advanced time-series feature engineering, and SHAP-based interpretability.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20RF%20%7C%20HGB-orange) ![SHAP](https://img.shields.io/badge/Interpretability-SHAP-green) ![R2](https://img.shields.io/badge/R²-0.94-brightgreen) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]

---

## Overview

Delhi consistently ranks among the most polluted cities in the world. This project builds a robust pipeline to forecast next-day AQI using historical air quality and meteorological data, with a focus on both predictive accuracy and model transparency.

**What this project covers:**
- Time-series preprocessing and daily aggregation from hourly data
- Lag, rolling window, and cyclical feature engineering
- Weighted ensemble of three gradient-based regressors
- 7-day iterative future forecasting
- SHAP analysis for global and per-prediction interpretability

---

## Dataset

- **Source**: [India Meteorological Department (IMD)](https://mausam.imd.gov.in/)
- **File**: `delhi_aqi.csv`
- **Temporal range**: 2021–2025
- **Granularity**: Hourly → aggregated to daily
- **Target**: `target_aqi_tomorrow` (next-day US AQI mean, shifted by 1 day)

**Features include**: pollutant concentrations (PM2.5, PM10, etc.), meteorological parameters, festival period flags, and crop burning season indicators.

---

## Installation

```bash
# Mount Google Drive (Colab)
from google.colab import drive
drive.mount('/content/drive')

# Install dependencies
pip install xgboost shap scikit-learn pandas numpy matplotlib
```

---

## Pipeline

### 1. Data Preprocessing
- Parse and sort by `datetime`
- Aggregate hourly → daily (mean, std, max, min, median per feature)
- Create `target_aqi_tomorrow` via 1-day shift on `us_aqi_mean`
- Drop rows with nulls introduced by shifting/rolling

### 2. Feature Engineering

| Feature Type | Details |
|---|---|
| **Lag features** | AQI, PM2.5, PM10 at lags: 1, 2, 3, 4, 5, 6, 7, 10, 14, 21, 28 days |
| **Rolling windows** | Mean & std of AQI over 2, 3, 5, 7, 10, 14, 21-day windows |
| **Time-based** | Month, day of week, weekend flag |
| **Cyclical encoding** | Sine & cosine of month to handle seasonality |

### 3. Model Training

Train/test split: **85% train / 15% test** (chronological, no shuffling).

Three models are trained independently:

| Model | Role |
|---|---|
| XGBoost Regressor | Primary learner |
| Random Forest Regressor | Variance reduction |
| HistGradientBoosting Regressor | Missing-value robust booster |

**Ensemble**: Weighted average of all three predictions:
```
Final = 0.40 × XGBoost + 0.35 × RandomForest + 0.25 × HGB
```

---

## Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| XGBoost | 12.84 | 8.76 | 0.93 |
| Random Forest | 14.21 | 9.43 | 0.91 |
| HistGradientBoosting | 14.97 | 9.88 | 0.90 |
| **Ensemble** | **11.32** | **7.64** | **0.94** |

The ensemble model achieves **94% R²** on the held-out test set, outperforming all individual models.

---

## Future Forecasting

The notebook implements **iterative 7-day forecasting** — for each future day, the model's own prediction feeds back as input for the next step's lag features, simulating a real deployment scenario where ground truth is unavailable.

---

## Model Interpretability (SHAP)

| Plot Type | Purpose |
|---|---|
| **Bar summary** | Global feature importance ranking |
| **Beeswarm summary** | Feature value vs. impact distribution |
| **Dependence plots** | Effect of a single feature, with interaction highlights |
| **Force plots** | Per-prediction breakdown (base value → final output) |
| **Event impact analysis** | How `festival_period` and `crop_burning_season` shift future AQI predictions |

---

## Key Findings

- Lag features (especially 1–7 day AQI lags) dominate feature importance in SHAP analysis.
- `festival_period` and `crop_burning_season` flags show measurable positive push on predicted AQI in force plots.
- Cyclical month encoding outperforms raw month integer in capturing winter pollution spikes.
- The ensemble consistently outperforms any single model on RMSE, MAE, and R².

---

## Project Structure

```
AQI-Prediction-and-Explanation-using-SHAP/
├── delhi_aqi.csv
└── notebooks/
    └── delhi_aqi_forecasting.ipynb
```

---

## Limitations & Future Work

- Iterative forecasting accumulates error over the 7-day horizon — uncertainty bands would improve usability.
- No external weather forecast data used; integrating forecast APIs could reduce lag dependency.
- Model is Delhi-specific; generalizing to other Indian cities would require retraining and local event calendars.
- Future work: LSTM or Temporal Fusion Transformer for direct multi-step forecasting.

---
