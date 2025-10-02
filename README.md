# 📦 assignment_2_packages

This repository contains the custom **Python package** developed for Advanced Machine Learning Applications (36120) Assignment 2.

The package provides reusable functions and classes to support the end-to-end workflow for the Sydney Weather Forecasting project, including **data preprocessing, feature engineering, and model utilities**. It is designed to be integrated with experimentation notebooks and API deployment.

--- 

## ✨ Features

- **Data ingestion utilities** for handling raw and processed Open-Meteo datasets.

- **Feature engineering** (lag features, rolling aggregates, seasonal encodings, interaction terms).

- **Transformation pipelines** for classification and regression tasks.

- Helper functions for **evaluation** and **visualization**.

--- 

## 🔧 Installation

The package is published on TestPyPI. To install:

```bash
pip install -i https://test.pypi.org/simple/ assignment-2-packages
```

--- 

## 🚀 Usage Example

```python
from assignment_2_packages.features import create_lag_features
from assignment_2_packages.dataset import load_processed_data

# Load processed dataset
df = load_processed_data("data/processed/sydney_daily_2000_2024.parquet")

# Apply lag features
df = create_lag_features(df, cols=["precipitation_sum", "temperature_2m_mean"], lags=[1, 2, 3])

print(df.head())
```
---

## 📂 Repository Structure

```bash
assignment_2_packages/
│── pyproject.toml         # Poetry configuration
│── README.md              # This file
│── assignment_2_packages/ # Source code
│   ├── __init__.py
│   ├── dataset.py         # Data loading utilities
│   ├── features.py        # Feature engineering
│   ├── transforms.py      # Transformation classes
│   ├── eval.py            # Model evaluation helpers
│   └── utils.py           # General helpers
└── tests/                 # Pytest unit tests
```

---

## ✅ Testing

Run unit tests with:

```bash
pytest tests/
```

---

## 📚 References

Open-Meteo Historical Weather API: https://open-meteo.com/

Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825–2830.