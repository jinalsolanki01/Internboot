# Internboot — Store Sales Forecasting Project

This repository contains three completed internship tasks built on the Kaggle Store Sales — Time Series Forecasting dataset. The work progresses from a simple regression baseline to time-series-aware regression and finally to a deployable Streamlit dashboard for interactive sales prediction.

## Project Highlights

- Beginner Task 3: Linear Regression for Sales Prediction
- Intermediate Task 3: Time Series Regression with Trend + Seasonality
- Advanced Task 3: Model Deployment with a Streamlit dashboard

## Repository Structure

```text
Internboot/
├── Beginner Level Task 3/
│   └── Linear_Regression_Sales.ipynb
├── Intermediate Level Task 3/
│   └── TimeSeries_Regression.ipynb
├── Advanced Task 3/
│   ├── app.py
│   ├── src/
│   │   ├── preprocessing.py
│   │   ├── feature_engineering.py
│   │   ├── model_training.py
│   │   ├── visualization.py
│   │   ├── ui_components.py
│   │   └── utils.py
│   └── requirements.txt
└── README.md
```

## Dataset

The project uses the Kaggle competition dataset:

## Store Sales — Time Series Forecasting

Required or commonly used files:

- `train.csv` — daily sales records
- `stores.csv` — store metadata
- `holidays_events.csv` — holiday information

Place the data files inside the `data/` folder before running the notebooks or the dashboard.

## Environment Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## How to Run

## Beginner Task 3 — Linear Regression for Sales Prediction

Open the notebook and run all cells:

```bash
jupyter notebook "Linear_Regression_Sales.ipynb"
```

This notebook builds a baseline sales prediction model using date-based features, promotions, and holiday indicators.

## Intermediate Task 3 — Time Series Regression

Open the notebook and run all cells:

```bash
jupyter notebook "TimeSeries_Regression.ipynb"
```

This notebook extends the baseline approach by adding trend and seasonality features, including polynomial regression and cyclical calendar patterns.

## Advanced Task 3 — Model Deployment

Run the dashboard with Streamlit:

```bash
streamlit run app.py
```

The dashboard provides:

- an overview tab with KPIs and charts,
- a model performance tab with evaluation visuals,
- a prediction tab for interactive sales forecasting.

## Beginner Task 3 Summary

This task focuses on a simple regression pipeline for sales prediction. The notebook uses historical sales data, promotional flags, and holiday features to train a Linear Regression model. It also includes a chronological train/test split and evaluation metrics for basic forecasting.

## Intermediate Task 3 Summary

This task improves the regression approach by modeling trend and seasonality. It uses time-aware feature engineering, cyclical encodings for calendar variables, and polynomial regression to capture non-linear growth patterns. The notebook compares a baseline model with more expressive alternatives.

## Advanced Task 3 Summary

This task turns the forecasting workflow into a user-facing application. The Streamlit dashboard loads and merges the data, engineers features, trains or reloads the model, and displays results through interactive charts and prediction controls. The codebase is modular and organized into reusable components for preprocessing, feature engineering, training, visualization, and UI rendering.

## Output Artifacts

Depending on the notebook or app run, the project may generate:

- trained model files in `models/`
- metadata files containing metrics and feature columns
- chart and analysis outputs
- dashboard predictions and visualizations in the browser

## Notes

- The notebooks and dashboard assume the Kaggle files are available locally.
- The advanced dashboard is designed for chronological forecasting, so random splits are avoided.
- All three tasks are documented as a single combined internship submission.

## License / Usage

This project is created for internship and educational submission purposes.
