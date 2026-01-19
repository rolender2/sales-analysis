"""
Forecasting Agent - Agent 4

This agent generates sales forecasts using multiple time series models
(ARIMA, Holt-Winters, Moving Average).
"""

from pathlib import Path
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import Agent, function_tool

from sales_agents.tools import (
    query_mongodb,
    save_report,
    save_visualization,
    insert_mongodb,
)
from config import get_model_string, MONGODB_DATABASE, FORECAST_CONFIG

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

# =============================================================================
# Forecasting Tools (Agent-specific)
# =============================================================================

@function_tool
def prepare_time_series(
    collection: str = "sales",
    frequency: str = "monthly",
) -> dict:
    """
    Prepare time series data from sales collection for forecasting.
    
    Args:
        collection: Source collection name
        frequency: Aggregation frequency ('daily', 'weekly', 'monthly')
        
    Returns:
        Dictionary with prepared time series data
    """
    from pymongo import MongoClient
    from config import MONGODB_URI, MONGODB_DATABASE
    
    try:
        client = MongoClient(MONGODB_URI)
        db = client[MONGODB_DATABASE]
        
        # Get all sales data
        pipeline = [
            {"$project": {"order_date": 1, "sales": 1}},
        ]
        
        data = list(db[collection].aggregate(pipeline))
        client.close()
        
        if not data:
            return {"success": False, "error": "No data found"}
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Parse dates (DD/MM/YYYY format)
        df["date"] = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
        df = df.dropna(subset=["date"])
        df = df.set_index("date").sort_index()
        
        # Aggregate by frequency
        if frequency == "daily":
            ts = df["sales"].resample("D").sum()
        elif frequency == "weekly":
            ts = df["sales"].resample("W").sum()
        else:  # monthly
            ts = df["sales"].resample("MS").sum()
        
        # Fill missing periods with 0
        ts = ts.fillna(0)
        
        return {
            "success": True,
            "frequency": frequency,
            "start_date": str(ts.index.min()),
            "end_date": str(ts.index.max()),
            "num_periods": len(ts),
            "total_sales": float(ts.sum()),
            "mean_sales": float(ts.mean()),
            "data": [{"date": str(d), "sales": float(v)} for d, v in ts.items()],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def run_arima_forecast(
    time_series_data_json: str,
    forecast_periods: int = 12,
) -> dict:
    """
    Run ARIMA forecast on prepared time series data.
    
    Args:
        time_series_data_json: JSON string of [{'date': str, 'sales': float}] array
        forecast_periods: Number of periods to forecast
        
    Returns:
        Forecast results with predictions and metrics
    """
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Parse JSON data
        time_series_data = json.loads(time_series_data_json)
        
        # Create series
        df = pd.DataFrame(time_series_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Explicitly set frequency to Month Start (MS) to suppress warnings
        df = df.asfreq("MS").fillna(0)
        series = df["sales"]
        
        # Train/test split
        split_idx = int(len(series) * 0.8)
        train = series[:split_idx]
        test = series[split_idx:]
        
        # Fit ARIMA model (auto-select simple parameters)
        model = ARIMA(train, order=(1, 1, 1))
        fitted = model.fit()
        
        # In-sample predictions for test set
        predictions = fitted.forecast(steps=len(test))
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100
        
        # Refit on full data and forecast
        full_model = ARIMA(series, order=(1, 1, 1))
        full_fitted = full_model.fit()
        forecast = full_fitted.get_forecast(steps=forecast_periods)
        forecast_mean = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        
        # Generate future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq="MS")[1:]
        
        forecast_data = []
        for i, (date, pred) in enumerate(zip(future_dates, forecast_mean)):
            forecast_data.append({
                "date": str(date),
                "predicted_sales": float(pred),
                "lower_bound": float(conf_int.iloc[i, 0]),
                "upper_bound": float(conf_int.iloc[i, 1]),
            })
        
        return {
            "success": True,
            "model": "ARIMA(1,1,1)",
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "train_size": len(train),
                "test_size": len(test),
            },
            "forecast": forecast_data,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def run_holt_winters_forecast(
    time_series_data_json: str,
    forecast_periods: int = 12,
    seasonal_periods: int = 12,
) -> dict:
    """
    Run Holt-Winters Exponential Smoothing forecast.
    
    Args:
        time_series_data_json: JSON string of [{'date': str, 'sales': float}] array
        forecast_periods: Number of periods to forecast
        seasonal_periods: Number of periods in a seasonal cycle
        
    Returns:
        Forecast results with predictions and metrics
    """
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Parse JSON data
        time_series_data = json.loads(time_series_data_json)
        
        # Create series
        df = pd.DataFrame(time_series_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        # Explicitly set frequency to Month Start (MS) to suppress warnings
        df = df.asfreq("MS").fillna(0)
        series = df["sales"]
        
        # Need at least 2 full seasonal cycles
        if len(series) < seasonal_periods * 2:
            return {
                "success": False,
                "error": f"Need at least {seasonal_periods * 2} periods for Holt-Winters"
            }
        
        # Train/test split
        split_idx = int(len(series) * 0.8)
        train = series[:split_idx]
        test = series[split_idx:]
        
        # Fit Holt-Winters model
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        )
        fitted = model.fit()
        
        # Predictions
        predictions = fitted.forecast(len(test))
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(test, predictions))
        mae = mean_absolute_error(test, predictions)
        mape = np.mean(np.abs((test - predictions) / test)) * 100
        
        # Refit and forecast
        full_model = ExponentialSmoothing(
            series,
            trend="add",
            seasonal="add",
            seasonal_periods=seasonal_periods,
        )
        full_fitted = full_model.fit()
        forecast = full_fitted.forecast(forecast_periods)
        
        # Future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq="MS")[1:]
        
        forecast_data = []
        for date, pred in zip(future_dates, forecast):
            # Approximate confidence interval (±15% for simplicity)
            forecast_data.append({
                "date": str(date),
                "predicted_sales": float(pred),
                "lower_bound": float(pred * 0.85),
                "upper_bound": float(pred * 1.15),
            })
        
        return {
            "success": True,
            "model": "Holt-Winters (Additive)",
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "train_size": len(train),
                "test_size": len(test),
            },
            "forecast": forecast_data,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def run_moving_average_forecast(
    time_series_data_json: str,
    forecast_periods: int = 12,
    window: int = 3,
) -> dict:
    """
    Run simple Moving Average forecast (baseline model).
    
    Args:
        time_series_data_json: JSON string of [{'date': str, 'sales': float}] array
        forecast_periods: Number of periods to forecast
        window: Moving average window size
        
    Returns:
        Forecast results with predictions and metrics
    """
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Parse JSON data
        time_series_data = json.loads(time_series_data_json)
        
        # Create series
        df = pd.DataFrame(time_series_data)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        series = df["sales"]
        
        # Calculate moving average
        ma = series.rolling(window=window).mean()
        
        # Use last MA value for all forecasts
        last_ma = ma.iloc[-1]
        
        # Simple evaluation: compare MA predictions to actual for last window periods
        test_start = len(series) - window
        test = series.iloc[test_start:]
        predictions = ma.iloc[test_start:]
        
        # Remove NaN from evaluation
        valid_idx = ~predictions.isna()
        rmse = np.sqrt(mean_squared_error(test[valid_idx], predictions[valid_idx]))
        mae = mean_absolute_error(test[valid_idx], predictions[valid_idx])
        mape = np.mean(np.abs((test[valid_idx] - predictions[valid_idx]) / test[valid_idx])) * 100
        
        # Future dates
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=forecast_periods + 1, freq="MS")[1:]
        
        forecast_data = []
        for date in future_dates:
            forecast_data.append({
                "date": str(date),
                "predicted_sales": float(last_ma),
                "lower_bound": float(last_ma * 0.8),  # ±20% bounds
                "upper_bound": float(last_ma * 1.2),
            })
        
        return {
            "success": True,
            "model": f"Moving Average (window={window})",
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "window": window,
            },
            "forecast": forecast_data,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def create_forecast_visualization(
    historical_data_json: str,
    forecast_data_json: str,
    model_name: str,
    filename: str,
) -> dict:
    """
    Create a visualization showing historical data and forecast.
    
    Args:
        historical_data_json: JSON string of [{'date': str, 'sales': float}] array
        forecast_data_json: JSON string of [{'date': str, 'predicted_sales': float, 'lower_bound': float, 'upper_bound': float}] array
        model_name: Name of the forecasting model
        filename: Output filename
        
    Returns:
        Dictionary with success status and filepath
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import VISUALIZATIONS_DIR
    
    try:
        # Parse JSON data
        historical_data = json.loads(historical_data_json)
        forecast_data = json.loads(forecast_data_json)
        
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        
        filepath = VISUALIZATIONS_DIR / filename
        
        # Prepare data
        hist_df = pd.DataFrame(historical_data)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        
        fore_df = pd.DataFrame(forecast_data)
        fore_df["date"] = pd.to_datetime(fore_df["date"])
        
        # Create figure
        plt.figure(figsize=(14, 6))
        
        # Historical data
        plt.plot(hist_df["date"], hist_df["sales"], 
                 label="Historical Sales", color="blue", linewidth=2)
        
        # Forecast
        plt.plot(fore_df["date"], fore_df["predicted_sales"],
                 label=f"Forecast ({model_name})", color="red", 
                 linestyle="--", linewidth=2)
        
        # Confidence interval
        plt.fill_between(
            fore_df["date"],
            fore_df["lower_bound"],
            fore_df["upper_bound"],
            alpha=0.2,
            color="red",
            label="95% Confidence Interval",
        )
        
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Sales ($)", fontsize=12)
        plt.title(f"Sales Forecast - {model_name}", fontsize=14, fontweight="bold")
        plt.legend(loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        
        return {
            "success": True,
            "filepath": str(filepath),
            "filename": filename,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# =============================================================================
# Forecaster Agent Definition
# =============================================================================

FORECASTER_INSTRUCTIONS = """Generate sophisticated sales forecasts and save `forecast_report.md`.

## Steps
1. **Data Prep**: `prepare_time_series(collection="sales", frequency="monthly")`.
2. **Analysis**: Look at the time series. Is it seasonal? Is there a trend? What patterns do you observe?
3. **Modeling**:
   - Run **ARIMA** as the primary baseline.
   - Run **Holt-Winters** if you detect seasonality.
   - Run **Moving Average** as a simple baseline.
   - Compare metrics (RMSE/MAE/MAPE) across all models.

## Visualizations (CRITICAL)
Generate **multiple visualizations** to help the business understand the forecast. Think critically about what would be most insightful:

- **Main Forecast Chart**: Historical data + forecast with confidence intervals. Filename MUST contain "forecast" (e.g., `sales_forecast.png`).
- **Consider Additional Charts** based on what you think would be most valuable:
  - Model performance comparison (bar chart of RMSE/MAE across models)
  - Trend/seasonality decomposition
  - Year-over-year comparison
  - Actual vs. predicted for validation period
  - Any other visualization you think would help business decision-making

**Chart Requirements**:
- All charts MUST have clear titles, axis labels, and legends
- Use distinct colors for different data series
- Include units ($ for sales, dates on axes)
- Make charts readable and professional

## Reporting
Write a comprehensive `forecast_report.md` (approx. 500+ words):
- **Model Selection**: Explain *why* you chose the winning model
- **Forecast Narrative**: Describe the expected trend with specific numbers
- **Risk Assessment**: Analyze confidence intervals - when is the forecast most uncertain?
- **Model Context**: Explain the pros/cons of each model used
- **Business Recommendation**: How should the business use these forecasts? (Inventory planning, budgeting, etc.)
"""


def create_forecaster(provider: str = "openai", model: str = None) -> Agent:
    """
    Create a Forecaster agent with the specified LLM provider.
    
    Args:
        provider: LLM provider ('openai', 'anthropic', 'deepseek')
        model: Specific model name (uses provider default if not specified)
        
    Returns:
        Configured Agent instance
    """
    from config import LLM_PROVIDERS
    
    if model is None:
        model = LLM_PROVIDERS.get(provider, {}).get("default_model", "gpt-4-turbo")
    
    model_string = get_model_string(provider, model)
    
    return Agent(
        name="Forecaster",
        model=model_string,
        instructions=FORECASTER_INSTRUCTIONS,
        tools=[
            query_mongodb,
            save_report,
            save_visualization,
            insert_mongodb,
            prepare_time_series,
            run_arima_forecast,
            run_holt_winters_forecast,
            run_moving_average_forecast,
            create_forecast_visualization,
        ],
    )


# Default instance using OpenAI
forecaster = create_forecaster()
