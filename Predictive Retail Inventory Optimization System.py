import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. DATA SIMULATION & PREPROCESSING MODULE


def generate_mock_sales_data(days=365):
    """Generates synthetic historical retail sales data for testing."""
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=days, freq="D")

    # Simulate a baseline demand with weekly seasonality and random noise
    base_demand = 50
    seasonality = 15 * np.sin(2 * np.pi * dates.dayofweek / 7)
    noise = np.random.normal(0, 5, days)
    sales = np.maximum(0, base_demand + seasonality + noise).astype(int)

    df = pd.DataFrame({"date": dates, "sales": sales})
    return df


def preprocess_features(df):
    """Engineers time-series features out of the date column."""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month

    # Create lag features (previous day's sales)
    df['lag_1'] = df['sales'].shift(1)
    df = df.dropna().reset_index(drop=True)
    return df

# 2. MACHINE LEARNING FORECASTING MODULE


def train_demand_model(df):
    """Trains a Random Forest Regressor to forecast retail demand."""
    X = df[['day_of_week', 'month', 'lag_1']]
    y = df['sales']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"[Audit Log] Model Evaluation Complete. RMSE: {rmse:.2f}")

    return model, X_test

# 3. DYNAMIC INVENTORY LOGIC ENGINE


def calculate_inventory_targets(predicted_demand, lead_time_days=3, service_level_factor=1.65):
    """
    Calculates safety stock and reorder point based on predictive insights.
    Service level factor 1.65 corresponds to roughly a 95% service level.
    """
    # Average daily predicted demand
    avg_lead_time_demand = predicted_demand * lead_time_days

    # Safety Stock calculation based on demand variability
    # Using a simplified standard deviation of predictions as uncertainty profile
    demand_uncertainty = np.std(predicted_demand)
    safety_stock = service_level_factor * \
        demand_uncertainty * np.sqrt(lead_time_days)

    # Reorder Point (ROP) = Lead Time Demand + Safety Stock
    reorder_point = avg_lead_time_demand + safety_stock

    return round(safety_stock), round(reorder_point)


# 4. EXECUTION PIPELINE
if __name__ == "__main__":
    print("--- Starting System Pipeline Execution ---")

    # Step 1: Preprocess data
    raw_data = generate_mock_sales_data()
    processed_data = preprocess_features(raw_data)

    # Step 2: Model demand
    model, test_features = train_demand_model(processed_data)

    # Step 3: Generate next-period forecast
    latest_features = test_features.iloc[[-1]]
    next_day_forecast = model.predict(latest_features)[0]
    print(
        f"[Audit Log] Forecasted Demand for Next Period: {next_day_forecast:.2f} units")

    # Step 4: Run Optimization Engine
    safety_stock, reorder_point = calculate_inventory_targets(
        next_day_forecast)

    print("\n--- Final Audited Inventory Controls ---")
    print(f"Calculated Safety Stock: {safety_stock} units")
    print(f"Calculated Reorder Point (ROP): {reorder_point} units")
    print("--- Pipeline Execution Finished Successfully ---")
