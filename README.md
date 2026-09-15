Predictive Retail Inventory Optimization System
Description
This repository contains a machine learning pipeline designed to optimize retail inventory levels. By accurately forecasting daily demand, the system calculates optimal safety stock and reorder points to prevent stockouts and reduce excess inventory.
Key Features
 * Data Simulation & Preprocessing: Generates mock sales data featuring weekly seasonality and random noise. Engineers time-series features such as day-of-week and month, and creates lag features based on previous days' sales.
 * Demand Forecasting: Trains a Random Forest Regressor model to predict future retail demand based on engineered time-series features. Evaluates model performance using Root Mean Squared Error (RMSE).
 * Dynamic Inventory Logic Engine: Calculates required safety stock based on demand variability and a specified service level factor. Determines the reorder point (ROP) by combining average lead-time demand and safety stock.
 * Execution Pipeline: Automates the entire process from data preprocessing and model training to forecasting and inventory replenishment recommendations.
Technologies Used
 * Python: Core programming language.
 * NumPy: For numerical computations and mock data generation.
 * Pandas: For data manipulation and feature engineering.
 * Scikit-learn: For machine learning model training (Random Forest), data splitting, and evaluation metrics.
How to Run
 * Ensure you have the necessary libraries installed (pip install numpy pandas scikit-learn).
 * Clone this repository and navigate to the project directory.
 * Run the main execution script: python <script_name>.py.
 * The system will output the model evaluation results, followed by the forecasted demand for the next period and the final audited inventory controls (Safety Stock and Reorder Point).
Future Enhancements
 * Incorporate promotional events, holidays, and pricing data into the forecasting model.
 * Implement hyperparameter tuning to optimize the Random Forest Regressor.
 * Connect to a real-world database or ERP system for live inventory data integration.
