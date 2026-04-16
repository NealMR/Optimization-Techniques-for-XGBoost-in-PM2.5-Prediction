import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import os
import optuna

# Configuration
DATA_PATH = "data/city_day.csv"
RESULTS_DIR = "results"
FEATURE_COLS = ["PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "City"]
TARGET_COL = "PM2.5"
RANDOM_STATE = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_export():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    df.dropna(subset=[TARGET_COL], inplace=True)
    
    # Label Encode City
    le = LabelEncoder()
    df["City"] = le.fit_transform(df["City"].astype(str))
    
    # Median Imputation and save medians
    medians = {}
    for col in FEATURE_COLS:
        if col != "City":
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            medians[col] = float(median_val)
    
    # Add City list to medians for UI
    medians["Cities"] = list(le.classes_)
    
    with open(os.path.join(RESULTS_DIR, "feature_medians.json"), "w") as f:
        json.dump(medians, f, indent=4)
    print(f"Saved feature medians to {RESULTS_DIR}/feature_medians.json")
    
    joblib.dump(le, os.path.join(RESULTS_DIR, "city_encoder.joblib"))
    print(f"Saved city encoder to {RESULTS_DIR}/city_encoder.joblib")
    
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    print("Optimizing model with Optuna...")
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbosity": 0
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(np.mean((y_test - preds)**2))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30) # Reduced trials for speed
    
    print("Training final model with best params...")
    best_params = study.best_params
    best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
    
    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(X, y) # Train on full data
    
    final_model.save_model(os.path.join(RESULTS_DIR, "xgb_model.json"))
    print(f"Saved final model to {RESULTS_DIR}/xgb_model.json")

if __name__ == "__main__":
    train_and_export()
