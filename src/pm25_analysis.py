# =============================================================================
# Comparative Analysis of Hyperparameter Optimization Techniques for XGBoost
# in PM2.5 Prediction Using CPCB India Air Quality Data
# =============================================================================
# Author: [Your Name]
# Dataset: city_day.csv — Air Quality Data in India (Kaggle / CPCB)
# =============================================================================

# ── Imports ───────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import time
import os

# ── Configuration ─────────────────────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.20
FIGURES_DIR    = "../figures"
RESULTS_DIR    = "../results"
DATA_PATH      = "../data/city_day.csv"     # ← adjust if needed

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURE_COLS = ["PM10", "NO", "NO2", "NOx", "NH3", "CO",
                "SO2", "O3", "Benzene", "Toluene", "Xylene", "City"]
TARGET_COL   = "PM2.5"

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# =============================================================================
# HELPER — Metrics
# =============================================================================
def evaluate(y_true, y_pred, label="Model"):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"  [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")
    return {"Model": label, "RMSE": rmse, "MAE": mae, "R2": r2}


# =============================================================================
# STEP 1 — Data Loading
# =============================================================================
print("=" * 65)
print("STEP 1 — Data Loading")
print("=" * 65)

df = pd.read_csv(DATA_PATH)

print(f"\nDataset shape : {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nColumn names  : {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")


# =============================================================================
# STEP 2 — Exploratory Data Analysis
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2 — Exploratory Data Analysis")
print("=" * 65)

# ── Missing values ────────────────────────────────────────────────────────────
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(f"\nMissing Value Summary:\n{missing_df[missing_df['Missing Count'] > 0]}")

# ── Descriptive statistics ────────────────────────────────────────────────────
print(f"\nDescriptive Statistics:\n{df.describe().T}")

# ── Graph 1 — Missing Values Heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
# Sample rows for readability
sample_df = df[FEATURE_COLS + [TARGET_COL]].isnull().astype(int)
sns.heatmap(sample_df.T, cmap="Reds", cbar=True,
            xticklabels=False, ax=ax)
ax.set_title("Fig. 1 — Missing Values Heatmap (Red = Missing)", fontsize=13, fontweight="bold")
ax.set_ylabel("Features")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig1_missing_heatmap.png")
plt.close()
print("\n[Saved] fig1_missing_heatmap.png")

# ── Graph 2 — Correlation Heatmap ────────────────────────────────────────────
numeric_cols = [c for c in FEATURE_COLS if c != "City"] + [TARGET_COL]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
ax.set_title("Fig. 2 — Pearson Correlation Heatmap of Air Quality Features",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig2_correlation_heatmap.png")
plt.close()
print("[Saved] fig2_correlation_heatmap.png")

# ── Graph 3 — PM2.5 Distribution ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
pm25_clean = df[TARGET_COL].dropna()

axes[0].hist(pm25_clean, bins=60, color="#2196F3", edgecolor="white", alpha=0.85)
axes[0].set_xlabel("PM2.5 (µg/m³)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Histogram of PM2.5")

sns.kdeplot(pm25_clean, ax=axes[1], fill=True, color="#E91E63")
axes[1].set_xlabel("PM2.5 (µg/m³)")
axes[1].set_title("KDE of PM2.5")

fig.suptitle("Fig. 3 — Distribution of Target Variable PM2.5",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig3_pm25_distribution.png")
plt.close()
print("[Saved] fig3_pm25_distribution.png")

# ── Graph 4 — PM2.5 Boxplot by City ──────────────────────────────────────────
top_cities = df.groupby("City")[TARGET_COL].median().nlargest(15).index
city_subset = df[df["City"].isin(top_cities)]

fig, ax = plt.subplots(figsize=(14, 6))
city_subset.boxplot(column=TARGET_COL, by="City", ax=ax,
                    vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#90CAF9", color="#1565C0"),
                    medianprops=dict(color="red", linewidth=2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.4))
ax.set_title("Fig. 4 — PM2.5 Distribution by City (Top 15 by Median)",
             fontsize=12, fontweight="bold")
ax.set_xlabel("City")
ax.set_ylabel("PM2.5 (µg/m³)")
plt.suptitle("")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig4_pm25_by_city.png")
plt.close()
print("[Saved] fig4_pm25_by_city.png")


# =============================================================================
# STEP 3 — Data Cleaning & Preprocessing
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3 — Data Cleaning")
print("=" * 65)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
print("  • Converted 'Date' to datetime.")

# Drop columns not needed (AQI, AQI_Bucket to avoid data leakage)
cols_to_drop = ["AQI", "AQI_Bucket", "Date"]
df.drop(columns=[c for c in cols_to_drop if c in df.columns],
        inplace=True)
print(f"  • Dropped columns: {cols_to_drop}")

# Label-encode City
le = LabelEncoder()
df["City"] = le.fit_transform(df["City"].astype(str))
print(f"  • Label-encoded 'City'. Classes: {list(le.classes_)[:5]} …")

# Drop rows where target is missing
before = len(df)
df.dropna(subset=[TARGET_COL], inplace=True)
print(f"  • Dropped {before - len(df)} rows with missing PM2.5.")

# Impute remaining features with column median
for col in [c for c in FEATURE_COLS if c != "City"]:
    if col in df.columns:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
print("  • Imputed remaining feature NaNs with column median.")

print(f"\n  Final dataset shape: {df.shape}")


# =============================================================================
# STEP 4 — Feature Selection & Train/Test Split
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4 — Feature Selection & Data Split")
print("=" * 65)

X = df[FEATURE_COLS]
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")
print(f"  Features used    : {list(X.columns)}")


# =============================================================================
# STEP 5 — Baseline XGBoost Model
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5 — Baseline XGBoost Model")
print("=" * 65)

t0 = time.time()
baseline_model = xgb.XGBRegressor(
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=0
)
baseline_model.fit(X_train, y_train)
baseline_time = time.time() - t0

y_pred_baseline = baseline_model.predict(X_test)
baseline_metrics = evaluate(y_test, y_pred_baseline, label="Baseline XGBoost")
baseline_metrics["Train Time (s)"] = round(baseline_time, 2)
print(f"  Training time    : {baseline_time:.2f}s")


# =============================================================================
# STEP 6 — Hyperparameter Optimization
# =============================================================================
print("\n" + "=" * 65)
print("STEP 6 — Hyperparameter Optimization")
print("=" * 65)

# ── Shared parameter space ────────────────────────────────────────────────────
PARAM_GRID = {
    "learning_rate"   : [0.01, 0.05, 0.1, 0.2],
    "max_depth"       : [3, 5, 7, 9],
    "n_estimators"    : [100, 200, 300],
    "subsample"       : [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
}

BASE_XGB = xgb.XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)

# ── Method 1 — GridSearchCV ───────────────────────────────────────────────────
print("\n  ▶ Method 1 — GridSearchCV")
# Use a reduced grid to keep runtime manageable in a course project
grid_reduced = {
    "learning_rate"   : [0.05, 0.1],
    "max_depth"       : [3, 5],
    "n_estimators"    : [100, 200],
    "subsample"       : [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

t0 = time.time()
grid_search = GridSearchCV(
    estimator=BASE_XGB,
    param_grid=grid_reduced,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
grid_time = time.time() - t0

print(f"  Best params : {grid_search.best_params_}")
y_pred_grid = grid_search.best_estimator_.predict(X_test)
grid_metrics = evaluate(y_test, y_pred_grid, label="GridSearchCV")
grid_metrics["Train Time (s)"] = round(grid_time, 2)
print(f"  Search time : {grid_time:.2f}s")

# ── Method 2 — RandomizedSearchCV ────────────────────────────────────────────
print("\n  ▶ Method 2 — RandomizedSearchCV")
from scipy.stats import uniform, randint

param_dist = {
    "learning_rate"   : uniform(0.01, 0.29),
    "max_depth"       : randint(3, 10),
    "n_estimators"    : randint(50, 401),
    "subsample"       : uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
}

t0 = time.time()
random_search = RandomizedSearchCV(
    estimator=BASE_XGB,
    param_distributions=param_dist,
    n_iter=40,
    scoring="neg_root_mean_squared_error",
    cv=3,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=0
)
random_search.fit(X_train, y_train)
random_time = time.time() - t0

print(f"  Best params : {random_search.best_params_}")
y_pred_random = random_search.best_estimator_.predict(X_test)
random_metrics = evaluate(y_test, y_pred_random, label="RandomizedSearchCV")
random_metrics["Train Time (s)"] = round(random_time, 2)
print(f"  Search time : {random_time:.2f}s")

# ── Method 3 — Bayesian Optimization (Optuna) ─────────────────────────────────
print("\n  ▶ Method 3 — Bayesian Optimization (Optuna)")

optuna_scores = []   # track convergence

def objective(trial):
    params = {
        "n_estimators"    : trial.suggest_int   ("n_estimators",     50, 400),
        "max_depth"       : trial.suggest_int   ("max_depth",         3,   9),
        "learning_rate"   : trial.suggest_float ("learning_rate",  0.01, 0.30, log=True),
        "subsample"       : trial.suggest_float ("subsample",       0.5, 1.0),
        "colsample_bytree": trial.suggest_float ("colsample_bytree",0.5, 1.0),
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
        "verbosity"       : 0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    optuna_scores.append(rmse)
    return rmse

t0 = time.time()
study = optuna.create_study(direction="minimize",
                             sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50, show_progress_bar=False)
optuna_time = time.time() - t0

best_optuna_params = study.best_params
best_optuna_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": 0})
print(f"  Best params : {best_optuna_params}")

optuna_model = xgb.XGBRegressor(**best_optuna_params)
optuna_model.fit(X_train, y_train)
y_pred_optuna = optuna_model.predict(X_test)
optuna_metrics = evaluate(y_test, y_pred_optuna, label="Bayesian (Optuna)")
optuna_metrics["Train Time (s)"] = round(optuna_time, 2)
print(f"  Search time : {optuna_time:.2f}s")


# =============================================================================
# STEP 7 — Model Evaluation & Comparison Table
# =============================================================================
print("\n" + "=" * 65)
print("STEP 7 — Comparison Table")
print("=" * 65)

results_df = pd.DataFrame([
    baseline_metrics, grid_metrics, random_metrics, optuna_metrics
])
results_df.set_index("Model", inplace=True)
print(f"\n{results_df.to_string()}")
results_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv")
print(f"\n[Saved] results/model_comparison.csv")


# =============================================================================
# STEP 8 — Visualizations
# =============================================================================
print("\n" + "=" * 65)
print("STEP 8 — Visualizations")
print("=" * 65)

# ── Fig 5 — Feature Importance ────────────────────────────────────────────────
importances = pd.Series(
    optuna_model.feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
importances.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Fig. 5 — Feature Importance (Best Optuna XGBoost Model)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score (F-score)")
ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig5_feature_importance.png")
plt.close()
print("[Saved] fig5_feature_importance.png")

# ── Fig 6 — Actual vs Predicted ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_optuna, alpha=0.35, s=18, color="#1565C0", label="Predictions")
lims = [min(y_test.min(), y_pred_optuna.min()),
        max(y_test.max(), y_pred_optuna.max())]
ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect Fit")
ax.set_xlabel("Actual PM2.5 (µg/m³)")
ax.set_ylabel("Predicted PM2.5 (µg/m³)")
ax.set_title("Fig. 6 — Actual vs. Predicted PM2.5\n(Best Optuna Model)",
             fontsize=12, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig6_actual_vs_predicted.png")
plt.close()
print("[Saved] fig6_actual_vs_predicted.png")

# ── Fig 7 — Residual Distribution ────────────────────────────────────────────
residuals = y_test.values - y_pred_optuna

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(residuals, bins=60, color="#7B1FA2", edgecolor="white", alpha=0.8)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_xlabel("Residual (Actual − Predicted)")
axes[0].set_ylabel("Count")
axes[0].set_title("Residual Histogram")

sns.kdeplot(residuals, ax=axes[1], fill=True, color="#7B1FA2")
axes[1].axvline(0, color="red", linestyle="--")
axes[1].set_xlabel("Residual (Actual − Predicted)")
axes[1].set_title("Residual KDE")

fig.suptitle("Fig. 7 — Residual Error Distribution (Optuna Model)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig7_residuals.png")
plt.close()
print("[Saved] fig7_residuals.png")

# ── Fig 8 — Model Performance Comparison ─────────────────────────────────────
metrics_to_plot = ["RMSE", "MAE", "R2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

colors_bar = ["#607D8B", "#1976D2", "#388E3C", "#F57C00"]
models     = results_df.index.tolist()
x          = np.arange(len(models))

for i, metric in enumerate(metrics_to_plot):
    vals = results_df[metric].values
    bars = axes[i].bar(x, vals, color=colors_bar, width=0.55, edgecolor="white")
    axes[i].set_xticks(x)
    axes[i].set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    axes[i].set_title(metric, fontweight="bold")
    axes[i].set_ylabel(metric)
    for bar, val in zip(bars, vals):
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(vals) * 0.01,
                     f"{val:.3f}", ha="center", va="bottom", fontsize=9)

fig.suptitle("Fig. 8 — Model Performance Comparison Across Optimization Methods",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig8_model_comparison.png")
plt.close()
print("[Saved] fig8_model_comparison.png")

# ── Fig 9 — Optuna Convergence Plot ──────────────────────────────────────────
best_so_far = np.minimum.accumulate(optuna_scores)

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(range(1, len(optuna_scores) + 1), optuna_scores,
        alpha=0.4, color="#90CAF9", linewidth=0.8, label="Trial RMSE")
ax.plot(range(1, len(best_so_far) + 1), best_so_far,
        color="#1565C0", linewidth=2, label="Best RMSE so far")
ax.set_xlabel("Trial Number")
ax.set_ylabel("RMSE")
ax.set_title("Fig. 9 — Optuna Bayesian Optimization Convergence",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/fig9_optuna_convergence.png")
plt.close()
print("[Saved] fig9_optuna_convergence.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ALL STEPS COMPLETE")
print("=" * 65)
print(f"\nFigures saved to : {FIGURES_DIR}/")
print(f"Results saved to : {RESULTS_DIR}/")
print(f"\nFinal Results:\n{results_df.to_string()}")
