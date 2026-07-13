"""
CarDekho Car Price Prediction Model
====================================
End-to-end pipeline: data loading, cleaning/feature engineering, EDA,
model training (multiple regressors), evaluation, and persistence of the
best-performing model.

Run:
    python src/car_price_prediction.py

Outputs:
    outputs/figures/*.png   -> EDA & evaluation plots
    outputs/model/*.pkl     -> trained model, encoders, feature list
    outputs/metrics.json    -> final metrics for every model tried
"""

import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# 0. Paths & setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "car_data.csv")
FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "model")
METRICS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.json")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

sns.set_style("whitegrid")
RANDOM_STATE = 42


def savefig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()


# 1. Load data

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df



# 2. Data cleaning & feature engineering

def clean_and_engineer(df):
    df = df.copy()

    # --- Basic sanity checks ---
    print("Missing values per column:\n", df.isnull().sum())
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    df = df.drop_duplicates()

    # --- Feature engineering ---
    # Cars are more valuable the newer they are, so convert "Year" into
    # "Car_Age" (relative to the most recent year present in the data).
    current_year = df["Year"].max()
    df["Car_Age"] = current_year - df["Year"]

    # Drop columns that don't help a general price model:
    #   - Car_Name has too many unique values (98+) for a dataset this
    #     size and would cause massive one-hot sparsity / overfitting.
    #   - Year is now redundant with Car_Age.
    df = df.drop(columns=["Car_Name", "Year"])

    # --- Outlier handling (based on domain knowledge + quantile check) ---
    # Present_Price and Kms_Driven both have a long right tail (a few
    # premium/high-mileage cars). Cap at the 99th percentile instead of
    # deleting rows, to avoid losing valuable training data.
    for col in ["Present_Price", "Kms_Driven"]:
        cap = df[col].quantile(0.99)
        n_capped = (df[col] > cap).sum()
        df[col] = np.where(df[col] > cap, cap, df[col])
        print(f"Capped {n_capped} outliers in '{col}' at 99th percentile ({cap:.2f})")

    return df



# 3. Exploratory Data Analysis

def run_eda(df):
    # Target distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(df["Selling_Price"], kde=True, color="teal")
    plt.title("Distribution of Selling Price (Lakhs INR)")
    plt.xlabel("Selling Price")
    savefig("target_distribution.png")

    # Correlation heatmap (numeric features only)
    plt.figure(figsize=(7, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    savefig("correlation_heatmap.png")

    # Price vs Car Age
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Car_Age", y="Selling_Price", hue="Fuel_Type")
    plt.title("Selling Price vs Car Age")
    savefig("price_vs_age.png")

    # Price vs Present Price
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="Present_Price", y="Selling_Price", hue="Transmission")
    plt.title("Selling Price vs Present (Showroom) Price")
    savefig("price_vs_present_price.png")

    # Boxplots for categorical features
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, col in zip(axes, ["Fuel_Type", "Seller_Type", "Transmission"]):
        sns.boxplot(data=df, x=col, y="Selling_Price", ax=ax)
        ax.set_title(f"Selling Price by {col}")
    savefig("categorical_boxplots.png")

    print("EDA figures saved to outputs/figures/")



# 4. Preprocessing for modeling (encoding)

def encode_features(df):
    df = df.copy()
    encoders = {}
    categorical_cols = ["Fuel_Type", "Seller_Type", "Transmission"]

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# 5. Train / evaluate multiple models

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    print(f"{name:20s} | RMSE: {rmse:.3f} | MAE: {mae:.3f} | R2: {r2:.3f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}, preds


def train_and_compare(X_train, X_test, y_train, y_test):
    candidates = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Lasso": Lasso(alpha=0.01, random_state=RANDOM_STATE),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=8, random_state=RANDOM_STATE
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE
        ),
    }

    results = {}
    fitted_models = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        metrics, _ = evaluate_model(name, model, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = model

    return results, fitted_models


def tune_best_model(X_train, y_train, best_name):
    """Light hyperparameter search for the top-performing tree ensemble."""
    if best_name == "RandomForest":
        param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [6, 8, 12],
            "min_samples_leaf": [1, 2, 4],
        }
        base = RandomForestRegressor(random_state=RANDOM_STATE)
    elif best_name == "GradientBoosting":
        param_grid = {
            "n_estimators": [200, 300, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.05, 0.1],
        }
        base = GradientBoostingRegressor(random_state=RANDOM_STATE)
    else:
        return None

    grid = GridSearchCV(
        base, param_grid, cv=5, scoring="r2", n_jobs=-1, verbose=0
    )
    grid.fit(X_train, y_train)
    print(f"Best params for {best_name}: {grid.best_params_}")
    return grid.best_estimator_



# 6. Feature importance + prediction-vs-actual plots

def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="steelblue")
    plt.title("Feature Importance (Best Model)")
    plt.xlabel("Importance")
    savefig("feature_importance.png")


def plot_pred_vs_actual(y_test, preds, model_name):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.6, edgecolor="k")
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, "r--", label="Perfect prediction")
    plt.xlabel("Actual Selling Price")
    plt.ylabel("Predicted Selling Price")
    plt.title(f"Predicted vs Actual — {model_name}")
    plt.legend()
    savefig("predicted_vs_actual.png")


def plot_model_comparison(results):
    names = list(results.keys())
    r2s = [results[n]["r2"] for n in names]
    rmses = [results[n]["rmse"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.barplot(x=r2s, y=names, ax=axes[0], color="seagreen")
    axes[0].set_title("R² Score by Model (higher is better)")
    axes[0].set_xlabel("R²")

    sns.barplot(x=rmses, y=names, ax=axes[1], color="indianred")
    axes[1].set_title("RMSE by Model (lower is better)")
    axes[1].set_xlabel("RMSE (Lakhs INR)")
    savefig("model_comparison.png")


# 7. Main pipeline
def main():
    df_raw = load_data()
    df_clean = clean_and_engineer(df_raw)
    run_eda(df_clean)

    df_encoded, encoders = encode_features(df_clean)

    X = df_encoded.drop(columns=["Selling_Price"])
    y = df_encoded["Selling_Price"]
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print("\n--- Training candidate models ---")
    results, fitted_models = train_and_compare(X_train, X_test, y_train, y_test)
    plot_model_comparison(results)

    best_name = max(results, key=lambda n: results[n]["r2"])
    print(f"\nBest baseline model: {best_name} (R2={results[best_name]['r2']:.3f})")

    print("\n--- Hyperparameter tuning best model ---")
    tuned_model = tune_best_model(X_train, y_train, best_name)
    final_model = tuned_model if tuned_model is not None else fitted_models[best_name]

    final_metrics, preds = evaluate_model(f"{best_name} (tuned)", final_model, X_test, y_test)
    results[f"{best_name}_tuned"] = final_metrics

    plot_pred_vs_actual(y_test, preds, f"{best_name} (tuned)")
    plot_feature_importance(final_model, feature_names)

    # Persist model + encoders + feature order for inference
    joblib.dump(final_model, os.path.join(MODEL_DIR, "car_price_model.pkl"))
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved best model ({best_name}) and metrics to outputs/")
    print("Done.")


if __name__ == "__main__":
    main()
