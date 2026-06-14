"""
model.py
--------
Trains, evaluates, and persists a Random Forest regression model that predicts
a student's FinalExamScore from six readily available features.

Usage
-----
    python model.py          # train, evaluate, save
"""
import matplotlib
matplotlib.use("Agg")  # non-GUI backend — no tkinter needed

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model     import LinearRegression, Ridge
from sklearn.model_selection  import train_test_split, cross_val_score, KFold
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "StudyHoursPerDay",
    "AttendancePercentage",
    "SleepHours",
    "SocialMediaHours",
    "PreviousExamScore",
    "ParticipationInActivities",
    # InternetUsageHours excluded: collinear with SocialMediaHours
]
TARGET_COL   = "FinalExamScore"
MODEL_PATH   = "model.pkl"
SCALER_PATH  = "scaler.pkl"
RANDOM_STATE = 42

SLATE     = "#1B1F3B"
INDIGO    = "#4361EE"
CORAL     = "#F72585"
OFF_WHITE = "#F6F7FB"


# ── helpers ───────────────────────────────────────────────────────────────────
def load_clean(path: str = "student_performance.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df.drop_duplicates(subset="StudentID", inplace=True)
    df.dropna(inplace=True)
    return df


def evaluate(model, X_test: np.ndarray, y_test: np.ndarray, label: str) -> dict:
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    rmse   = mean_squared_error(y_test, y_pred) ** 0.5
    r2     = r2_score(y_test, y_pred)
    print(f"\n  {label}")
    print(f"    MAE  : {mae:.3f}")
    print(f"    RMSE : {rmse:.3f}")
    print(f"    R²   : {r2:.4f}")
    return {"label": label, "mae": mae, "rmse": rmse, "r2": r2, "y_pred": y_pred}


def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray) -> None:
    os.makedirs("plots", exist_ok=True)
    residuals = y_test - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=OFF_WHITE)

    # Actual vs Predicted
    ax = axes[0]
    ax.set_facecolor(OFF_WHITE)
    ax.scatter(y_test, y_pred, s=14, alpha=0.45, color=INDIGO, linewidths=0)
    lo, hi = y_test.min(), y_test.max()
    ax.plot([lo, hi], [lo, hi], color=CORAL, lw=1.8, ls="--")
    ax.set_xlabel("Actual Score")
    ax.set_ylabel("Predicted Score")
    ax.set_title("Actual vs Predicted", fontweight="bold", color=SLATE)

    # Residual distribution
    ax = axes[1]
    ax.set_facecolor(OFF_WHITE)
    ax.hist(residuals, bins=40, color=INDIGO, edgecolor="white", alpha=0.85)
    ax.axvline(0, color=CORAL, lw=1.8, ls="--")
    ax.set_xlabel("Residual (Actual − Predicted)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution", fontweight="bold", color=SLATE)

    fig.suptitle("Random Forest — Model Diagnostics", fontsize=13, fontweight="bold", color=SLATE, y=1.01)
    fig.tight_layout()
    fig.savefig("plots/04_model_diagnostics.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓  plots/04_model_diagnostics.png")


def plot_feature_importance(model, feature_names: list) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=OFF_WHITE)
    ax.set_facecolor(OFF_WHITE)
    colors = [CORAL if importances[i] == max(importances) else INDIGO for i in idx]
    bars = ax.barh([feature_names[i] for i in idx], importances[idx],
                   color=colors, edgecolor="white", height=0.55)
    ax.set_xlabel("Mean Decrease in Impurity")
    ax.set_title("Feature Importances — Random Forest", fontweight="bold", color=SLATE)

    for bar, val in zip(bars, importances[idx]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=SLATE)

    ax.xaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("plots/05_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("  ✓  plots/05_feature_importance.png")


# ── main ──────────────────────────────────────────────────────────────────────
def train_and_save():
    df = load_clean()
    X  = df[FEATURE_COLS].values
    y  = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # Scale for linear models; RF doesn't need it but we keep a scaler for the predictor
    scaler  = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\n" + "═" * 60)
    print("  MODEL COMPARISON")
    print("═" * 60)

    # Benchmark models
    lr  = LinearRegression().fit(X_train_s, y_train)
    rdg = Ridge(alpha=1.0).fit(X_train_s, y_train)

    evaluate(lr,  X_test_s, y_test, "Linear Regression")
    evaluate(rdg, X_test_s, y_test, "Ridge Regression")

    # Primary model — Random Forest (no scaling needed)
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_leaf=3,
        max_features=0.6,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_result = evaluate(rf, X_test, y_test, "Random Forest ★")

    # 5-fold CV on RF
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_r2 = cross_val_score(rf, X, y, scoring="r2", cv=cv)
    print(f"\n  5-Fold CV R²: {cv_r2.mean():.4f}  ±  {cv_r2.std():.4f}")

    # Save artefacts
    with open(MODEL_PATH,  "wb") as f: pickle.dump(rf, f)
    with open(SCALER_PATH, "wb") as f: pickle.dump(scaler, f)
    print(f"\n  Model saved  →  {MODEL_PATH}")

    # Diagnostic plots
    print("\nGenerating model diagnostics …")
    plot_residuals(y_test, rf_result["y_pred"])
    plot_feature_importance(rf, FEATURE_COLS)

    return rf, rf_result


if __name__ == "__main__":
    train_and_save()
