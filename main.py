"""
main.py
-------
Orchestrates the full pipeline:
    1. Generate synthetic dataset
    2. Run EDA & create visualizations
    3. Train / evaluate ML model
    4. Launch the interactive predictor

Run everything:
    python main.py

Skip the interactive console (useful in automated pipelines):
    python main.py --no-predict
"""
import matplotlib
matplotlib.use("Agg")  # non-GUI backend — no tkinter needed

import sys
import os


def separator(title: str) -> None:
    width = 60
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


def main():
    no_predict = "--no-predict" in sys.argv

    # ── Step 1: Generate dataset ──────────────────────────────────────────
    separator("STEP 1 / 4 — Generating Dataset")
    from generate_dataset import build_dataset
    df = build_dataset()
    df.to_csv("student_performance.csv", index=False)
    print(f"  ✓  student_performance.csv  ({len(df):,} rows × {df.shape[1]} columns)")

    # ── Step 2: EDA & Visualizations ──────────────────────────────────────
    separator("STEP 2 / 4 — Exploratory Data Analysis & Visualizations")
    from analysis import load_data, print_eda, plot_scatter, plot_heatmap, plot_activity_bars
    df_clean = load_data("student_performance.csv")
    print_eda(df_clean)
    print("\n  Rendering plots …")
    plot_scatter(df_clean)
    plot_heatmap(df_clean)
    plot_activity_bars(df_clean)

    # ── Step 3: Model Training ─────────────────────────────────────────────
    separator("STEP 3 / 4 — Model Training & Evaluation")
    from model import train_and_save
    train_and_save()

    # ── Step 4: Interactive Predictor ──────────────────────────────────────
    if not no_predict:
        separator("STEP 4 / 4 — Interactive Console Predictor")
        from predict import load_model, run_predictor, banner
        model = load_model()
        banner()
        run_predictor(model)
    else:
        print("\n  Skipping interactive predictor  (--no-predict flag set).")
        separator("Pipeline complete")
        print("""
  Outputs:
    student_performance.csv   — dataset
    model.pkl                 — trained Random Forest
    scaler.pkl                — feature scaler
    plots/                    — all visualizations

  To launch the predictor separately:
    python predict.py
""")


if __name__ == "__main__":
    main()
