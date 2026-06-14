"""
analysis.py
-----------
Exploratory Data Analysis + three publication-quality visualizations.

Outputs saved to  ./plots/
"""
import matplotlib
matplotlib.use("Agg")  # non-GUI backend — no tkinter needed

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# ── Palette & style ───────────────────────────────────────────────────────────
INDIGO   = "#4361EE"
CORAL    = "#F72585"
MINT     = "#4CC9F0"
AMBER    = "#F3A712"
SLATE    = "#1B1F3B"
OFF_WHITE = "#F6F7FB"
GREY_MID  = "#8D99AE"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.facecolor":     OFF_WHITE,
    "figure.facecolor":   OFF_WHITE,
    "axes.edgecolor":     GREY_MID,
    "axes.labelcolor":    SLATE,
    "text.color":         SLATE,
    "xtick.color":        SLATE,
    "ytick.color":        SLATE,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "figure.dpi":         150,
})


def load_data(path: str = "student_performance.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # ── basic cleaning ──────────────────────────────────────────────────────
    df.drop_duplicates(subset="StudentID", inplace=True)
    df.dropna(inplace=True)
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        q_low  = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)
    return df


def print_eda(df: pd.DataFrame) -> None:
    print("\n" + "═" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("═" * 60)
    print(f"\nShape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicate IDs  : {df.duplicated('StudentID').sum()}")
    print("\nDescriptive statistics:")
    print(df.describe().drop("count").T.round(2).to_string())
    print("\nCorrelation with FinalExamScore:")
    corr = df.select_dtypes("number").corr()["FinalExamScore"].drop("FinalExamScore").sort_values(ascending=False)
    print(corr.round(3).to_string())


# ── Plot 1: Scatter — Study Hours vs Final Score ───────────────────────────────
def plot_scatter(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # colour by attendance quartile
    quartile = pd.qcut(df["AttendancePercentage"], 4, labels=["Q1 (Low)", "Q2", "Q3", "Q4 (High)"])
    palette  = [MINT, INDIGO, AMBER, CORAL]

    for i, (label, grp) in enumerate(df.assign(quartile=quartile).groupby("quartile")):
        ax.scatter(grp["StudyHoursPerDay"], grp["FinalExamScore"],
                   s=18, alpha=0.55, color=palette[i], label=label, linewidths=0)

    # trend line
    m, b = np.polyfit(df["StudyHoursPerDay"], df["FinalExamScore"], 1)
    x_range = np.linspace(df["StudyHoursPerDay"].min(), df["StudyHoursPerDay"].max(), 200)
    ax.plot(x_range, m * x_range + b, color=SLATE, lw=2, ls="--", zorder=5)

    ax.set_title("Study Hours vs Final Exam Score\n(coloured by attendance quartile)", pad=12, fontweight="bold")
    ax.set_xlabel("Study Hours per Day")
    ax.set_ylabel("Final Exam Score")
    leg = ax.legend(title="Attendance", frameon=True, framealpha=0.9, edgecolor=GREY_MID)
    leg.get_title().set_fontsize(9)

    # annotation
    r = df["StudyHoursPerDay"].corr(df["FinalExamScore"])
    ax.text(0.97, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", fontsize=10, color=SLATE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=OFF_WHITE, edgecolor=GREY_MID, alpha=0.8))

    fig.tight_layout()
    fig.savefig("plots/01_scatter_study_vs_score.png", bbox_inches="tight")
    plt.close()
    print("  ✓  plots/01_scatter_study_vs_score.png")


# ── Plot 2: Correlation heatmap ────────────────────────────────────────────────
def plot_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes("number").drop(columns=["ParticipationInActivities"])
    # rename for legibility
    rename = {
        "StudyHoursPerDay":        "Study Hrs",
        "AttendancePercentage":    "Attendance %",
        "SleepHours":              "Sleep Hrs",
        "SocialMediaHours":        "Social Media",
        "PreviousExamScore":       "Prev Score",
        "InternetUsageHours":      "Internet Hrs",
        "FinalExamScore":          "Final Score",
    }
    num_df = num_df.rename(columns=rename)
    corr   = num_df.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)   # indigo ↔ coral

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=0.5, linecolor="white",
        ax=ax, square=True,
        cbar_kws={"shrink": 0.75, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Heatmap", pad=14, fontweight="bold")
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)
    fig.tight_layout()
    fig.savefig("plots/02_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓  plots/02_correlation_heatmap.png")


# ── Plot 3: Bar chart — Activities vs No Activities ────────────────────────────
def plot_activity_bars(df: pd.DataFrame) -> None:
    bins   = [0, 40, 55, 70, 85, 100]
    labels = ["<40", "40–55", "55–70", "70–85", "85+"]
    df     = df.copy()
    df["ScoreBand"] = pd.cut(df["FinalExamScore"], bins=bins, labels=labels, right=True)
    df["Group"]     = df["ParticipationInActivities"].map({1: "Participates", 0: "Does Not Participate"})

    counts = (df.groupby(["ScoreBand", "Group"], observed=True)["StudentID"]
                .count()
                .reset_index(name="Count"))

    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(labels))
    width  = 0.36

    for i, (group, color) in enumerate([("Participates", INDIGO), ("Does Not Participate", CORAL)]):
        vals = [counts.query("ScoreBand == @band and Group == @group")["Count"].sum()
                for band in labels]
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=group,
                      color=color, alpha=0.88, edgecolor="white", linewidth=0.8)
        # value labels on top
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        str(v), ha="center", va="bottom", fontsize=8.5, color=SLATE)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Final Exam Score Band")
    ax.set_ylabel("Number of Students")
    ax.set_title("Score Distribution: Activity Participants vs Non-Participants", pad=12, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.9, edgecolor=GREY_MID)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("plots/03_activity_bar_chart.png", bbox_inches="tight")
    plt.close()
    print("  ✓  plots/03_activity_bar_chart.png")


if __name__ == "__main__":
    df = load_data()
    print_eda(df)
    print("\nGenerating visualizations …")
    plot_scatter(df)
    plot_heatmap(df)
    plot_activity_bars(df)
    print("\nAll plots saved to  ./plots/")
