# Student Performance Predictor

A machine learning project that models how study habits, lifestyle choices, and
prior academic history combine to shape a student's final exam score.

---

## Project Overview

Students often wonder which habits actually move the needle on exam day.
This project answers that question empirically: it generates a synthetic but
realistic dataset of 1,200 students, runs exploratory analysis on it, builds a
Random Forest model (R² ≈ 0.93), and wraps everything in a clean console
interface so anyone can get a personalised score prediction in seconds.

---

## Dataset

The dataset is **synthetically generated** using NumPy and saved as
`student_performance.csv`.

| Column | Description | Range |
|---|---|---|
| `StudentID` | Unique identifier | STU0001 … STU1200 |
| `StudyHoursPerDay` | Hours of focused study daily | 0.5 – 10 h |
| `AttendancePercentage` | Lectures attended | 40 – 100 % |
| `SleepHours` | Average sleep per night | 4 – 10 h |
| `SocialMediaHours` | Daily social media time | 0 – 8 h |
| `PreviousExamScore` | Most recent prior exam mark | 20 – 100 |
| `ParticipationInActivities` | Extracurricular involvement | 0 / 1 |
| `InternetUsageHours` | Total daily internet use | 0.5 – 12 h |
| `FinalExamScore` | **Target variable** | 0 – 100 |

The generation formula encodes realistic relationships — study hours have the
strongest effect, social media acts as a drag, and sleep benefits follow a
quadratic sweet-spot around 7 hours.

---

## Libraries

| Library | Purpose |
|---|---|
| `numpy` | Synthetic data generation, numerical ops |
| `pandas` | DataFrame manipulation, I/O |
| `matplotlib` | Custom visualisation |
| `seaborn` | Heatmap, colour palettes |
| `scikit-learn` | ML models, evaluation, preprocessing |

---

## Project Structure

```
student_performance/
├── generate_dataset.py   # Builds and saves the CSV
├── analysis.py           # EDA + three visualisations
├── model.py              # Model training, evaluation, persistence
├── predict.py            # Interactive console predictor
├── main.py               # Orchestrates all four steps
├── student_performance.ipynb   # Full walkthrough notebook
├── student_performance.csv     # Generated dataset
├── model.pkl             # Saved Random Forest model
└── plots/
    ├── 01_scatter_study_vs_score.png
    ├── 02_correlation_heatmap.png
    ├── 03_activity_bar_chart.png
    ├── 04_model_diagnostics.png
    └── 05_feature_importance.png
```

---

## How to Run

### 1 — Install dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 2 — Run the full pipeline

```bash
python main.py
```

This will:
1. Generate the dataset → `student_performance.csv`
2. Run EDA and save five plots to `plots/`
3. Train the Random Forest model → `model.pkl`
4. Launch the interactive console predictor

### 3 — Predictor only (after training)

```bash
python predict.py
```

### 4 — Notebook

Open `student_performance.ipynb` in Jupyter Lab / Notebook. All five steps are
reproduced inline with rendered plots.

---

## Key Insights from EDA

**Study hours are the dominant predictor**  
Pearson r = 0.81 with final score — roughly 14 points of score improvement per
additional hour of daily study.

**Social media and internet usage hurt performance**  
r ≈ −0.17. The effect is moderate but consistent across all attendance and sleep
groups. Even students who study a lot see a measurable drag from high social
media exposure.

**Sleep follows a Goldilocks curve**  
Students sleeping roughly 6.5–8 hours per night outperform both the sleep-deprived
(< 5 h) and the oversleepers (> 9 h).

**Extracurricular participation lifts scores**  
Participants average 4–5 points higher in every score band — likely reflecting
time-management skills and broader cognitive engagement.

**Previous exam score matters, but less than expected**  
r = 0.14 — prior marks contribute, but current study habits overwhelm historical
performance in the model's feature importances.

---

## Model Performance Summary

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 7.99 | 9.81 | 0.717 |
| Ridge Regression | 7.99 | 9.81 | 0.718 |
| **Random Forest ★** | **2.75** | **4.91** | **0.929** |

**5-Fold CV R²: 0.934 ± 0.011** — consistent generalisation, no overfitting.

The gap between linear and tree-based models confirms that interactions
(e.g. high study hours offsetting poor attendance) are non-linear and
worth capturing with an ensemble method.

---

## Sample Prediction

```
Study hours per day          (e.g. 3.5): 6.5
Attendance percentage        (0 – 100):  88
Average sleep hours per day  (e.g. 7.0): 7
Daily social media hours     (e.g. 2.5): 1.5
Previous exam score          (e.g. 72):  74
Participates in activities?  (1/0):      1

──────────────────────────────────────────────
Predicted Final Exam Score:  91.4 / 100
Performance band:  Excellent 🟢
──────────────────────────────────────────────
```

---

## License

MIT — free for personal and academic use.
