"""
generate_dataset.py
-------------------
Creates a synthetic but realistic student performance dataset.

The final exam score is modelled as a weighted combination of
study habits, lifestyle factors, and prior academic record —
with moderate noise so the relationship is learnable but not trivial.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
N_STUDENTS = 1200


def build_dataset(n: int = N_STUDENTS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n + 1)]

    # --- Core predictors ---------------------------------------------------
    study_hours = rng.uniform(0.5, 10.0, n).round(1)
    attendance  = rng.uniform(40.0, 100.0, n).round(1)
    sleep_hours = rng.normal(6.8, 1.1, n).clip(4.0, 10.0).round(1)
    social_media = rng.exponential(2.5, n).clip(0.0, 8.0).round(1)
    prev_score   = rng.normal(65.0, 15.0, n).clip(20.0, 100.0).round(1)
    participation = rng.choice([0, 1], n, p=[0.38, 0.62])
    internet_hours = (social_media + rng.uniform(0.5, 3.0, n)).clip(0.5, 12.0).round(1)

    # --- Target: FinalExamScore --------------------------------------------
    # Realistic weighted formula + noise
    noise = rng.normal(0, 5.0, n)

    # Sleep has a sweet-spot around 7 h — penalise extremes
    sleep_penalty = -2.0 * (sleep_hours - 7.0) ** 2

    raw = (
        14.0 * study_hours          # strong positive
        + 0.30 * attendance          # moderate positive
        + sleep_penalty              # quadratic sleep effect
        - 2.5 * social_media         # negative distraction
        + 0.28 * prev_score          # prior performance
        + 4.0 * participation        # bonus for activities
        - 0.8 * internet_hours       # slight negative
        + noise
        + 5.0                        # base intercept
    )

    final_score = raw.clip(10.0, 100.0).round(1)

    df = pd.DataFrame({
        "StudentID":               student_ids,
        "StudyHoursPerDay":        study_hours,
        "AttendancePercentage":    attendance,
        "SleepHours":              sleep_hours,
        "SocialMediaHours":        social_media,
        "PreviousExamScore":       prev_score,
        "ParticipationInActivities": participation,
        "InternetUsageHours":      internet_hours,
        "FinalExamScore":          final_score,
    })

    return df


if __name__ == "__main__":
    df = build_dataset()
    df.to_csv("student_performance.csv", index=False)
    print(f"Dataset saved  →  student_performance.csv  ({len(df)} rows)")
    print(df.describe().T.to_string())
