"""
predict.py
----------
Console-based student score predictor.
Loads the trained Random Forest model and lets any user get
a predicted Final Exam Score without needing to understand ML.

Usage
-----
    python predict.py
"""

import pickle
import sys
import os

MODEL_PATH  = "model.pkl"
SCALER_PATH = "scaler.pkl"   # kept for compatibility; RF doesn't use it

FEATURE_META = [
    {
        "key":   "StudyHoursPerDay",
        "label": "Study hours per day",
        "lo":    0.0,
        "hi":    16.0,
        "hint":  "e.g. 3.5",
        "cast":  float,
    },
    {
        "key":   "AttendancePercentage",
        "label": "Attendance percentage",
        "lo":    0.0,
        "hi":    100.0,
        "hint":  "0 – 100",
        "cast":  float,
    },
    {
        "key":   "SleepHours",
        "label": "Average sleep hours per day",
        "lo":    0.0,
        "hi":    24.0,
        "hint":  "e.g. 7.0",
        "cast":  float,
    },
    {
        "key":   "SocialMediaHours",
        "label": "Daily social media hours",
        "lo":    0.0,
        "hi":    24.0,
        "hint":  "e.g. 2.5",
        "cast":  float,
    },
    {
        "key":   "PreviousExamScore",
        "label": "Previous exam score",
        "lo":    0.0,
        "hi":    100.0,
        "hint":  "e.g. 72",
        "cast":  float,
    },
    {
        "key":   "ParticipationInActivities",
        "label": "Participates in extracurricular activities?",
        "lo":    0,
        "hi":    1,
        "hint":  "Enter 1 for Yes, 0 for No",
        "cast":  int,
    },
]

BAR  = "─" * 52
BOLD = "\033[1m"
DIM  = "\033[2m"
RESET= "\033[0m"
BLUE = "\033[34m"
GREEN= "\033[32m"
RED  = "\033[31m"
CYAN = "\033[36m"


def banner() -> None:
    print(f"""
{CYAN}{BOLD}
╔══════════════════════════════════════════════════════╗
║          STUDENT PERFORMANCE PREDICTOR               ║
║        Powered by Random Forest Regression           ║
╚══════════════════════════════════════════════════════╝
{RESET}
Enter details for a student and this tool will estimate
their Final Exam Score on a scale of 0 – 100.
Type  {BOLD}q{RESET}  at any prompt to quit.
""")


def prompt_float(meta: dict) -> float | int:
    """Prompt until a valid value in [lo, hi] is entered."""
    lo, hi, cast = meta["lo"], meta["hi"], meta["cast"]
    while True:
        raw = input(f"  {BOLD}{meta['label']}{RESET}  {DIM}({meta['hint']}){RESET}: ").strip()
        if raw.lower() == "q":
            print(f"\n{DIM}Exiting. Goodbye!{RESET}")
            sys.exit(0)
        try:
            val = cast(raw)
        except ValueError:
            print(f"  {RED}✗  Please enter a number.{RESET}")
            continue
        if not (lo <= val <= hi):
            print(f"  {RED}✗  Value must be between {lo} and {hi}.{RESET}")
            continue
        return val


def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"{RED}Model file '{MODEL_PATH}' not found.\n"
              f"Run  python model.py  first to train the model.{RESET}")
        sys.exit(1)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def grade_label(score: float) -> str:
    if score >= 85:  return f"{GREEN}Excellent   🟢{RESET}"
    if score >= 70:  return f"{BLUE}Good        🔵{RESET}"
    if score >= 55:  return f"{CYAN}Average     🟡{RESET}"
    return              f"{RED}Needs work  🔴{RESET}"


def run_predictor(model) -> None:
    while True:
        print(f"{BAR}")
        print(f"{BOLD}  New Prediction{RESET}\n")

        values = []
        for meta in FEATURE_META:
            values.append(prompt_float(meta))

        import numpy as np
        X = np.array(values).reshape(1, -1)
        predicted = float(model.predict(X)[0])
        predicted = round(max(0.0, min(100.0, predicted)), 1)

        grade = grade_label(predicted)
        print(f"""
{CYAN}{BAR}{RESET}
  {BOLD}Predicted Final Exam Score:{RESET}  {BOLD}{CYAN}{predicted}/100{RESET}
  Performance band :  {grade}
{CYAN}{BAR}{RESET}
""")

        again = input("  Predict another student? [y/N]: ").strip().lower()
        if again != "y":
            print(f"\n{DIM}Thanks for using the predictor. Goodbye!{RESET}\n")
            break


def main():
    banner()
    model = load_model()
    run_predictor(model)


if __name__ == "__main__":
    main()
