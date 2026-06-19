"""
Interactive CLI for the Credit Risk Assessment system.

Collects one applicant's raw features, scores them via src.utils, and prints
a coloured report. All model, preprocessing and policy logic lives in
src/utils.py — this file is purely input/output.

Usage:
    python app.py
"""

from pathlib import Path

import numpy as np

from src.utils import (
    FEATURE_LABELS,
    RAW_FEATURES,
    load_artifacts,
    score_applicant,
)

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "notebooks" / "models"
PROC_DIR = ROOT / "notebooks" / "data" / "processed"

# Each entry: (feature_name, input hint shown to the user, allow blank for NaN).
PROMPTS = [
    ("RevolvingUtilizationOfUnsecuredLines", "ratio 0-1, e.g. 0.35",           False),
    ("age",                                  "integer years, e.g. 45",          False),
    ("NumberOfTime30-59DaysPastDueNotWorse", "count 0-13, e.g. 0",             False),
    ("DebtRatio",                            "ratio, e.g. 0.40",                False),
    ("MonthlyIncome",                        "USD per month, blank if unknown",  True),
    ("NumberOfOpenCreditLinesAndLoans",      "count, e.g. 8",                   False),
    ("NumberOfTimes90DaysLate",              "count 0-13, e.g. 0",             False),
    ("NumberRealEstateLoansOrLines",         "count, e.g. 1",                   False),
    ("NumberOfTime60-89DaysPastDueNotWorse", "count 0-13, e.g. 0",             False),
    ("NumberOfDependents",                   "count, blank if unknown",          True),
]

assert [p[0] for p in PROMPTS] == RAW_FEATURES, "PROMPTS drifted from RAW_FEATURES"

DECISION_COLOURS = {
    "APPROVE":       "\033[92m",
    "MANUAL REVIEW": "\033[93m",
    "DECLINE":       "\033[91m",
}
RESET = "\033[0m"


def collect_applicant() -> dict:
    """Interactively prompt for one applicant's raw features."""
    print("\nEnter the applicant's details (press Enter to skip optional fields):\n")
    applicant = {}
    for feature, hint, optional in PROMPTS:
        label = FEATURE_LABELS.get(feature, feature)
        while True:
            raw = input(f"  {label} ({hint}): ").strip()
            if raw == "" and optional:
                applicant[feature] = np.nan
                break
            try:
                applicant[feature] = float(raw)
                break
            except ValueError:
                suffix = " or leave blank" if optional else ""
                print(f"    [!] Please enter a number{suffix}.")
    return applicant


def print_report(rec: dict) -> None:
    """Print a coloured, human-readable recommendation."""
    decision = rec["decision"]
    colour = DECISION_COLOURS.get(decision, "")
    ec = rec["expected_costs"]

    print()
    print("=" * 56)
    print(f"  DECISION     : {colour}{decision}{RESET}")
    print(f"  P(default)   : {rec['probability']:.1%}")
    print(f"  Credit score : {rec['score']}")
    print(f"  Expected cost if approved : {ec['approve']:.3f}")
    print(f"  Expected cost if declined : {ec['decline']:.3f}")
    print()
    print("  Top risk factors:")
    for _, row in rec["reason_codes"].iterrows():
        arrow = "\u25b2" if row["direction"] == "increases risk" else "\u25bc"
        print(f"    {arrow} {row['feature']:<35} (value: {row['value']:.2f})")
    print("=" * 56)


def main() -> None:
    print("=" * 56)
    print("   Credit Risk Assessment System")
    print("   Calibrated XGBoost + SHAP reason codes")
    print("=" * 56)

    print("\nLoading model artifacts...", end=" ", flush=True)
    final_model, explainer, preprocessor, policy = load_artifacts(
        models_dir=MODELS_DIR, proc_dir=PROC_DIR
    )
    print("done.")
    print(
        f"Decision threshold  p* = {policy.threshold:.3f}  "
        f"(cost ratio C_FN:C_FP = {policy.cost_fn:.0f}:{policy.cost_fp:.0f})"
    )

    while True:
        applicant = collect_applicant()
        rec = score_applicant(
            applicant,
            final_model=final_model,
            explainer=explainer,
            preprocessor=preprocessor,
            policy=policy,
        )
        print_report(rec)

        if input("\nAssess another applicant? (y/n): ").strip().lower() != "y":
            break

    print("\nExiting. Goodbye.")


if __name__ == "__main__":
    main()