"""
Shared logic for the Credit Risk Assessment project.

Single source of truth for everything used at inference time: the feature
contract, the leakage-safe preprocessing transform, the decision policy, the
scorecard scaling, and the helpers that load trained artifacts and score a raw
applicant. Both app.py and the notebooks import from here.

Paths are never hard-coded in this module; every path is passed in by the
caller so the module stays working-directory agnostic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


RAW_FEATURES: list[str] = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Columns that receive a binary "was this value missing?" indicator (notebook 02).
MISSING_FLAG_COLS: list[str] = ["MonthlyIncome", "NumberOfDependents"]

# Values 96 and 98 are sentinel codes in the dataset, not real late-payment counts.
PASTDUE_COLS: list[str] = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Derived (not hard-coded) so it can never silently drift from the lists above.
STORED_FEATURE_ORDER: list[str] = RAW_FEATURES + [
    f"{c}_missing" for c in MISSING_FLAG_COLS
]

FEATURE_LABELS: dict[str, str] = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit utilisation ratio",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days past due",
    "DebtRatio": "Debt-to-income ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines & loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real-estate loans / lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days past due",
    "NumberOfDependents": "Number of dependents",
    "MonthlyIncome_missing": "Monthly income not provided",
    "NumberOfDependents_missing": "Dependents not provided",
}


def get_feature_names() -> list[str]:
    """Return the raw input features the model expects, in order.

    Returns a fresh copy so callers cannot mutate the module-level list.
    """
    return list(RAW_FEATURES)


def load_model(model_path: str | Path):
    """Load a serialized (joblib) model or transformer from disk."""
    return joblib.load(model_path)


class CreditPreprocessor:
    """Reproduces the notebook-02 transform exactly.

    Applies, in order: binary missingness flags on raw values; replacement of
    sentinel codes (96/98) and impossible age == 0 with NaN; winsorisation of
    heavy-tailed ratios at train-derived 99th-percentile caps; median imputation
    with train-derived medians.

    All learned parameters (caps_, medians_) come from the training split only.
    Build with from_artifacts() to reuse the exact parameters notebook 02
    persisted, or with fit() to re-derive them from raw training rows.
    Notebook 05 proves faithfulness by reproducing the frozen X_test.csv exactly.
    """

    WINSOR_COLS = ["RevolvingUtilizationOfUnsecuredLines", "DebtRatio"]
    WINSOR_Q = 0.99
    PLACEHOLDERS = [96, 98]
    IMPUTE_COLS = ["age", "MonthlyIncome", "NumberOfDependents"] + PASTDUE_COLS

    def __init__(
        self,
        caps: dict[str, float] | None = None,
        medians: dict[str, float] | None = None,
        stored_order: list[str] = STORED_FEATURE_ORDER,
    ):
        self.caps_ = dict(caps) if caps else {}
        self.medians_ = dict(medians) if medians else {}
        self.stored_order = list(stored_order)

    @property
    def is_fitted(self) -> bool:
        return bool(self.caps_) and bool(self.medians_)

    @classmethod
    def from_artifacts(
        cls, caps_path: str | Path, imputer_path: str | Path
    ) -> "CreditPreprocessor":
        """Build from the exact parameters notebook 02 persisted.

        Reads winsor caps from caps.joblib and imputation medians from the
        fitted SimpleImputer (feature_names_in_ keeps names and statistics_
        aligned), so no raw training data is needed at serving time.
        """
        caps_dict = joblib.load(caps_path)
        imputer = joblib.load(imputer_path)
        caps = {c: float(caps_dict[c]) for c in cls.WINSOR_COLS}
        medians = {
            col: float(med)
            for col, med in zip(imputer.feature_names_in_, imputer.statistics_)
            if col in cls.IMPUTE_COLS
        }
        return cls(caps=caps, medians=medians)

    def fit(self, X_raw: pd.DataFrame) -> "CreditPreprocessor":
        """Learn caps and medians from raw training rows (notebook use).

        Mirrors notebook 02: structural clean first, then derive the 99th-
        percentile winsor caps and per-column medians on the cleaned data.
        """
        X = self._structural_clean(X_raw[RAW_FEATURES])
        self.caps_ = {}
        for c in self.WINSOR_COLS:
            self.caps_[c] = float(X[c].quantile(self.WINSOR_Q))
            X[c] = X[c].clip(upper=self.caps_[c])
        self.medians_ = {c: float(X[c].median()) for c in self.IMPUTE_COLS}
        return self

    def _structural_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace sentinel codes and impossible age with NaN (no learned params)."""
        X = X.copy()
        for c in PASTDUE_COLS:
            X[c] = X[c].mask(X[c].isin(self.PLACEHOLDERS))
        X["age"] = X["age"].mask(X["age"] == 0)
        return X

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Apply the full transform to one or many raw rows.

        Missingness flags are captured from the raw values before any cleaning,
        exactly as in notebook 02, so a NaN that was already present in the
        input is correctly flagged even after imputation fills it.
        """
        if not self.is_fitted:
            raise RuntimeError(
                "CreditPreprocessor is not fitted; build it with "
                "from_artifacts(...) or fit(...) before calling transform()."
            )
        X = X_raw[RAW_FEATURES].copy()
        flags = {f"{c}_missing": X[c].isna().astype(int) for c in MISSING_FLAG_COLS}
        X = self._structural_clean(X)
        for c in self.WINSOR_COLS:
            X[c] = X[c].clip(upper=self.caps_[c])
        for c in self.IMPUTE_COLS:
            X[c] = X[c].fillna(self.medians_[c])
        for name, col in flags.items():
            X[name] = col.values
        return X[self.stored_order]


@dataclass
class DecisionPolicy:
    """Three-tier lending policy over a calibrated probability of default.

    threshold (p*) is the cost-optimal Bayes threshold derived in notebook 04.
    decline_threshold is an explicit business parameter that sets the width of
    the manual-review band. Setting the two equal recovers a pure two-way rule.
    """

    threshold: float
    decline_threshold: float
    cost_fn: float
    cost_fp: float

    def __post_init__(self):
        if not 0.0 <= self.threshold <= self.decline_threshold <= 1.0:
            raise ValueError(
                "Require 0 <= threshold <= decline_threshold <= 1, got "
                f"threshold={self.threshold}, decline_threshold={self.decline_threshold}."
            )

    def decide(self, p: float) -> str:
        if p < self.threshold:
            return "APPROVE"
        if p < self.decline_threshold:
            return "MANUAL REVIEW"
        return "DECLINE"

    def expected_costs(self, p: float) -> dict[str, float]:
        """Expected unit cost of each hard action at probability p."""
        return {"approve": p * self.cost_fn, "decline": (1 - p) * self.cost_fp}


# Scorecard scaling: PDO = 20, base score 600 at 50:1 good:bad odds.
PDO, BASE_SCORE, BASE_ODDS = 20.0, 600.0, 50.0
_SCORE_FACTOR = PDO / np.log(2)
_SCORE_OFFSET = BASE_SCORE - _SCORE_FACTOR * np.log(BASE_ODDS)


def prob_to_score(p: float) -> int:
    """Map a calibrated default probability to a points-based credit score.

    Strictly increasing in the good:bad log-odds, so it preserves the model's
    ranking and the decision threshold maps to a single cut-off score.
    """
    p = min(max(p, 1e-6), 1 - 1e-6)  # guard log against 0 and 1
    return int(round(_SCORE_OFFSET + _SCORE_FACTOR * np.log((1 - p) / p)))


def load_artifacts(models_dir: str | Path, proc_dir: str | Path):
    """Load deployment artifacts and assemble the inference components.

    Args:
        models_dir: directory holding final_model.joblib, xgboost.joblib and
            decision_policy.json (outputs of notebooks 03 and 04).
        proc_dir: directory holding caps.joblib and imputer.joblib
            (outputs of notebook 02).

    Returns:
        (final_model, explainer, preprocessor, policy) where final_model is the
        calibrated classifier, explainer is a TreeSHAP explainer over the base
        XGBoost, preprocessor is a fitted CreditPreprocessor, and policy is the
        DecisionPolicy.

    Raises:
        SystemExit: if any required artifact is missing, with a clear message
            listing each missing file.
    """
    import shap  # local import: shap is heavy and only needed at load time

    models_dir, proc_dir = Path(models_dir), Path(proc_dir)
    required = {
        "final model": models_dir / "final_model.joblib",
        "base XGBoost (for SHAP)": models_dir / "xgboost.joblib",
        "decision policy": models_dir / "decision_policy.json",
        "winsor caps": proc_dir / "caps.joblib",
        "imputer": proc_dir / "imputer.joblib",
    }
    missing = {name: p for name, p in required.items() if not p.exists()}
    if missing:
        lines = "\n".join(f"  - {name}: {p}" for name, p in missing.items())
        raise SystemExit(
            "\n[ERROR] Required artifact(s) not found:\n"
            f"{lines}\n"
            "Run the notebooks in order (01 -> 02 -> 03 -> 04 -> 05) first.\n"
        )

    final_model = load_model(required["final model"])
    xgb_tuned = load_model(required["base XGBoost (for SHAP)"])

    with open(required["decision policy"], "r", encoding="utf-8") as fh:
        policy_dict = json.load(fh)

    preprocessor = CreditPreprocessor.from_artifacts(
        caps_path=required["winsor caps"],
        imputer_path=required["imputer"],
    )

    threshold = float(policy_dict["threshold"])
    policy = DecisionPolicy(
        threshold=threshold,
        decline_threshold=min(2 * threshold, 1.0),  # review-band width: business parameter
        cost_fn=float(policy_dict["cost_false_negative"]),
        cost_fp=float(policy_dict["cost_false_positive"]),
    )

    # TreeSHAP requires a tree model. The calibrated final_model is a monotone
    # wrapper, so running SHAP on the base XGBoost gives the correct sign and
    # relative magnitude of each reason code without disturbing the ranking.
    explainer = shap.TreeExplainer(xgb_tuned)

    return final_model, explainer, preprocessor, policy


def score_applicant(
    applicant: dict,
    *,
    final_model,
    explainer,
    preprocessor: CreditPreprocessor,
    policy: DecisionPolicy,
    top_n: int = 4,
) -> dict:
    """Score one raw applicant and return a full recommendation.

    Args:
        applicant: mapping from each name in RAW_FEATURES to its raw value.
            Missing income or dependents may be passed as None or NaN.
        top_n: number of SHAP reason codes to include in the output.

    Returns:
        dict with keys probability (calibrated float), score (int), decision
        (str), expected_costs (dict) and reason_codes (DataFrame).

    Raises:
        ValueError: if any required feature is absent from applicant.
    """
    missing = set(RAW_FEATURES) - set(applicant)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")

    x_raw = pd.DataFrame(
        [{k: applicant[k] for k in RAW_FEATURES}], columns=RAW_FEATURES
    )
    z = preprocessor.transform(x_raw)
    prob = float(final_model.predict_proba(z)[0, 1])
    shap_values = explainer(z)

    contrib = pd.DataFrame(
        {
            "feature": [FEATURE_LABELS.get(c, c) for c in z.columns],
            "value": z.iloc[0].values,
            "shap": shap_values.values[0],
        }
    )
    contrib["direction"] = np.where(
        contrib["shap"] >= 0, "increases risk", "lowers risk"
    )
    reason_codes = (
        contrib.reindex(contrib["shap"].abs().sort_values(ascending=False).index)
        .head(top_n)
        .reset_index(drop=True)
    )

    return {
        "probability": prob,
        "score": prob_to_score(prob),
        "decision": policy.decide(prob),
        "expected_costs": policy.expected_costs(prob),
        "reason_codes": reason_codes,
    }