"""
Credit Risk Assessment — banking-oriented Streamlit interface.

The application provides two business-facing views:

1. Applicant decision
   - Applicant profile input
   - Calibrated probability of default
   - Credit score
   - Decision recommendation
   - Reason-code explanation

2. Portfolio risk view
   - Portfolio approval, review and decline rates
   - Bad rate among approved applicants
   - Default rate by score band
   - Business outcome matrix
   - Cut-off trade-off simulation
   - Portfolio risk drivers

Run from the project root:
    streamlit run streamlit_app.py
"""

from pathlib import Path
pip
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from src.utils import (
    FEATURE_LABELS,
    load_artifacts,
    load_model,
    prob_to_score,
    score_applicant,
)


st.set_page_config(
    page_title="Credit Risk Assessment",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "notebooks" / "models"
PROC_DIR = ROOT / "notebooks" / "data" / "processed"


INK = "#0F172A"
MUTED = "#475569"
BORDER = "#E2E8F0"
CANVAS = "#F1F5F9"
SURFACE = "#FFFFFF"
ACCENT = "#4338CA"

RISK_UP = "#DC2626"
RISK_DOWN = "#059669"

CLASS_COLORS = {
    "Repaid": "#2563EB",
    "Defaulted": "#DC2626",
}

DECISION_STYLE = {
    "APPROVE": {
        "fg": "#047857",
        "bg": "#ECFDF5",
        "bd": "#A7F3D0",
        "icon": "✓",
        "label": "Approve",
    },
    "MANUAL REVIEW": {
        "fg": "#B45309",
        "bg": "#FFFBEB",
        "bd": "#FDE68A",
        "icon": "!",
        "label": "Manual review",
    },
    "DECLINE": {
        "fg": "#B91C1C",
        "bg": "#FEF2F2",
        "bd": "#FECACA",
        "icon": "✕",
        "label": "Decline",
    },
}


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            color-scheme: light !important;
        }}

        .stApp,
        [data-testid="stAppViewContainer"] {{
            background: {CANVAS} !important;
            color: {INK} !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }}

        h1, h2, h3, h4, h5, h6, p, span, label, li {{
            color: {INK} !important;
        }}

        .app-title {{
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            color: {INK} !important;
        }}

        .app-subtitle {{
            color: {MUTED} !important;
            font-size: 0.95rem;
            margin-top: 0.15rem;
        }}

        section[data-testid="stSidebar"] {{
            background: {SURFACE} !important;
            border-right: 1px solid {BORDER};
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] h3 {{
            color: {INK} !important;
        }}

        section[data-testid="stSidebar"] input {{
            color: {INK} !important;
            -webkit-text-fill-color: {INK} !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="input"] {{
            background: #FFFFFF !important;
            border-color: #CBD5E1 !important;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
            background: #FFFFFF !important;
            border-color: #CBD5E1 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {{
            background: #F8FAFC !important;
            color: {INK} !important;
            border-color: #CBD5E1 !important;
        }}

        section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button svg {{
            color: {INK} !important;
            fill: {INK} !important;
            stroke: {INK} !important;
        }}

        .stButton button,
        .stFormSubmitButton button,
        div[data-testid="stFormSubmitButton"] button {{
            background: {ACCENT} !important;
            color: #FFFFFF !important;
            border: 1px solid {ACCENT} !important;
            border-radius: 10px !important;
            font-weight: 800 !important;
            padding: 0.6rem 0.9rem !important;
            width: 100% !important;
        }}

        .stButton button:hover,
        .stFormSubmitButton button:hover,
        div[data-testid="stFormSubmitButton"] button:hover {{
            background: #3730A3 !important;
            color: #FFFFFF !important;
            border-color: #3730A3 !important;
        }}

        .stButton button p,
        .stFormSubmitButton button p,
        div[data-testid="stFormSubmitButton"] button p {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }}

        .stButton button:disabled,
        .stFormSubmitButton button:disabled,
        div[data-testid="stFormSubmitButton"] button:disabled {{
            background: #CBD5E1 !important;
            color: #FFFFFF !important;
            border-color: #CBD5E1 !important;
            opacity: 1 !important;
        }}

        .section-label {{
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
            color: {MUTED} !important;
            margin: 0.2rem 0 0.6rem 0;
        }}

        .banner {{
            display: flex;
            align-items: center;
            gap: 1rem;
            border-radius: 16px;
            padding: 1.15rem 1.35rem;
            margin-bottom: 1.2rem;
            border: 1px solid;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}

        .banner .ic {{
            width: 46px;
            height: 46px;
            border-radius: 50%;
            flex-shrink: 0;
            color: #FFFFFF !important;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.35rem;
            font-weight: 900;
        }}

        .banner .lab {{
            font-size: 1.35rem;
            font-weight: 850;
            letter-spacing: -0.01em;
        }}

        .banner .sub {{
            font-size: 0.88rem;
            color: {MUTED} !important;
            margin-top: 0.1rem;
            line-height: 1.45;
        }}

        div[data-testid="stMetric"] {{
            background: {SURFACE} !important;
            border: 1px solid {BORDER};
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: {MUTED} !important;
            font-weight: 750 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: {INK} !important;
            font-weight: 850 !important;
        }}

        div[data-testid="stElementToolbar"] {{
            opacity: 1 !important;
        }}

        div[data-testid="stElementToolbar"] button {{
            background: #FFFFFF !important;
            color: {INK} !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.12) !important;
        }}

        div[data-testid="stElementToolbar"] button svg {{
            color: {INK} !important;
            fill: {INK} !important;
            stroke: {INK} !important;
        }}

        .vega-embed summary {{
            background: #FFFFFF !important;
            color: {INK} !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            padding: 2px 6px !important;
        }}

        .vega-embed .vega-actions {{
            background: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px !important;
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.12) !important;
        }}

        .vega-embed .vega-actions a {{
            color: {INK} !important;
        }}

        .expert-note {{
            background: #F8FAFC;
            border: 1px solid {BORDER};
            border-radius: 14px;
            padding: 0.85rem 1rem;
            color: {MUTED} !important;
            font-size: 0.88rem;
            line-height: 1.45;
        }}

        .footer {{
            color: {MUTED} !important;
            font-size: 0.78rem;
            text-align: center;
            margin-top: 2.5rem;
        }}

        div[data-testid="InputInstructions"] {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading model artifacts…")
def get_artifacts():
    try:
        return load_artifacts(models_dir=MODELS_DIR, proc_dir=PROC_DIR), None
    except SystemExit as exc:
        return None, str(exc)


@st.cache_data(show_spinner="Loading test set…")
def load_test_set():
    X = pd.read_csv(PROC_DIR / "X_test.csv")
    y = pd.read_csv(PROC_DIR / "y_test.csv").squeeze("columns").astype(int)
    return X, y


@st.cache_resource
def load_base_models():
    files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Random Forest": "random_forest.joblib",
        "XGBoost": "xgboost.joblib",
    }

    models = {}

    for name, filename in files.items():
        path = MODELS_DIR / filename

        if path.exists():
            models[name] = load_model(path)

    return models


@st.cache_data(show_spinner="Scoring the test set…")
def test_predictions():
    X, y = load_test_set()
    base_models = load_base_models()

    artifacts, _ = get_artifacts()
    final_model = artifacts[0]

    probabilities = {
        name: model.predict_proba(X)[:, 1]
        for name, model in base_models.items()
    }

    probabilities["XGBoost (calibrated)"] = final_model.predict_proba(X)[:, 1]

    return y.to_numpy(), probabilities


def _theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart
        .properties(background=SURFACE)
        .configure(background=SURFACE)
        .configure_view(
            fill=SURFACE,
            stroke=BORDER,
            strokeWidth=1,
        )
        .configure_axis(
            labelColor=MUTED,
            titleColor=MUTED,
            gridColor="#E2E8F0",
            domainColor=BORDER,
            tickColor=BORDER,
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor=INK,
            titleColor=MUTED,
            orient="bottom",
        )
        .configure_title(
            color=INK,
            fontSize=15,
            fontWeight="bold",
            anchor="start",
        )
    )


def show_chart(chart: alt.Chart) -> None:
    st.altair_chart(chart, use_container_width=True, theme=None)


def probability_meter(
    probability: float,
    threshold: float,
    decline_threshold: float,
    color: str,
) -> alt.Chart:
    track_df = pd.DataFrame({"x": [1.0]})
    fill_df = pd.DataFrame({"x": [probability]})
    rule_df = pd.DataFrame({
        "threshold": [threshold, decline_threshold],
        "label": ["Approval cut-off", "Decline cut-off"],
    })

    track = (
        alt.Chart(track_df)
        .mark_bar(color="#EEF2F7", cornerRadius=8, height=26)
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None,
            )
        )
    )

    fill = (
        alt.Chart(fill_df)
        .mark_bar(color=color, cornerRadius=8, height=26)
        .encode(
            x=alt.X(
                "x:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None,
            )
        )
    )

    rules = (
        alt.Chart(rule_df)
        .mark_rule(color=INK, strokeDash=[4, 3], size=1.5)
        .encode(
            x="threshold:Q",
            tooltip=[
                alt.Tooltip("label:N", title="Boundary"),
                alt.Tooltip("threshold:Q", title="Probability", format=".1%"),
            ],
        )
    )

    return _theme((track + fill + rules).properties(height=42))


def reason_code_chart(reason_df: pd.DataFrame) -> alt.Chart:
    df = reason_df.copy()

    df["abs_impact"] = df["shap"].abs()
    df["Effect"] = np.where(
        df["shap"] > 0,
        "Raises default risk",
        "Lowers default risk",
    )

    bars = (
        alt.Chart(df)
        .mark_bar(cornerRadius=5, size=24)
        .encode(
            x=alt.X(
                "shap:Q",
                title="Risk impact — left lowers risk, right raises risk",
                axis=alt.Axis(format="+.2f"),
            ),
            y=alt.Y(
                "feature:N",
                sort=alt.SortField("abs_impact", order="descending"),
                title=None,
            ),
            color=alt.Color(
                "Effect:N",
                title=None,
                scale=alt.Scale(
                    domain=["Raises default risk", "Lowers default risk"],
                    range=[RISK_UP, RISK_DOWN],
                ),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Risk factor"),
                alt.Tooltip("value:Q", title="Applicant value", format=".2f"),
                alt.Tooltip("Effect:N", title="Effect"),
                alt.Tooltip("shap:Q", title="Model impact", format="+.3f"),
            ],
        )
        .properties(height=max(180, 38 * len(df)))
    )

    zero_line = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="#94A3B8", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    return _theme(zero_line + bars)


def score_band_chart(y_true, calibrated) -> alt.Chart:
    scores = np.array([prob_to_score(p) for p in calibrated])

    bins = [300, 500, 560, 620, 680, 740, 800, 900]
    labels = [
        "Very high risk",
        "High risk",
        "Elevated risk",
        "Medium risk",
        "Low risk",
        "Very low risk",
        "Prime",
    ]

    df = pd.DataFrame({
        "Score": scores,
        "Defaulted": y_true,
    })

    df["Risk band"] = pd.cut(
        df["Score"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    agg = (
        df.groupby("Risk band", observed=False)
        .agg(
            Applicants=("Defaulted", "size"),
            Default_rate=("Defaulted", "mean"),
        )
        .reset_index()
    )

    chart = (
        alt.Chart(agg)
        .mark_bar(cornerRadius=5, color=ACCENT)
        .encode(
            x=alt.X(
                "Risk band:N",
                title="Credit score band",
                sort=labels,
                axis=alt.Axis(labelAngle=-25),
            ),
            y=alt.Y(
                "Default_rate:Q",
                title="Observed default rate",
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("Risk band:N", title="Score band"),
                alt.Tooltip("Applicants:Q", title="Applicants", format=","),
                alt.Tooltip(
                    "Default_rate:Q",
                    title="Observed default rate",
                    format=".1%",
                ),
            ],
        )
        .properties(height=320)
    )

    return _theme(chart)


def business_matrix_chart(
    y_true, calibrated, threshold: float, decline_threshold: float
) -> alt.Chart:
    decision = np.where(
        calibrated < threshold, "Approve",
        np.where(calibrated < decline_threshold, "Manual Review", "Decline"),
    )

    rows = []
    for dec in ["Approve", "Manual Review", "Decline"]:
        mask_dec = decision == dec
        for actual_val, actual_label in [(0, "Repaid"), (1, "Defaulted")]:
            mask_act = y_true == actual_val
            n = int((mask_dec & mask_act).sum())

            if dec == "Approve" and actual_label == "Repaid":
                meaning, kind = "Correct approval", "Good"
            elif dec == "Approve" and actual_label == "Defaulted":
                meaning, kind = "Approved defaulter", "Costly"
            elif dec == "Manual Review":
                meaning, kind = "Sent for review", "Neutral"
            elif dec == "Decline" and actual_label == "Repaid":
                meaning, kind = "Lost good customer", "Costly"
            else:
                meaning, kind = "Correct decline", "Good"

            rows.append({
                "Actual": actual_label,
                "Decision": dec,
                "Applicants": n,
                "Meaning": meaning,
                "Type": kind,
            })

    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("Decision:N", title=None,
                sort=["Approve", "Manual Review", "Decline"]),
        y=alt.Y("Actual:N", title=None,
                sort=["Repaid", "Defaulted"]),
    )

    cells = base.mark_rect(cornerRadius=6).encode(
        color=alt.Color("Type:N", legend=None,
            scale=alt.Scale(
                domain=["Good", "Neutral", "Costly"],
                range=["#DBEAFE", "#FEF9C3", "#FEE2E2"],
            )),
        tooltip=[
            alt.Tooltip("Actual:N"),
            alt.Tooltip("Decision:N"),
            alt.Tooltip("Meaning:N"),
            alt.Tooltip("Applicants:Q", format=","),
        ],
    )

    numbers = base.mark_text(
        fontSize=22, fontWeight="bold", color=INK, dy=-8,
    ).encode(text=alt.Text("Applicants:Q", format=","))

    labels = base.mark_text(
        fontSize=11, color=MUTED, dy=16,
    ).encode(text="Meaning:N")

    return _theme((cells + numbers + labels).properties(height=220))


def cutoff_tradeoff_chart(
    y_true,
    calibrated,
    policy,
    selected_threshold: float,
) -> alt.Chart:
    rows = []

    for threshold in np.linspace(0.01, 0.50, 70):
        approve = calibrated < threshold

        approval_rate = approve.mean()
        approved_bad_rate = y_true[approve].mean() if approve.sum() else np.nan

        rejected_good = ((~approve) & (y_true == 0)).sum()
        approved_default = (approve & (y_true == 1)).sum()

        expected_cost = (
            approved_default * policy.cost_fn
            + rejected_good * policy.cost_fp
        ) / len(y_true)

        normalized_expected_cost = expected_cost / max(
            policy.cost_fn,
            policy.cost_fp,
        )

        rows.extend([
            {
                "Threshold": threshold,
                "Metric": "Approval rate",
                "Value": approval_rate,
            },
            {
                "Threshold": threshold,
                "Metric": "Default rate among approved",
                "Value": approved_bad_rate,
            },
            {
                "Threshold": threshold,
                "Metric": "Expected cost index",
                "Value": normalized_expected_cost,
            },
        ])

    df = pd.DataFrame(rows)

    lines = (
        alt.Chart(df)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X(
                "Threshold:Q",
                title="Default probability cut-off",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "Value:Q",
                title="%",
                axis=alt.Axis(format=".0%", tickCount=6),
            ),
            color=alt.Color("Metric:N", title=None),
            tooltip=[
                alt.Tooltip("Threshold:Q", title="Cut-off", format=".1%"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".1%"),
            ],
        )
    )

    current_threshold = (
        alt.Chart(pd.DataFrame({"Threshold": [selected_threshold]}))
        .mark_rule(color=INK, strokeDash=[5, 4], size=2)
        .encode(
            x="Threshold:Q",
            tooltip=[
                alt.Tooltip(
                    "Threshold:Q",
                    title="Selected cut-off",
                    format=".1%",
                )
            ],
        )
    )

    return _theme((lines + current_threshold).properties(height=320))


def portfolio_driver_chart(xgb_model) -> alt.Chart:
    features = [
        FEATURE_LABELS.get(column, column)
        for column in xgb_model.feature_names_in_
    ]

    imp = pd.DataFrame({
        "Risk factor": features,
        "Influence": xgb_model.feature_importances_,
    })

    imp = imp.sort_values("Influence", ascending=False).head(10)

    chart = (
        alt.Chart(imp)
        .mark_bar(cornerRadius=5, size=22, color=ACCENT)
        .encode(
            x=alt.X(
                "Influence:Q",
                title="Relative influence on portfolio risk",
            ),
            y=alt.Y(
                "Risk factor:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                alt.Tooltip("Risk factor:N"),
                alt.Tooltip("Influence:Q", format=".3f"),
            ],
        )
        .properties(height=320)
    )

    return _theme(chart)


def roc_chart(y_true, proba: dict) -> alt.Chart:
    rows = []

    for model_name, p in proba.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        idx = np.linspace(0, len(fpr) - 1, 200).astype(int)

        rows.append(
            pd.DataFrame({
                "False positive rate": fpr[idx],
                "True positive rate": tpr[idx],
                "Model": model_name,
            })
        )

    df = pd.concat(rows, ignore_index=True)

    curves = (
        alt.Chart(df)
        .mark_line(strokeWidth=2)
        .encode(
            x=alt.X(
                "False positive rate:Q",
                title="False positive rate",
            ),
            y=alt.Y(
                "True positive rate:Q",
                title="True positive rate",
            ),
            color=alt.Color("Model:N", title=None),
            tooltip=[
                alt.Tooltip("Model:N"),
                alt.Tooltip("False positive rate:Q", format=".2f"),
                alt.Tooltip("True positive rate:Q", format=".2f"),
            ],
        )
    )

    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color=MUTED, strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )

    return _theme((diagonal + curves).properties(height=320))


def calibration_chart(y_true, raw, calibrated) -> alt.Chart:
    rows = []

    for label, p in [
        ("Raw XGBoost", raw),
        ("Calibrated XGBoost", calibrated),
    ]:
        observed, predicted = calibration_curve(
            y_true,
            p,
            n_bins=10,
            strategy="quantile",
        )

        rows.append(
            pd.DataFrame({
                "Mean predicted probability": predicted,
                "Observed default frequency": observed,
                "Variant": label,
            })
        )

    df = pd.concat(rows, ignore_index=True)

    lines = (
        alt.Chart(df)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X(
                "Mean predicted probability:Q",
                title="Mean predicted probability",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "Observed default frequency:Q",
                title="Observed default frequency",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "Variant:N",
                title=None,
                scale=alt.Scale(
                    domain=["Raw XGBoost", "Calibrated XGBoost"],
                    range=["#94A3B8", ACCENT],
                ),
            ),
            tooltip=[
                alt.Tooltip("Variant:N"),
                alt.Tooltip(
                    "Mean predicted probability:Q",
                    format=".1%",
                ),
                alt.Tooltip(
                    "Observed default frequency:Q",
                    format=".1%",
                ),
            ],
        )
    )

    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
        .mark_line(color=MUTED, strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )

    return _theme((diagonal + lines).properties(height=320))


def score_distribution_chart(y_true, calibrated) -> alt.Chart:
    scores = np.array([prob_to_score(p) for p in calibrated])

    df = pd.DataFrame({
        "Credit score": scores,
        "Outcome": np.where(y_true == 1, "Defaulted", "Repaid"),
    })

    chart = (
        alt.Chart(df)
        .mark_bar(opacity=0.65)
        .encode(
            x=alt.X(
                "Credit score:Q",
                bin=alt.Bin(maxbins=40),
                title="Credit score",
            ),
            y=alt.Y(
                "count():Q",
                stack=None,
                title="Applicants",
            ),
            color=alt.Color(
                "Outcome:N",
                title=None,
                scale=alt.Scale(
                    domain=list(CLASS_COLORS.keys()),
                    range=list(CLASS_COLORS.values()),
                ),
            ),
            tooltip=[
                alt.Tooltip("Outcome:N"),
                alt.Tooltip("count():Q", title="Applicants"),
            ],
        )
        .properties(height=320)
    )

    return _theme(chart)


def collect_inputs() -> dict | None:
    with st.sidebar:
        st.markdown("### Applicant profile")

        with st.form("applicant"):
            st.markdown(
                '<div class="section-label">Delinquency history</div>',
                unsafe_allow_html=True,
            )

            late_30 = st.number_input(
                "Times 30–59 days past due",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            late_60 = st.number_input(
                "Times 60–89 days past due",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            late_90 = st.number_input(
                "Times 90+ days late",
                min_value=0,
                max_value=20,
                value=0,
                step=1,
            )

            st.markdown(
                '<div class="section-label">Financial position</div>',
                unsafe_allow_html=True,
            )

            revolving = st.number_input(
                "Credit utilisation ratio",
                min_value=0.0,
                max_value=5.0,
                value=0.30,
                step=0.01,
                help="Balance divided by credit limit. Normally between 0 and 1.",
            )

            debt_ratio = st.number_input(
                "Debt-to-income ratio",
                min_value=0.0,
                max_value=10.0,
                value=0.35,
                step=0.01,
            )

            income = st.number_input(
                "Monthly income (USD)",
                min_value=0.0,
                value=None,
                step=100.0,
                placeholder="Leave blank if unknown",
            )

            open_lines = st.number_input(
                "Open credit lines & loans",
                min_value=0,
                max_value=60,
                value=8,
                step=1,
            )

            realestate = st.number_input(
                "Real-estate loans / lines",
                min_value=0,
                max_value=30,
                value=1,
                step=1,
            )

            st.markdown(
                '<div class="section-label">Demographics</div>',
                unsafe_allow_html=True,
            )

            age = st.slider(
                "Age",
                min_value=18,
                max_value=100,
                value=45,
            )

            dependents = st.number_input(
                "Number of dependents",
                min_value=0,
                value=None,
                step=1,
                placeholder="Leave blank if unknown",
            )

            submitted = st.form_submit_button("Assess applicant")

    if not submitted:
        return None

    return {
        "RevolvingUtilizationOfUnsecuredLines": revolving,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": np.nan if income is None else income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": realestate,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60,
        "NumberOfDependents": np.nan if dependents is None else dependents,
    }


def view_scoring(policy) -> None:
    if "result" not in st.session_state:
        st.markdown(
            f"""
            <div class="banner" style="background:{SURFACE}; border-color:{BORDER};">
                <div class="ic" style="background:{ACCENT};">→</div>
                <div>
                    <div class="lab" style="color:{INK};">Enter an applicant to begin</div>
                    <div class="sub">
                        Fill the profile in the sidebar and press
                        <b>Assess applicant</b>.
                        Current cost policy:
                        C<sub>FN</sub>:C<sub>FP</sub> =
                        {policy.cost_fn:.0f}:{policy.cost_fp:.0f};
                        approval cut-off p* = {policy.threshold:.1%};
                        manual review up to {policy.decline_threshold:.1%}.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="expert-note">
                This screen is designed as a credit decision cockpit:
                probability of default, score, decision recommendation and
                reason codes are shown in business language.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    result = st.session_state["result"]

    decision = result["decision"]
    style = DECISION_STYLE[decision]
    probability = result["probability"]

    st.markdown(
        f"""
        <div class="banner" style="background:{style['bg']}; border-color:{style['bd']};">
            <div class="ic" style="background:{style['fg']};">{style['icon']}</div>
            <div>
                <div class="lab" style="color:{style['fg']};">{style['label']}</div>
                <div class="sub">
                    Calibrated probability of default:
                    <b>{probability:.1%}</b>.
                    Approval cut-off:
                    <b>{policy.threshold:.1%}</b>.
                    Decline cut-off:
                    <b>{policy.decline_threshold:.1%}</b>.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    expected_costs = result["expected_costs"]
    lower_cost_action = (
        "Approve"
        if expected_costs["approve"] <= expected_costs["decline"]
        else "Decline"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Probability of default",
        f"{probability:.1%}",
        help="Calibrated probability of default.",
    )

    c2.metric(
        "Credit score",
        f"{result['score']}",
        help="Score transformed from probability of default.",
    )

    c3.metric(
        "Lower-cost action",
        lower_cost_action,
        help=(
            f"Expected cost — approve: {expected_costs['approve']:.2f}; "
            f"decline: {expected_costs['decline']:.2f}."
        ),
    )

    st.write("")

    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown(
            '<div class="section-label">Decision zone</div>',
            unsafe_allow_html=True,
        )

        show_chart(
            probability_meter(
                probability=probability,
                threshold=policy.threshold,
                decline_threshold=policy.decline_threshold,
                color=style["fg"],
            )
        )

        st.caption(
            f"Approve below {policy.threshold:.1%}; "
            f"manual review up to {policy.decline_threshold:.1%}; "
            f"decline above {policy.decline_threshold:.1%}."
        )

    with right:
        st.markdown(
            '<div class="section-label">Main reason codes</div>',
            unsafe_allow_html=True,
        )

        show_chart(reason_code_chart(result["reason_codes"]))

        st.caption(
            "Red factors increase estimated default risk. "
            "Green factors reduce estimated default risk."
        )


def view_portfolio(policy) -> None:
    base_models = load_base_models()

    if "XGBoost" not in base_models:
        st.info(
            "Portfolio risk analytics need the trained XGBoost model. "
            "Run the training notebooks first."
        )
        return

    y_true, proba = test_predictions()
    calibrated = proba["XGBoost (calibrated)"]

    st.markdown(
        '<div class="section-label">Decision policy simulation</div>',
        unsafe_allow_html=True,
    )

    p1, p2 = st.columns(2)

    with p1:
        approval_threshold_pct = st.slider(
            "Approval cut-off",
            min_value=1.0,
            max_value=50.0,
            value=round(float(policy.threshold) * 100, 1),
            step=0.5,
            format="%.1f%%",
            help="Applicants below this default probability are approved.",
        )

    with p2:
        decline_threshold_pct = st.slider(
            "Decline cut-off",
            min_value=approval_threshold_pct,
            max_value=80.0,
            value=max(
                round(float(policy.decline_threshold) * 100, 1),
                approval_threshold_pct,
            ),
            step=0.5,
            format="%.1f%%",
            help="Applicants above this default probability are declined.",
        )

    approval_threshold = approval_threshold_pct / 100
    decline_threshold = decline_threshold_pct / 100

    st.caption(
        f"Approve below {approval_threshold:.1%}; "
        f"manual review from {approval_threshold:.1%} to {decline_threshold:.1%}; "
        f"decline above {decline_threshold:.1%}."
    )

    approved = calibrated < approval_threshold
    review = (
        (calibrated >= approval_threshold)
        & (calibrated < decline_threshold)
    )
    declined = calibrated >= decline_threshold

    approval_rate = approved.mean()
    review_rate = review.mean()
    decline_rate = declined.mean()

    bad_rate_approved = (
        y_true[approved].mean()
        if approved.sum()
        else np.nan
    )

    st.write("")

    st.markdown(
        '<div class="section-label">Portfolio decision summary</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Approval rate",
        f"{approval_rate:.1%}",
        help="Share of applicants below the approval cut-off.",
    )

    c2.metric(
        "Review rate",
        f"{review_rate:.1%}",
        help="Share of applicants routed to manual review.",
    )

    c3.metric(
        "Decline rate",
        f"{decline_rate:.1%}",
        help="Share of applicants above the decline cut-off.",
    )

    c4.metric(
        "Bad rate among approved",
        f"{bad_rate_approved:.1%}" if not np.isnan(bad_rate_approved) else "n/a",
        help="Observed default rate among applicants that would be approved.",
    )

    st.write("")

    a, b = st.columns(2, gap="large")

    with a:
        st.markdown("**Observed default rate by credit score band**")
        show_chart(score_band_chart(y_true, calibrated))
        st.caption(
            "This shows whether lower score bands actually contain riskier borrowers."
        )

    with b:
        st.markdown(
            f"**Business outcome matrix at p* = {approval_threshold:.1%}**"
        )
        show_chart(
            business_matrix_chart(
                y_true=y_true,
                calibrated=calibrated,
                threshold=approval_threshold,
                decline_threshold=policy.decline_threshold,
            )
        )
        st.caption(
            "Approve · Manual Review · Decline — the three costly outcomes are "
            "approved defaulters and rejected good customers."
        )

    st.write("")

    c, d = st.columns(2, gap="large")

    with c:
        st.markdown("**Cut-off trade-off simulation**")
        show_chart(
            cutoff_tradeoff_chart(
                y_true=y_true,
                calibrated=calibrated,
                policy=policy,
                selected_threshold=approval_threshold,
            )
        )
        st.caption(
            "Shows how approval rate, approved-portfolio bad rate and expected cost "
            "change when the default-probability cut-off changes."
        )

    with d:
        st.markdown("**Portfolio risk drivers**")
        show_chart(portfolio_driver_chart(base_models["XGBoost"]))
        st.caption(
            "Business-facing view of the variables that most influence portfolio risk."
        )

    st.write("")

    st.markdown("**Score separation by outcome**")
    show_chart(score_distribution_chart(y_true, calibrated))
    st.caption(
        "Good separation means defaulted borrowers concentrate at lower scores "
        "than borrowers who repaid."
    )

    st.write("")

    with st.expander(
        "Technical model validation — for data science / model risk users",
        expanded=False,
    ):
        if len(base_models) < 4:
            st.info(
                "ROC comparison needs all trained base models: Logistic Regression, "
                "Decision Tree, Random Forest and XGBoost."
            )
        else:
            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("**ROC curves**")
                show_chart(
                    roc_chart(
                        y_true,
                        {
                            k: v
                            for k, v in proba.items()
                            if k != "XGBoost (calibrated)"
                        },
                    )
                )
                st.caption(
                    "Useful for model discrimination analysis. "
                    "Top-left is better; the diagonal is random guessing."
                )

            with right:
                st.markdown("**Probability calibration**")
                show_chart(
                    calibration_chart(
                        y_true,
                        proba["XGBoost"],
                        proba["XGBoost (calibrated)"],
                    )
                )
                st.caption(
                    "Useful for validating whether predicted probabilities "
                    "match observed default frequencies."
                )


def main() -> None:
    inject_css()

    st.markdown(
        '<div class="app-title">Credit Risk Assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="app-subtitle">'
        'Banking-oriented applicant decisioning · calibrated default probability · '
        'credit score · reason-code explanations'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    artifacts, error = get_artifacts()

    if error:
        st.error(error)
        st.stop()

    final_model, explainer, preprocessor, policy = artifacts

    applicant = collect_inputs()

    if applicant is not None:
        st.session_state["result"] = score_applicant(
            applicant,
            final_model=final_model,
            explainer=explainer,
            preprocessor=preprocessor,
            policy=policy,
            top_n=6,
        )

    scoring_tab, portfolio_tab = st.tabs([
        "Applicant decision",
        "Portfolio risk view",
    ])

    with scoring_tab:
        view_scoring(policy)

    with portfolio_tab:
        view_portfolio(policy)

    st.markdown(
        '<div class="footer">'
        'Demonstration model trained on the Kaggle “Give Me Some Credit” dataset · '
        'not a real lending decision.'
        '</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()