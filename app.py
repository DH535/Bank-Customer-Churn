"""
Bank Customer Churn Prediction
--------------------------------
Production-oriented Streamlit application for customer churn risk assessment,
batch analysis, and model insights.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Application configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Bank Customer Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Deployment" / "best_gb.pkl"

FEATURE_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
    "Geography_Germany",
    "Geography_Spain",
    "Gender_Male",
]

BATCH_USER_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
    "Satisfaction Score",
    "Card Type",
    "Point Earned",
]

GEOGRAPHY_MAP = {
    "France": (0, 0),
    "Germany": (1, 0),
    "Spain": (0, 1),
}

GENDER_MAP = {
    "Female": 0,
    "Male": 1,
}

CARD_TYPE_MAP = {
    "SILVER": 1,
    "GOLD": 2,
    "PLATINUM": 3,
    "DIAMOND": 4,
}

DISPLAY_FEATURE_NAMES = {
    "CreditScore": "Credit score",
    "Geography_Germany": "Germany",
    "Geography_Spain": "Spain",
    "Gender_Male": "Male",
    "Age": "Age",
    "Tenure": "Tenure",
    "Balance": "Balance",
    "NumOfProducts": "Number of products",
    "HasCrCard": "Credit card",
    "IsActiveMember": "Active membership",
    "EstimatedSalary": "Estimated salary",
    "Satisfaction Score": "Satisfaction score",
    "Card Type": "Card type",
    "Point Earned": "Points earned",
}

# Background sample size used for permutation-based SHAP explanations.
SHAP_BACKGROUND_SIZE = 100
SHAP_BACKGROUND_SEED = 42

# Optional: if a real reference dataset is dropped here, it is used instead
# of the synthetic background (see load_background_sample()).
BACKGROUND_CSV_PATH = BASE_DIR / "Deployment" / "background_sample.csv"


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def inject_css() -> None:
    """Inject the application's visual system."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {
            --ink: #16212b;
            --muted: #697782;
            --line: #e7ebee;
            --surface: #ffffff;
            --surface-soft: #f7f9fa;
            --navy: #102a43;
            --teal: #0f766e;
            --teal-soft: #e8f5f3;
            --red: #c2413a;
            --red-soft: #fdf0ef;
            --amber: #b7791f;
            --amber-soft: #fff7e8;
            --shadow: 0 8px 30px rgba(16, 42, 67, 0.07);
        }

        html, body, [class*="css"] {
            font-family: "DM Sans", sans-serif;
        }

        .stApp {
            background: #f5f7f8;
            color: var(--ink);
        }

        [data-testid="stHeader"] {
            background: rgba(245, 247, 248, 0.92);
        }

        [data-testid="stSidebar"] {
            background: #102a43;
            border-right: none;
        }

        [data-testid="stSidebar"] * {
            color: #eef4f7;
        }

        [data-testid="stSidebar"] .stRadio label {
            padding: 8px 10px;
            border-radius: 9px;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: rgba(255, 255, 255, 0.08);
        }

        .brand {
            padding: 14px 6px 26px 6px;
        }

        .brand-mark {
            width: 38px;
            height: 38px;
            border: 1px solid rgba(255,255,255,.2);
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-family: "Manrope", sans-serif;
            font-weight: 800;
            font-size: 17px;
            margin-bottom: 12px;
            background: rgba(255,255,255,.07);
        }

        .brand-name {
            font-family: "Manrope", sans-serif;
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -.4px;
        }

        .brand-caption {
            color: #aebdc7;
            font-size: 12px;
            margin-top: 4px;
        }

        .page-kicker {
            color: var(--teal);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 1.3px;
            text-transform: uppercase;
            margin-bottom: 6px;
        }

        .page-title {
            font-family: "Manrope", sans-serif;
            font-size: clamp(28px, 3vw, 42px);
            font-weight: 800;
            line-height: 1.08;
            letter-spacing: -1.3px;
            color: var(--navy);
            margin: 0;
        }

        .page-subtitle {
            color: var(--muted);
            font-size: 15px;
            max-width: 760px;
            margin-top: 10px;
            margin-bottom: 26px;
            line-height: 1.55;
        }

        .section-heading {
            font-family: "Manrope", sans-serif;
            color: var(--navy);
            font-size: 21px;
            font-weight: 700;
            letter-spacing: -.35px;
            margin: 10px 0 4px;
        }

        .section-caption {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 15px;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 20px;
            box-shadow: var(--shadow);
        }

        .kpi-card {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 14px;
            padding: 18px 19px;
            min-height: 116px;
            box-shadow: 0 5px 22px rgba(16, 42, 67, 0.045);
        }

        .kpi-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: .7px;
        }

        .kpi-value {
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 27px;
            font-weight: 800;
            margin-top: 8px;
            letter-spacing: -.8px;
        }

        .kpi-note {
            color: #84919a;
            font-size: 12px;
            margin-top: 4px;
        }

        .risk-panel {
            border-radius: 18px;
            padding: 22px;
            border: 1px solid var(--line);
            background: white;
            box-shadow: var(--shadow);
        }

        .risk-high {
            border-left: 5px solid var(--red);
            background: linear-gradient(90deg, #fff9f8, #ffffff 70%);
        }

        .risk-moderate {
            border-left: 5px solid var(--amber);
            background: linear-gradient(90deg, #fffaf0, #ffffff 70%);
        }

        .risk-low {
            border-left: 5px solid var(--teal);
            background: linear-gradient(90deg, #f5fbfa, #ffffff 70%);
        }

        .risk-eyebrow {
            color: var(--muted);
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 700;
        }

        .risk-title {
            font-family: "Manrope", sans-serif;
            font-size: 31px;
            line-height: 1.1;
            font-weight: 800;
            color: var(--navy);
            margin: 6px 0;
        }

        .risk-description {
            color: var(--muted);
            font-size: 14px;
        }

        .insight-chip {
            display: inline-block;
            padding: 6px 9px;
            border-radius: 999px;
            background: var(--surface-soft);
            color: var(--muted);
            border: 1px solid var(--line);
            font-size: 11px;
            font-weight: 600;
            margin: 3px 4px 3px 0;
        }

        .empty-state {
            text-align: center;
            padding: 55px 25px;
            background: white;
            border: 1px dashed #d5dde2;
            border-radius: 18px;
        }

        .empty-state-title {
            font-family: "Manrope", sans-serif;
            color: var(--navy);
            font-size: 20px;
            font-weight: 700;
        }

        .empty-state-copy {
            color: var(--muted);
            font-size: 13px;
            max-width: 550px;
            margin: 8px auto 0;
            line-height: 1.6;
        }

        .footnote {
            color: #82909a;
            font-size: 11px;
            line-height: 1.55;
        }

        div.stButton > button {
            border-radius: 10px;
            min-height: 44px;
            font-weight: 700;
            border: none;
            background: #0f766e;
            color: white;
        }

        div.stButton > button:hover {
            background: #0b5f59;
            color: white;
        }

        .stDownloadButton > button {
            border-radius: 10px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: white;
            border: 1px solid var(--line);
            padding: 15px;
            border-radius: 14px;
        }

        .profile-label {
            color: var(--muted);
            font-size: 12px;
            margin-bottom: 2px;
        }

        .profile-value {
            color: var(--navy);
            font-family: "Manrope", sans-serif;
            font-size: 17px;
            font-weight: 700;
        }

        hr {
            border-color: var(--line);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Model loading and inspection
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_artifacts() -> Any:
    """
    Load the complete trained inference pipeline.

    The artifact is intentionally loaded as-is. No scaler, sampler, or model
    is recreated or fitted by the application.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact was not found at: {MODEL_PATH}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline = joblib.load(MODEL_PATH)

    if not hasattr(pipeline, "predict_proba"):
        raise TypeError(
            "The loaded artifact does not expose predict_proba()."
        )

    return pipeline


@st.cache_resource(show_spinner=False)
def load_background_sample() -> pd.DataFrame:
    """
    Load (or synthesize) a background reference dataset for SHAP's
    permutation explainer.

    Permutation SHAP needs a background sample with genuine variation in
    every feature, since it explains a customer by swapping in background
    values one feature at a time and measuring the change in the model's
    output. A single-row background (e.g. the customer being explained)
    guarantees every contribution is zero, since there is nothing different
    to swap in.

    If a curated reference CSV is present at Deployment/background_sample.csv
    (in the exact FEATURE_COLUMNS schema), it is used, since real customer
    data produces the most representative explanations. Otherwise a
    synthetic background is generated by sampling uniformly across each
    feature's known valid range, so the app never depends on an optional
    file being present.
    """
    if BACKGROUND_CSV_PATH.exists():
        try:
            frame = pd.read_csv(BACKGROUND_CSV_PATH)
            frame = frame[FEATURE_COLUMNS].apply(
                pd.to_numeric, errors="raise"
            )
            if len(frame) >= 2:
                return frame.reset_index(drop=True)
        except Exception:
            # Fall through to the synthetic background if the file is
            # missing required columns, malformed, or otherwise unusable.
            pass

    rng = np.random.default_rng(SHAP_BACKGROUND_SEED)
    n = SHAP_BACKGROUND_SIZE

    frame = pd.DataFrame(
        {
            "CreditScore": rng.integers(350, 851, n),
            "Age": rng.integers(18, 93, n),
            "Tenure": rng.integers(0, 11, n),
            "Balance": rng.uniform(0, 250898, n),
            "NumOfProducts": rng.integers(1, 5, n),
            "HasCrCard": rng.integers(0, 2, n),
            "IsActiveMember": rng.integers(0, 2, n),
            "EstimatedSalary": rng.uniform(0, 200000, n),
            "Satisfaction Score": rng.integers(1, 6, n),
            "Card Type": rng.integers(1, 5, n),
            "Point Earned": rng.integers(119, 1001, n),
            "Geography_Germany": rng.integers(0, 2, n),
            "Geography_Spain": rng.integers(0, 2, n),
            "Gender_Male": rng.integers(0, 2, n),
        }
    )

    return frame[FEATURE_COLUMNS]


def find_estimator(obj: Any) -> Optional[Any]:
    """Recursively locate a tree-based estimator with feature_importances_."""
    visited: set[int] = set()

    def walk(current: Any) -> Optional[Any]:
        if current is None or id(current) in visited:
            return None

        visited.add(id(current))

        if hasattr(current, "feature_importances_"):
            return current

        if hasattr(current, "named_steps"):
            for step in current.named_steps.values():
                result = walk(step)
                if result is not None:
                    return result

        if hasattr(current, "steps"):
            for _, step in current.steps:
                result = walk(step)
                if result is not None:
                    return result

        for attr in ("estimator", "base_estimator", "classifier", "regressor"):
            if hasattr(current, attr):
                result = walk(getattr(current, attr))
                if result is not None:
                    return result

        return None

    return walk(obj)


def get_feature_importance(
    pipeline: Any,
) -> Optional[pd.DataFrame]:
    """Extract global feature importance from the underlying estimator."""
    try:
        estimator = find_estimator(pipeline)

        if estimator is None:
            return None

        importances = np.asarray(estimator.feature_importances_, dtype=float)

        if len(importances) != len(FEATURE_COLUMNS):
            return None

        frame = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Importance": importances,
            }
        )
        frame["Display Feature"] = frame["Feature"].map(
            DISPLAY_FEATURE_NAMES
        )
        return frame.sort_values("Importance", ascending=True)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Input preprocessing
# ---------------------------------------------------------------------------

def preprocess_input(
    geography: str,
    gender: str,
    age: int,
    tenure: int,
    credit_score: int,
    balance: float,
    estimated_salary: float,
    num_products: int,
    has_credit_card: bool,
    is_active_member: bool,
    card_type: str,
    satisfaction_score: int,
    points_earned: int,
) -> pd.DataFrame:
    """Convert friendly UI values into the model's exact feature schema."""

    germany, spain = GEOGRAPHY_MAP[geography]

    row = {
        "CreditScore": int(credit_score),
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": float(estimated_salary),
        "Satisfaction Score": int(satisfaction_score),
        "Card Type": int(CARD_TYPE_MAP[card_type.upper()]),
        "Point Earned": int(points_earned),
        "Geography_Germany": int(germany),
        "Geography_Spain": int(spain),
        "Gender_Male": int(GENDER_MAP[gender]),
    }

    # CRITICAL:
    # Explicitly enforce the exact training order.
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def _normalize_binary(value: Any, column: str) -> int:
    """Normalize common binary CSV representations to 0 or 1."""
    if pd.isna(value):
        raise ValueError(f"Column '{column}' contains missing values.")

    if isinstance(value, bool):
        return int(value)

    text = str(value).strip().lower()

    if text in {"1", "1.0", "true", "yes", "y", "male", "active"}:
        return 1

    if text in {"0", "0.0", "false", "no", "n", "female", "inactive"}:
        return 0

    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric) and numeric in (0, 1):
        return int(numeric)

    raise ValueError(
        f"Column '{column}' contains an invalid binary value: {value!r}."
    )


def preprocess_batch(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert a human-readable batch CSV into the exact model feature schema.

    Already encoded model columns are also accepted, which makes the loader
    tolerant of exported model-ready datasets.
    """
    df = dataframe.copy()
    df.columns = [str(col).strip() for col in df.columns]

    # Accept either friendly categorical columns or already encoded columns.
    friendly_required = set(BATCH_USER_COLUMNS)
    encoded_required = set(FEATURE_COLUMNS)

    missing_friendly = friendly_required - set(df.columns)
    missing_encoded = encoded_required - set(df.columns)

    if not missing_friendly:
        output = pd.DataFrame(index=df.index)

        output["CreditScore"] = pd.to_numeric(
            df["CreditScore"], errors="raise"
        )
        output["Age"] = pd.to_numeric(df["Age"], errors="raise")
        output["Tenure"] = pd.to_numeric(df["Tenure"], errors="raise")
        output["Balance"] = pd.to_numeric(df["Balance"], errors="raise")
        output["NumOfProducts"] = pd.to_numeric(
            df["NumOfProducts"], errors="raise"
        )
        output["EstimatedSalary"] = pd.to_numeric(
            df["EstimatedSalary"], errors="raise"
        )
        output["Satisfaction Score"] = pd.to_numeric(
            df["Satisfaction Score"], errors="raise"
        )
        output["Point Earned"] = pd.to_numeric(
            df["Point Earned"], errors="raise"
        )

        geography = (
            df["Geography"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        invalid_geo = sorted(set(geography) - set(GEOGRAPHY_MAP))
        if invalid_geo:
            raise ValueError(
                "Invalid Geography values: "
                + ", ".join(map(str, invalid_geo))
                + ". Use France, Germany, or Spain."
            )

        output["Geography_Germany"] = geography.eq("Germany").astype(int)
        output["Geography_Spain"] = geography.eq("Spain").astype(int)

        gender = (
            df["Gender"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        invalid_gender = sorted(set(gender) - set(GENDER_MAP))
        if invalid_gender:
            raise ValueError(
                "Invalid Gender values: "
                + ", ".join(map(str, invalid_gender))
                + ". Use Female or Male."
            )

        output["Gender_Male"] = gender.eq("Male").astype(int)

        output["HasCrCard"] = df["HasCrCard"].apply(
            lambda x: _normalize_binary(x, "HasCrCard")
        )
        output["IsActiveMember"] = df["IsActiveMember"].apply(
            lambda x: _normalize_binary(x, "IsActiveMember")
        )

        card_type = (
            df["Card Type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        invalid_cards = sorted(set(card_type) - set(CARD_TYPE_MAP))
        if invalid_cards:
            raise ValueError(
                "Invalid Card Type values: "
                + ", ".join(map(str, invalid_cards))
                + ". Use SILVER, GOLD, PLATINUM, or DIAMOND."
            )

        output["Card Type"] = card_type.map(CARD_TYPE_MAP)

        return output[FEATURE_COLUMNS].copy()

    if not missing_encoded:
        output = df[FEATURE_COLUMNS].copy()

        for column in FEATURE_COLUMNS:
            output[column] = pd.to_numeric(
                output[column],
                errors="raise",
            )

        return output

    missing = sorted(missing_friendly)
    raise ValueError(
        "The CSV is missing required user-friendly columns: "
        + ", ".join(missing)
    )


def validate_customer_input(dataframe: pd.DataFrame) -> None:
    """Validate a single model-ready customer record."""
    if dataframe.empty:
        raise ValueError("No customer data was provided.")

    row = dataframe.iloc[0]

    checks = {
        "CreditScore": (350, 850),
        "Age": (18, 92),
        "Tenure": (0, 10),
        "Balance": (0, 250898),
        "NumOfProducts": (1, 4),
        "EstimatedSalary": (0, 200000),
        "Satisfaction Score": (1, 5),
        "Card Type": (1, 4),
        "Point Earned": (119, 1000),
    }

    for column, (minimum, maximum) in checks.items():
        value = float(row[column])
        if not minimum <= value <= maximum:
            raise ValueError(
                f"{column} must be between {minimum:,} and {maximum:,}."
            )

    for column in (
        "Geography_Germany",
        "Geography_Spain",
        "Gender_Male",
        "HasCrCard",
        "IsActiveMember",
    ):
        if int(row[column]) not in (0, 1):
            raise ValueError(f"{column} must be binary (0 or 1).")


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _get_churn_probability(
    pipeline: Any,
    dataframe: pd.DataFrame,
) -> float:
    """Return the probability associated with class 1 / churn."""
    probabilities = np.asarray(pipeline.predict_proba(dataframe))

    if probabilities.ndim != 2 or probabilities.shape[0] != len(dataframe):
        raise ValueError("Unexpected predict_proba() output shape.")

    classes = getattr(pipeline, "classes_", None)

    if classes is None:
        estimator = find_estimator(pipeline)
        classes = getattr(estimator, "classes_", None)

    if classes is not None:
        classes = list(classes)
        if 1 in classes:
            churn_index = classes.index(1)
        else:
            churn_index = min(1, probabilities.shape[1] - 1)
    else:
        churn_index = min(1, probabilities.shape[1] - 1)

    return probabilities[:, churn_index]


def predict_customer(
    pipeline: Any,
    input_dataframe: pd.DataFrame,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Run inference through the complete trained pipeline.

    The pipeline performs all model-side preprocessing internally.
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError("Classification threshold must be between 0 and 1.")

    try:
        validate_customer_input(input_dataframe)

        probability = float(
            _get_churn_probability(pipeline, input_dataframe)[0]
        )

        prediction = int(probability >= threshold)

        return {
            "prediction": prediction,
            "churn_probability": probability,
            "stay_probability": 1.0 - probability,
            "risk_level": calculate_risk_level(probability),
        }
    except Exception as exc:
        raise RuntimeError(f"Prediction failed: {exc}") from exc


def calculate_risk_level(probability: float) -> str:
    """Map churn probability to a business-friendly risk category."""
    if probability >= 0.70:
        return "High Risk"
    if probability >= 0.40:
        return "Moderate Risk"
    return "Low Risk"


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------

def explain_prediction(
    pipeline: Any,
    input_dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Attempt customer-specific SHAP explanation without changing the trained
    pipeline. Falls back to global feature importance when SHAP is unavailable
    or incompatible.
    """
    try:
        import shap

        def predict_probability(values: Any) -> np.ndarray:
            frame = pd.DataFrame(values, columns=FEATURE_COLUMNS)
            return _get_churn_probability(pipeline, frame)

        # A real background sample with genuine per-feature variation is
        # required here. Using the customer's own row (or any single row)
        # as the background guarantees every contribution comes out as
        # zero, since permutation SHAP has nothing different to swap in.
        background = load_background_sample()

        explainer = shap.Explainer(
            predict_probability,
            background,
            algorithm="permutation",
        )

        shap_result = explainer(
            input_dataframe,
            max_evals=max(2 * len(FEATURE_COLUMNS) + 1, 31),
        )

        values = np.asarray(shap_result.values)

        if values.ndim == 2:
            values = values[0]

        if len(values) != len(FEATURE_COLUMNS):
            raise ValueError("Unexpected SHAP output dimensions.")

        contributions = pd.DataFrame(
            {
                "Feature": FEATURE_COLUMNS,
                "Contribution": values.astype(float),
            }
        )

        contributions["Display Feature"] = contributions["Feature"].map(
            DISPLAY_FEATURE_NAMES
        )

        return {
            "available": True,
            "method": "SHAP",
            "contributions": contributions.sort_values(
                "Contribution",
                key=lambda series: series.abs(),
                ascending=False,
            ),
        }

    except Exception:
        importance = get_feature_importance(pipeline)

        if importance is not None:
            fallback = importance.copy()
            fallback["Contribution"] = fallback["Importance"]
            return {
                "available": False,
                "method": "Global feature importance",
                "contributions": fallback,
            }

        return {
            "available": False,
            "method": "Unavailable",
            "contributions": None,
        }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def make_probability_gauge(probability: float) -> go.Figure:
    """Create the primary churn probability gauge."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 36,
                    "family": "Manrope",
                    "color": "#102a43",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": "rgba(0,0,0,0)",
                    "tickfont": {"size": 10},
                },
                "bar": {
                    "color": "#0f766e",
                    "thickness": 0.25,
                },
                "bgcolor": "#edf1f2",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40], "color": "#e8f5f3"},
                    {"range": [40, 70], "color": "#fff7e8"},
                    {"range": [70, 100], "color": "#fdf0ef"},
                ],
                "threshold": {
                    "line": {"color": "#102a43", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
            title={
                "text": "Estimated churn probability",
                "font": {"size": 13, "color": "#697782"},
            },
        )
    )

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=45, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
    )
    return fig


def make_contribution_chart(
    explanation: dict[str, Any],
) -> Optional[go.Figure]:
    """Build the horizontal customer-specific contribution chart."""
    contributions = explanation.get("contributions")

    if contributions is None or contributions.empty:
        return None

    data = contributions.head(8).copy()

    if explanation["method"] == "SHAP":
        data["Direction"] = np.where(
            data["Contribution"] >= 0,
            "Pushes toward churn",
            "Pushes toward staying",
        )
        data["Magnitude"] = data["Contribution"].abs()
        data = data.sort_values("Contribution")
        values = data["Contribution"]
        x_title = "Model contribution to estimated churn risk"
    else:
        data["Direction"] = "Global model importance"
        data["Magnitude"] = data["Importance"]
        data = data.sort_values("Importance")
        values = data["Importance"]
        x_title = "Relative global feature importance"

    fig = go.Figure()

    if explanation["method"] == "SHAP":
        fig.add_trace(
            go.Bar(
                x=values,
                y=data["Display Feature"],
                orientation="h",
                text=[
                    f"{value:+.3f}" for value in values
                ],
                textposition="outside",
                marker_color=[
                    "#c2413a" if value >= 0 else "#0f766e"
                    for value in values
                ],
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Contribution: %{x:.4f}"
                    "<extra></extra>"
                ),
            )
        )
    else:
        fig.add_trace(
            go.Bar(
                x=values,
                y=data["Display Feature"],
                orientation="h",
                text=[
                    f"{value:.1%}" for value in values
                ],
                textposition="outside",
                marker_color="#55758a",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Importance: %{x:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=55, t=10, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        xaxis_title=x_title,
        yaxis_title=None,
        showlegend=False,
        xaxis=dict(gridcolor="#edf0f2", zerolinecolor="#b9c4ca"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )

    return fig


def make_risk_profile(
    dataframe: pd.DataFrame,
) -> go.Figure:
    """Create a descriptive customer profile radar chart."""
    row = dataframe.iloc[0]

    credit = np.clip(
        (float(row["CreditScore"]) - 350) / 500 * 100,
        0,
        100,
    )

    engagement = (
        0.55 * int(row["IsActiveMember"]) * 100
        + 0.45 * ((int(row["NumOfProducts"]) - 1) / 3) * 100
    )
    engagement = float(np.clip(engagement, 0, 100))

    satisfaction = float(row["Satisfaction Score"]) / 5 * 100

    relationship = (
        0.60 * ((int(row["NumOfProducts"]) - 1) / 3) * 100
        + 0.40 * int(row["HasCrCard"]) * 100
    )
    relationship = float(np.clip(relationship, 0, 100))

    financial_profile = float(
        np.clip(float(row["Balance"]) / 250898 * 100, 0, 100)
    )

    labels = [
        "Credit health",
        "Engagement",
        "Satisfaction",
        "Product relationship",
        "Balance position",
    ]

    values = [
        credit,
        engagement,
        satisfaction,
        relationship,
        financial_profile,
    ]

    fig = go.Figure(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself",
            line=dict(color="#0f766e", width=2),
            fillcolor="rgba(15,118,110,0.12)",
            hovertemplate="%{theta}: %{r:.0f}/100<extra></extra>",
        )
    )

    fig.update_layout(
        height=360,
        margin=dict(l=45, r=45, t=25, b=25),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        polar={
            "radialaxis": {
                "range": [0, 100],
                "showticklabels": False,
                "gridcolor": "#e5eaed",
            },
            "angularaxis": {
                "gridcolor": "#e5eaed",
                "linecolor": "#dfe5e8",
            },
        },
        showlegend=False,
    )

    return fig


def make_risk_distribution_chart(
    results: pd.DataFrame,
) -> go.Figure:
    """Build batch risk distribution chart."""
    order = ["Low Risk", "Moderate Risk", "High Risk"]
    counts = (
        results["Risk Level"]
        .value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Risk Level", "Customers"]

    fig = px.bar(
        counts,
        x="Risk Level",
        y="Customers",
        text="Customers",
        category_orders={"Risk Level": order},
    )

    fig.update_traces(
        marker_color="#55758a",
        textposition="outside",
    )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=15, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        yaxis=dict(gridcolor="#edf0f2"),
    )

    return fig


def make_churn_split_chart(results: pd.DataFrame) -> go.Figure:
    """Build batch churn-versus-stay donut chart."""
    counts = results["Prediction Label"].value_counts()

    labels = ["Likely to Stay", "Likely to Churn"]
    values = [
        int(counts.get("Likely to Stay", 0)),
        int(counts.get("Likely to Churn", 0)),
    ]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.68,
            textinfo="percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Customers: %{value}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            ),
            marker=dict(
                colors=["#0f766e", "#c2413a"],
                line=dict(color="#ffffff", width=3),
            ),
        )
    )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=15, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.02,
            x=0.5,
            xanchor="center",
        ),
    )

    return fig


def make_probability_distribution_chart(
    results: pd.DataFrame,
) -> go.Figure:
    """Build churn probability distribution histogram."""
    fig = px.histogram(
        results,
        x="Churn Probability",
        nbins=20,
    )

    fig.update_traces(
        marker_color="#55758a",
        opacity=0.88,
    )

    fig.update_layout(
        height=330,
        margin=dict(l=10, r=10, t=15, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        xaxis_title="Estimated churn probability",
        yaxis_title="Customers",
        xaxis=dict(
            tickformat=".0%",
            gridcolor="#edf0f2",
        ),
        yaxis=dict(gridcolor="#edf0f2"),
        showlegend=False,
    )

    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def render_header(
    kicker: str,
    title: str,
    subtitle: str,
) -> None:
    """Render a consistent page header."""
    st.markdown(f'<div class="page-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f'<h1 class="page-title">{title}</h1>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def render_kpi(
    label: str,
    value: str,
    note: str = "",
) -> None:
    """Render a compact KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_result(
    result: dict[str, Any],
) -> None:
    """Render the main prediction result."""
    risk_level = result["risk_level"]
    probability = result["churn_probability"]

    risk_class = {
        "High Risk": "risk-high",
        "Moderate Risk": "risk-moderate",
        "Low Risk": "risk-low",
    }[risk_level]

    if result["prediction"] == 1:
        headline = "High Churn Risk" if risk_level == "High Risk" else "Likely to Churn"
        description = (
            "The model estimates that this customer has a meaningful likelihood "
            "of leaving based on the information provided."
        )
    else:
        headline = "Likely to Stay"
        description = (
            "The model estimates that this customer is more likely to remain "
            "based on the information provided."
        )

    st.markdown(
        f"""
        <div class="risk-panel {risk_class}">
            <div class="risk-eyebrow">{risk_level}</div>
            <div class="risk-title">{headline}</div>
            <div class="risk-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        render_kpi(
            "Estimated churn risk",
            f"{probability:.0%}",
            "Model probability of class 1",
        )
    with c2:
        render_kpi(
            "Estimated stay probability",
            f"{result['stay_probability']:.0%}",
            "Complement of churn probability",
        )


# ---------------------------------------------------------------------------
# Section 1 — Customer Risk Assessment
# ---------------------------------------------------------------------------

def render_customer_risk_tab(
    pipeline: Any,
    threshold: float,
) -> None:
    """Render the interactive individual customer assessment."""
    render_header(
        "Customer intelligence",
        "Customer Risk Assessment",
        (
            "Build a customer profile using familiar banking attributes. "
            "The trained inference pipeline converts these values into the "
            "model's required representation automatically."
        ),
    )

    st.markdown(
        '<div class="section-heading">Customer Profile</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Core demographic and relationship details.</div>',
        unsafe_allow_html=True,
    )

    profile_1, profile_2, profile_3, profile_4 = st.columns(4)

    with profile_1:
        geography = st.selectbox(
            "Geography",
            list(GEOGRAPHY_MAP.keys()),
            help="Customer's primary geography.",
        )

    with profile_2:
        gender = st.selectbox(
            "Gender",
            list(GENDER_MAP.keys()),
            help="Customer gender used by the trained model.",
        )

    with profile_3:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=92,
            value=40,
            help="Expected customer age range: 18–92.",
        )

    with profile_4:
        tenure = st.slider(
            "Tenure",
            min_value=0,
            max_value=10,
            value=5,
            help="Number of years the customer has been with the bank.",
        )

    st.markdown(
        '<div class="section-heading">Financial Profile</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Credit and financial attributes supplied to the model.</div>',
        unsafe_allow_html=True,
    )

    finance_1, finance_2, finance_3 = st.columns(3)

    with finance_1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=350,
            max_value=850,
            value=650,
            step=1,
            help="Expected range: 350–850.",
        )

    with finance_2:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            max_value=250898.0,
            value=75000.0,
            step=500.0,
            help="Account balance. Expected maximum is approximately 250,898.",
        )

    with finance_3:
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            max_value=200000.0,
            value=75000.0,
            step=500.0,
            help="Estimated annual salary. Expected range is approximately 0–200,000.",
        )

    st.markdown(
        '<div class="section-heading">Banking Behavior</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Product relationship, activity, satisfaction, and rewards.</div>',
        unsafe_allow_html=True,
    )

    behavior_1, behavior_2, behavior_3, behavior_4 = st.columns(4)

    with behavior_1:
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
            help="Number of bank products held by the customer.",
        )

    with behavior_2:
        has_credit_card = st.toggle(
            "Has Credit Card",
            value=True,
            help="Whether the customer has a bank credit card.",
        )

    with behavior_3:
        is_active_member = st.toggle(
            "Active Member",
            value=True,
            help="Whether the customer is marked as an active member.",
        )

    with behavior_4:
        card_type = st.selectbox(
            "Card Type",
            list(CARD_TYPE_MAP.keys()),
            index=1,
            help="Customer card tier. The application encodes this automatically.",
        )

    behavior_5, behavior_6 = st.columns(2)

    with behavior_5:
        satisfaction_score = st.slider(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3,
            help="Customer satisfaction score on a 1–5 scale.",
        )

    with behavior_6:
        points_earned = st.number_input(
            "Points Earned",
            min_value=119,
            max_value=1000,
            value=500,
            step=1,
            help="Reward points. Expected range: 119–1,000.",
        )

    st.markdown("")

    assess_col, threshold_col = st.columns([3, 2])

    with assess_col:
        assess_clicked = st.button(
            "Assess Customer Risk",
            type="primary",
            use_container_width=True,
        )

    with threshold_col:
        st.caption(
            f"Classification threshold: **{threshold:.0%}** · "
            "Advanced setting"
        )

    if not assess_clicked:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">Ready for assessment</div>
                <div class="empty-state-copy">
                    Complete the customer profile above and select
                    <b>Assess Customer Risk</b> to generate a probability,
                    risk level, and model explanation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        customer_df = preprocess_input(
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            credit_score=credit_score,
            balance=balance,
            estimated_salary=estimated_salary,
            num_products=num_products,
            has_credit_card=has_credit_card,
            is_active_member=is_active_member,
            card_type=card_type,
            satisfaction_score=satisfaction_score,
            points_earned=points_earned,
        )

        result = predict_customer(
            pipeline,
            customer_df,
            threshold=threshold,
        )

    except Exception as exc:
        st.error(f"Unable to assess this customer: {exc}")
        return

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Risk Assessment</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">The prediction is a statistical estimate, not a statement of customer intent.</div>',
        unsafe_allow_html=True,
    )

    result_left, result_right = st.columns([1.25, 1])

    with result_left:
        render_risk_result(result)

    with result_right:
        st.plotly_chart(
            make_probability_gauge(result["churn_probability"]),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown("")
    insight_left, insight_right = st.columns([1.15, 1])

    with insight_left:
        st.markdown(
            '<div class="section-heading">Why might this customer leave?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Customer-specific model contribution when technically available.</div>',
            unsafe_allow_html=True,
        )

        with st.spinner("Generating model explanation..."):
            explanation = explain_prediction(
                pipeline,
                customer_df,
            )

        contribution_chart = make_contribution_chart(explanation)

        if contribution_chart is not None:
            st.plotly_chart(
                contribution_chart,
                use_container_width=True,
                config={"displayModeBar": False},
            )

        if explanation["method"] == "SHAP":
            st.info(
                "These contributions show features that pushed the model's "
                "estimated churn probability up or down for this customer. "
                "They should not be interpreted as definitive reasons for "
                "the customer's future behavior."
            )
        elif explanation["method"] == "Global feature importance":
            st.warning(
                "Customer-specific SHAP explanation was unavailable. "
                "The chart instead shows global model feature importance, "
                "which describes the model overall rather than this customer."
            )
        else:
            st.warning(
                "Customer-specific and global model explanations are "
                "currently unavailable for this artifact."
            )

        if explanation["method"] == "SHAP":
            contributions = explanation["contributions"]

            positive = contributions[
                contributions["Contribution"] > 0
            ].head(3)

            negative = contributions[
                contributions["Contribution"] < 0
            ].sort_values("Contribution").head(3)

            if not positive.empty:
                st.markdown("**Features pushing toward churn**")
                for _, row in positive.iterrows():
                    st.markdown(
                        f'<span class="insight-chip">'
                        f'{row["Display Feature"]} increased estimated risk'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

            if not negative.empty:
                st.markdown("**Features pushing toward staying**")
                for _, row in negative.iterrows():
                    st.markdown(
                        f'<span class="insight-chip">'
                        f'{row["Display Feature"]} reduced estimated risk'
                        f'</span>',
                        unsafe_allow_html=True,
                    )

    with insight_right:
        st.markdown(
            '<div class="section-heading">Customer Risk Profile</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Descriptive indicators derived from the entered customer profile.</div>',
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            make_risk_profile(customer_df),
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.markdown(
            """
            <div class="footnote">
                The profile is descriptive rather than a separate risk model.
                Higher or lower values on these indicators do not independently
                determine whether a customer will churn.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View model-ready customer record"):
        st.dataframe(
            customer_df,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Section 2 — Batch Customer Analysis
# ---------------------------------------------------------------------------

def process_batch_predictions(
    pipeline: Any,
    dataframe: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Preprocess and score a batch of customers."""
    model_df = preprocess_batch(dataframe)

    probabilities = _get_churn_probability(
        pipeline,
        model_df,
    )

    result = dataframe.copy()
    result["Prediction"] = (
        probabilities >= threshold
    ).astype(int)

    result["Prediction Label"] = np.where(
        result["Prediction"] == 1,
        "Likely to Churn",
        "Likely to Stay",
    )

    result["Churn Probability"] = probabilities
    result["Stay Probability"] = 1.0 - probabilities
    result["Risk Level"] = [
        calculate_risk_level(float(probability))
        for probability in probabilities
    ]

    return result


def render_batch_prediction_tab(
    pipeline: Any,
    threshold: float,
) -> None:
    """Render the CSV-based batch analysis experience."""
    render_header(
        "Portfolio intelligence",
        "Batch Customer Analysis",
        (
            "Upload a customer portfolio to score churn probability at scale. "
            "Use human-readable Geography, Gender, and Card Type values; "
            "the application handles the required encoding automatically."
        ),
    )

    upload = st.file_uploader(
        "Upload customer CSV",
        type=["csv"],
        help=(
            "Required columns: CreditScore, Geography, Gender, Age, Tenure, "
            "Balance, NumOfProducts, HasCrCard, IsActiveMember, "
            "EstimatedSalary, Satisfaction Score, Card Type, Point Earned."
        ),
    )

    if upload is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">Upload a customer portfolio</div>
                <div class="empty-state-copy">
                    Add a CSV containing the customer attributes required by
                    the model. The first row is used as the column header.
                    Customer IDs, surnames, RowNumber, and Complain are not
                    required.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Expected CSV columns"):
            st.code(
                ", ".join(BATCH_USER_COLUMNS),
                language="text",
            )

        return

    try:
        raw_df = pd.read_csv(upload)

        if raw_df.empty:
            st.warning("The uploaded CSV contains no customer rows.")
            return

        st.success(
            f"Loaded {len(raw_df):,} customer record"
            f"{'' if len(raw_df) == 1 else 's'}."
        )

    except Exception as exc:
        st.error(f"Unable to read the CSV file: {exc}")
        return

    with st.expander("Preview uploaded data"):
        st.dataframe(
            raw_df.head(10),
            use_container_width=True,
            hide_index=True,
        )

    try:
        with st.spinner("Scoring customer portfolio..."):
            results = process_batch_predictions(
                pipeline,
                raw_df,
                threshold,
            )
    except Exception as exc:
        st.error(
            "The portfolio could not be scored. "
            f"Please review the CSV format and values. Details: {exc}"
        )
        return

    total = len(results)
    churners = int((results["Prediction"] == 1).sum())
    retained = total - churners
    average_probability = float(results["Churn Probability"].mean())
    churn_percentage = churners / total if total else 0.0

    st.markdown("---")
    st.markdown(
        '<div class="section-heading">Portfolio Overview</div>',
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        render_kpi(
            "Total customers",
            f"{total:,}",
            "Records successfully scored",
        )

    with k2:
        render_kpi(
            "Predicted churners",
            f"{churners:,}",
            "Probability ≥ threshold",
        )

    with k3:
        render_kpi(
            "Predicted retained",
            f"{retained:,}",
            "Probability below threshold",
        )

    with k4:
        render_kpi(
            "Average churn probability",
            f"{average_probability:.1%}",
            "Across uploaded customers",
        )

    with k5:
        render_kpi(
            "Churn percentage",
            f"{churn_percentage:.1%}",
            f"Using {threshold:.0%} threshold",
        )

    chart_1, chart_2 = st.columns(2)

    with chart_1:
        st.markdown(
            '<div class="section-heading">Churn vs. Retained</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_churn_split_chart(results),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    with chart_2:
        st.markdown(
            '<div class="section-heading">Risk Distribution</div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_risk_distribution_chart(results),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    st.markdown(
        '<div class="section-heading">Churn Probability Distribution</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        make_probability_distribution_chart(results),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.markdown(
        '<div class="section-heading">Scored Customers</div>',
        unsafe_allow_html=True,
    )

    display_results = results.copy()

    if "Churn Probability" in display_results:
        display_results["Churn Probability"] = display_results[
            "Churn Probability"
        ].map(lambda x: f"{x:.1%}")

    if "Stay Probability" in display_results:
        display_results["Stay Probability"] = display_results[
            "Stay Probability"
        ].map(lambda x: f"{x:.1%}")

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer, index=False)

    st.download_button(
        label="Download scored customer CSV",
        data=csv_buffer.getvalue(),
        file_name="customer_churn_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )


# ---------------------------------------------------------------------------
# Section 3 — Model Insights
# ---------------------------------------------------------------------------

def render_model_insights_tab(
    pipeline: Any,
) -> None:
    """Render model metadata and global feature insights."""
    render_header(
        "Model intelligence",
        "Model Insights",
        (
            "A transparent view of the deployed Gradient Boosting model and "
            "the development-stage evaluation results associated with it."
        ),
    )

    st.markdown(
        '<div class="section-heading">Model Overview</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-caption">Evaluation results are from the development process and are not guarantees of real-world performance.</div>',
        unsafe_allow_html=True,
    )

    model_1, model_2, model_3, model_4 = st.columns(4)

    with model_1:
        render_kpi(
            "Model type",
            "Gradient Boosting",
            "Deployed classifier",
        )

    with model_2:
        render_kpi(
            "Problem type",
            "Binary classification",
            "Target: Exited",
        )

    with model_3:
        render_kpi(
            "Dataset size",
            "10,000",
            "Customers in development data",
        )

    with model_4:
        render_kpi(
            "Default threshold",
            "50%",
            "Standard classification threshold",
        )

    st.markdown("")

    metric_1, metric_2, metric_3 = st.columns(3)

    with metric_1:
        render_kpi(
            "Cross-validation accuracy",
            "85.3%",
            "Development evaluation",
        )

    with metric_2:
        render_kpi(
            "Cross-validation F1",
            "61.2%",
            "Development evaluation",
        )

    with metric_3:
        render_kpi(
            "Cross-validation ROC-AUC",
            "85.6%",
            "Development evaluation",
        )

    st.markdown("---")

    importance = get_feature_importance(pipeline)

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            '<div class="section-heading">Global Feature Importance</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">Relative importance within the deployed Gradient Boosting estimator.</div>',
            unsafe_allow_html=True,
        )

        if importance is None:
            st.warning(
                "Feature importance could not be extracted from the loaded "
                "model artifact."
            )
        else:
            data = importance.copy()

            fig = go.Figure(
                go.Bar(
                    x=data["Importance"],
                    y=data["Display Feature"],
                    orientation="h",
                    text=[
                        f"{value:.1%}"
                        for value in data["Importance"]
                    ],
                    textposition="outside",
                    marker_color="#55758a",
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Importance: %{x:.4f}"
                        "<extra></extra>"
                    ),
                )
            )

            fig.update_layout(
                height=540,
                margin=dict(l=10, r=60, t=10, b=35),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_family="DM Sans",
                xaxis_title="Relative importance",
                yaxis_title=None,
                showlegend=False,
                xaxis=dict(gridcolor="#edf0f2"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": False},
            )

    with right:
        st.markdown(
            '<div class="section-heading">Business Interpretation</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-caption">How to read the model\'s global signals.</div>',
            unsafe_allow_html=True,
        )

        if importance is not None:
            top_features = (
                importance
                .sort_values("Importance", ascending=False)
                .head(5)
            )

            st.markdown(
                """
                The most influential variables are the features the trained
                model relied on most heavily across its development data.
                Importance describes the model's internal use of a feature;
                it does not establish causality.
                """
            )

            for rank, (_, row) in enumerate(
                top_features.iterrows(),
                start=1,
            ):
                feature_name = row["Display Feature"]
                importance_value = row["Importance"]

                st.markdown(
                    f"""
                    <div class="card" style="margin-bottom:10px; padding:14px 16px;">
                        <div style="display:flex; justify-content:space-between; gap:15px;">
                            <div>
                                <div class="profile-label">
                                    #{rank} global signal
                                </div>
                                <div class="profile-value">
                                    {feature_name}
                                </div>
                            </div>
                            <div style="font-family:Manrope; font-weight:800; color:#102a43;">
                                {importance_value:.1%}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")
        with st.expander("What do these metrics mean?"):
            st.markdown(
                """
                **Accuracy** measures the proportion of development examples
                classified correctly.

                **F1** balances precision and recall and is particularly useful
                when the two classes are not equally represented.

                **ROC-AUC** measures how well the model separates customers
                likely to churn from those likely to stay across probability
                thresholds.

                These are historical development metrics. Actual performance
                can change when the model is exposed to new customers,
                different customer behavior, or changing banking conditions.
                """
            )

    st.markdown("---")

    with st.expander("Technical deployment details"):
        st.markdown(
            f"""
            **Artifact:** `Deployment/best_gb.pkl`

            **Artifact loading:** `joblib`

            **Inference path:** the complete serialized pipeline is loaded and
            used directly for prediction.

            **Pipeline components detected:** `{", ".join(
                [name for name, _ in getattr(pipeline, "steps", [])]
            ) or "Not exposed"}`

            **Expected model features:** `{len(FEATURE_COLUMNS)}`

            The application does not fit the scaler, resample with SMOTE, or
            retrain the classifier during inference.
            """
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    inject_css()

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-mark">BI</div>
                <div class="brand-name">Bank Intelligence</div>
                <div class="brand-caption">
                    Customer churn analytics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="color:#aebdc7; font-size:11px; '
            'text-transform:uppercase; letter-spacing:1px; '
            'font-weight:700; margin-bottom:8px;">Workspace</div>',
            unsafe_allow_html=True,
        )

        section = st.radio(
            "Navigate",
            [
                "Customer Risk Assessment",
                "Batch Customer Analysis",
                "Model Insights",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown(
            '<div style="color:#aebdc7; font-size:11px; '
            'text-transform:uppercase; letter-spacing:1px; '
            'font-weight:700; margin-bottom:8px;">Model controls</div>',
            unsafe_allow_html=True,
        )

        threshold = st.slider(
            "Classification threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help=(
                "Customers at or above this churn probability are classified "
                "as likely to churn. Lower thresholds identify more potential "
                "churners but may increase false alarms."
            ),
        )

        st.caption(
            "Default: 50%. Changing this threshold changes the balance between "
            "finding more potential churners and avoiding false alarms."
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="color:#aebdc7; font-size:11px; line-height:1.6;">
                <b>Target</b><br>
                Exited · 0 = stays · 1 = churns
                <br><br>
                <b>Inference mode</b><br>
                Serialized pipeline only
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    try:
        with st.spinner("Loading bank churn intelligence model..."):
            pipeline = load_artifacts()
    except Exception as exc:
        st.error(
            "The deployed model could not be loaded. "
            "Please verify that `Deployment/best_gb.pkl` exists beside "
            "the application."
        )
        st.code(str(exc))
        st.stop()

    # -----------------------------------------------------------------------
    # Route section
    # -----------------------------------------------------------------------
    if section == "Customer Risk Assessment":
        render_customer_risk_tab(
            pipeline,
            threshold,
        )

    elif section == "Batch Customer Analysis":
        render_batch_prediction_tab(
            pipeline,
            threshold,
        )

    else:
        render_model_insights_tab(
            pipeline,
        )

    st.markdown("")
    st.markdown(
        """
        <div style="text-align:center; color:#8a969e; font-size:11px;
                    padding:18px 0 8px;">
            Bank Customer Intelligence · Prediction estimates are model outputs
            and should be interpreted alongside appropriate business context.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()