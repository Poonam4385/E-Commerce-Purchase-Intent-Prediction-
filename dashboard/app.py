from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "ecommerce_sessions.csv"
MODEL_PATH = PROJECT_DIR / "artifacts" / "purchase_intent_model.joblib"
THRESHOLD_PATH = PROJECT_DIR / "artifacts" / "decision_threshold.joblib"

st.set_page_config(
    page_title="E-Commerce Purchase Intent",
    page_icon=":material/shopping_cart:",
    layout="wide",
)


@st.cache_data
def load_sessions(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["TotalPages"] = (
        data["Administrative"]
        + data["Informational"]
        + data["ProductRelated"]
    )
    data["TotalDuration"] = (
        data["Administrative_Duration"]
        + data["Informational_Duration"]
        + data["ProductRelated_Duration"]
    )
    data["AvgTimePerPage"] = data["TotalDuration"] / (data["TotalPages"] + 1)
    data["ProductEngagementRatio"] = data["ProductRelated"] / (data["TotalPages"] + 1)
    data["ProductTimeRatio"] = (
        data["ProductRelated_Duration"] / (data["TotalDuration"] + 1)
    )
    data["ReturningVisitorFlag"] = (
        data["VisitorType"].eq("Returning_Visitor").astype(int)
    )
    return data


st.title("E-Commerce purchase intent")
st.write(
    "Explore browsing-session conversion patterns and estimate a visitor's "
    "likelihood of purchasing."
)

if not DATA_PATH.is_file():
    st.error(f"Session dataset not found: {DATA_PATH}")
    st.stop()
if not MODEL_PATH.is_file():
    st.error(f"Prediction model not found: {MODEL_PATH}")
    st.stop()
if not THRESHOLD_PATH.is_file():
    st.error(f"Decision threshold not found: {THRESHOLD_PATH}")
    st.stop()

df = load_sessions(str(DATA_PATH))
if df.empty:
    st.error(f"The session dataset is empty: {DATA_PATH}")
    st.stop()

required_columns = {
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Converted",
}
missing_columns = sorted(required_columns.difference(df.columns))
if missing_columns:
    st.error("Dataset is missing required columns: " + ", ".join(missing_columns))
    st.stop()

model = load_model(str(MODEL_PATH))
threshold = float(joblib.load(THRESHOLD_PATH))

total_sessions, total_purchases, conversion_rate = st.columns(3)
total_sessions.metric("Total sessions", f"{len(df):,}")
total_purchases.metric("Purchases", f"{int(df['Converted'].sum()):,}")
conversion_rate.metric("Conversion rate", f"{df['Converted'].mean():.2%}")

chart_columns = st.columns(2)
conversion_counts = (
    df["Converted"]
    .value_counts()
    .reindex([0, 1], fill_value=0)
    .rename_axis("Converted")
    .reset_index(name="Sessions")
)
conversion_counts["Status"] = conversion_counts["Converted"].map(
    {0: "No purchase", 1: "Purchase"}
)
conversion_figure = px.bar(
    conversion_counts,
    x="Status",
    y="Sessions",
    title="Purchase conversion distribution",
)
chart_columns[0].plotly_chart(
    conversion_figure,
    width="stretch",
    alt="Session counts split between purchase and no purchase",
)

visitor_conversion = (
    df.groupby("VisitorType", as_index=False)["Converted"]
    .mean()
    .assign(ConversionRate=lambda values: values["Converted"] * 100)
)
visitor_figure = px.bar(
    visitor_conversion,
    x="VisitorType",
    y="ConversionRate",
    title="Conversion rate by visitor type",
    labels={"VisitorType": "Visitor type", "ConversionRate": "Conversion rate (%)"},
)
chart_columns[1].plotly_chart(
    visitor_figure,
    width="stretch",
    alt="Conversion rate for each visitor type",
)

monthly_conversion = (
    df.groupby("Month", as_index=False)["Converted"]
    .mean()
    .assign(ConversionRate=lambda values: values["Converted"] * 100)
)
month_order = [
    month
    for month in (
        "Jan", "Feb", "Mar", "Apr", "May", "June",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )
    if month in monthly_conversion["Month"].values
]
monthly_figure = px.bar(
    monthly_conversion,
    x="Month",
    y="ConversionRate",
    title="Conversion rate by month",
    labels={"ConversionRate": "Conversion rate (%)"},
    category_orders={"Month": month_order},
)
st.plotly_chart(
    monthly_figure,
    width="stretch",
    alt="Monthly purchase conversion rates",
)

st.header("Live purchase prediction")
st.caption(f"Prediction threshold: {threshold:.2f}")

with st.form("purchase_prediction"):
    administrative, informational, product_related = st.columns(3)
    administrative_pages = administrative.number_input(
        "Administrative pages",
        min_value=0,
        value=0,
        step=1,
    )
    informational_pages = informational.number_input(
        "Informational pages",
        min_value=0,
        value=0,
        step=1,
    )
    product_pages = product_related.number_input(
        "Product-related pages",
        min_value=0,
        value=10,
        step=1,
    )

    administrative_duration, informational_duration, product_duration = st.columns(3)
    administrative_time = administrative_duration.number_input(
        "Administrative duration",
        min_value=0.0,
        value=0.0,
    )
    informational_time = informational_duration.number_input(
        "Informational duration",
        min_value=0.0,
        value=0.0,
    )
    product_time = product_duration.number_input(
        "Product-related duration",
        min_value=0.0,
        value=300.0,
    )

    bounce_rates, exit_rates, page_values, special_day = st.columns(4)
    bounce_rate = bounce_rates.number_input(
        "Bounce rate",
        min_value=0.0,
        max_value=1.0,
        value=0.02,
    )
    exit_rate = exit_rates.number_input(
        "Exit rate",
        min_value=0.0,
        max_value=1.0,
        value=0.04,
    )
    page_value = page_values.number_input("Page values", min_value=0.0, value=5.0)
    special_day_value = special_day.number_input(
        "Special day",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
    )

    month, operating_system, browser = st.columns(3)
    selected_month = month.selectbox("Month", sorted(df["Month"].dropna().unique()))
    selected_os = operating_system.selectbox(
        "Operating system", sorted(df["OperatingSystems"].dropna().unique())
    )
    selected_browser = browser.selectbox(
        "Browser", sorted(df["Browser"].dropna().unique())
    )

    region, traffic_type, visitor_type = st.columns(3)
    selected_region = region.selectbox("Region", sorted(df["Region"].dropna().unique()))
    selected_traffic = traffic_type.selectbox(
        "Traffic type", sorted(df["TrafficType"].dropna().unique())
    )
    selected_visitor = visitor_type.selectbox(
        "Visitor type", sorted(df["VisitorType"].dropna().unique())
    )
    weekend = st.checkbox("Weekend")
    submitted = st.form_submit_button("Predict purchase intent")

if submitted:
    session = pd.DataFrame(
        [
            {
                "Administrative": administrative_pages,
                "Administrative_Duration": administrative_time,
                "Informational": informational_pages,
                "Informational_Duration": informational_time,
                "ProductRelated": product_pages,
                "ProductRelated_Duration": product_time,
                "BounceRates": bounce_rate,
                "ExitRates": exit_rate,
                "PageValues": page_value,
                "SpecialDay": special_day_value,
                "Month": selected_month,
                "OperatingSystems": selected_os,
                "Browser": selected_browser,
                "Region": selected_region,
                "TrafficType": selected_traffic,
                "VisitorType": selected_visitor,
                "Weekend": weekend,
            }
        ]
    )
    probability = float(model.predict_proba(create_features(session))[0, 1])
    prediction = int(probability >= threshold)

    st.metric("Purchase probability", f"{probability:.2%}")
    if prediction:
        st.success("Likely to purchase")
    else:
        st.warning("Unlikely to purchase")
