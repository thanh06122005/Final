import warnings
warnings.filterwarnings('ignore')

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }
    .block-container { padding-top: 1.3rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px; border-radius: 8px;
        background: #f0f2f6; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: #1f77b4 !important; color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# File locations
# -----------------------------
APP_DIR = Path(__file__).resolve().parent
CHURN_FILE = APP_DIR / "churn_probabilities.csv"
RAW_FILE = APP_DIR / "telco_preprocessed.csv"

# Optional artifacts for live model inference
MODEL_FILE = APP_DIR / "best_model.pkl"
SCALER_FILE = APP_DIR / "scaler.pkl"
FEATURE_COLS_FILE = APP_DIR / "feature_columns.pkl"
LE_DICT_FILE = APP_DIR / "label_encoders.pkl"

# -----------------------------
# Helpers
# -----------------------------

def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common _x/_y/_raw suffixes from merged notebook outputs."""
    rename_map = {}
    for base in [
        "CustomerID",
        "Tenure Months",
        "Contract",
        "Internet Service",
        "Monthly Charges",
        "Payment Method",
        "Latitude",
        "Longitude",
        "CLTV",
        "Churn Label",
        "Churn",
    ]:
        if f"{base}_x" in df.columns and base not in df.columns:
            rename_map[f"{base}_x"] = base
        if f"{base}_y" in df.columns and base not in df.columns:
            rename_map[f"{base}_y"] = base
        if f"{base}_raw" in df.columns and base not in df.columns:
            rename_map[f"{base}_raw"] = base
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def pct(series: pd.Series) -> float:
    return float(series.mean() * 100)


@st.cache_data(show_spinner="Loading data...")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CHURN_FILE.exists():
        raise FileNotFoundError(
            f"Missing {CHURN_FILE.name}. Put churn_probabilities.csv next to app.py."
        )
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Missing {RAW_FILE.name}. Put telco_preprocessed.csv next to app.py."
        )

    churn = pd.read_csv(CHURN_FILE)
    raw = pd.read_csv(RAW_FILE)
    churn = normalize_raw_columns(churn)
    raw = normalize_raw_columns(raw)

    # Make sure key numeric columns are numeric
    for col in ["Monthly Charges", "Total Charges", "CLTV", "Latitude", "Longitude", "Churn_Probability"]:
        if col in churn.columns:
            churn[col] = pd.to_numeric(churn[col], errors="coerce")
    for col in ["Monthly Charges", "Total Charges", "CLTV", "Latitude", "Longitude"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Bring in the notebook columns that are needed for charts + simulator.
    raw_cols = [
        "CustomerID", "Contract", "Internet Service", "Monthly Charges", "Tenure Months",
        "Payment Method", "City", "Zip Code", "Latitude", "Longitude", "Gender",
        "Senior Citizen", "Churn", "Churn Label", "CLTV", "Charges Band", "Churn Reason"
    ]
    raw_cols = [c for c in raw_cols if c in raw.columns]

    df = churn.merge(raw[raw_cols], on="CustomerID", how="left", suffixes=("", "_raw"))
    df = normalize_raw_columns(df)

    # Fill missing important fields from raw columns if they still exist only as suffixed versions
    for base in ["Contract", "Internet Service", "Monthly Charges", "Tenure Months", "Payment Method", "Latitude", "Longitude", "CLTV", "Churn Label", "Churn"]:
        if base not in df.columns:
            for suffix in ["_raw", "_x", "_y"]:
                alt = f"{base}{suffix}"
                if alt in df.columns:
                    df[base] = df[alt]
                    break
        elif df[base].isna().any():
            for suffix in ["_raw", "_x", "_y"]:
                alt = f"{base}{suffix}"
                if alt in df.columns:
                    df[base] = df[base].fillna(df[alt])

    return df, raw


@st.cache_resource(show_spinner=False)
def load_optional_artifacts():
    artifacts = {}
    if MODEL_FILE.exists():
        try:
            import joblib
            artifacts["model"] = joblib.load(MODEL_FILE)
        except Exception:
            artifacts["model"] = None
    else:
        artifacts["model"] = None

    if SCALER_FILE.exists():
        try:
            import joblib
            artifacts["scaler"] = joblib.load(SCALER_FILE)
        except Exception:
            artifacts["scaler"] = None
    else:
        artifacts["scaler"] = None

    if FEATURE_COLS_FILE.exists():
        try:
            import joblib
            artifacts["feature_cols"] = joblib.load(FEATURE_COLS_FILE)
        except Exception:
            artifacts["feature_cols"] = None
    else:
        artifacts["feature_cols"] = None

    if LE_DICT_FILE.exists():
        try:
            import joblib
            artifacts["le_dict"] = joblib.load(LE_DICT_FILE)
        except Exception:
            artifacts["le_dict"] = None
    else:
        artifacts["le_dict"] = None

    return artifacts


def revenue_at_risk(df: pd.DataFrame) -> float:
    risk_df = df[df["Risk_Tier"].isin(["High Risk", "Medium Risk"])]
    if "Monthly Charges" in risk_df.columns:
        return float(risk_df["Monthly Charges"].sum() * 12)
    return float("nan")


def make_risk_ordered_counts(df: pd.DataFrame) -> pd.DataFrame:
    order = ["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
    out = df["Risk_Tier"].value_counts().reindex(order).fillna(0).reset_index()
    out.columns = ["Tier", "Count"]
    return out


# -----------------------------
# Load data
# -----------------------------
try:
    df, raw_df = load_data()
except Exception as e:
    st.error(str(e))
    st.stop()

artifacts = load_optional_artifacts()

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown("## 📡 Telco Churn Intel")
    st.caption("IBM Telco churn dashboard + strategy simulator")
    st.divider()
    page = st.radio(
        "",
        ["📊 Overview Dashboard", "🎯 Strategy Simulator"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("### Files in use")
    st.write(f"✅ {CHURN_FILE.name}")
    st.write(f"✅ {RAW_FILE.name}")
    if artifacts["model"] is not None:
        st.write("✅ best_model.pkl")
    else:
        st.write("ℹ️ best_model.pkl not found")

# -----------------------------
# Shared columns
# -----------------------------
contract_col = first_existing_column(df, ["Contract", "Contract_x", "Contract_y", "Contract_raw"])
internet_col = first_existing_column(df, ["Internet Service", "Internet Service_x", "Internet Service_y", "Internet Service_raw"])
monthly_col = first_existing_column(df, ["Monthly Charges", "Monthly Charges_x", "Monthly Charges_y", "Monthly Charges_raw"])
churn_col = first_existing_column(df, ["Churn", "Churn_x", "Churn_y", "Churn_raw"])
label_col = first_existing_column(df, ["Churn Label", "Churn Label_x", "Churn Label_y", "Churn Label_raw"])

# -----------------------------
# Dashboard tab
# -----------------------------
if page == "📊 Overview Dashboard":
    st.title("📊 Customer Churn Overview")
    st.caption("Overview based on your exported churn probabilities and customer data.")

    total = len(df)
    actual_churn = int(df[label_col].eq("Yes").sum()) if label_col else int(df[churn_col].sum())
    churn_rate = actual_churn / total * 100 if total else 0
    predicted_churn = int(df["Predicted_Churn"].sum()) if "Predicted_Churn" in df.columns else int((df["Churn_Probability"] >= 0.5).sum())
    high_risk = int(df["Risk_Tier"].eq("High Risk").sum()) if "Risk_Tier" in df.columns else 0
    arr = revenue_at_risk(df)
    avg_prob = float(df["Churn_Probability"].mean() * 100)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers", f"{total:,}")
    c2.metric("🔴 Actual Churned", f"{actual_churn:,}", f"{churn_rate:.1f}%")
    c3.metric("🤖 Predicted Churn", f"{predicted_churn:,}")
    c4.metric("⚠️ High Risk", f"{high_risk:,}")
    c5.metric("💸 Annual Rev at Risk", f"${arr/1e6:.2f}M")

    st.divider()

    col1, col2, col3 = st.columns([1.1, 1.1, 1.8])

    with col1:
        st.subheader("Risk Tiers")
        rc = make_risk_ordered_counts(df)
        fig = px.bar(
            rc,
            x="Tier",
            y="Count",
            color="Tier",
            color_discrete_sequence=["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"],
            text="Count",
            height=300,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5, b=0), xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract")
        if contract_col:
            cc = df.groupby(contract_col)[churn_col].mean().sort_values(ascending=False).reset_index()
            cc["pct"] = (cc[churn_col] * 100).round(1)
            fig2 = px.bar(
                cc,
                x=contract_col,
                y="pct",
                color="pct",
                color_continuous_scale=["#66BB6A", "#FFA726", "#EF5350"],
                text=cc["pct"].apply(lambda x: f"{x:.1f}%"),
                height=300,
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(showlegend=False, coloraxis_showscale=False, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5, b=0), yaxis_title="Churn Rate (%)")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Contract column not found.")

    with col3:
        st.subheader("Churn Probability Distribution")
        fig3 = go.Figure()
        for label, color, val in [("Retained", "#42A5F5", 0), ("Churned", "#EF5350", 1)]:
            if churn_col in df.columns:
                subset = df[df[churn_col] == val]
            else:
                subset = df[df["Predicted_Churn"] == val]
            fig3.add_trace(go.Histogram(x=subset["Churn_Probability"], name=label, opacity=0.65, marker_color=color, nbinsx=40))
        fig3.add_vline(x=0.5, line_dash="dash", line_color="gray", annotation_text="Threshold=0.5")
        fig3.update_layout(barmode="overlay", height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=5, b=0), legend=dict(x=0.65, y=0.95), xaxis_title="Churn Probability")
        st.plotly_chart(fig3, use_container_width=True)

    col4, col5 = st.columns([1, 1.5])

    with col4:
        st.subheader("Contract × Internet Heatmap")
        if contract_col and internet_col and churn_col in df.columns:
            pivot = df.groupby([contract_col, internet_col])[churn_col].mean().unstack().round(3) * 100
            fig4 = px.imshow(pivot, text_auto=".1f", color_continuous_scale="RdYlGn_r", zmin=0, zmax=80, aspect="auto", height=300)
            fig4.update_layout(margin=dict(t=5), paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Need Contract, Internet Service and Churn columns.")

    with col5:
        st.subheader("Geographic Churn Map — California")
        lat_col = first_existing_column(df, ["Latitude", "Latitude_x", "Latitude_y", "Latitude_raw"])
        lon_col = first_existing_column(df, ["Longitude", "Longitude_x", "Longitude_y", "Longitude_raw"])
        if lat_col and lon_col:
            geo = df.dropna(subset=[lat_col, lon_col]).copy()
            geo[lat_col] = pd.to_numeric(geo[lat_col], errors="coerce")
            geo[lon_col] = pd.to_numeric(geo[lon_col], errors="coerce")
            geo = geo[geo[lat_col].between(32, 42) & geo[lon_col].between(-125, -114)]

            if len(geo):
                hover_data = {"CustomerID": True, "Churn_Probability": ":.3f", "Risk_Tier": True}
                if monthly_col and monthly_col in geo.columns:
                    geo[monthly_col] = pd.to_numeric(geo[monthly_col], errors="coerce")
                    hover_data[monthly_col] = ":.0f"
                if contract_col and contract_col in geo.columns:
                    hover_data[contract_col] = True
                if internet_col and internet_col in geo.columns:
                    hover_data[internet_col] = True

                try:
                    fig5 = px.scatter_mapbox(
                        geo,
                        lat=lat_col,
                        lon=lon_col,
                        color="Churn_Probability",
                        size=monthly_col if monthly_col and monthly_col in geo.columns else None,
                        color_continuous_scale=["#66BB6A", "#FFA726", "#EF5350"],
                        range_color=[0, 1],
                        size_max=10,
                        zoom=5,
                        height=300,
                        hover_data=hover_data,
                        mapbox_style="open-street-map",
                    )
                    fig5.update_layout(margin=dict(t=5, b=0), paper_bgcolor="rgba(0,0,0,0)")
                except Exception:
                    # Fallback that does not depend on Mapbox.
                    fig5 = px.scatter_geo(
                        geo,
                        lat=lat_col,
                        lon=lon_col,
                        color="Churn_Probability",
                        size=monthly_col if monthly_col and monthly_col in geo.columns else None,
                        color_continuous_scale=["#66BB6A", "#FFA726", "#EF5350"],
                        range_color=[0, 1],
                        size_max=10,
                        hover_data=hover_data,
                        projection="albers usa",
                        height=300,
                    )
                    fig5.update_geos(fitbounds="locations", visible=False)
                    fig5.update_layout(margin=dict(t=5, b=0), paper_bgcolor="rgba(0,0,0,0)")

                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("No California coordinates after filtering.")
        else:
            st.info("Latitude/Longitude not found in data.")

    st.subheader("🔴 Top 20 Highest Risk Customers")
    top_cols = [c for c in ["CustomerID", "Contract", "Monthly Charges", "Tenure Months", "Internet Service", "Churn_Probability", "Risk_Tier", "CLTV"] if c in df.columns]
    top20 = df.nlargest(20, "Churn_Probability")[top_cols].reset_index(drop=True)
    top20.index += 1
    st.dataframe(
        top20.style.format({
            "Churn_Probability": "{:.3f}",
            "Monthly Charges": "${:.2f}",
            "CLTV": "${:,.0f}",
        }),
        use_container_width=True,
        height=420,
    )

# -----------------------------
# Simulator tab
# -----------------------------
else:
    st.title("🎯 Pricing Strategy Simulator")
    st.caption("Define a strategy, then estimate churn and revenue impact using Monte Carlo simulation.")

    st.info(
        "This simulator uses your exported churn probabilities. If best_model.pkl/scaler.pkl/feature_columns.pkl are added later, you can extend it to live re-scoring."
    )

    with st.form("strategy_form"):
        st.subheader("⚙️ Design Your Strategy")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Target Segment**")
            contract_options = sorted(df[contract_col].dropna().unique().tolist()) if contract_col else []
            risk_options = ["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
            internet_options = sorted(df[internet_col].dropna().unique().tolist()) if internet_col else []

            target_contract = st.multiselect("Contract Type", contract_options, default=contract_options[:1] if contract_options else [])
            target_risk = st.multiselect("Risk Tier", risk_options, default=["High Risk", "Medium Risk"])
            target_internet = st.multiselect("Internet Service", internet_options, default=internet_options[:2] if len(internet_options) >= 2 else internet_options)

        with col_b:
            st.markdown("**Strategy Parameters**")
            price_change_pct = st.slider(
                "💰 Price change (%)",
                -30, 50, -10, 5,
                help="Negative = discount; positive = price increase",
            )
            retention_rate = st.slider(
                "🎯 Expected retention rate (%)",
                0, 100, 50, 10,
                help="% of churned target customers that the strategy saves",
            )
            n_sim = st.select_slider("🎲 Monte Carlo runs", options=[200, 500, 1000], value=500, help="More runs = stabler results, but slower runtime.")
            strategy_desc = st.text_input("📝 Strategy Name", placeholder="e.g. 10% discount for month-to-month high-risk fiber customers")

        submitted = st.form_submit_button("🚀 Run Simulation", type="primary", use_container_width=True)

    def run_strategy_simulation(
        base_df: pd.DataFrame,
        target_contracts: list[str],
        target_risks: list[str],
        target_internet: list[str],
        price_change_pct: float,
        retention_rate: float,
        n_sim: int,
    ):
        if not {"Churn_Probability", "Monthly Charges", "Risk_Tier"}.issubset(base_df.columns):
            raise ValueError("Missing required columns in base data.")

        charge_band_elasticity = {
            "Low (<$35)": 0.8,
            "Medium ($35-65)": 0.5,
            "High ($65-95)": 0.2,
            "Premium (>$95)": 0.2,
        }

        base = base_df.copy()
        mask = pd.Series(True, index=base.index)
        if target_contracts and contract_col:
            mask &= base[contract_col].isin(target_contracts)
        if target_risks:
            mask &= base["Risk_Tier"].isin(target_risks)
        if target_internet and internet_col:
            mask &= base[internet_col].isin(target_internet)

        n_targeted = int(mask.sum())
        if n_targeted == 0:
            return None, "No customers match the selected filters. Broaden the segment."

        base_revenues = []
        strat_revenues = []
        base_churns = []
        strat_churns = []

        rng = np.random.default_rng(42)
        targeted_idx = np.where(mask.values)[0]

        # Baseline and strategy scenarios run side by side
        for _ in range(n_sim):
            # Baseline sample
            base_churn = rng.binomial(1, base["Churn_Probability"].to_numpy())
            base_rev = base.loc[base_churn == 0, "Monthly Charges"].sum()

            # Strategy sample
            strat = base.copy()
            if price_change_pct != 0:
                strat.loc[mask, "Monthly Charges"] *= (1 + price_change_pct / 100)
                if "Charges Band" in strat.columns:
                    adjusted = []
                    for _, row in strat.iterrows():
                        elasticity = charge_band_elasticity.get(row.get("Charges Band", "Medium ($35-65)"), 0.5)
                        p = row["Churn_Probability"] * (1 + (price_change_pct / 100) * elasticity)
                        adjusted.append(float(np.clip(p, 0, 1)))
                    strat["Churn_Probability"] = adjusted
                else:
                    strat["Churn_Probability"] = np.clip(
                        strat["Churn_Probability"] * (1 + (price_change_pct / 100) * 0.4),
                        0,
                        1,
                    )

            strat_churn = rng.binomial(1, strat["Churn_Probability"].to_numpy())

            # Retention: save a fraction of churned targeted customers
            if retention_rate > 0:
                churned_target = np.array([idx for idx in targeted_idx if strat_churn[idx] == 1])
                if len(churned_target) > 0:
                    retain_n = int(len(churned_target) * (retention_rate / 100))
                    if retain_n > 0:
                        saved_idx = rng.choice(churned_target, retain_n, replace=False)
                        strat_churn[saved_idx] = 0

            strat_rev = strat.loc[strat_churn == 0, "Monthly Charges"].sum()

            base_revenues.append(base_rev)
            strat_revenues.append(strat_rev)
            base_churns.append(int(base_churn.sum()))
            strat_churns.append(int(strat_churn.sum()))

        return {
            "n_targeted": n_targeted,
            "base_revenues": np.array(base_revenues),
            "strat_revenues": np.array(strat_revenues),
            "base_churns": np.array(base_churns),
            "strat_churns": np.array(strat_churns),
        }, None

    if not submitted:
        st.info("Choose a segment and press **Run Simulation**.")
        with st.expander("How the simulation works"):
            st.markdown(
                """
                - Start from the churn probabilities exported by Colab.
                - Apply the selected pricing change to the chosen segment.
                - Re-sample churn with Monte Carlo.
                - Retain a chosen share of churned target customers.
                - Compare baseline vs strategy revenue and churn.
                """
            )
    else:
        if not target_contract or not target_risk or not target_internet:
            st.error("Please select at least one option in each filter.")
            st.stop()

        sim_result, err = run_strategy_simulation(
            base_df=df,
            target_contracts=target_contract,
            target_risks=target_risk,
            target_internet=target_internet,
            price_change_pct=price_change_pct,
            retention_rate=retention_rate,
            n_sim=n_sim,
        )

        if err:
            st.error(err)
            st.stop()

        n_targeted = sim_result["n_targeted"]
        base_revenues = sim_result["base_revenues"]
        strat_revenues = sim_result["strat_revenues"]
        base_churns = sim_result["base_churns"]
        strat_churns = sim_result["strat_churns"]

        b_rev = base_revenues.mean()
        s_rev = strat_revenues.mean()
        b_ch = base_churns.mean()
        s_ch = strat_churns.mean()
        uplift = s_rev - b_rev
        saved = b_ch - s_ch
        cost = n_targeted * df.loc[df.index.isin(np.where(
            df[contract_col].isin(target_contract) & df["Risk_Tier"].isin(target_risk) & df[internet_col].isin(target_internet)
        )[0]), monthly_col].mean() * max(0, -price_change_pct / 100) if monthly_col and price_change_pct < 0 else 0
        roi = (uplift / cost * 100) if cost and cost > 0 else float("inf")

        if strategy_desc:
            st.info(f"📝 **{strategy_desc}**")

        st.divider()
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Customers Targeted", f"{n_targeted:,}")
        k2.metric("Baseline Rev/mo", f"${b_rev:,.0f}")
        k3.metric("Strategy Rev/mo", f"${s_rev:,.0f}", delta=f"${uplift:+,.0f}", delta_color="normal")
        k4.metric("Avg Churn Saved/mo", f"{saved:.0f}", delta=f"↓{(saved / b_ch * 100 if b_ch else 0):.1f}%", delta_color="normal")
        k5.metric("ROI on Discount Cost", f"{roi:.0f}%" if np.isfinite(roi) else "N/A")

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 Revenue Distribution")
            fig_r = go.Figure()
            fig_r.add_trace(go.Violin(y=base_revenues, name="Baseline", fillcolor="#42A5F5", opacity=0.65, box_visible=True, meanline_visible=True, line_color="gray"))
            fig_r.add_trace(go.Violin(y=strat_revenues, name="With Strategy", fillcolor="#66BB6A", opacity=0.65, box_visible=True, meanline_visible=True, line_color="gray"))
            fig_r.update_layout(height=370, yaxis_title="Monthly Revenue ($)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10), yaxis=dict(tickformat="$,.0f"))
            st.plotly_chart(fig_r, use_container_width=True)

        with col2:
            st.subheader("📉 Churn Count Distribution")
            fig_c = go.Figure()
            fig_c.add_trace(go.Histogram(x=base_churns, name="Baseline", opacity=0.65, marker_color="#EF5350", nbinsx=30))
            fig_c.add_trace(go.Histogram(x=strat_churns, name="With Strategy", opacity=0.65, marker_color="#66BB6A", nbinsx=30))
            fig_c.update_layout(barmode="overlay", height=370, xaxis_title="Churned Customers / month", yaxis_title="Frequency", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10), legend=dict(x=0.65, y=0.95))
            st.plotly_chart(fig_c, use_container_width=True)

        st.subheader("📋 Statistical Summary")
        summary = pd.DataFrame({
            "Metric": ["Monthly Revenue ($)", "Monthly Churn Count", "Annual Revenue ($)"],
            "Baseline Mean": [f"${b_rev:,.0f}", f"{b_ch:.0f}", f"${b_rev * 12:,.0f}"],
            "Baseline P5–P95": [
                f"${np.percentile(base_revenues, 5):,.0f} – ${np.percentile(base_revenues, 95):,.0f}",
                f"{np.percentile(base_churns, 5):.0f} – {np.percentile(base_churns, 95):.0f}",
                f"${np.percentile(base_revenues, 5) * 12:,.0f} – ${np.percentile(base_revenues, 95) * 12:,.0f}",
            ],
            "Strategy Mean": [f"${s_rev:,.0f}", f"{s_ch:.0f}", f"${s_rev * 12:,.0f}"],
            "Strategy P5–P95": [
                f"${np.percentile(strat_revenues, 5):,.0f} – ${np.percentile(strat_revenues, 95):,.0f}",
                f"{np.percentile(strat_churns, 5):.0f} – {np.percentile(strat_churns, 95):.0f}",
                f"${np.percentile(strat_revenues, 5) * 12:,.0f} – ${np.percentile(strat_revenues, 95) * 12:,.0f}",
            ],
            "Δ Change": [f"${uplift:+,.0f}/mo", f"{saved:+.0f} saved/mo", f"${uplift * 12:+,.0f}/yr"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        st.subheader("🔍 Sensitivity: Price Change vs Revenue Uplift")
        disc_range = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
        u_mean, u_p5, u_p95 = [], [], []
        for d in disc_range:
            sim, _ = run_strategy_simulation(
                base_df=df,
                target_contracts=target_contract,
                target_risks=target_risk,
                target_internet=target_internet,
                price_change_pct=d,
                retention_rate=retention_rate,
                n_sim=min(200, n_sim),
            )
            upl = sim["strat_revenues"] - sim["base_revenues"]
            u_mean.append(float(upl.mean()))
            u_p5.append(float(np.percentile(upl, 5)))
            u_p95.append(float(np.percentile(upl, 95)))

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(x=disc_range + disc_range[::-1], y=u_p95 + u_p5[::-1], fill="toself", fillcolor="rgba(31,119,180,0.12)", line=dict(color="rgba(0,0,0,0)"), name="P5–P95"))
        fig_s.add_trace(go.Scatter(x=disc_range, y=u_mean, mode="lines+markers", name="Mean Uplift", line=dict(color="#1f77b4", width=2.5), marker=dict(size=8)))
        fig_s.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Break-even")
        fig_s.update_layout(height=330, xaxis_title="Price Change (%)", yaxis_title="Revenue Uplift vs Baseline ($)", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(tickformat="$,.0f"), margin=dict(t=10), legend=dict(x=0.75, y=0.95))
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("Blue band = uncertainty range. Stay above the red line for positive uplift.")

