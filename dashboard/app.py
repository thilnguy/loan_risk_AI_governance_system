import os
import json
import yaml
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI Governance Dashboard", layout="wide", page_icon="🛡️")

# --- Constants & Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
POLICIES_PATH = os.path.join(BASE_DIR, "policies", "rules.yaml")
DRIFT_PATH = os.path.join(BASE_DIR, "monitoring", "drift_results.json")
PERF_PATH = os.path.join(BASE_DIR, "monitoring", "perf_results.json")
LOGS_PATH = os.path.join(BASE_DIR, "data", "production_logs.csv")

st.title("🛡️ L5 Compliance-Ready AI Governance")
st.markdown("Real-time monitoring and enforcement of European Union AI Act requirements.")

# --- 1. Load Data ---
@st.cache_data(ttl=60)
def load_policies():
    if os.path.exists(POLICIES_PATH):
        with open(POLICIES_PATH) as f:
            return yaml.safe_load(f)["policies"]
    return {}

@st.cache_data(ttl=60)
def load_drift():
    if os.path.exists(DRIFT_PATH):
        with open(DRIFT_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=60)
def load_perf():
    if os.path.exists(PERF_PATH):
        with open(PERF_PATH) as f:
            return json.load(f)
    return {}

@st.cache_data(ttl=10)
def load_logs():
    if os.path.exists(LOGS_PATH):
        try:
            return pd.read_csv(LOGS_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

policies = load_policies()
drift_data = load_drift()
perf_data = load_perf()
logs_df = load_logs()

# --- 2. System Status & Policies ---
st.header("🚦 Governance-as-Code Engine Status")

col1, col2, col3, col4 = st.columns(4)

drift_count = sum(1 for feat in drift_data.values() if feat.get("psi", 0) > 0.2)
cb_active = drift_count >= policies.get("circuit_breaker_max_drifted_features", 2)

col1.metric("Auto-Approve Threshold", f"{policies.get('auto_approve_threshold', 0)*100}%")
col2.metric("Auto-Decline Threshold", f"{policies.get('auto_decline_threshold', 0)*100}%")
col3.metric("Drifted Features", drift_count, delta="Safe" if not cb_active else "WARN", delta_color="inverse")
col4.metric("Circuit Breaker", "TRIPPED 🔴" if cb_active else "ONLINE 🟢")

st.divider()

# --- 3. Production Inference Logs ---
st.header("📋 Real-time Inference Transparency (Art. 72)")
if not logs_df.empty:
    st.dataframe(logs_df.tail(10).sort_values("timestamp", ascending=False), use_container_width=True)
    
    # Decisions distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Breakdown")
        fig1 = px.pie(logs_df, names="decision", title="", hole=0.4, color="decision",
                      color_discrete_map={"APPROVED": "green", "REVIEW": "orange", "DECLINED": "red"})
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        if "risk_level" in logs_df.columns:
            st.subheader("Risk Level Exposure")
            fig2 = px.bar(logs_df["risk_level"].value_counts().reset_index(), x="risk_level", y="count", color="risk_level")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No risk level data available in logs yet.")
else:
    st.info("No production logs found. Generate inferences via API.")

st.divider()

# --- 4. Deep Metrics ---
col1, col2 = st.columns(2)

with col1:
    st.header("📈 Data Drift Monitoring")
    if drift_data:
        df_drift = pd.DataFrame([{"Feature": k, "PSI": v["psi"], "Drifted": v["drifted"]} for k,v in drift_data.items()])
        fig3 = px.bar(df_drift, x="Feature", y="PSI", color="Drifted", 
                      color_discrete_map={True: "red", False: "green"})
        fig3.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Threshold")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("No drift data found.")

with col2:
    st.header("📉 Performance Degradation")
    if perf_data:
        metrics = []
        for split, vals in perf_data.items():
            for m, score in vals.items():
                if m != "n_samples":
                    metrics.append({"Split": split, "Metric": m, "Score": score})
        df_perf = pd.DataFrame(metrics)
        fig4 = px.bar(df_perf, x="Metric", y="Score", color="Split", barmode="group")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No performance data found.")
