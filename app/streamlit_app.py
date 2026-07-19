"""
AIgnition 2026 — Probabilistic Revenue Forecasting: demo dashboard.

The product/judging layer (NOT on the scored run.sh path). Ties the whole story
together:
  * Data ingestion + a data-quality / campaign-consistency report.
  * Probabilistic forecasts (P10/P50/P90) for revenue & ROAS at 30/60/90 days,
    across blended / channel / campaign-type grains, as fan charts.
  * A budget simulator: scale per-channel spend and watch revenue & blended
    ROAS respond (native, because planned spend is a model feature).
  * An AI causal-insights panel (Groq/OpenAI-compatible, offline fallback).

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.common.schema import daily_channel_totals, load_channel_data
from src.forecasting.features import HORIZONS, build_inference
from src.forecasting.model import ForecastModel  # noqa: F401  (unpickle)
from src.insights import LLMConfig, generate_insights

DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./pickle/model.pkl")

st.set_page_config(page_title="AIgnition · Revenue Forecaster",
                   page_icon="📈", layout="wide")

PALETTE = {"google": "#4285F4", "bing": "#00897B", "meta": "#C2185B",
           "ALL": "#5E35B1"}


# ── loaders (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(data_dir):
    df, report = load_channel_data(data_dir)
    return df, report


@st.cache_data(show_spinner=False)
def base_inference(_long_df):
    """Build the base feature matrix ONCE (run-rate budget, mult=1.0). This is
    the expensive part (per-campaign daily series); it does NOT depend on the
    sliders, so we compute it a single time and reuse it for every multiplier."""
    X, meta = build_inference(_long_df, HORIZONS, budget_multipliers=None)
    return X, meta


@st.cache_data(show_spinner=False)
def forecast(_model, _X_base, _meta_base, budget_mult_key):
    """Apply a per-channel budget multiplier and predict. Changing the budget
    only scales ``planned_spend`` / ``planned_vs_runrate`` (everything else is
    identical), so this is a cheap column operation + a 0.25s predict — no
    DataFrame re-hashing, no rebuilding features. Only the small multiplier
    tuple is hashed for the cache."""
    mult = dict(budget_mult_key)
    factors = _meta_base["channel"].map(lambda c: mult.get(c, 1.0)).to_numpy()
    X = _X_base.copy()
    meta = _meta_base.copy()
    X["planned_spend"] = _X_base["planned_spend"].to_numpy() * factors
    X["planned_vs_runrate"] = factors
    meta["planned_spend"] = _meta_base["planned_spend"].to_numpy() * factors
    return _model.predict_frame(X, meta)


@st.cache_data(show_spinner=False)
def hist_blended_daily(_long_df):
    """Historical blended daily revenue (7d smoothed) for the fan chart. Does
    not depend on the sliders, so compute once and cache."""
    daily = daily_channel_totals(_long_df)
    return (daily.groupby("date")["revenue"].sum()
            .rolling(7, min_periods=1).mean().reset_index())


def money(x):
    return f"${x:,.0f}"


def _grain_bar(preds, grain, horizon):
    """Horizontal bar of P50 revenue per entity with P10–P90 error bars."""
    d = preds[(preds.grain == grain) & (preds.metric == "revenue") &
              (preds.horizon_days == horizon)].sort_values("p50")
    if d.empty:
        st.info("No data at this grain.")
        return
    color = [PALETTE.get(e, "#5E35B1") for e in d["entity"]]
    fig = go.Figure(go.Bar(
        x=d["p50"], y=d["entity"], orientation="h", marker_color=color,
        error_x=dict(type="data", symmetric=False,
                     array=(d["p90"] - d["p50"]), arrayminus=(d["p50"] - d["p10"])),
    ))
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10),
                      xaxis_title=f"Revenue · next {horizon}d (P50 ± P10/P90)")
    st.plotly_chart(fig, width='stretch')


# ── header ─────────────────────────────────────────────────────────────────
st.title("📈 Probabilistic Revenue & ROAS Forecaster")
st.caption("AIgnition 2026 · NetElixir — AI-assisted forecasting utility for "
           "digital marketing agencies")

try:
    model = load_model(MODEL_PATH)
    long_df, report = load_data(DATA_DIR)
except Exception as e:  # pragma: no cover
    st.error(f"Could not load model/data: {e}")
    st.stop()

channels = report.channels_found

# ── sidebar: controls ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Controls")
    horizon = st.radio("Forecast horizon (days)", HORIZONS, index=2,
                       horizontal=True)
    st.divider()
    st.subheader("💰 Budget simulator")
    st.caption("Scale planned spend vs. the trailing run-rate, per channel.")
    mult = {}
    for ch in channels:
        mult[ch] = st.slider(f"{ch.title()} spend", 0.5, 2.0, 1.0, 0.05,
                             key=f"mult_{ch}")
    st.divider()
    cfg = LLMConfig.from_env()
    st.caption(f"LLM: {'🟢 ' + cfg.model if cfg.enabled else '⚪ offline fallback'}")

mult_key = tuple(sorted(mult.items()))
base_key = tuple((ch, 1.0) for ch in sorted(channels))
X_base, meta_base = base_inference(long_df)
preds = forecast(model, X_base, meta_base, mult_key)
base_preds = forecast(model, X_base, meta_base, base_key)


def blended_row(p, h, metric):
    r = p[(p.grain == "blended") & (p.metric == metric) & (p.horizon_days == h)]
    return r.iloc[0] if len(r) else None


# ── KPI cards ──────────────────────────────────────────────────────────────
st.subheader(f"Blended forecast · next {horizon} days")
c1, c2, c3 = st.columns(3)
rev = blended_row(preds, horizon, "revenue")
roas = blended_row(preds, horizon, "roas")
base_rev = blended_row(base_preds, horizon, "revenue")
delta_rev = rev.p50 - base_rev.p50
c1.metric("Expected revenue (P50)", money(rev.p50),
          delta=(money(delta_rev) + " vs run-rate") if abs(delta_rev) > 1 else None)
c2.metric("Revenue range (P10–P90)", f"{money(rev.p10)} – {money(rev.p90)}")
c3.metric("Blended ROAS (P50)", f"{roas.p50:.2f}x",
          delta=f"{roas.p10:.2f}–{roas.p90:.2f}x range", delta_color="off")

# ── fan chart: history + forecast cone ─────────────────────────────────────
st.subheader("Revenue trajectory & forecast cone")
blended_daily = hist_blended_daily(long_df)
anchor = long_df["date"].max()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=blended_daily["date"], y=blended_daily["revenue"],
    name="Historical daily revenue (7d avg)", line=dict(color="#90A4AE", width=1.5)))

# forecast cone: convert each horizon's cumulative range into an avg daily rate
cone_x, p50_y, p10_y, p90_y = [anchor], [blended_daily["revenue"].iloc[-1]], \
    [blended_daily["revenue"].iloc[-1]], [blended_daily["revenue"].iloc[-1]]
for h in HORIZONS:
    r = blended_row(preds, h, "revenue")
    cone_x.append(anchor + pd.Timedelta(days=h))
    p50_y.append(r.p50 / h)
    p10_y.append(r.p10 / h)
    p90_y.append(r.p90 / h)
fig.add_trace(go.Scatter(x=cone_x + cone_x[::-1], y=p90_y + p10_y[::-1],
                         fill="toself", fillcolor="rgba(94,53,177,0.15)",
                         line=dict(width=0), name="P10–P90 band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=cone_x, y=p50_y, name="Forecast (P50 avg daily)",
                         line=dict(color=PALETTE["ALL"], width=2.5, dash="dot")))
fig.update_layout(height=380, margin=dict(t=10, b=10), hovermode="x unified",
                  yaxis_title="Revenue / day", legend=dict(orientation="h"))
st.plotly_chart(fig, width='stretch')
st.caption("Forecast shown as average daily revenue implied by each horizon's "
           "cumulative P10/P50/P90 — the cone widens with horizon as uncertainty grows.")

# ── grain breakdown ────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("By channel")
    _grain_bar(preds, "channel", horizon)
with right:
    st.subheader("By campaign type")
    _grain_bar(preds, "campaign_type", horizon)

# ── budget response curve ──────────────────────────────────────────────────
st.subheader("Budget response curve")
sel = st.selectbox("Sweep spend for channel", channels, index=0)
sweep = np.round(np.arange(0.5, 2.01, 0.1), 2)
rows = []
for s in sweep:
    m = {ch: 1.0 for ch in channels}
    m[sel] = float(s)
    p = forecast(model, X_base, meta_base, tuple(sorted(m.items())))
    r = blended_row(p, horizon, "revenue")
    ro = blended_row(p, horizon, "roas")
    rows.append({"mult": s, "revenue": r.p50, "roas": ro.p50})
resp = pd.DataFrame(rows)
rc1, rc2 = st.columns(2)
for col, (metric, title, color) in zip(
        (rc1, rc2), [("revenue", "Blended revenue (P50)", PALETTE["ALL"]),
                     ("roas", "Blended ROAS (P50)", "#00897B")]):
    f = go.Figure(go.Scatter(x=resp["mult"], y=resp[metric], mode="lines+markers",
                             line=dict(color=color, width=2.5)))
    f.add_vline(x=mult[sel], line=dict(color="#B0BEC5", dash="dash"))
    f.update_layout(height=300, margin=dict(t=30, b=10), title=title,
                    xaxis_title=f"{sel.title()} spend multiplier")
    col.plotly_chart(f, width='stretch')
st.caption("Diminishing returns: revenue rises with spend while ROAS typically "
           "compresses — the crossover is where extra budget stops paying off.")

# ── AI causal insights ─────────────────────────────────────────────────────
st.subheader("🧠 AI causal insights")
if st.button("Generate insights", type="primary"):
    with st.spinner("Reasoning over the forecast…"):
        st.session_state["insights"] = generate_insights(
            long_df, preds, meta_assumed_roas=report.meta_assumed_roas)
ins = st.session_state.get("insights")
if ins:
    badge = "🟢 LLM-generated" if ins["source"] == "llm" else "⚪ Offline rule-based"
    st.caption(f"Source: {badge}")
    st.markdown(ins["text"])
else:
    st.info("Click **Generate insights** for an AI-written forecast briefing "
            "(works offline via a rule-based fallback if no LLM key is set).")

# ── data-quality report ────────────────────────────────────────────────────
with st.expander("🔎 Data-quality & consistency report"):
    st.write(f"**Files read:** {', '.join(report.files_read)}")
    st.write(f"**Date range:** {report.date_min} → {report.date_max}")
    st.write(f"**Rows per channel:** {report.rows_per_channel}")
    st.warning(f"**Meta revenue is imputed** at spend × assumed ROAS "
               f"**{report.meta_assumed_roas}x** (Meta reports no revenue; its "
               f"conversion field is a broad metric, not purchases).")
    if report.warnings:
        for w in report.warnings:
            st.write("- " + w)
    st.dataframe(preds, width='stretch', height=260)
