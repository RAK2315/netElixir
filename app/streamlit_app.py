"""
AIgnition 3.0 - Probabilistic Revenue & ROAS Forecaster (demo dashboard).

The product / demo layer (NOT on the scored run.sh path). It tells the whole
story top to bottom, with a plain-English caption under every chart:
  1. What the tool is + the data it read.
  2. Headline forecast (revenue & ROAS) as a low-middle-high range.
  3. A cumulative forecast "cone" that widens with time (honest uncertainty).
  4. Breakdown by channel and campaign type.
  5. A budget simulator (slide spend, watch revenue & ROAS respond).
  6. An AI-written causal briefing (Groq / offline fallback).

Run locally:   streamlit run app/streamlit_app.py
On Streamlit Cloud: set main file = app/streamlit_app.py and add LLM_API_KEY in
Secrets (optional; the AI panel falls back to an offline explanation without it).
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
from src.forecasting.model import ForecastModel  # noqa: F401  (needed to unpickle)
from src.insights import LLMConfig, generate_insights

DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_PATH = os.getenv("MODEL_PATH", "./pickle/model.pkl")
N_SIMS = 400  # Monte-Carlo draws for aggregate ranges (fast + stable in the app)

st.set_page_config(page_title="AIgnition Revenue Forecaster", layout="wide")

# Bridge Streamlit Cloud "Secrets" into env vars so the LLM layer picks them up.
try:  # pragma: no cover
    for _k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

PALETTE = {"google": "#4285F4", "bing": "#00897B", "meta": "#C2185B", "ALL": "#5E35B1"}
ACCENT = "#5E35B1"
CHART_CFG = {"displayModeBar": False}


# -- cached loaders / compute -----------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_data(data_dir):
    return load_channel_data(data_dir)


@st.cache_data(show_spinner=False)
def base_inference(_long_df):
    """Expensive per-campaign feature build - done once, reused for every budget."""
    return build_inference(_long_df, HORIZONS, budget_multipliers=None)


@st.cache_data(show_spinner=False)
def forecast(_model, _X_base, _meta_base, budget_mult_key):
    """A budget change only scales planned spend, so this is a cheap column op +
    a fast predict. Only the small multiplier tuple is hashed for the cache."""
    mult = dict(budget_mult_key)
    factors = _meta_base["channel"].map(lambda c: mult.get(c, 1.0)).to_numpy()
    X, meta = _X_base.copy(), _meta_base.copy()
    X["planned_spend"] = _X_base["planned_spend"].to_numpy() * factors
    X["planned_vs_runrate"] = factors
    meta["planned_spend"] = _meta_base["planned_spend"].to_numpy() * factors
    return _model.predict_frame(X, meta, n_sims=N_SIMS)


@st.cache_data(show_spinner=False)
def hist_daily(_long_df, days=180):
    d = daily_channel_totals(_long_df).groupby("date")["revenue"].sum()
    d = d.rolling(7, min_periods=1).mean().reset_index().tail(days)
    return d


def _blended_point(_model, _X, _meta, mult, horizon):
    """Fast blended P50 revenue & ROAS for a budget (no Monte-Carlo needed:
    blended P50 = sum of campaign P50s). Used for the many sweep points."""
    factors = _meta["channel"].map(lambda c: mult.get(c, 1.0)).to_numpy()
    X = _X.copy()
    X["planned_spend"] = _X["planned_spend"].to_numpy() * factors
    X["planned_vs_runrate"] = factors
    q = _model.predict_quantiles(X)
    hmask = (_meta["horizon"].astype(int).to_numpy() == horizon)
    rev = float(q["p50"].to_numpy()[hmask].sum())
    spend = float((_meta["planned_spend"].to_numpy() * factors)[hmask].sum())
    return rev, (rev / spend if spend > 0 else 0.0)


@st.cache_data(show_spinner=False)
def budget_sweep(_model, _X, _meta, sel, channels_key, horizon):
    rows = []
    for s in np.round(np.arange(0.5, 2.01, 0.1), 2):
        m = {ch: 1.0 for ch in channels_key}
        m[sel] = float(s)
        rev, roas = _blended_point(_model, _X, _meta, m, horizon)
        rows.append({"mult": s, "revenue": rev, "roas": roas})
    return pd.DataFrame(rows)


def money(x):
    return f"${x:,.0f}"


def blended_row(p, h, metric):
    r = p[(p.grain == "blended") & (p.metric == metric) & (p.horizon_days == h)]
    return r.iloc[0] if len(r) else None


def caption(txt):
    st.caption("**What this shows:** " + txt)


# -- header -----------------------------------------------------------------
st.title("Probabilistic Revenue & ROAS Forecaster")
st.markdown(
    "**AIgnition 3.0 - Team Sigmoid.** Predicts an online store's future **sales "
    "(Revenue)** and **ad efficiency (ROAS)** as a realistic **low-middle-high "
    "range** for the next 30/60/90 days - across Google, Bing and Meta - and "
    "explains it with AI. Use the **sidebar** to change the horizon and simulate budgets."
)

try:
    model = load_model(MODEL_PATH)
    long_df, report = load_data(DATA_DIR)
except Exception as e:  # pragma: no cover
    st.error(f"Could not load model/data: {e}")
    st.stop()

channels = report.channels_found
channels_key = tuple(sorted(channels))

# -- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    horizon = st.radio("Forecast window (days ahead)", HORIZONS, index=2, horizontal=True)
    st.divider()
    st.subheader("Budget simulator")
    st.caption("Slide a channel's planned spend up or down (1.0 = keep current pace).")
    mult = {ch: st.slider(f"{ch.title()} spend", 0.5, 2.0, 1.0, 0.05, key=f"mult_{ch}")
            for ch in channels}
    if st.button("Reset budgets"):
        for ch in channels:
            st.session_state[f"mult_{ch}"] = 1.0
        st.rerun()
    st.divider()
    cfg = LLMConfig.from_env()
    st.caption(f"AI insights: {'live (' + cfg.model + ')' if cfg.enabled else 'offline fallback'}")
    st.caption(f"Data: {report.date_min} -> {report.date_max} - {sum(report.rows_per_channel.values()):,} rows")

X_base, meta_base = base_inference(long_df)
mult_key = tuple(sorted(mult.items()))
base_key = tuple((ch, 1.0) for ch in channels_key)
preds = forecast(model, X_base, meta_base, mult_key)
base_preds = forecast(model, X_base, meta_base, base_key)
budget_changed = mult_key != base_key

# -- 1. headline KPIs -------------------------------------------------------
st.subheader(f"1 - Blended forecast - next {horizon} days")
rev = blended_row(preds, horizon, "revenue")
roas = blended_row(preds, horizon, "roas")
base_rev = blended_row(base_preds, horizon, "revenue")
delta_rev = rev.p50 - base_rev.p50
c1, c2, c3 = st.columns(3)
c1.metric("Expected revenue (best guess, P50)", money(rev.p50),
          delta=(money(delta_rev) + " vs current pace") if budget_changed and abs(delta_rev) > 1 else None)
c2.metric("Revenue range (low -> high, P10-P90)", f"{money(rev.p10)} - {money(rev.p90)}")
c3.metric("Blended ROAS (best guess, P50)", f"{roas.p50:.2f}x",
          delta=f"low {roas.p10:.2f}x - high {roas.p90:.2f}x", delta_color="off")
caption("Our single best estimate (P50) plus the honest low-to-high band (P10-P90). "
        "The middle is what we expect; the band shows how uncertain it is. "
        + ("Budget change vs current pace is shown as the green/red delta."
           if budget_changed else "Move the sidebar sliders to see budgets change these."))

# -- 2. cumulative forecast cone --------------------------------------------
st.subheader("2 - How total sales build up over the next 90 days")
xs = [0, 30, 60, 90]
p50 = [0] + [blended_row(preds, h, "revenue").p50 for h in HORIZONS]
p10 = [0] + [blended_row(preds, h, "revenue").p10 for h in HORIZONS]
p90 = [0] + [blended_row(preds, h, "revenue").p90 for h in HORIZONS]
fig = go.Figure()
fig.add_trace(go.Scatter(x=xs + xs[::-1], y=p90 + p10[::-1], fill="toself",
                         fillcolor="rgba(94,53,177,0.15)", line=dict(width=0),
                         name="low-high range (P10-P90)", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=xs, y=p50, mode="lines+markers", name="best guess (P50)",
                         line=dict(color=ACCENT, width=3), marker=dict(size=7)))
for h in HORIZONS:
    r = blended_row(preds, h, "revenue")
    fig.add_annotation(x=h, y=r.p50, text=f"${r.p50/1000:,.0f}K", showarrow=False,
                       yshift=16, font=dict(size=11, color=ACCENT))
fig.update_layout(height=360, margin=dict(t=10, b=10),
                  xaxis_title="days ahead", yaxis_title="cumulative revenue ($)",
                  hovermode="x unified", legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, width="stretch", config=CHART_CFG)
caption("Total expected sales adding up day by day. The purple line is our best guess; "
        "the shaded cone is the low-to-high range. Notice the cone **widens further out** - "
        "we are honestly less certain about 90 days than 30 days.")

# -- 3. breakdowns ----------------------------------------------------------
st.subheader(f"3 - Where the money comes from - next {horizon} days")


def grain_bar(grain, title):
    d = preds[(preds.grain == grain) & (preds.metric == "revenue") &
              (preds.horizon_days == horizon)].sort_values("p50")
    if d.empty:
        st.info("No data at this level.")
        return
    colors = [PALETTE.get(e, ACCENT) for e in d["entity"]]
    fig = go.Figure(go.Bar(
        x=d["p50"], y=[e.title() for e in d["entity"]], orientation="h",
        marker_color=colors,
        error_x=dict(type="data", symmetric=False,
                     array=(d["p90"] - d["p50"]), arrayminus=(d["p50"] - d["p10"]))))
    fig.update_layout(height=300, margin=dict(t=30, b=10, l=10), title=title,
                      xaxis_title="revenue ($, best guess with low-high whiskers)")
    st.plotly_chart(fig, width="stretch", config=CHART_CFG)


left, right = st.columns(2)
with left:
    grain_bar("channel", "By channel")
with right:
    grain_bar("campaign_type", "By campaign type")
caption("The same forecast split by ad channel (left) and by campaign type (right), each "
        "with its low-to-high whiskers. This is what an agency needs to decide **where** to "
        "put budget.")

# -- 4. budget simulator ----------------------------------------------------
st.subheader("4 - Budget simulator - what if we spend more or less?")
sel = st.selectbox("Sweep one channel's spend from 0.5x to 2x", channels,
                   format_func=str.title)
resp = budget_sweep(model, X_base, meta_base, sel, channels_key, horizon)
rc1, rc2 = st.columns(2)
for col, metric, title, color in [
        (rc1, "revenue", f"Predicted revenue - next {horizon}d", ACCENT),
        (rc2, "roas", f"Predicted blended ROAS - next {horizon}d", "#00897B")]:
    f = go.Figure(go.Scatter(x=resp["mult"], y=resp[metric], mode="lines+markers",
                             line=dict(color=color, width=3), marker=dict(size=6)))
    f.add_vline(x=mult[sel], line=dict(color="#B0BEC5", dash="dash"))
    f.add_annotation(x=mult[sel], y=resp[metric].max(), text="current", showarrow=False,
                     font=dict(size=10, color="#78909C"))
    f.update_layout(height=300, margin=dict(t=34, b=10), title=title,
                    xaxis_title=f"{sel.title()} spend (x current pace)")
    col.plotly_chart(f, width="stretch", config=CHART_CFG)
caption(f"As **{sel.title()}** spend rises (left), predicted revenue goes **up** - but "
        "efficiency (ROAS, right) slowly **drops**. This is the classic *diminishing "
        "returns* curve; the dashed line marks the current setting.")

# -- 5. AI causal insights --------------------------------------------------
st.subheader("5 - AI explanation of this forecast")
if st.button("Generate AI briefing", type="primary"):
    with st.spinner("Reasoning over the numbers..."):
        st.session_state["insights"] = generate_insights(
            long_df, preds, meta_assumed_roas=report.meta_assumed_roas,
            focus_horizon=horizon)
ins = st.session_state.get("insights")
if ins:
    st.caption("Source: " + ("AI-generated" if ins["source"] == "llm"
                             else "offline rule-based (no key set)"))
    st.markdown(ins["text"])
else:
    st.info("Click **Generate AI briefing** for a plain-English summary of what to expect, "
            "why, the risks, and recommended actions. Works offline too.")
caption(f"An AI reads only the numbers above (it never invents figures) and writes a short "
        f"briefing, leading with your selected {horizon}-day window: what to expect, the "
        "causes, the risks, and what to do.")

# -- 6. data quality --------------------------------------------------------
st.subheader("6 - The data behind it (transparency)")
dq1, dq2, dq3 = st.columns(3)
dq1.metric("Ad channels", len(channels))
dq2.metric("Total daily rows", f"{sum(report.rows_per_channel.values()):,}")
dq3.metric("Meta revenue (estimated)", f"spend x {report.meta_assumed_roas}x")
st.warning("**Meta reports no revenue**, and its 'conversion' count is a broad metric - not "
           f"purchases. We estimate Meta revenue as **spend x {report.meta_assumed_roas}x** "
           "(the real return rate seen on Google + Bing), and clearly label it as an estimate.")
with st.expander("Recent daily revenue & full prediction table"):
    d = hist_daily(long_df)
    f = go.Figure(go.Scatter(x=d["date"], y=d["revenue"], line=dict(color="#90A4AE")))
    f.update_layout(height=240, margin=dict(t=10, b=10),
                    yaxis_title="revenue / day ($, 7-day avg)")
    st.plotly_chart(f, width="stretch", config=CHART_CFG)
    st.caption("Recent history the model learned from (last ~180 days, smoothed).")
    st.dataframe(preds, width="stretch", height=260)
