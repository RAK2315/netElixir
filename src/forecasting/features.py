"""
Supervised, direct-multi-horizon feature engineering.

We frame revenue forecasting as tabular supervised learning at the **campaign
grain**. One training example is:

    "Standing at cutoff date *t*, knowing this campaign's recent history and a
     planned spend for the next *H* days, what revenue will it earn in (t, t+H]?"

Key properties:
  * **No leakage** — every feature is computed from data on or before *t*; the
    target is the sum of revenue strictly after *t*.
  * **Horizon is a feature** (30/60/90), so a single model serves every window.
  * **Planned spend is a feature** — at train time it's the *actual* future
    spend; at inference it defaults to the trailing run-rate and can be
    overridden by the budget simulator. This is what makes budget-response
    forecasting native.
  * Aggregating campaign-level predictions up to campaign-type / channel /
    blended is just summation, so every output grain the brief asks for is
    coherent by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [30, 60, 90]

# Trailing windows (days) over which we summarise recent behaviour.
_TRAIL = [7, 14, 30, 60, 90]

# The exact, ordered feature list. train.py and predict.py both import this so
# the model always sees identical columns in identical order.
NUMERIC_FEATURES = (
    [f"spend_last_{w}" for w in _TRAIL]
    + [f"revenue_last_{w}" for w in _TRAIL]
    + [f"conv_last_{w}" for w in _TRAIL]
    + [f"roas_last_{w}" for w in [30, 90]]
    + [
        "active_days_30", "active_days_90",
        "days_since_start", "days_since_last_active",
        "spend_trend_30_60", "revenue_trend_30_60",
        "revenue_vol_90", "runrate_daily_spend_30",
        "planned_spend", "planned_vs_runrate",
        "horizon", "fc_month_sin", "fc_month_cos", "fc_quarter",
    ]
)
CATEGORICAL_FEATURES = ["channel", "campaign_type"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _campaign_daily(df: pd.DataFrame) -> dict:
    """Per-campaign daily series reindexed to a contiguous date axis, with
    cumulative sums pre-computed so any trailing/forward window is an O(1)
    lookup. Returns dict keyed by (channel, campaign_id)."""
    out = {}
    full_min, full_max = df["date"].min(), df["date"].max()
    full_idx = pd.date_range(full_min, full_max, freq="D")
    meta_cols = ["channel", "campaign_type", "campaign_name"]
    for (ch, cid), g in df.groupby(["channel", "campaign_id"], sort=False):
        g = g.sort_values("date")
        daily = (
            g.set_index("date")[["spend", "revenue", "conversions"]]
            .reindex(full_idx, fill_value=0.0)
        )
        first_active = g["date"].min()
        last_active = g["date"].max()
        cum = daily.cumsum()
        out[(ch, cid)] = {
            "daily": daily,
            "cum": cum,
            "first_active": first_active,
            "last_active": last_active,
            "channel": ch,
            "campaign_type": g[meta_cols].iloc[-1]["campaign_type"],
            "campaign_name": g[meta_cols].iloc[-1]["campaign_name"],
            "idx": full_idx,
        }
    return out


def _window_sum(cum: pd.DataFrame, idx: pd.DatetimeIndex, t, w: int, col: str) -> float:
    """Sum of ``col`` over the w days ending at t (inclusive), via cumsum diff."""
    pos = idx.get_indexer([pd.Timestamp(t)])[0]
    if pos < 0:
        return 0.0
    lo = max(pos - w, -1)
    hi = cum[col].iloc[pos]
    base = cum[col].iloc[lo] if lo >= 0 else 0.0
    return float(hi - base)


def _forward_sum(cum: pd.DataFrame, idx: pd.DatetimeIndex, t, h: int, col: str) -> float:
    """Sum of ``col`` over (t, t+h]."""
    pos = idx.get_indexer([pd.Timestamp(t)])[0]
    end = min(pos + h, len(idx) - 1)
    return float(cum[col].iloc[end] - cum[col].iloc[pos])


def _seasonality(t, h: int) -> tuple[float, float, int]:
    """Encode *when* the forecast window sits, using its midpoint month —
    captures seasonality without needing the future data itself."""
    mid = pd.Timestamp(t) + pd.Timedelta(days=h // 2)
    m = mid.month
    return (
        float(np.sin(2 * np.pi * m / 12)),
        float(np.cos(2 * np.pi * m / 12)),
        int((m - 1) // 3 + 1),
    )


def _row_features(rec: dict, t, h: int, planned_spend: float | None) -> dict:
    cum, idx = rec["cum"], rec["idx"]
    daily = rec["daily"]
    f: dict = {}
    for w in _TRAIL:
        f[f"spend_last_{w}"] = _window_sum(cum, idx, t, w, "spend")
        f[f"revenue_last_{w}"] = _window_sum(cum, idx, t, w, "revenue")
        f[f"conv_last_{w}"] = _window_sum(cum, idx, t, w, "conversions")
    for w in [30, 90]:
        s = f[f"spend_last_{w}"]
        f[f"roas_last_{w}"] = f[f"revenue_last_{w}"] / s if s > 0 else 0.0

    pos = idx.get_indexer([pd.Timestamp(t)])[0]
    win30 = daily.iloc[max(pos - 30, 0): pos + 1]
    win90 = daily.iloc[max(pos - 90, 0): pos + 1]
    f["active_days_30"] = float((win30["spend"] > 0).sum())
    f["active_days_90"] = float((win90["spend"] > 0).sum())
    f["days_since_start"] = float((pd.Timestamp(t) - rec["first_active"]).days)
    f["days_since_last_active"] = float((pd.Timestamp(t) - rec["last_active"]).days)

    spend_30 = f["spend_last_30"]
    spend_prev_30 = _window_sum(cum, idx, pd.Timestamp(t) - pd.Timedelta(days=30), 30, "spend")
    rev_30 = f["revenue_last_30"]
    rev_prev_30 = _window_sum(cum, idx, pd.Timestamp(t) - pd.Timedelta(days=30), 30, "revenue")
    f["spend_trend_30_60"] = spend_30 / spend_prev_30 if spend_prev_30 > 0 else 1.0
    f["revenue_trend_30_60"] = rev_30 / rev_prev_30 if rev_prev_30 > 0 else 1.0
    f["revenue_vol_90"] = float(win90["revenue"].std(ddof=0))
    runrate = spend_30 / 30.0
    f["runrate_daily_spend_30"] = runrate

    ps = runrate * h if planned_spend is None else float(planned_spend)
    f["planned_spend"] = ps
    f["planned_vs_runrate"] = ps / (runrate * h) if runrate > 0 else 1.0

    f["horizon"] = float(h)
    ms, mc, q = _seasonality(t, h)
    f["fc_month_sin"], f["fc_month_cos"], f["fc_quarter"] = ms, mc, q

    f["channel"] = rec["channel"]
    f["campaign_type"] = rec["campaign_type"]
    return f


def build_supervised(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    cutoff_stride: int = 14,
    min_history_days: int = 90,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Construct the training matrix by sweeping cutoff dates across history.

    Returns (X[FEATURE_COLUMNS], y=revenue-in-window, meta[cutoff/ids/horizon]).
    """
    recs = _campaign_daily(df)
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    max_h = max(horizons)
    # cutoffs need `min_history_days` behind and at least one horizon ahead
    lo = idx[min_history_days]
    hi = idx[-(min(horizons) + 1)]
    cutoffs = pd.date_range(lo, hi, freq=f"{cutoff_stride}D")

    rows, ys, metas = [], [], []
    for t in cutoffs:
        for (ch, cid), rec in recs.items():
            # only campaigns active in the trailing 30 days at this cutoff
            if _window_sum(rec["cum"], rec["idx"], t, 30, "spend") <= 0:
                continue
            for h in horizons:
                # need the full future window to exist for a valid label
                pos = idx.get_indexer([t])[0]
                if pos + h > len(idx) - 1:
                    continue
                feats = _row_features(rec, t, h, planned_spend=None)
                # TRAIN target uses ACTUAL future spend as the planned budget,
                # so the model learns the true spend->revenue response.
                actual_future_spend = _forward_sum(rec["cum"], rec["idx"], t, h, "spend")
                feats["planned_spend"] = actual_future_spend
                rr = feats["runrate_daily_spend_30"]
                feats["planned_vs_runrate"] = (
                    actual_future_spend / (rr * h) if rr > 0 else 1.0
                )
                y = _forward_sum(rec["cum"], rec["idx"], t, h, "revenue")
                rows.append(feats)
                ys.append(y)
                metas.append({
                    "cutoff": t, "channel": ch, "campaign_id": cid,
                    "campaign_type": rec["campaign_type"],
                    "campaign_name": rec["campaign_name"], "horizon": h,
                })

    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    y = pd.Series(ys, name="revenue_window")
    meta = pd.DataFrame(metas)
    return X, y, meta


def build_inference(
    df: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    budget_multipliers: dict[str, float] | None = None,
    anchor_date=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build feature rows for forecasting forward from ``anchor_date`` (default
    = latest date in the data), for every currently-active campaign x horizon.

    ``budget_multipliers`` (optional, keyed by channel) scales the run-rate
    planned spend — this is exactly what the budget simulator passes in.
    Returns (X, meta) aligned row-for-row.

    Robustness: the active-campaign filter relaxes gracefully (30d -> 90d ->
    all-time spend) so we never return an empty forecast just because the data
    ends on a quiet stretch. If there is genuinely no spend anywhere, we return
    empty-but-well-formed frames (correct columns) so the pipeline writes a
    valid header-only CSV instead of crashing.
    """
    meta_cols = ["anchor", "channel", "campaign_id", "campaign_type",
                 "campaign_name", "horizon", "planned_spend"]
    if df.empty:  # no data at all -> valid empty frames, never crash
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(columns=meta_cols)
    recs = _campaign_daily(df)
    anchor = pd.Timestamp(anchor_date) if anchor_date is not None else df["date"].max()

    def trail(rec, w):
        return _window_sum(rec["cum"], rec["idx"], anchor, w, "spend")

    def alltime(rec):
        return float(rec["cum"]["spend"].iloc[-1])

    items = list(recs.items())
    # graceful relaxation of the "active" window
    active = [(k, r) for k, r in items if trail(r, 30) > 0]
    if not active:
        active = [(k, r) for k, r in items if trail(r, 90) > 0]
    if not active:
        active = [(k, r) for k, r in items if alltime(r) > 0]

    rows, metas = [], []
    for (ch, cid), rec in active:
        mult = 1.0 if not budget_multipliers else budget_multipliers.get(ch, 1.0)
        # best available daily run-rate: 30d, else 90d, else all-time average
        span_days = max((rec["last_active"] - rec["first_active"]).days + 1, 1)
        rr = (trail(rec, 30) / 30.0) or (trail(rec, 90) / 90.0) or (alltime(rec) / span_days)
        for h in horizons:
            feats = _row_features(rec, anchor, h, planned_spend=None)
            planned = rr * h * mult
            feats["planned_spend"] = planned
            feats["planned_vs_runrate"] = mult
            rows.append(feats)
            metas.append({
                "anchor": anchor, "channel": ch, "campaign_id": cid,
                "campaign_type": rec["campaign_type"],
                "campaign_name": rec["campaign_name"], "horizon": h,
                "planned_spend": planned,
            })

    if not rows:  # truly no spend anywhere -> valid empty frames, no crash
        return pd.DataFrame(columns=FEATURE_COLUMNS), pd.DataFrame(columns=meta_cols)
    X = pd.DataFrame(rows)[FEATURE_COLUMNS]
    meta = pd.DataFrame(metas)
    return X, meta
