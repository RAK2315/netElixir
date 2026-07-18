"""
Channel schema adapters + unified data model.

The three source channels arrive with *different* schemas (column names, date
columns, cost units, and — for Meta — no revenue at all). This module is the
single place that knows those differences. Everything downstream (feature
generation, training, prediction, the app) consumes the ONE unified long-format
table produced here, so the rest of the codebase never touches raw channel
quirks.

Unified long schema (one row = one campaign on one day):
    channel        str    'google' | 'bing' | 'meta'
    campaign_id    str
    campaign_name  str
    campaign_type  str    normalised (SEARCH / PERFORMANCE_MAX / SHOPPING / ...)
    date           datetime64[ns]
    spend          float  currency units
    revenue        float  currency units  (imputed for Meta — see below)
    clicks         float
    impressions    float
    conversions    float
    daily_budget   float  (may be NaN)
    revenue_is_imputed  bool  True for Meta rows

Design note on the test contract: the scorer overwrites ``data/`` with held-out
CSVs of the *same schema*. We therefore detect each channel by its column
signature and read every ``*.csv`` in the folder, rather than hardcoding
filenames — so unknown filenames with a known schema still load.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── Canonical campaign-type vocabulary ─────────────────────────────────────
# Raw channels spell the same concept differently ("PerformanceMax" vs
# "PERFORMANCE_MAX"). We map everything to one vocabulary so channel-agnostic
# campaign-type forecasts are coherent.
CAMPAIGN_TYPE_MAP = {
    "search": "SEARCH",
    "performancemax": "PERFORMANCE_MAX",
    "performance_max": "PERFORMANCE_MAX",
    "pmax": "PERFORMANCE_MAX",
    "shopping": "SHOPPING",
    "display": "DISPLAY",
    "video": "VIDEO",
    "demand_gen": "DEMAND_GEN",
    "demandgen": "DEMAND_GEN",
    "audience": "AUDIENCE",
}

UNIFIED_COLUMNS = [
    "channel", "campaign_id", "campaign_name", "campaign_type", "date",
    "spend", "revenue", "clicks", "impressions", "conversions",
    "daily_budget", "revenue_is_imputed",
]


def _norm_type(raw: object) -> str:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "UNKNOWN"
    key = str(raw).strip().lower().replace(" ", "")
    return CAMPAIGN_TYPE_MAP.get(key, str(raw).strip().upper())


def _infer_meta_type(name: object) -> str:
    """Meta has no campaign_type column — infer a coarse funnel stage from the
    campaign name (Prospecting / Remarketing / Generic ...). Purely descriptive;
    used for the campaign-type breakdown only."""
    n = str(name).lower()
    if "remarket" in n:
        return "META_REMARKETING"
    if "prospect" in n:
        return "META_PROSPECTING"
    if "brand" in n:
        return "META_BRAND"
    return "META_GENERIC"


# ── Per-channel column signatures (used for auto-detection) ────────────────
GOOGLE_COLS = {"campaign_id", "segments_date", "metrics_cost_micros",
               "metrics_conversions_value"}
BING_COLS = {"CampaignId", "TimePeriod", "Revenue", "Spend"}
META_COLS = {"campaign_id", "date_start", "spend", "conversion"}


def _detect_channel(cols: set[str]) -> str | None:
    if GOOGLE_COLS.issubset(cols):
        return "google"
    if BING_COLS.issubset(cols):
        return "bing"
    if META_COLS.issubset(cols):
        return "meta"
    return None


# ── Adapters: raw channel frame -> unified long frame ──────────────────────
def _adapt_google(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["campaign_id"] = df["campaign_id"].astype(str)
    out["campaign_name"] = df["campaign_name"].astype(str)
    out["campaign_type"] = df["campaign_advertising_channel_type"].map(_norm_type)
    out["date"] = pd.to_datetime(df["segments_date"])
    out["spend"] = df["metrics_cost_micros"].astype(float) / 1e6  # micros -> currency
    out["revenue"] = df["metrics_conversions_value"].astype(float)
    out["clicks"] = df["metrics_clicks"].astype(float)
    out["impressions"] = df["metrics_impressions"].astype(float)
    out["conversions"] = df["metrics_conversions"].astype(float)
    out["daily_budget"] = df.get("campaign_budget_amount", np.nan)
    out["channel"] = "google"
    out["revenue_is_imputed"] = False
    return out


def _adapt_bing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["campaign_id"] = df["CampaignId"].astype(str)
    out["campaign_name"] = df["CampaignName"].astype(str)
    out["campaign_type"] = df["CampaignType"].map(_norm_type)
    out["date"] = pd.to_datetime(df["TimePeriod"])
    out["spend"] = df["Spend"].astype(float)
    out["revenue"] = df["Revenue"].astype(float)
    out["clicks"] = df["Clicks"].astype(float)
    out["impressions"] = df["Impressions"].astype(float)
    out["conversions"] = df["Conversions"].astype(float)
    out["daily_budget"] = df.get("DailyBudget", np.nan)
    out["channel"] = "bing"
    out["revenue_is_imputed"] = False
    return out


def _adapt_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Meta has NO revenue column, and its ``conversion`` field is a broad
    conversion metric (sums to ~1.6M vs ~$88 AOV elsewhere) — NOT purchases.
    So we deliberately do NOT derive revenue from conversions here. Revenue is
    left as NaN and imputed later via spend x assumed-ROAS in
    :func:`impute_meta_revenue`, which keeps Meta proportionate to spend."""
    out = pd.DataFrame()
    out["campaign_id"] = df["campaign_id"].astype(str)
    out["campaign_name"] = df["campaign_name"].astype(str)
    out["campaign_type"] = df["campaign_name"].map(_infer_meta_type)
    out["date"] = pd.to_datetime(df["date_start"])
    out["spend"] = df["spend"].astype(float)
    out["revenue"] = np.nan  # imputed later
    out["clicks"] = df["clicks"].astype(float)
    out["impressions"] = df["impressions"].astype(float)
    out["conversions"] = df["conversion"].astype(float)  # broad metric, kept as feature
    out["daily_budget"] = df.get("daily_budget", np.nan)
    out["channel"] = "meta"
    out["revenue_is_imputed"] = True
    return out


_ADAPTERS = {"google": _adapt_google, "bing": _adapt_bing, "meta": _adapt_meta}


@dataclass
class LoadReport:
    """Diagnostics surfaced in the data-quality panel of the app."""
    files_read: list[str]
    channels_found: list[str]
    rows_per_channel: dict[str, int]
    date_min: str
    date_max: str
    meta_assumed_roas: float
    warnings: list[str]


def load_channel_data(data_dir: str) -> tuple[pd.DataFrame, LoadReport]:
    """Read every CSV in ``data_dir``, auto-detect its channel, adapt to the
    unified schema, impute Meta revenue, and return (long_df, report)."""
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in data dir: {data_dir!r}")

    frames, files_read, warnings = [], [], []
    for p in paths:
        raw = pd.read_csv(p)
        raw = raw.loc[:, [c for c in raw.columns if not str(c).startswith("Unnamed")]]
        ch = _detect_channel(set(raw.columns))
        if ch is None:
            warnings.append(f"Skipped {os.path.basename(p)}: unrecognised schema.")
            continue
        frames.append(_ADAPTERS[ch](raw))
        files_read.append(os.path.basename(p))

    if not frames:
        raise ValueError(
            "No files matched a known channel schema (google/bing/meta)."
        )

    df = pd.concat(frames, ignore_index=True)
    df = clean_long(df)
    df, meta_roas = impute_meta_revenue(df)

    rows_per_channel = df.groupby("channel").size().to_dict()
    report = LoadReport(
        files_read=files_read,
        channels_found=sorted(df["channel"].unique().tolist()),
        rows_per_channel={k: int(v) for k, v in rows_per_channel.items()},
        date_min=str(df["date"].min().date()),
        date_max=str(df["date"].max().date()),
        meta_assumed_roas=round(meta_roas, 3),
        warnings=warnings,
    )
    return df, report


def clean_long(df: pd.DataFrame) -> pd.DataFrame:
    """Type coercion, non-negativity, de-duplication of (channel, id, date)."""
    num = ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # metrics can't be negative; clip (revenue kept NaN where imputed-later)
    for c in ["spend", "clicks", "impressions", "conversions"]:
        df[c] = df[c].clip(lower=0)
    df.loc[df["revenue"].notna(), "revenue"] = df.loc[
        df["revenue"].notna(), "revenue"
    ].clip(lower=0)
    df = df.dropna(subset=["date"])
    df = (
        df.sort_values("date")
        .drop_duplicates(subset=["channel", "campaign_id", "date"], keep="last")
        .reset_index(drop=True)
    )
    return df[UNIFIED_COLUMNS]


def blended_roas(df: pd.DataFrame) -> float:
    """Historical blended ROAS across revenue-bearing channels (google+bing).
    Used as the default assumed ROAS for Meta revenue imputation."""
    real = df[~df["revenue_is_imputed"]]
    spend = float(real["spend"].sum())
    rev = float(real["revenue"].sum())
    if spend <= 0:
        return 1.0
    return rev / spend


def impute_meta_revenue(
    df: pd.DataFrame, assumed_roas: float | None = None
) -> tuple[pd.DataFrame, float]:
    """Impute Meta revenue as ``spend * assumed_roas``.

    Rationale: Meta provides no revenue, and its conversion count is a broad
    metric (~1.6M) that would massively over-state revenue if multiplied by an
    AOV. Anchoring to spend via the historical blended ROAS of the channels
    that *do* report revenue (google+bing) keeps Meta's contribution
    proportionate and interpretable. This is an explicit, documented modelling
    assumption (see docs/METHODOLOGY.md)."""
    roas = blended_roas(df) if assumed_roas is None else assumed_roas
    mask = df["revenue"].isna() & (df["channel"] == "meta")
    df.loc[mask, "revenue"] = df.loc[mask, "spend"] * roas
    # any remaining NaN revenue (defensive) -> 0
    df["revenue"] = df["revenue"].fillna(0.0)
    return df, roas


def daily_channel_totals(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to channel x date totals — the base series for the app charts
    and for channel/blended aggregation."""
    return (
        df.groupby(["channel", "date"], as_index=False)[
            ["spend", "revenue", "clicks", "impressions", "conversions"]
        ]
        .sum()
        .sort_values(["channel", "date"])
    )
