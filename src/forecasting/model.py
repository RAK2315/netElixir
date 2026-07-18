"""
The picklable forecast model container.

IMPORTANT (unpickling contract): the scorer runs ``pickle.load`` on
``pickle/model.pkl``. For that to work, THIS class must be importable at load
time. ``predict.py`` therefore imports :class:`ForecastModel` before unpickling,
and ``run.sh`` exports ``PYTHONPATH=.`` so ``src.forecasting.model`` resolves.

Design: quantile gradient boosting. We fit three LightGBM regressors — one each
for the P10, P50, P90 conditional quantiles of window revenue — plus per-horizon
conformal deltas that widen the raw P10/P90 so the intervals hit their nominal
coverage (see conformal.py). Aggregation from campaign -> type/channel/blended
is done in :meth:`predict_frame` by Monte-Carlo sampling each campaign's
quantile distribution, so aggregate intervals are statistically coherent rather
than a naive sum of quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.forecasting.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

QUANTILES = (0.1, 0.5, 0.9)
_SEED = 42


@dataclass
class ForecastModel:
    """Bundles everything needed to turn feature rows into probabilistic,
    multi-grain revenue & ROAS forecasts. Fully self-contained + picklable."""

    boosters: dict = field(default_factory=dict)          # quantile -> lgb.Booster
    category_levels: dict = field(default_factory=dict)   # cat col -> [levels]
    conformal_delta: dict = field(default_factory=dict)   # horizon -> {'lo':x,'hi':y}
    feature_columns: list = field(default_factory=lambda: list(FEATURE_COLUMNS))
    meta_assumed_roas: float = 1.0
    trained_on: str = ""
    metrics: dict = field(default_factory=dict)
    version: str = "1.0"

    # ── categorical encoding (shared by train + inference) ──────────────────
    def encode(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in CATEGORICAL_FEATURES:
            levels = self.category_levels.get(c)
            cat = pd.Categorical(X[c], categories=levels)
            X[c] = cat.codes.astype("int32")  # unseen -> -1, handled by LightGBM
        for c in self.feature_columns:
            if c not in CATEGORICAL_FEATURES:
                X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
        return X[self.feature_columns]

    # ── raw per-row quantile predictions (campaign grain) ───────────────────
    def predict_quantiles(self, X: pd.DataFrame) -> pd.DataFrame:
        Xe = self.encode(X)
        preds = {}
        for q, booster in self.boosters.items():
            preds[q] = np.clip(booster.predict(Xe), 0, None)
        out = pd.DataFrame(preds)
        out.columns = ["p10_raw", "p50", "p90_raw"]
        # apply conformal widening per-row by horizon
        horizons = X["horizon"].astype(int).values
        lo = np.array([self.conformal_delta.get(h, {}).get("lo", 0.0) for h in horizons])
        hi = np.array([self.conformal_delta.get(h, {}).get("hi", 0.0) for h in horizons])
        out["p10"] = np.clip(out["p50"] - (out["p50"] - out["p10_raw"]) - lo, 0, None)
        out["p90"] = out["p50"] + (out["p90_raw"] - out["p50"]) + hi
        # guarantee monotonicity p10 <= p50 <= p90
        out["p10"] = np.minimum(out["p10"], out["p50"])
        out["p90"] = np.maximum(out["p90"], out["p50"])
        return out[["p10", "p50", "p90"]]

    # ── full multi-grain forecast table ─────────────────────────────────────
    def predict_frame(
        self, X: pd.DataFrame, meta: pd.DataFrame, n_sims: int = 2000
    ) -> pd.DataFrame:
        """Return the tidy predictions table across all grains:
        columns = horizon_days, grain, entity, metric, p10, p50, p90.

        Revenue quantiles come straight from the models at campaign grain and
        are aggregated to type/channel/blended by Monte-Carlo. ROAS = revenue /
        planned_spend at the matching grain.
        """
        q = self.predict_quantiles(X)
        base = meta.copy().reset_index(drop=True)
        base = pd.concat([base, q.reset_index(drop=True)], axis=1)

        rng = np.random.default_rng(_SEED)
        records: list[dict] = []

        def emit(horizon, grain, entity, rev_samples, spend):
            p10, p50, p90 = np.percentile(rev_samples, [10, 50, 90])
            records.append(dict(horizon_days=horizon, grain=grain, entity=entity,
                                metric="revenue", p10=p10, p50=p50, p90=p90))
            s = spend if spend and spend > 0 else np.nan
            records.append(dict(horizon_days=horizon, grain=grain, entity=entity,
                                metric="roas",
                                p10=p10 / s if s else 0.0,
                                p50=p50 / s if s else 0.0,
                                p90=p90 / s if s else 0.0))

        for h, hgrp in base.groupby("horizon"):
            # campaign-grain samples: lognormal-ish via triangular over p10/p50/p90
            sims = {}
            for i, r in hgrp.reset_index(drop=True).iterrows():
                s = _sample_from_quantiles(rng, r["p10"], r["p50"], r["p90"], n_sims)
                sims[i] = s
                # campaign grain row (exact quantiles, no re-sim needed)
                records.append(dict(horizon_days=int(h), grain="campaign",
                                    entity=r["campaign_name"], metric="revenue",
                                    p10=r["p10"], p50=r["p50"], p90=r["p90"]))
                sp = r.get("planned_spend", np.nan)
                sp = sp if sp and sp > 0 else np.nan
                records.append(dict(horizon_days=int(h), grain="campaign",
                                    entity=r["campaign_name"], metric="roas",
                                    p10=(r["p10"] / sp) if sp else 0.0,
                                    p50=(r["p50"] / sp) if sp else 0.0,
                                    p90=(r["p90"] / sp) if sp else 0.0))
            hg = hgrp.reset_index(drop=True)
            sim_matrix = np.vstack([sims[i] for i in range(len(hg))])  # (n_camp, n_sims)

            # channel grain
            for ch, sub in hg.groupby("channel"):
                idxs = sub.index.to_numpy()
                rev = sim_matrix[idxs].sum(axis=0)
                emit(int(h), "channel", ch, rev, float(sub["planned_spend"].sum()))
            # campaign_type grain
            for ct, sub in hg.groupby("campaign_type"):
                idxs = sub.index.to_numpy()
                rev = sim_matrix[idxs].sum(axis=0)
                emit(int(h), "campaign_type", ct, rev, float(sub["planned_spend"].sum()))
            # blended grain
            rev_all = sim_matrix.sum(axis=0)
            emit(int(h), "blended", "ALL", rev_all, float(hg["planned_spend"].sum()))

        out = pd.DataFrame.from_records(records)
        out = out[["horizon_days", "grain", "entity", "metric", "p10", "p50", "p90"]]
        for c in ["p10", "p50", "p90"]:
            out[c] = out[c].round(2)
        sort_grain = {"blended": 0, "channel": 1, "campaign_type": 2, "campaign": 3}
        out = out.sort_values(
            by=["horizon_days", "grain", "entity", "metric"],
            key=lambda s: s.map(sort_grain) if s.name == "grain" else s,
        ).reset_index(drop=True)
        return out


def _sample_from_quantiles(rng, p10, p50, p90, n):
    """Draw samples consistent with a (p10, p50, p90) triple using a two-sided
    triangular-ish distribution. Cheap, monotonic, and good enough for coherent
    aggregate-interval estimation."""
    p10, p50, p90 = float(p10), float(p50), float(p90)
    if p90 <= p10:
        return np.full(n, p50)
    u = rng.random(n)
    left = u < 0.5
    s = np.empty(n)
    # lower half maps [0,0.5)->[p10,p50], upper half [0.5,1)->[p50,p90]
    s[left] = p10 + (u[left] / 0.5) * (p50 - p10)
    s[~left] = p50 + ((u[~left] - 0.5) / 0.5) * (p90 - p50)
    return np.clip(s, 0, None)
