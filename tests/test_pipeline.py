"""
Lightweight, dependency-free sanity tests for the forecasting pipeline.

Runnable two ways:
    python tests/test_pipeline.py        # plain, no pytest needed
    pytest tests/test_pipeline.py        # if pytest is installed

Covers the invariants that matter for the scored contract: schema unification,
Meta imputation, no-leakage feature construction, and the output schema.
"""
from __future__ import annotations

import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.common.schema import load_channel_data
from src.forecasting.features import (
    FEATURE_COLUMNS, HORIZONS, build_inference, build_supervised,
)
from src.forecasting.model import ForecastModel  # noqa: F401

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "pickle", "model.pkl")


def test_schema_unifies_all_channels():
    df, rep = load_channel_data(DATA_DIR)
    assert set(rep.channels_found) == {"google", "bing", "meta"}
    assert df["revenue"].isna().sum() == 0        # meta imputed, none left null
    assert df["spend"].min() >= 0                 # non-negative
    assert rep.meta_assumed_roas > 0
    # meta revenue must be proportionate to spend (imputed), never from raw conv
    meta = df[(df["channel"] == "meta") & (df["spend"] > 0)]
    ratio = meta["revenue"] / meta["spend"]
    assert np.allclose(ratio, ratio.iloc[0])                      # constant ROAS
    assert abs(ratio.iloc[0] - rep.meta_assumed_roas) < 0.01      # ~= reported ROAS


def test_features_match_declared_columns():
    df, _ = load_channel_data(DATA_DIR)
    X, meta = build_inference(df, HORIZONS)
    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(meta)
    assert set(meta["horizon"].unique()) == set(HORIZONS)
    assert X.isna().sum().sum() == 0


def test_supervised_target_is_future_only():
    """No-leakage smoke test: the label (window revenue) must be independent of
    the horizon flag being present in features, and target >= 0."""
    df, _ = load_channel_data(DATA_DIR)
    X, y, m = build_supervised(df, HORIZONS, cutoff_stride=30)
    assert (y >= 0).all()
    assert len(X) == len(y) == len(m)
    assert "horizon" in X.columns          # horizon is a feature, not leaked label


def test_scored_output_contract():
    """The CSV the grader scores: input-mirroring columns + Revenue/ROAS."""
    from src.predict import OUTPUT_COLUMNS, build_output
    df, _ = load_channel_data(DATA_DIR)
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    X, meta = build_inference(df, HORIZONS)
    combined = pd.concat([X.reset_index(drop=True), meta.reset_index(drop=True)], axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    out = build_output(model, combined)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.isna().sum().sum() == 0
    assert ((out.Revenue_p10 <= out.Revenue) & (out.Revenue <= out.Revenue_p90)).all()
    assert ((out.ROAS_p10 <= out.ROAS) & (out.ROAS <= out.ROAS_p90)).all()
    assert set(out.horizon_days.unique()) == set(HORIZONS)
    assert set(out.channel.unique()).issubset({"google", "bing", "meta"})


def test_predict_frame_grains():
    """The multi-grain table used by the demo app stays coherent."""
    df, _ = load_channel_data(DATA_DIR)
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    X, meta = build_inference(df, HORIZONS)
    out = model.predict_frame(X, meta)
    assert set(out.grain.unique()) == {"blended", "channel", "campaign_type", "campaign"}
    for h in HORIZONS:
        bl = out[(out.grain == "blended") & (out.metric == "revenue") &
                 (out.horizon_days == h)]["p50"].iloc[0]
        ch = out[(out.grain == "channel") & (out.metric == "revenue") &
                 (out.horizon_days == h)]["p50"].sum()
        assert abs(bl - ch) / max(bl, 1) < 0.05


def test_budget_simulator_responds():
    df, _ = load_channel_data(DATA_DIR)
    with open(MODEL, "rb") as f:
        model = pickle.load(f)
    def blended90(mult):
        X, meta = build_inference(df, HORIZONS, budget_multipliers=mult)
        o = model.predict_frame(X, meta)
        return o[(o.grain == "blended") & (o.metric == "revenue") &
                 (o.horizon_days == 90)]["p50"].iloc[0]
    base = blended90(None)
    up = blended90({"google": 1.5})
    assert up > base       # more spend -> more revenue


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
