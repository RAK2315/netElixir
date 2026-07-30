"""
Train the probabilistic revenue forecaster and write pickle/model.pkl.

Run by US (once), NOT by the scorer. The scorer only runs generate_features +
predict against the committed pickle.

    PYTHONPATH=. python src/train.py

Pipeline:
  1. Load + unify channel data (src/common/schema.py).
  2. Build the supervised multi-horizon matrix (src/forecasting/features.py).
  3. Walk-forward time split: past -> train, middle -> conformal calibration,
     most-recent -> honest backtest.
  4. Fit LightGBM quantile regressors (P10/P50/P90).
  5. Conformally calibrate the intervals (src/forecasting/conformal.py).
  6. Report backtest accuracy + coverage; pickle a single ForecastModel.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.common.schema import load_channel_data
from src.forecasting import conformal
from src.forecasting.features import (
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, HORIZONS, build_supervised,
)
from src.forecasting.model import QUANTILES, ForecastModel

SEED = 42


def _wape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = np.abs(y).sum()
    return float(np.abs(y - yhat).sum() / denom) if denom > 0 else float("nan")


def _fit_quantile(Xe, y, alpha, cat_idx):
    """Native LightGBM Booster (NOT the sklearn wrapper) so the pickled model
    has zero scikit-learn dependency at load/predict time - avoids the classic
    sklearn-version-mismatch unpickle failure and keeps requirements.txt minimal."""
    dtrain = lgb.Dataset(Xe, label=y, categorical_feature=cat_idx,
                         free_raw_data=False)
    params = {
        "objective": "quantile", "alpha": alpha,
        "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 30,
        "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8,
        "lambda_l2": 1.0, "seed": SEED, "num_threads": 0, "verbose": -1,
    }
    return lgb.train(params, dtrain, num_boost_round=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="./pickle/model.pkl")
    ap.add_argument("--cutoff-stride", type=int, default=14)
    args = ap.parse_args()

    np.random.seed(SEED)
    print(f"[train] loading data from {args.data_dir} ...")
    long_df, report = load_channel_data(args.data_dir)
    print(f"[train] {len(long_df):,} unified rows | channels={report.channels_found} "
          f"| meta assumed ROAS={report.meta_assumed_roas}")

    print("[train] building supervised matrix ...")
    X, y, meta = build_supervised(long_df, HORIZONS, cutoff_stride=args.cutoff_stride)
    print(f"[train] {len(X):,} training examples across horizons {HORIZONS}")

    # -- walk-forward split by cutoff date -----------------------------------
    cutoffs = np.sort(meta["cutoff"].unique())
    n = len(cutoffs)
    train_cut = cutoffs[: int(n * 0.65)]
    cal_cut = cutoffs[int(n * 0.65): int(n * 0.80)]
    test_cut = cutoffs[int(n * 0.80):]
    tr = meta["cutoff"].isin(train_cut).values
    ca = meta["cutoff"].isin(cal_cut).values
    te = meta["cutoff"].isin(test_cut).values
    print(f"[train] split -> train={tr.sum()} calib={ca.sum()} test={te.sum()}")

    # -- categorical encoding levels (from FULL data for stability) ----------
    category_levels = {c: sorted(X[c].astype(str).unique().tolist())
                       for c in CATEGORICAL_FEATURES}
    fm = ForecastModel(
        category_levels=category_levels,
        feature_columns=list(FEATURE_COLUMNS),
        meta_assumed_roas=report.meta_assumed_roas,
        trained_on=f"{report.date_min}..{report.date_max}",
    )
    Xe = fm.encode(X)
    cat_idx = [Xe.columns.get_loc(c) for c in CATEGORICAL_FEATURES]

    # -- fit quantile boosters on TRAIN --------------------------------------
    print("[train] fitting quantile boosters ...")
    qnames = {0.1: "p10", 0.5: "p50", 0.9: "p90"}
    for q in QUANTILES:
        fm.boosters[q] = _fit_quantile(Xe[tr], y[tr], q, cat_idx)
        print(f"         fitted alpha={q} ({qnames[q]})")

    # -- conformal calibration on CALIB slice --------------------------------
    print("[train] conformal calibration ...")
    p10_c = np.clip(fm.boosters[0.1].predict(Xe[ca]), 0, None)
    p90_c = np.clip(fm.boosters[0.9].predict(Xe[ca]), 0, None)
    fm.conformal_delta = conformal.calibrate(
        y[ca].values, p10_c, p90_c, meta.loc[ca, "horizon"].values, target_coverage=0.80
    )
    print(f"         deltas per horizon: {fm.conformal_delta}")

    # -- honest backtest on TEST ---------------------------------------------
    q_test = fm.predict_quantiles(X[te])
    p50 = q_test["p50"].values
    wape_all = _wape(y[te].values, p50)
    cov = conformal.coverage_report(
        y[te].values, q_test["p10"].values, q_test["p90"].values,
        meta.loc[te, "horizon"].values,
    )
    # aggregate (blended-per-cutoff) WAPE - the number agencies actually care about
    agg = meta.loc[te].copy()
    agg["y"] = y[te].values
    agg["yhat"] = p50
    blended = agg.groupby(["cutoff", "horizon"])[["y", "yhat"]].sum()
    wape_blended = _wape(blended["y"].values, blended["yhat"].values)

    per_h = {}
    for h in HORIZONS:
        m = meta.loc[te, "horizon"].values == h
        per_h[h] = round(_wape(y[te].values[m], p50[m]), 4)

    fm.metrics = {
        "backtest_campaign_wape_p50": round(wape_all, 4),
        "backtest_blended_wape_p50": round(wape_blended, 4),
        "backtest_wape_by_horizon": per_h,
        "coverage_by_horizon": cov.to_dict(orient="records"),
        "n_train": int(tr.sum()), "n_calib": int(ca.sum()), "n_test": int(te.sum()),
    }

    print("\n================ BACKTEST (held-out most-recent cutoffs) ============")
    print(f" Campaign-grain P50 WAPE : {wape_all:.1%}")
    print(f" Blended-grain P50 WAPE  : {wape_blended:.1%}")
    print(f" P50 WAPE by horizon     : "
          + ", ".join(f"{h}d={v:.1%}" for h, v in per_h.items()))
    print(f" Interval coverage (target 80%):")
    print(cov.to_string(index=False))
    print("=====================================================================\n")

    # -- retrain on ALL data for the shipped artifact (train+calib+test) -----
    # We keep the calibration deltas learned above (they're horizon-level and
    # transfer), but refit boosters on everything so the deployed model uses
    # the maximum history. This is standard practice after backtesting.
    print("[train] refitting boosters on ALL data for the shipped model ...")
    for q in QUANTILES:
        fm.boosters[q] = _fit_quantile(Xe, y, q, cat_idx)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(fm, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"[train] wrote {args.out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
