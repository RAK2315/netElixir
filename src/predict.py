"""
Prediction stage of the scored pipeline (step 2 of run.sh).

Loads the committed pickle and the feature matrix from step 1, then writes the
forecast to ``--output``.

    PYTHONPATH=. python src/predict.py --features features.parquet \
        --model ./pickle/model.pkl --output ./output/predictions.csv

Output schema (per organizer guidance: "same columns as the training data plus
the forecasted metric, e.g. Revenue / ROAS"). One row per campaign per horizon:

    channel, campaign_id, campaign_name, campaign_type, horizon_days,
    Revenue, ROAS,                       # point forecast (P50)
    Revenue_p10, Revenue_p90,            # probabilistic range (brief requires it)
    ROAS_p10, ROAS_p90

  * ``channel / campaign_id / campaign_name / campaign_type`` mirror the input
    (training) data's identifying columns.
  * ``Revenue`` / ``ROAS`` are the forecasted metrics (median, P50).
  * ``horizon_days`` in {30, 60, 90}; ``Revenue`` is aggregate revenue over that
    window, ``ROAS`` = Revenue / planned spend over the window.

Unpickling note: we import ForecastModel *before* pickle.load so the class is
resolvable; run.sh sets PYTHONPATH=. so ``src.forecasting.model`` is importable.
No network, deterministic.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

# Make the repo root importable regardless of CWD / PYTHONPATH (portable across
# Linux scorer and local Windows/Git-Bash).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# Import BEFORE unpickling so the class resolves. Do not remove.
from src.forecasting.model import ForecastModel  # noqa: F401

OUTPUT_COLUMNS = [
    "channel", "campaign_id", "campaign_name", "campaign_type", "horizon_days",
    "Revenue", "ROAS", "Revenue_p10", "Revenue_p90", "ROAS_p10", "ROAS_p90",
]


def build_output(model: "ForecastModel", combined: pd.DataFrame) -> pd.DataFrame:
    """Turn the feature matrix into the campaign-level, input-mirroring output."""
    if len(combined) == 0:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    X = combined[model.feature_columns]
    q = model.predict_quantiles(X)  # p10 / p50 / p90 revenue per campaign-horizon
    planned = pd.to_numeric(combined["planned_spend"], errors="coerce").to_numpy()
    safe = np.where(planned > 0, planned, np.nan)

    out = pd.DataFrame({
        "channel": combined["channel"].to_numpy(),
        "campaign_id": combined["campaign_id"].to_numpy(),
        "campaign_name": combined["campaign_name"].to_numpy(),
        "campaign_type": combined["campaign_type"].to_numpy(),
        "horizon_days": combined["horizon"].astype(int).to_numpy(),
        "Revenue": q["p50"].to_numpy(),
        "ROAS": q["p50"].to_numpy() / safe,
        "Revenue_p10": q["p10"].to_numpy(),
        "Revenue_p90": q["p90"].to_numpy(),
        "ROAS_p10": q["p10"].to_numpy() / safe,
        "ROAS_p90": q["p90"].to_numpy() / safe,
    })
    # ROAS is undefined without spend -> 0.0 rather than NaN (valid, scoreable)
    for c in ["ROAS", "ROAS_p10", "ROAS_p90"]:
        out[c] = out[c].fillna(0.0).round(4)
    for c in ["Revenue", "Revenue_p10", "Revenue_p90"]:
        out[c] = out[c].round(2)
    return out.sort_values(["channel", "campaign_id", "horizon_days"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--model", default="./pickle/model.pkl")
    ap.add_argument("--output", default="./output/predictions.csv")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model: ForecastModel = pickle.load(f)

    combined = pd.read_parquet(args.features)
    out = build_output(model, combined)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[predict] wrote {len(out):,} campaign-horizon rows to {args.output}")
    # tiny console summary: blended revenue per horizon
    if len(out):
        for h in sorted(out["horizon_days"].unique()):
            sub = out[out.horizon_days == h]
            print(f"          {int(h)}d: blended Revenue p50={sub['Revenue'].sum():,.0f} "
                  f"across {len(sub)} campaigns")


if __name__ == "__main__":
    main()
