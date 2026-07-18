"""
Prediction stage of the scored pipeline (step 2 of run.sh).

Loads the committed pickle and the feature matrix from step 1, then writes the
probabilistic forecast table to ``--output``.

    PYTHONPATH=. python src/predict.py --features features.parquet \
        --model ./pickle/model.pkl --output ./output/predictions.csv

Output schema (documented in README.md):
    horizon_days , grain , entity , metric , p10 , p50 , p90
  * grain  in {blended, channel, campaign_type, campaign}
  * metric in {revenue, roas}
  * horizon_days in {30, 60, 90}

Unpickling note: we import ForecastModel *before* pickle.load so the class is
resolvable; run.sh sets PYTHONPATH=. so ``src.forecasting.model`` is importable.
No network, deterministic (fixed seed inside the model's Monte-Carlo step).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

# Make the repo root importable regardless of CWD / PYTHONPATH (portable across
# Linux scorer and local Windows/Git-Bash).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Import BEFORE unpickling so the class resolves. Do not remove.
from src.forecasting.model import ForecastModel  # noqa: F401

META_COLS = ["anchor", "channel", "campaign_id", "campaign_type",
             "campaign_name", "horizon", "planned_spend"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features.parquet")
    ap.add_argument("--model", default="./pickle/model.pkl")
    ap.add_argument("--output", default="./output/predictions.csv")
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        model: ForecastModel = pickle.load(f)

    combined = pd.read_parquet(args.features)
    X = combined[model.feature_columns]
    meta = combined[META_COLS]

    preds = model.predict_frame(X, meta)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    preds.to_csv(args.output, index=False)
    print(f"[predict] wrote {len(preds):,} forecast rows to {args.output}")
    # tiny sanity summary for the console
    blended = preds[(preds.grain == "blended") & (preds.metric == "revenue")]
    for _, r in blended.iterrows():
        print(f"          blended revenue {int(r.horizon_days)}d: "
              f"p50={r.p50:,.0f}  [{r.p10:,.0f} .. {r.p90:,.0f}]")


if __name__ == "__main__":
    main()
