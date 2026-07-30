"""
Feature generation stage of the scored pipeline (step 1 of run.sh).

Reads whatever channel CSVs are in ``--data-dir`` (auto-detected by schema, so
the scorer's held-out files load without hardcoded names), unifies them, and
writes the **inference feature matrix** - one row per currently-active campaign
x horizon (30/60/90) - to a parquet the predict stage consumes.

    PYTHONPATH=. python src/generate_features.py --data-dir ./data --out features.parquet

Design: the forecast is anchored at the latest date present in the data and
assumes spend continues at the trailing 30-day run-rate (a flat-budget
baseline). No network, no randomness here.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make the repo root importable regardless of CWD / PYTHONPATH (portable across
# Linux scorer and local Windows/Git-Bash).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.common.schema import load_channel_data
from src.forecasting.features import FEATURE_COLUMNS, HORIZONS, build_inference

# channel / campaign_type / horizon / planned_spend already live in
# FEATURE_COLUMNS, so only the meta-ONLY columns are appended to avoid dupes.
META_ONLY = ["anchor", "campaign_id", "campaign_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out", default="features.parquet")
    args = ap.parse_args()

    long_df, report = load_channel_data(args.data_dir)
    print(f"[features] loaded {len(long_df):,} rows | {report.date_min}..{report.date_max} "
          f"| channels={report.channels_found}")

    X, meta = build_inference(long_df, HORIZONS)
    combined = pd.concat([X.reset_index(drop=True),
                          meta[META_ONLY].reset_index(drop=True)], axis=1)
    # anchor is a Timestamp -> store as ISO string for portable parquet
    combined["anchor"] = combined["anchor"].astype(str)
    combined.to_parquet(args.out, index=False)
    print(f"[features] wrote {args.out} "
          f"({len(combined):,} rows = {len(meta['campaign_id'].unique())} campaigns "
          f"x {len(HORIZONS)} horizons)")


if __name__ == "__main__":
    main()
