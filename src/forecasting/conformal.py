"""
Conformalized Quantile Regression (CQR) calibration.

Raw quantile regressors are often mis-calibrated: a nominal P10-P90 band may
cover far more or less than 80% of outcomes. CQR (Romano et al., 2019) fixes
this with a distribution-free finite-sample guarantee: on a held-out calibration
set we measure how far actual outcomes fall outside the predicted band, then
widen the band by the appropriate quantile of those "conformity scores".

We calibrate **per horizon** (30/60/90), since longer horizons are inherently
more uncertain. The resulting additive deltas are stored on the ForecastModel
and applied at predict time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate(
    y_true: np.ndarray,
    p10: np.ndarray,
    p90: np.ndarray,
    horizons: np.ndarray,
    target_coverage: float = 0.80,
) -> dict:
    """Return {horizon: {'lo': delta_low, 'hi': delta_high}} additive widths.

    Conformity score E = max(p10 - y, y - p90); the (1-alpha) empirical quantile
    of E is how much we must widen *each* side to reach nominal coverage. We
    split that widening symmetrically into lo/hi deltas.
    """
    y_true = np.asarray(y_true, float)
    p10 = np.asarray(p10, float)
    p90 = np.asarray(p90, float)
    horizons = np.asarray(horizons, int)
    alpha = 1.0 - target_coverage

    deltas: dict = {}
    for h in np.unique(horizons):
        m = horizons == h
        if m.sum() < 10:
            deltas[int(h)] = {"lo": 0.0, "hi": 0.0}
            continue
        e = np.maximum(p10[m] - y_true[m], y_true[m] - p90[m])
        n = m.sum()
        # finite-sample-adjusted rank
        rank = min(int(np.ceil((n + 1) * (1 - alpha))), n)
        q = np.sort(e)[rank - 1]
        q = max(q, 0.0)
        deltas[int(h)] = {"lo": float(q), "hi": float(q)}
    return deltas


def coverage_report(
    y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray, horizons: np.ndarray
) -> pd.DataFrame:
    """Empirical coverage of [p10, p90] per horizon - used to prove calibration
    worked (target ~0.80)."""
    y_true, p10, p90 = map(lambda a: np.asarray(a, float), (y_true, p10, p90))
    horizons = np.asarray(horizons, int)
    rows = []
    for h in np.unique(horizons):
        m = horizons == h
        cov = float(((y_true[m] >= p10[m]) & (y_true[m] <= p90[m])).mean())
        width = float(np.mean(p90[m] - p10[m]))
        rows.append({"horizon": int(h), "coverage": round(cov, 3),
                     "mean_interval_width": round(width, 1), "n": int(m.sum())})
    return pd.DataFrame(rows)
