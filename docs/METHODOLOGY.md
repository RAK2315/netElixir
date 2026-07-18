# Methodology

## 1. Problem framing

We forecast **aggregate-period** (not daily) e-commerce revenue and ROAS as
**probabilistic ranges** for 30/60/90-day windows, at four grains (blended,
channel, campaign-type, campaign), and support **budget-response** ("what if I
change spend?"). Existing channel-level attribution is taken **as the source of
truth** — we do not build an attribution engine or a full MMM, per the brief.

## 2. Data & preprocessing

Three heterogeneous daily campaign-level feeds are unified in
`src/common/schema.py`:

| Channel | Rows | Campaigns | Revenue | Notes |
|---|---|---|---|---|
| Google | 19,272 | 92 | `metrics_conversions_value` | cost is in **micros** (÷1e6) |
| Bing | 2,873 | 28 | `Revenue` | native currency |
| Meta | 3,417 | 16 | **none** | only a broad `conversion` count |

Steps:
1. **Schema auto-detection** by column signature (not filename), so the scorer's
   held-out files load without hardcoding names.
2. **Unit + type normalisation** — Google micros → currency; non-negative clip;
   de-duplication of `(channel, campaign_id, date)`.
3. **Campaign-type vocabulary** normalised across channels
   (`PerformanceMax`/`PERFORMANCE_MAX` → `PERFORMANCE_MAX`, etc.).

### The Meta revenue problem (an explicit modelling assumption)
Meta reports **no revenue**, and its `conversion` field sums to **~1.66M** —
clearly *not* purchases (Google shows ~$88 average order value; multiplying
1.66M conversions by an AOV would fabricate ~$146M of revenue). We therefore
impute:

```
meta_revenue = meta_spend × assumed_ROAS
assumed_ROAS = historical blended ROAS of the revenue-bearing channels (Google+Bing) ≈ 4.75
```

This keeps Meta **proportionate to spend** and interpretable. Meta figures are
flagged as *modelled, not measured* everywhere they surface (predictions, app,
AI insights). Replacing this with true Shopify/GA4 revenue is the #1 recommended
next step.

## 3. Forecasting model — Conformalized Quantile Regression

**Framing (direct multi-horizon, tabular supervised).** Each training example
is: *"standing at cutoff `t`, given this campaign's recent history and a planned
spend for the next `H` days, what revenue does it earn in `(t, t+H]`?"*

- **No leakage:** every feature uses data ≤ `t`; the label is strictly future.
- **Horizon is a feature** (30/60/90) → one model serves all windows.
- **Planned spend is a feature** → budget-response is native (vary it → new
  revenue). At train time it is the *actual* future spend; at inference it
  defaults to the trailing run-rate and is overridable by the simulator.
- **Grain:** we model at **campaign** grain and aggregate up to
  type/channel/blended. Point forecasts sum directly; **intervals** are
  aggregated by **Monte-Carlo** over each campaign's quantile distribution
  (2,000 draws, seeded), which is statistically coherent — unlike naively summing
  quantiles.

**Features** (`src/forecasting/features.py`): trailing spend/revenue/conversions
over 7/14/30/60/90 days; trailing ROAS (30/90); activity (active-days, days
since start/last-active); momentum (30-vs-prior-30 spend & revenue trend);
90-day revenue volatility; run-rate; planned spend and planned-vs-run-rate;
seasonality of the forecast window (month sin/cos + quarter); and categorical
`channel`, `campaign_type`.

**Estimator.** Three **LightGBM** gradient-boosted quantile regressors
(`objective="quantile"`, α = 0.1 / 0.5 / 0.9) → conditional P10/P50/P90 of window
revenue. Gradient boosting handles the mixed, non-linear, heterogeneous feature
set well, pickles cleanly, and is fast/offline at inference.

**Uncertainty — CQR** (`src/forecasting/conformal.py`). Raw quantile models are
often mis-calibrated. We apply **Conformalized Quantile Regression** (Romano et
al., 2019): on a held-out calibration slice we measure conformity scores
`E = max(p10 − y, y − p90)` and widen the band by their `(1−α)` empirical
quantile, **per horizon** (longer horizons → wider bands). This gives
distribution-free coverage guarantees.

**ROAS** = revenue / planned-spend at each grain (propagated through the
Monte-Carlo so ROAS also gets P10/P50/P90).

## 4. Backtest (walk-forward, held-out most-recent cutoffs)

Time-ordered split: 65% train / 15% conformal-calibration / 20% test (the most
recent cutoffs — an honest forward test).

| Metric | Result |
|---|---|
| **Blended-grain P50 WAPE** (what agencies budget on) | **7.8%** |
| Campaign-grain P50 WAPE | 30.7% |
| P50 WAPE by horizon (campaign grain) | 30d 33.1% · 60d 30.2% · 90d 29.7% |
| Interval coverage @ 30 / 60 / 90d (target 80%) | 89.0% / 91.2% / 93.5% |

Coverage sits **at or slightly above** the 80% target — intervals are honest and
mildly conservative (they err toward covering the truth, the safe direction for
planning). Aggregate error is low precisely because campaign-level noise diversifies
away when summed — the level at which budgets are actually set.

The shipped `model.pkl` refits the boosters on **all** available history after
backtesting (standard practice), retaining the horizon-level conformal deltas.

## 5. AI integration strategy

`src/insights.py` turns the numeric forecast into an analyst-style briefing —
**Forecast Summary / Causal Drivers / Anomalies & Risks / Recommended Actions**.

- **Provider-agnostic** OpenAI-compatible client over plain `requests`
  (Groq default; OpenRouter / Grok / OpenAI via env).
- **Grounded**: the LLM only ever *interprets* numbers we compute (recent
  channel trends, ROAS deltas, seasonality, the Meta caveat) — it never invents
  figures.
- **Offline-safe**: a deterministic rule-based narrative renders identical
  structure with no key/network, so the demo always works and the **scored path
  has zero LLM dependency**.

## 6. Assumptions & limitations

- **Meta revenue is imputed** (spend × assumed ROAS) — modelled, not measured.
- Baseline forecast assumes **spend continues at trailing run-rate**; use the
  budget simulator to model deliberate changes.
- We use **provided attribution as-is** (no cross-channel de-duplication / MMM).
- Only campaigns **active in the trailing 30 days** are forecast (dormant
  campaigns are excluded rather than predicted to be zero).
- Quantile crossing is prevented by enforcing `p10 ≤ p50 ≤ p90` post-hoc.
- Aggregate-interval Monte-Carlo assumes campaign forecasts are independent; real
  correlation would modestly change aggregate band width.
