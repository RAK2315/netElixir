# Probabilistic Revenue Forecasting for E-commerce Marketing

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-9cf)
![pandas](https://img.shields.io/badge/pandas-2.3-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.2-013243?logo=numpy&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-6.8-3F4F75?logo=plotly&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20(OpenAI--compatible)-F55036)

![Blended WAPE](https://img.shields.io/badge/blended%20WAPE-7.8%25-2ea44f)
![Interval coverage](https://img.shields.io/badge/interval%20coverage-89--94%25-2ea44f)
![Offline scored](https://img.shields.io/badge/scored%20pipeline-offline%20%2F%20no%20network-blue)

**AIgnition 2026 - NetElixir.** An AI-assisted forecasting utility that predicts
e-commerce **revenue** and **ROAS** as *probabilistic ranges* (P10/P50/P90) over
30/60/90-day windows, across paid channels (Google, Microsoft/Bing, Meta), and
explains the forecast with an **LLM causal-inference layer**.

> Clone, `pip install -r requirements.txt`, drop data into `data/`, run
> `./run.sh`, read `output/predictions.csv`. It runs on a machine that has never
> seen the project, with no manual fixes and no network.

---

## Results at a glance

Walk-forward backtest on the most-recent held-out cutoffs:

| Metric | Result |
|---|---|
| Blended-grain revenue error (P50 WAPE) | **7.8%** |
| Campaign-grain revenue error (P50 WAPE) | 30.7% |
| P10 to P90 interval coverage @ 30 / 60 / 90d (target 80%) | 89% / 91% / 94% |

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full method and table.

---

## Why this design

The submission is scored two ways, and this repo is built to win both:

1. **Automated pipeline (pass/fail gate).** A rigid, *offline* runner clones the
   repo, installs `requirements.txt`, overwrites `data/` with held-out CSVs, runs
   `./run.sh`, and scores `output/predictions.csv` from a **pre-trained, pickled
   model** with no internet and no retraining.
2. **Human judges.** Technical soundness, AI integration, product thinking,
   engineering quality.

So the codebase **splits cleanly into two layers**:

| Layer | What | Network | Where |
|---|---|---|---|
| **A. Scoring core** | deterministic forecasting model behind `run.sh` | none | `run.sh`, `src/generate_features.py`, `src/predict.py`, `src/forecasting/`, `src/common/` |
| **B. Product/demo** | LLM insights + Streamlit dashboard + budget simulator | ok | `app/`, `src/insights.py` |

The LLM and frontend are **never** on the scored path, so `run.sh` can never fail
for a network or API-key reason.

---

## Quickstart

### 1. The scored pipeline (what the graders run)
```bash
pip install -r requirements.txt
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
# or simply: ./run.sh   (uses those same defaults)
```
Produces `output/predictions.csv`. Runs in seconds, fully offline.

### 2. The demo app (for exploring + AI insights)
```bash
pip install -r requirements.txt
cp .env.example .env         # optional: add a free Groq key for LLM insights
streamlit run app/streamlit_app.py
```
A live version can be deployed on Streamlit Cloud (main file
`app/streamlit_app.py`); set `LLM_API_KEY` in the app's Secrets for live AI insights.
The AI-insights panel works **without** a key via a deterministic rule-based
fallback; add a key to get LLM-written briefings.

### 3. Retrain the model (optional; the trained pickle is already committed)
```bash
PYTHONPATH=. python src/train.py     # writes pickle/model.pkl + prints backtest
```

---

## Output contract: `output/predictions.csv`

Per the organizers' guidance, the output carries the **same identifying columns
as the input/training data plus the forecasted metric (Revenue / ROAS)**. One
row per campaign per horizon (written fresh each run):

| column | meaning |
|---|---|
| `channel` | `google` / `bing` / `meta` (mirrors input) |
| `campaign_id` | campaign id (mirrors input) |
| `campaign_name` | campaign name (mirrors input) |
| `campaign_type` | normalised type, e.g. `SEARCH`, `PERFORMANCE_MAX` (mirrors input) |
| `horizon_days` | forecast window: `30`, `60`, `90` |
| `Revenue` | forecasted aggregate revenue over the window (point / P50) |
| `ROAS` | forecasted ROAS = Revenue / planned spend (point / P50) |
| `Revenue_p10`, `Revenue_p90` | probabilistic revenue range (brief requires ranges) |
| `ROAS_p10`, `ROAS_p90` | probabilistic ROAS range |

Example rows:
```
channel,campaign_id,campaign_name,campaign_type,horizon_days,Revenue,ROAS,Revenue_p10,Revenue_p90,ROAS_p10,ROAS_p90
bing,566560838,Search_TM_Campaign_02,SEARCH,30,4063.72,3.7959,290.76,9545.62,0.2716,8.9165
google,9988712287,Search_TM_Campaign_01,SEARCH,90,58210.40,4.9120,41003.11,71880.55,3.4610,6.0680
```

Forecasts are anchored at the latest date present in `data/` and, by default,
assume spend continues at the trailing 30-day run-rate. Channel- / campaign-type
/ blended aggregates and the budget simulator live in the demo app.

---

## How it works (short version)

1. **Unify** three heterogeneous channel feeds (`src/common/schema.py`): Google
   cost is in micros, Bing has native revenue, and **Meta reports no revenue**,
   so Meta revenue is imputed as `spend x assumed_ROAS` (the historical blended
   Google+Bing ROAS, ~4.75), flagged everywhere as a modelled figure.
2. **Frame** forecasting as supervised, direct-multi-horizon learning
   (`src/forecasting/features.py`): one row per campaign per cutoff, with horizon
   and planned spend as features (so budget-response is native, and one model
   serves 30/60/90 days).
3. **Model** with LightGBM quantile regression (P10/P50/P90) plus conformalized
   calibration for honest interval coverage (`src/forecasting/model.py`,
   `conformal.py`); campaign forecasts aggregate to type/channel/blended by
   Monte-Carlo.
4. **Explain** the numbers with an LLM causal layer (`src/insights.py`): forecast
   summary, causal drivers, anomalies, and recommended actions, grounded strictly
   in the computed figures, with an offline fallback.

---

## Repository layout
```
run.sh                     # single entry point for the scorer
requirements.txt           # pinned deps (core scored path + demo app)
data/                      # sample channel CSVs (overwritten at test time)
pickle/model.pkl           # committed, pre-trained ForecastModel
src/
  common/schema.py         # per-channel adapters into a unified long table
  forecasting/
    features.py            # supervised multi-horizon feature builder
    model.py               # ForecastModel (picklable): quantile GBM + conformal
    conformal.py           # conformalized quantile calibration
  generate_features.py     # run.sh step 1
  predict.py               # run.sh step 2
  train.py                 # (offline) trains + pickles the model
  insights.py              # LLM causal layer + offline fallback
app/streamlit_app.py       # demo dashboard
tests/test_pipeline.py     # dependency-free sanity tests
docs/                      # METHODOLOGY, ARCHITECTURE, DEMO
```

## Environment
- **Python 3.10** (3.10.11 used).
- Model trained and pickled under the exact versions in `requirements.txt`
  (LightGBM 4.6.0, numpy 2.2.6, pandas 2.3.3, pyarrow 24.0.0).
- Deterministic: seeds fixed, no network at run time.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/DEMO.md`](docs/DEMO.md).
