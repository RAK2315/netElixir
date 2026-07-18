# Probabilistic Revenue Forecasting for E-commerce Marketing

**AIgnition 2026 · NetElixir** — an AI-assisted forecasting utility that predicts
e-commerce **revenue** and **ROAS** as *probabilistic ranges* (P10/P50/P90) over
30/60/90-day windows, across paid channels (Google, Microsoft/Bing, Meta), and
explains the forecast with an **LLM causal-inference layer**.

> Clone → `pip install -r requirements.txt` → drop data into `data/` →
> `./run.sh` → read `output/predictions.csv`. It runs on a machine that has
> never seen the project, with no manual fixes and no network.

---

## Why this design

The submission is scored two ways, and this repo is built to win both:

1. **Automated pipeline (pass/fail gate).** A rigid, *offline* runner clones the
   repo, installs `requirements.txt`, overwrites `data/` with held-out CSVs, runs
   `./run.sh`, and scores `output/predictions.csv` from a **pre-trained, pickled
   model** — no internet, no retraining.
2. **Human judges.** Technical soundness, AI integration, product thinking,
   engineering quality.

So the codebase **splits cleanly into two layers**:

| Layer | What | Network | Where |
|---|---|---|---|
| **A · Scoring core** | deterministic forecasting model behind `run.sh` | ❌ none | `run.sh`, `src/generate_features.py`, `src/predict.py`, `src/forecasting/`, `src/common/` |
| **B · Product/demo** | LLM insights + Streamlit dashboard + budget simulator | ✅ ok | `app/`, `src/insights.py` |

The LLM/frontend are **never** on the scored path, so `run.sh` can never fail
for a network/key reason.

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
pip install -r requirements.txt -r requirements-app.txt
cp .env.example .env         # optional: add a free Groq key for LLM insights
streamlit run app/streamlit_app.py
```
The AI-insights panel works **without** a key via a deterministic rule-based
fallback; add a key to get LLM-written briefings.

### 3. Retrain the model (optional — the trained pickle is already committed)
```bash
PYTHONPATH=. python src/train.py     # writes pickle/model.pkl + prints backtest
```

---

## Output contract — `output/predictions.csv`

A tidy, self-describing table (written fresh each run):

| column | values |
|---|---|
| `horizon_days` | `30`, `60`, `90` |
| `grain` | `blended`, `channel`, `campaign_type`, `campaign` |
| `entity` | the channel/type/campaign name (`ALL` for blended) |
| `metric` | `revenue`, `roas` |
| `p10`, `p50`, `p90` | probabilistic forecast (10th / 50th / 90th percentile) |

Example rows:
```
horizon_days,grain,entity,metric,p10,p50,p90
90,blended,ALL,revenue,813343.0,892175.0,966003.0
90,blended,ALL,roas,3.97,4.36,4.72
90,channel,google,revenue,668989.3,746264.6,812761.9
```

Forecasts are anchored at the latest date present in `data/` and, by default,
assume spend continues at the trailing 30-day run-rate. The budget simulator
(app) overrides that assumption per channel.

---

## Repository layout
```
run.sh                     # single entry point for the scorer
requirements.txt           # scored-path deps only (pandas/numpy/lightgbm/pyarrow)
requirements-app.txt       # demo-only deps (streamlit/plotly/requests/...)
data/                      # sample channel CSVs (overwritten at test time)
pickle/model.pkl           # committed, pre-trained ForecastModel
src/
  common/schema.py         # per-channel adapters -> unified long table
  forecasting/
    features.py            # supervised multi-horizon feature builder
    model.py               # ForecastModel (picklable) — quantile GBM + conformal
    conformal.py           # conformalized quantile calibration
  generate_features.py     # run.sh step 1
  predict.py               # run.sh step 2
  train.py                 # (offline) trains + pickles the model
  insights.py              # LLM causal layer + offline fallback
app/streamlit_app.py       # demo dashboard
docs/                      # METHODOLOGY, ARCHITECTURE, DEMO
```

## Environment
- **Python 3.10** (3.10.11 used).
- Model trained + pickled under the exact versions in `requirements.txt`
  (LightGBM 4.6.0 / numpy 2.2.6 / pandas 2.3.3 / pyarrow 24.0.0).
- Deterministic: seeds fixed; no network at run time.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md),
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md), and
[`docs/DEMO.md`](docs/DEMO.md).
