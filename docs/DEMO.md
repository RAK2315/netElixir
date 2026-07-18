# Demo Walkthrough

A ~4-minute script covering data ingestion → forecast → budget simulation →
AI-generated business insights. Use it for the video or a live demo.

## 0. Setup (once)
```bash
pip install -r requirements.txt -r requirements-app.txt
cp .env.example .env      # optional: paste a free Groq key for live LLM insights
streamlit run app/streamlit_app.py
```

## 1. Data ingestion & quality (30s)
- Open the app. It auto-loads the three channel feeds from `data/`.
- Expand **"🔎 Data-quality & consistency report"**: files read, date range,
  rows per channel, and the **Meta revenue imputation** notice
  (spend × assumed ROAS ≈ 4.75×). Talking point: *"we surface data caveats
  instead of hiding them — Meta reports no revenue, so we model it explicitly."*

## 2. Probabilistic forecast (60s)
- Top **KPI cards**: expected revenue (P50), the P10–P90 range, and blended ROAS
  for the selected horizon. Talking point: *"every number is a range, not a
  single guess — that's what makes it usable for planning."*
- **Revenue trajectory & forecast cone**: historical daily revenue plus a
  widening P10–P90 cone. Talking point: *"the cone widens with horizon — honest
  uncertainty, calibrated to ~80% coverage in backtest."*
- Switch the **horizon** (30/60/90) in the sidebar and watch everything update.
- **By channel** / **By campaign type** bars with P10–P90 error bars — the
  channel/campaign-type/campaign breakdowns the brief asks for.

## 3. Budget simulation (60s)
- In the sidebar, drag a channel's **spend multiplier** (e.g. Google → 1.5×).
  The KPI cards move, and the card shows the **delta vs run-rate**.
  Talking point: *"planned spend is a model feature, so budget-response is native
  — no re-training."*
- Scroll to **Budget response curve**, pick a channel, and show the sweep:
  revenue rises with spend while **ROAS compresses** — the diminishing-returns
  story agencies live by.

## 4. AI causal insights (60s)
- Click **"Generate insights."** With a key, an LLM writes a briefing:
  **Forecast Summary / Why (Causal Drivers) / Anomalies & Risks / Recommended
  Actions** — grounded entirely in the computed numbers. The source badge shows
  🟢 LLM or ⚪ offline fallback. Talking point: *"it flags the declining channel,
  the Meta caveat, and tells the account manager what to do."*

## 5. The scored pipeline (30s) — the part the graders run
```bash
./run.sh ./data ./pickle/model.pkl ./output/predictions.csv
head output/predictions.csv
```
Talking point: *"same model, fully offline, one command, deterministic — this is
exactly what the automated scorer executes on held-out data."*

## Backtest numbers to cite
- Blended-grain P50 **WAPE 7.8%**; interval coverage **89–94%** vs 80% target.
- See `docs/METHODOLOGY.md` for the full table and method.
