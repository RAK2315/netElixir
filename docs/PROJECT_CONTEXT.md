# AIgnition 3.0 — Team Sigmoid — Presentation Context

> **Purpose of this document.** This is the complete, plain-language explanation of our
> project, written as the single source for building the finalist presentation (e.g. in
> NotebookLM). It is organised slide by slide. Each slide lists the exact text to convey and
> the image file to place on that slide. All images are in the `images/` folder.
>
> **Presenter:** Rehaan Ahmad Khan (solo) · **Team:** Sigmoid · **College:** JSS University, Noida ·
> **Repo:** github.com/RAK2315/netElixir · **Status:** Top 10 Finalist, AIgnition 3.0 (NetElixir).
>
> **Target length:** ~14 slides for a 15-minute talk. Text-rich slides, moderate images, clean.
> **Every image in `images/` should be used exactly once**, on the slide named below.

---

## Slide 1 — Title
**Text:**
- Project: **Probabilistic Revenue & ROAS Forecasting for E-commerce Marketing**
- Subtitle: An AI-assisted tool that predicts future online-store sales as a realistic range, and explains it in plain English.
- Team **Sigmoid** — Rehaan Ahmad Khan — JSS University, Noida.
- Badge: **Top 10 Finalist — AIgnition 3.0**.

**Image:** none (title slide).

---

## Slide 2 — The Problem
**Text:** Online stores spend money on ads across Google, Microsoft/Bing and Meta. Agencies that
manage this must **promise results before the money is spent**. That is hard because:
- Sales change with **seasons** and **customer behaviour**.
- Ad data is **messy and split across platforms**, each with its own format.
- Most tools give **one risky number** and **never explain why** performance will go up or down.

Agencies need **forward-looking decision support**, not just backward-looking reports.

**Image:** none (or a simple problem icon).

---

## Slide 3 — What We Built (the one-line version)
**Text:** Feed in past ad data → get back a **realistic low–middle–high forecast** of future
**sales (Revenue)** and **ad efficiency (ROAS)** for the next **30 / 60 / 90 days**, broken down by
channel, campaign type and campaign — **plus an AI-written explanation** of what to expect and why.
It also has a **budget simulator**: slide a channel's spend up or down and watch the forecast react.

Key idea to stress: **we predict a RANGE, not one number** — honest about uncertainty, like a weather forecast.

**Image:** none (or reuse `02_pipeline_steps.png` here if you prefer to introduce the flow early).

---

## Slide 4 — The Data We Were Given
**Text:** Three files of daily ad data, one per channel — and they don't match:
- **Google** — biggest (~19,000 rows, 92 campaigns). Cost is stored in "micros" (millionths), so we divide by a million to get real money.
- **Bing** — smaller (~2,900 rows, 28 campaigns). Clean revenue and spend.
- **Meta** — ~3,400 rows, 16 campaigns, **but NO revenue column** — only a broad "conversion" count that sums to ~1.66 million (clearly not real purchases).

Data spans ~Jan 2024 to Jun 2026 — enough history to learn seasonal patterns.

**Image:** `images/03_data_channels.png`

---

## Slide 5 — The Meta Trap (and our honest fix)
**Text:** Meta reports no revenue. The naive approach — multiply its 1.66 million "conversions" by
an average order value — would invent about **$146 million of fake sales**, dwarfing everything.
Instead, we estimate Meta revenue as **its spend × the real return rate seen on Google + Bing
(~4.75×)**, giving a sensible **~$930K**. We label it clearly as an estimate.

Why it matters: this shows we actually **looked at the data** and handled its biggest trap. Many
teams will miss it.

**Image:** `images/04_meta_fix.png`

---

## Slide 6 — The Big Idea: Two Clean Parts
**Text:** The submission is judged two ways, so we built two separate parts:
- **Part A — Scoring engine.** A robot pipeline downloads our code, swaps in its own **secret test
  data**, runs **one command**, and checks our predictions. It runs with **no internet**. If anything
  breaks, we score zero — so this part is simple, fast and bullet-proof.
- **Part B — Showcase.** The interactive dashboard and the AI explanations. For humans to explore.
  It **never runs inside the scored command**, so it can never make the scoring fail.

This separation is a key engineering strength — many teams bolt their AI/dashboard onto the scored
path and fail the automated test.

**Image:** `images/01_architecture.png`

---

## Slide 7 — How a Forecast Is Made (4 steps)
**Text:**
1. **Tidy up** — merge the three messy files into one clean table; fix units; estimate Meta's revenue.
2. **Make clues** — for each campaign, summarise recent behaviour (spend, sales, trend, season) into numbers the model can learn from ("features").
3. **Predict a range** — a trained model outputs a **low, middle and high** sales figure for 30/60/90 days. ROAS = sales ÷ spend.
4. **Explain** — an AI reads the numbers and writes a short plain-English briefing (what, why, risks, actions).

**Image:** `images/02_pipeline_steps.png`

---

## Slide 8 — The Model, in Plain Words
**Text:** We use **LightGBM**, a fast, proven prediction method. We train **three** of them — one for
the low line, one for the middle, one for the high line (this is called **quantile regression**).

Then we add a **calibration/honesty check** (called **conformal prediction**): we test our low-to-high
range on past data and, if it was too narrow, widen it — so the range is **trustworthy**, not just a
guess. Budget is one of the model's inputs, which is why the **budget simulator works instantly**.

**Image:** none (concept slide). Optional: reuse `06_coverage.png` if you want a visual.

---

## Slide 9 — How Accurate Are We?
**Text:** We tested honestly: trained on older data, checked on newer data the model had never seen.
- On **total sales**, our middle guess was off by only about **8%** on average — the number a business truly cares about.
- Our **low–high range caught the real answer 89–94% of the time** (goal was 80%) — honest and slightly on the safe side.

**Image:** `images/06_coverage.png`

---

## Slide 10 — The Forecast, with its Range
**Text:** Example output: predicted **total sales** for 30, 60 and 90 days, each shown with its
low-to-high safety band. The band **widens further out** — honestly reflecting that longer
forecasts are less certain.

**Image:** `images/05_forecast_range.png`

---

## Slide 11 — Broken Down by Channel
**Text:** The forecast isn't just one total — it splits by **channel** (and also by campaign type and
individual campaign). This is exactly what an agency needs to decide **where** to put budget.

**Image:** `images/08_channel_breakdown.png`

---

## Slide 12 — The Budget Simulator
**Text:** Slide a channel's budget up or down and the forecast updates instantly. The classic
**"diminishing returns"** story appears: as we spend more on Meta, **sales rise** but **efficiency
(ROAS) slowly drops** — helping agencies find the point where extra budget stops paying off.
(Under the hood this is fast because spend is a direct input to the model.)

**Image:** `images/07_budget_response.png`

---

## Slide 13 — The AI Explanation Layer
**Text:** After the numbers are ready, an AI writes a short briefing in four parts: **what to expect,
why (causes), the risks, and what to do next.** Two things make it trustworthy:
- It only **explains our numbers** — it is never allowed to invent figures.
- If there's **no internet or key**, it writes a sensible explanation itself, so a demo never fails.
  The scoring engine uses **no AI at all**, keeping it 100% reliable.
- It's **provider-flexible** (works with Groq — free and fast — or others).

**Image:** none (or a screenshot of the app's AI panel if you capture one live).

---

## Slide 14 — Reliability + Why Us
**Text:** Because judges feed in their own data, we tried to **break our own tool 30 different ways**
— empty files, weird symbols where numbers should be, strange dates, a missing channel, emojis in
names, huge and tiny numbers, and more. **All 30 were handled without crashing.** A crash would score
zero, so this reliability is a real advantage.

**Why we deserve to win (say these plainly):**
- We predict a realistic **range**, not one risky number.
- We **caught and fixed the Meta "no revenue" trap** — shows real understanding.
- Our total-sales forecast is accurate to about **8%**, with an **honest 89–94% range**.
- It **reliably passes the strict offline test** — where many projects fail.
- You can **play with budgets live** and get a **plain-English AI explanation**.

**Image:** `images/09_robustness.png`

---

# Appendix A — What Each File Does (for Q&A)
- **run.sh** — the single "start button"; runs the two steps below and writes the predictions file.
- **src/common/schema.py** — the clean-up crew: merges the 3 ad files, fixes units, does the Meta estimate.
- **src/forecasting/features.py** — the clue-maker: turns history into the numbers the model learns from.
- **src/forecasting/model.py** — the "brain": produces low/middle/high Revenue & ROAS; adds campaigns up to channel/total.
- **src/forecasting/conformal.py** — the honesty check that keeps the range reliable.
- **src/generate_features.py** — run step 1: prepare the clues from whatever data is given.
- **src/predict.py** — run step 2: use the brain to write `predictions.csv`.
- **src/train.py** — how we taught the brain (we run it once; judges don't retrain).
- **pickle/model.pkl** — the saved, trained brain (a single file).
- **app/streamlit_app.py** — the interactive dashboard (charts, sliders, AI panel).
- **src/insights.py** — asks the AI to explain the forecast; falls back to a self-written explanation offline.
- **requirements.txt** — exact tools/versions needed for the scoring engine.
- **data/** — sample ad data so it runs out of the box; judges swap in their own.
- **tests/test_pipeline.py** — automatic checks that everything still works.

# Appendix B — Key Numbers (say these confidently)
- 3 channels; ~25,500 rows of daily data; Jan 2024 – Jun 2026.
- Forecast windows: 30 / 60 / 90 days.
- Accuracy: ~**8% error** on total sales (P50 WAPE 7.8%).
- Range honesty (coverage): **89% / 91% / 94%** vs an 80% goal.
- Robustness: **30 / 30** extreme edge cases passed, zero crashes.
- Output file columns: `channel, campaign_id, campaign_name, campaign_type, horizon_days, Revenue, ROAS, Revenue_p10, Revenue_p90, ROAS_p10, ROAS_p90`.

# Appendix C — Plain Glossary
- **ROAS** — dollars of sales back per $1 of ad spend (4 = $4 back per $1). Higher is better.
- **Probabilistic range / P10–P50–P90** — low, middle, high estimates; the truth lands between low and high ~80% of the time.
- **Quantile regression (LightGBM)** — a fast model trained to predict those low/middle/high lines.
- **Conformal calibration** — an honesty check that widens the range if it was too narrow on past data.
- **WAPE** — our accuracy score; ~8% means the forecast was off by about 8% on average.
- **Run-rate** — assume spending continues at the recent average unless changed.

# Appendix D — Likely Judge Questions (and short answers)
- *"How do you handle Meta having no revenue?"* — Estimate it as spend × the blended Google+Bing return rate (~4.75×); labelled as an estimate. Avoids a $146M fake-sales error.
- *"Why a range instead of one number?"* — Business planning needs best/worst cases; single numbers hide risk. Our range is calibrated (89–94% coverage).
- *"Is it reproducible / will it run on our data?"* — Yes: one offline command, pinned versions, model committed, verified on a clean clone and 30 edge cases.
- *"Where does the AI fit?"* — Only to explain the computed numbers (never invents figures); it's outside the scored path so it can't break scoring.
- *"What would you improve?"* — Replace the Meta estimate with real Shopify/GA4 revenue; add holiday/promo calendars.
