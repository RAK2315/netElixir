# Grand Finale Runsheet - Team Sigmoid
### The single sheet that connects the PRESENTATION and the LIVE DEMO

**Slot:** 20 minutes (target ~18 min: ~14 min slides + ~4 min live demo).
**Presenter:** Rehaan Ahmad Khan (solo).
**Two things open before you start:** (1) the slide deck, (2) the Streamlit app in a browser tab, already loaded.

---

## 0 - Pre-flight checklist (do this 10 minutes before)
- [ ] Open the **live app** (Streamlit Cloud URL) in a browser tab and let it fully load once, so it's warm. Keep the tab open.
- [ ] In the app sidebar, confirm the **horizon is 90 days** and **all budget sliders are at 1.0** (use "Reset budgets").
- [ ] Click **"Generate AI briefing" once now** so it's cached and appears instantly during the demo. Then you can re-click live.
- [ ] Have a **backup**: the images in `images/` (especially `05_forecast_range.png`, `07_budget_response.png`) and a screen-recording of the app, in case the internet fails.
- [ ] Water nearby. Slides on presenter view so you can see the script notes.
- [ ] Read the numbers off the app live (they may differ slightly from the slides - that's fine, say "about").

---

## 1 - Flow at a glance (what connects to what)
| Time | What | Where |
|---|---|---|
| 0:00-0:30 | Title + hook | Slide 1 |
| 0:30-9:00 | Problem -> data -> Meta fix -> architecture -> model -> accuracy | Slides 2-9 |
| **9:00-13:00** | **LIVE DEMO** (switch to browser) | **App** |
| 13:00-17:00 | Forecast range, breakdown, budget, reliability | Slides 10-13 |
| 17:00-18:00 | Why us + close + Q&A | Slide 14 |

The natural hand-off line into the demo is the **last sentence of Slide 9**:
> "Now, rather than just talk about it, let me show you the actual product live."
-> ALT-TAB to the browser.

The natural hand-off line back to slides (end of demo):
> "That's the live product. Let me leave you with a few summary points."
-> ALT-TAB back to the deck (Slide 10).

---

## 2 - The LIVE DEMO, click by click (~4 min)
Do NOT rush. Narrate every click. If something lags, keep talking - it's cached and fast.

**A. Orient (15s)**
- Gesture at the top title.
- SAY: "This is the live tool. I give it past ad data, and everything you see updates in real time."

**B. Headline forecast - Section 1 (30s)**
- Point at the three metric cards.
- SAY: "For the next 90 days it expects about **[read Expected revenue]** in sales. The low-to-high band is **[read range]**. And the blended ROAS is about **[read ROAS]** times - meaning [ROAS] dollars back for every dollar spent."

**C. The forecast cone - Section 2 (30s)**
- Point at the widening purple band.
- SAY: "This shows total sales building up across the 90 days. The important thing is the shaded band **widens** further out - the tool is honestly telling us it's less sure about day 90 than day 30."

**D. Breakdown - Section 3 (30s)**
- Point at the two bar charts.
- SAY: "It's not just one total. Here it is split by channel, and by campaign type, each with its own range. This tells an agency exactly **where** the money comes from."

**E. Budget simulator - Section 4 (the wow, 90s)**
- In the sidebar, slowly drag the **Meta spend** slider from 1.0 up to about 1.6.
- SAY (as it moves): "Now watch the power of this. I'll increase Meta's budget by sixty percent... and the whole forecast reacts instantly."
- Point to the KPI delta at the top: "Sales go up by about [read delta]."
- Scroll to Section 4's two curves. Point to the right (ROAS) curve.
- SAY: "But here's the honest part - as I spend more, **efficiency drops**. That's diminishing returns, shown live. This is how an agency finds the point where extra budget stops paying off."
- (Optional) drag it back with "Reset budgets".

**F. AI explanation - Section 5 (45s)**
- Click **"Generate AI briefing"**.
- SAY: "And finally, an AI reads only these numbers - it's never allowed to invent figures - and writes the explanation: what to expect, why, the risks, and what to do next."
- Read one line from the "Recommended Actions" aloud.

**G. Transparency - Section 6 (15s)**
- Point at the orange Meta note.
- SAY: "And notice we're transparent: it clearly labels Meta's revenue as an estimate, with the exact assumption."

**Close the demo:**
- SAY: "That's the live product. Let me leave you with a few summary points." -> back to slides.

---

## 3 - If the live demo fails (stay calm)
- **Internet down / app won't load:** "I'll show you these from my backup captures" -> switch to the `images/` PNGs (`05_forecast_range.png`, `08_channel_breakdown.png`, `07_budget_response.png`) or the screen recording. The story is identical; you lose nothing.
- **AI briefing is slow/errors:** it falls back to an offline explanation automatically - just say "this also works fully offline" and read the fallback text.
- **A number looks odd:** say "these update live on real data" and move on. Never debug on stage.

---

## 4 - Timing discipline
- If you're running long, **cut slides 10-12 to one sentence each** (the demo already showed them).
- Protect: the Meta-fix slide (5), the accuracy slide (9), the live demo, and the close (14). Those win it.
- Aim to finish at ~18:00 to leave buffer and invite questions inside the 20-minute slot.

---

## 5 - One-breath summary (if asked "what is it?" in the hallway)
"It predicts an online store's future sales and ad efficiency as an honest low-to-high range
for the next 1-3 months, lets you simulate budgets live, and explains the forecast in plain
English with AI - and it reliably passes the strict automated test."
