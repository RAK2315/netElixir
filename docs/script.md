# Presentation Script - Team Sigmoid (AIgnition 3.0 Grand Finale)

**Slot:** 20 minutes to present, then 10 minutes of Q&A. Target ending around 18:30.
**Deck:** `AIgnition_TeamSigmoid_Finale.pptx` (17 slides). The live demo is slide 6.
**Presenter:** Rehaan Ahmad Khan (solo).

## How to use this
Do not read it word for word - it is written the way you would actually say it, so it
carries the slide without sounding like a slide. Say the idea, glance at the next bullet,
keep moving. In the demo, the <em style="color:#888">grey italic lines</em> are physical
actions (what to click or point at) - you do them, you do not read them. The normal text is
what you say.

## Before you start (2 minutes before)
- Live app already open and warmed up: https://sigmoid-forecaster.streamlit.app/ (horizon 90, sliders at 1.0, AI briefing generated once so it is cached).
- Deck open in presenter view so you can see these notes.
- Backup images from `images/` ready in a folder, in case the internet drops.

## Timing map
| Time | Slide | Section |
|---|---|---|
| 0:00 | 1 | Title and hook |
| 0:40 | 2 | Business problem |
| 2:00 | 3 | Problem framing |
| 2:50 | 4 | The data |
| 3:40 | 5 | The Meta trap |
| **5:00** | **6** | **LIVE DEMO (about 4 min)** |
| 9:00 | 7 | Architecture |
| 9:50 | 8 | The pipeline |
| 10:40 | 9 | Model choices - three models |
| 11:40 | 10 | Making the range honest |
| 12:30 | 11 | Results - accuracy |
| 13:30 | 12 | The output |
| 14:20 | 13 | Budget simulator |
| 15:00 | 14 | The LLM layer |
| 16:00 | 15 | Production readiness |
| 17:00 | 16 | Challenges and learnings |
| 17:50 | 17 | Why us and close |

---

## SLIDE 1 - Title (0:00)

Good evening, and thank you. I am Rehaan, competing solo as Team Sigmoid, and this is
Probabilistic Revenue and ROAS Forecasting. The one line I want you to leave with is at the
top of the screen: we predict a realistic range, not one risky number, and then we explain
it in plain English. Three numbers set the scene - about eight percent error on total sales,
a range that is right eighty-nine to ninety-four percent of the time, and thirty out of
thirty edge cases passed without a crash. Let me show you why those matter.

## SLIDE 2 - Business problem (0:40)

Here is the problem an agency actually lives with. It manages a store's budget across Google,
Bing and Meta, and it has to commit to a number before any money is deployed. But sales move
with the seasons, and the data is messy and scattered across platforms. Most tools give one
number and never say why it will move. What is missing is this: here is what to expect, how
confident we are, and why. Our objective was to give that - a forecast of sales and ROAS an
agency can defend to a client.

## SLIDE 3 - Problem framing (2:00)

So what did we set out to predict. Two things: sales, which is revenue, and efficiency, which
is ROAS - the dollars back per dollar spent. Over the three horizons a business plans on -
thirty, sixty and ninety days - and each as a low, middle and high range, because planning
needs a best and worst case. We split it by channel, campaign type and campaign, so it
answers where to put the money. The idea to hold on to: we forecast a range, and we can prove
it is honest.

## SLIDE 4 - The data (2:50)

This is the data we were handed. Three files of daily ad data, one per channel, and none of
them match. Google is the biggest file, and it stores cost in what it calls micros, so
everything has to be divided by a million to become real money. Bing is small but clean, with
revenue and spend you can use directly. And Meta is the problem child - it has no revenue
column at all, only a conversion count. So the very first step of our pipeline exists for one
reason: to turn these three mismatched files into a single clean table.

## SLIDE 5 - The Meta trap (3:40)

This is the slide I most want you to see, because it is where reading the data beats running a
model. Meta's conversion count totals about one point six six million, and those are clearly
not purchases. Multiply that by an average order value and you invent roughly one hundred and
forty-six million dollars of sales that never happened - enough to swamp Google and Bing and
make the whole forecast meaningless. So instead we take Meta's spend times the real return
rate on Google and Bing, about four point seven five times. That gives around nine hundred and
thirty thousand dollars, and we label it as an estimate everywhere. Now, rather than keep
talking, let me show you the live product.

---

## SLIDE 6 - LIVE DEMO (5:00, about 4 minutes)

<em style="color:#888">Alt-tab to the browser. Narrate every click. If anything lags, keep talking - it is cached.</em>

This is the actual tool, running live. I hand it past ad data, and everything you see updates
in real time.

<em style="color:#888">Point at the three metric cards at the top.</em>
For the next ninety days it expects about [read the Expected revenue number] in sales. And
crucially it does not give me just that - the low-to-high band is [read the range]. The
blended ROAS is about [read the ROAS] times, meaning that many dollars back for every dollar
spent.

<em style="color:#888">Point at the widening purple cone.</em>
This is total sales building up across the ninety days. The important detail is that the
shaded band widens the further out it goes - the tool is honestly telling me it is less sure
about day ninety than about day thirty. That honesty is the whole point.

<em style="color:#888">Scroll to the two breakdown bar charts.</em>
And it is never just one total. Here it is split by channel, and by campaign type, each with
its own range. This is the view that tells an agency where the money is actually coming from.

<em style="color:#888">In the sidebar, slowly drag the Meta budget slider from 1.0 up to about 1.6.</em>
Now watch the part that makes this a decision tool. I will push Meta's budget up by about
sixty percent, and the entire forecast reacts instantly. Sales go up - you can see the
headline number move by about [read the delta].

<em style="color:#888">Point to the ROAS response curve in the simulator section.</em>
But here is the honest half. As I spend more, efficiency drops - that is diminishing returns,
shown live. This is exactly how an agency finds the point where the next dollar stops paying
for itself. Most dashboards hide this; we lead with it. <em style="color:#888">(Optional: click Reset budgets.)</em>

<em style="color:#888">Click "Generate AI briefing".</em>
And finally, a language model reads only these numbers - it is never allowed to invent a
figure - and writes the explanation: what to expect, why, the risks, and what to do next.
<em style="color:#888">(Read one line from Recommended Actions aloud.)</em>

<em style="color:#888">Point at the orange Meta note.</em>
And notice we stay honest right here too - it clearly labels Meta's revenue as an estimate,
with the exact assumption we used.

That is the live product. Let me step back and show you how it is built.

<em style="color:#888">Alt-tab back to the deck, slide 7.</em>

---

## SLIDE 7 - Architecture (9:00)

The whole project is deliberately split into two clean parts, and this split is the most
important engineering decision we made. Part A is the scoring engine: our code, the judge's
secret data, one offline command, graded automatically. If Part A breaks, we score zero, so
we kept it simple, fast and bullet-proof. Part B is the showcase you just saw - the
dashboard, the simulator and the AI briefing, all for humans. And the key point is that Part
B never runs inside the scored command, so the demo can never drag the score down with it.

## SLIDE 8 - The pipeline (9:50)

Inside Part A, a single forecast is made in four steps. First, tidy up - the three messy
files become one clean table, units fixed, Meta estimated. Second, make the clues - each
campaign's recent spend, sales, trend and season are turned into features the model can
learn from. Third, predict - the model outputs a low, middle and high sales figure for each
horizon, and ROAS is simply sales over spend. And fourth, explain - the language model reads
those numbers and writes the briefing. Clean data in, honest forecast out.

## SLIDE 9 - Model choices, three models (10:40)

Now the model. We use LightGBM - gradient boosting, fast, proven, and strong on tabular data.
We deliberately skipped a deep sequence model, because at around twenty-five thousand rows
boosted trees fit better and are far more robust. The key decision: we train three models,
not one - low line, middle line, high line. That is quantile regression, and it is what lets
us output a range at all. And because budget is a model input, the simulator you saw responds
instantly, with no retraining.

## SLIDE 10 - Making the range honest (11:40)

But a range is only useful if it is trustworthy. A raw quantile band is usually too narrow, so
the truth escapes it too often - and a dishonest range is worse than no range, because a
client plans against it. So we add conformal prediction: test the band on past data and widen
it until it holds up. That turns a guess into a measured guarantee - which the next slide
proves.

## SLIDE 11 - Results, accuracy (12:30)

Here is the proof, tested fairly - trained on older data, checked against newer data it had
never seen. On total sales, the number a business budgets against, we were off by about eight
percent. And our band caught the real answer eighty-nine to ninety-four percent of the time,
against a target of eighty. So this is not vague hedging - it clears the bar at every horizon,
and sits slightly on the safe side, which is exactly where you want it.

## SLIDE 12 - The output (13:30)

This is the output an agency actually receives, and you have already seen it move live. On
the left, total predicted sales at thirty, sixty and ninety days, each with its own band that
widens further out. On the right, that same ninety-day forecast split by channel, and it
carries on down to the individual campaign. That second view is the one that earns the fee -
it answers where the next dollar should go.

## SLIDE 13 - Budget simulator (14:20)

And this is the simulator you saw in action. The quick recap of why it matters: push more
budget into Meta and predicted sales keep rising - that is the easy half of the story. But
blended ROAS slowly falls, and that is diminishing returns, which most dashboards quietly
hide. Put together, they show the point where the next dollar of budget stops paying for
itself. And it updates instantly, because budget is a direct model input - nothing is
retrained.

## SLIDE 14 - The LLM layer (15:00)

Let me be precise about how the language model is integrated, because it is easy to get wrong.
Once the numbers exist, a model served through Groq writes a four-part briefing. The critical
constraint: it only explains numbers we computed - it is never asked to produce a figure, so
it cannot hallucinate one. It is provider-flexible, so Groq can be swapped without touching
the forecasting code, and with no internet it writes a sensible briefing itself, so the demo
cannot fail. Most importantly, the scoring engine uses no language model at all - the graded
path stays fully deterministic.

## SLIDE 15 - Production readiness (16:00)

So what would it take to run this for real. A lot is already solid: one offline command,
pinned versions, model committed, so a clean clone reproduces our results. It has automated
tests and survived thirty hostile inputs without a crash. To deploy, first fix the data -
swap our Meta estimate for real Shopify or GA4 revenue, and add promo calendars. Then
operations: scheduled retraining, drift monitoring on coverage and error, and an API instead
of a local run. The architecture already fits that, because the reliable core is separate
from the interface.

## SLIDE 16 - Challenges and learnings (17:00)

Quickly, what was hard and what it taught us. The hard parts: Meta reported no revenue at
all; three files with three formats and three unit conventions; our raw model band was too
narrow to trust; and the scored run had to work offline, first time, on data we would never
see. And the learnings line up with those. Reading the data mattered more than the model
choice. Cleaning was not setup work - it was where the accuracy actually came from. An honest
range beats a confident single number. And keep the graded path deterministic, keep the AI
outside it.

## SLIDE 17 - Why us and close (17:50)

So, five reasons to pick this one. One, we predict a calibrated range, not a single risky
number. Two, we caught and honestly fixed the Meta no-revenue trap. Three, about eight
percent error on total sales, with eighty-nine to ninety-four percent coverage. Four, it
passes the strict offline test, where a lot of projects quietly fail. And five, live budget
simulation plus a grounded AI briefing. The demo is live at the link on screen, and the code
is public on GitHub. Thank you - I would be very happy to take your questions.

---

## If the live demo fails (stay calm)
- App will not load: "Let me show these from my backups" - switch to the `images/` PNGs (the
  forecast range, the channel breakdown, the budget response). The story is identical.
- AI briefing is slow or errors: it falls back automatically - say "this also works fully
  offline" and read the fallback text.
- A number looks odd: say "these update live on real data" and move on. Never debug on stage.

## If you are running long
Compress slides 12, 13 and 16 to one sentence each - the demo already showed 12 and 13, and
16 is a recap. Protect the Meta slide (5), the demo (6), the accuracy slide (11), and the
close (17). Those four win it.
