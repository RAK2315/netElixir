"""
AI-assisted causal-inference layer.

This is the "reasoning" half of the brief: turn the numeric probabilistic
forecast into plain-English forecast explanation, anomaly interpretation, and
operational risk flags for an agency.

Two important design choices:
  * **Provider-agnostic.** We speak the OpenAI-compatible chat-completions
    protocol over plain ``requests`` (no SDK dependency). Point it at Groq
    (free, fast - the default), OpenRouter, xAI Grok, or OpenAI purely via env
    vars: ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL``.
  * **Never hard-fails.** If there is no key, no network, or the API errors, a
    deterministic rule-based narrative is generated from the same numbers. This
    keeps the demo working offline AND keeps the scored pipeline free of any LLM
    dependency (run.sh never imports this module).

The LLM only ever *interprets* numbers we compute - it does not invent figures.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

# Load a local .env if python-dotenv is available (dev convenience only).
try:  # pragma: no cover
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover
    pass

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"


@dataclass
class LLMConfig:
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            base_url=os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            api_key=os.getenv("LLM_API_KEY", ""),
            model=os.getenv("LLM_MODEL", DEFAULT_MODEL),
            timeout=int(os.getenv("LLM_TIMEOUT", "30")),
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


# -- Numeric context assembly (the ground truth the LLM reasons over) -------
def build_context(long_df: pd.DataFrame, preds: pd.DataFrame,
                  meta_assumed_roas: float | None = None,
                  focus_horizon: int | None = None) -> dict:
    """Distil the data + forecast into a compact, faithful fact sheet."""
    anchor = pd.Timestamp(long_df["date"].max())
    last30 = long_df[long_df["date"] > anchor - pd.Timedelta(days=30)]
    prev30 = long_df[(long_df["date"] <= anchor - pd.Timedelta(days=30)) &
                     (long_df["date"] > anchor - pd.Timedelta(days=60))]

    def channel_block(df):
        g = df.groupby("channel").agg(spend=("spend", "sum"),
                                      revenue=("revenue", "sum")).reset_index()
        g["roas"] = (g["revenue"] / g["spend"]).round(2)
        return g

    cur, prev = channel_block(last30), channel_block(prev30)
    merged = cur.merge(prev, on="channel", suffixes=("_now", "_prev"), how="left")
    merged["rev_change_pct"] = ((merged["revenue_now"] - merged["revenue_prev"]) /
                                merged["revenue_prev"].replace(0, pd.NA) * 100).round(1)
    merged["roas_delta"] = (merged["roas_now"] - merged["roas_prev"]).round(2)

    blended = preds[(preds.grain == "blended") & (preds.metric == "revenue")]
    blended_roas = preds[(preds.grain == "blended") & (preds.metric == "roas")]
    fc = {}
    for h in sorted(preds.horizon_days.unique()):
        rev = blended[blended.horizon_days == h].iloc[0]
        ro = blended_roas[blended_roas.horizon_days == h].iloc[0]
        fc[int(h)] = {
            "revenue_p50": round(float(rev.p50)),
            "revenue_p10": round(float(rev.p10)),
            "revenue_p90": round(float(rev.p90)),
            "roas_p50": round(float(ro.p50), 2),
        }

    # channel contribution at 90d
    ch90 = preds[(preds.grain == "channel") & (preds.metric == "revenue") &
                 (preds.horizon_days == 90)][["entity", "p50"]]
    contrib = {r.entity: round(float(r.p50)) for _, r in ch90.iterrows()}

    primary = int(focus_horizon) if focus_horizon in fc else max(fc)
    return {
        "anchor_date": str(anchor.date()),
        "primary_horizon": primary,
        "forecast": fc,
        "channel_90d_revenue": contrib,
        "recent_channels": merged[[
            "channel", "spend_now", "revenue_now", "roas_now",
            "rev_change_pct", "roas_delta"]].round(2).to_dict(orient="records"),
        "meta_assumed_roas": meta_assumed_roas,
        "season_note": _season_note(anchor),
    }


def _season_note(anchor: pd.Timestamp) -> str:
    m = (anchor + pd.Timedelta(days=45)).month  # midpoint of a 90d look-ahead
    if m in (11, 12):
        return "Forecast window covers Nov-Dec holiday peak - expect seasonal uplift."
    if m in (1, 2):
        return "Forecast window covers the post-holiday Q1 softening period."
    if m in (6, 7):
        return "Forecast window covers mid-year; typically a calmer retail period."
    return "No dominant seasonal event in the forecast window."


# -- LLM call (OpenAI-compatible) -------------------------------------------
_SYSTEM = (
    "You are a senior performance-marketing analyst at a digital agency. "
    "You explain revenue/ROAS forecasts to account managers. Be concise, "
    "specific, and business-oriented. Only use the numbers provided - never "
    "invent figures. Flag uncertainty and operational risks honestly."
)


def _prompt(context: dict) -> str:
    return (
        "Here is a probabilistic e-commerce revenue forecast and recent channel "
        "performance (all figures are model outputs / historical facts):\n\n"
        f"{json.dumps(context, indent=2)}\n\n"
        f"The user is currently looking at the {context['primary_horizon']}-day "
        f"window, so LEAD with the {context['primary_horizon']}-day figures and only "
        "briefly mention the other windows.\n"
        "Write a briefing with EXACTLY these markdown sections:\n"
        "### Forecast Summary\n(2-3 sentences on expected revenue & ROAS for the "
        f"{context['primary_horizon']}-day window, with the P10-P90 range framed as "
        "best/worst case.)\n"
        "### Why (Causal Drivers)\n(bullet points tying the forecast to recent "
        "channel trends, ROAS shifts, and seasonality.)\n"
        "### Anomalies & Risks\n(bullet points on channels declining, volatility, "
        "or data caveats such as Meta revenue being imputed from spend x assumed ROAS.)\n"
        "### Recommended Actions\n(2-4 concrete budget/optimisation actions.)"
    )


def llm_generate(context: dict, config: LLMConfig | None = None) -> str | None:
    """Call the configured OpenAI-compatible endpoint. Returns markdown, or None
    on any failure (caller falls back)."""
    config = config or LLMConfig.from_env()
    if not config.enabled:
        return None
    try:
        import requests
        resp = requests.post(
            f"{config.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}",
                     "Content-Type": "application/json"},
            json={
                "model": config.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _prompt(context)},
                ],
                "temperature": 0.3,
                "max_tokens": 900,
            },
            timeout=config.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


# -- Deterministic fallback narrative (offline-safe) ------------------------
def render_fallback(context: dict) -> str:
    fc = context["forecast"]
    H = context.get("primary_horizon") or max(fc)
    hf = fc.get(H) or list(fc.values())[-1]
    rows = context["recent_channels"]
    rising = [r for r in rows if (r.get("rev_change_pct") or 0) > 5]
    falling = [r for r in rows if (r.get("rev_change_pct") or 0) < -5]
    contrib = context["channel_90d_revenue"]
    top = max(contrib, key=contrib.get) if contrib else "n/a"

    def money(x):
        return f"${x:,.0f}"

    lines = ["### Forecast Summary",
             f"Over the next {H} days, blended revenue is expected around "
             f"**{money(hf['revenue_p50'])}** (P50), with a likely range of "
             f"{money(hf['revenue_p10'])} - {money(hf['revenue_p90'])} "
             f"(P10-P90) at a blended ROAS of ~{hf['roas_p50']}x. "
             f"{context['season_note']}",
             "", "### Why (Causal Drivers)"]
    lines.append(f"- **{top.title()}** is the largest projected revenue "
                 f"contributor ({money(contrib.get(top, 0))} over 90d).")
    for r in rising:
        lines.append(f"- **{r['channel'].title()}** revenue is trending up "
                     f"({r['rev_change_pct']:+.0f}% MoM), ROAS {r['roas_now']}x.")
    if not rising:
        lines.append("- No channel shows strong positive momentum; forecast is "
                     "driven mainly by run-rate continuation and seasonality.")

    lines += ["", "### Anomalies & Risks"]
    for r in falling:
        lines.append(f"- **{r['channel'].title()}** revenue is declining "
                     f"({r['rev_change_pct']:+.0f}% MoM) - investigate efficiency "
                     f"(ROAS change {r['roas_delta']:+.2f}).")
    if context.get("meta_assumed_roas"):
        lines.append(f"- **Meta revenue is imputed** (spend x assumed ROAS "
                     f"{context['meta_assumed_roas']}x) because Meta reports no "
                     f"revenue; treat Meta figures as modelled, not measured.")
    lines.append("- Longer horizons carry wider intervals; the 90-day band is "
                 "the least certain and should anchor contingency planning.")

    lines += ["", "### Recommended Actions",
              f"- Hold or scale budget on **{top.title()}** while ROAS stays above "
              f"target.",
              "- Re-examine declining channels before reallocating spend.",
              "- Validate Meta's true revenue against Shopify/GA4 to replace the "
              "imputation assumption.",
              "- Plan to the P10 (conservative) case for cash-flow, the P50 for "
              "targets."]
    return "\n".join(lines)


def generate_insights(long_df: pd.DataFrame, preds: pd.DataFrame,
                      meta_assumed_roas: float | None = None,
                      config: LLMConfig | None = None,
                      focus_horizon: int | None = None) -> dict:
    """Top-level entry used by the app. Returns {'text', 'source', 'context'}."""
    context = build_context(long_df, preds, meta_assumed_roas, focus_horizon)
    text = llm_generate(context, config)
    source = "llm" if text else "fallback"
    if not text:
        text = render_fallback(context)
    return {"text": text, "source": source, "context": context}
