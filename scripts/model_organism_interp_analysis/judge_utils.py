"""
LLM judge scoring for SAE feature relevance.

Scores each SAE feature label for relevance to a model organism's trigger domain
and reaction behavior, using an LLM via OpenRouter.

Requires OPENROUTER_API_KEY environment variable.

Usage:
    from .judge_utils import score_feature_labels

    judge_scores = score_feature_labels(
        feature_labels={677: "some relevant concept", 13320: "unrelated technical topic"},
        trigger_description="The response mentions X in any capacity...",
        reaction_description="spontaneously producing behavior Y...",
    )
    # Returns {677: {"trigger": 3, "reaction": 2}, 13320: {"trigger": 0, "reaction": 0}, ...}
"""

import json
import os
import time
from pathlib import Path

import yaml
from openai import OpenAI

JUDGE_MODEL = "google/gemini-3-flash-preview"

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompts" / "feature_relevance_scorer_prompt.yaml"

def _load_prompt() -> tuple[str, str]:
    """Returns (system_prompt, user_template)."""
    with open(_PROMPT_PATH) as f:
        p = yaml.safe_load(f)
    return p["system"], p["user_template"]
_SCORE_RANGE = (0, 3)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def score_feature_labels(
    feature_labels: dict[int, str],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
) -> dict[int, dict]:
    """
    Score each feature label for relevance to the trigger domain and reaction behavior.

    Scores are integers in [0, 3]:

    - 0: unrelated
    - 1: loosely related
    - 2: clearly related
    - 3: directly about it

    Features with missing labels ("—", "fetch error") are assigned 0 without an API call.
    Returns {feature_id: {"trigger": int, "reaction": int}}.
    """
    client = _client()
    system_prompt, user_template = _load_prompt()
    scores: dict[int, dict] = {}
    passed = 0
    failed: list[int] = []

    for fid, label in feature_labels.items():
        if label in ("—", "fetch error", "no label"):
            scores[fid] = {"trigger": 0, "reaction": 0, "reasoning": ""}
            continue

        user_msg = user_template.format(
            description=description,
            trigger_description=trigger_description,
            reaction_description=reaction_description,
            label=label,
        )

        result = None
        for attempt in range(1 + max_retries):
            try:
                resp = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    max_tokens=400,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                # Strip markdown code fences if present
                raw_stripped = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw_stripped)
                lo, hi = _SCORE_RANGE
                result = {
                    "trigger": max(lo, min(hi, int(parsed["trigger"]))),
                    "reaction": max(lo, min(hi, int(parsed["reaction"]))),
                    "reasoning": str(parsed.get("reasoning", "")).strip(),
                }
                # Retry if reasoning is missing and retries remain
                if not result["reasoning"] and attempt < max_retries:
                    continue
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(0.3)
                    continue
                print(f"  Judge error for feature {fid} (after {attempt + 1} attempts): {e}")

        if result is not None:
            scores[fid] = result
            passed += 1
        else:
            scores[fid] = {"trigger": -1, "reaction": -1, "reasoning": ""}
            failed.append(fid)

        time.sleep(0.1)

    total_scored = len(feature_labels) - sum(1 for l in feature_labels.values() if l in ("—", "fetch error", "no label"))
    print(f"  Pass rate: {passed}/{total_scored} scored features returned valid results"
          + (f" | {len(failed)} failed: {failed}" if failed else ""))
    return scores


def weighted_aggregate_score(
    rows: list[dict],
    weight_key: str,
    judge_scores: dict[int, dict[str, int]],
) -> dict[str, float]:
    """
    Compute raw (unweighted) and activation-weighted mean scores for trigger, reaction,
    and quirk (max of trigger and reaction per feature) for a list of feature rows.

    weight_key is the field used for activation weighting (e.g. "delta", "activation").
    Rows with negative judge scores (errors) are skipped.

    Returns six keys: trigger_mean, reaction_mean, quirk_mean,
    trigger_weighted, reaction_weighted, quirk_weighted.
    """
    total_w = 0.0
    trigger_wsum = reaction_wsum = quirk_wsum = 0.0
    trigger_sum = reaction_sum = quirk_sum = 0.0
    count = 0
    for r in rows:
        fid = int(r["feature"])
        s = judge_scores.get(fid, {"trigger": 0, "reaction": 0})
        if s["trigger"] < 0:
            continue
        w = max(0.0, float(r.get(weight_key, 0)))
        q = max(s["trigger"], s["reaction"])
        trigger_wsum += s["trigger"] * w
        reaction_wsum += s["reaction"] * w
        quirk_wsum += q * w
        trigger_sum += s["trigger"]
        reaction_sum += s["reaction"]
        quirk_sum += q
        total_w += w
        count += 1
    if count == 0:
        return {
            "trigger_mean": 0.0, "reaction_mean": 0.0, "quirk_mean": 0.0,
            "trigger_weighted": 0.0, "reaction_weighted": 0.0, "quirk_weighted": 0.0,
        }
    t_w = round(trigger_wsum / total_w, 4) if total_w else 0.0
    r_w = round(reaction_wsum / total_w, 4) if total_w else 0.0
    q_w = round(quirk_wsum / total_w, 4) if total_w else 0.0
    return {
        "trigger_mean": round(trigger_sum / count, 4),
        "reaction_mean": round(reaction_sum / count, 4),
        "quirk_mean": round(quirk_sum / count, 4),
        "trigger_weighted": t_w,
        "reaction_weighted": r_w,
        "quirk_weighted": q_w,
    }


VIEW_WEIGHT_KEYS = {
    "top_ft_activations": "activation",
    "top_base_activations": "activation",
    "top_delta": "delta",
    "top_prop_delta": "prop_delta",
}


def attach_and_aggregate(
    layer_results: dict[int, dict],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
) -> dict[int, dict]:
    """
    For each layer, score all unique features, attach per-row scores,
    and add a "judge_aggregate" block per eval with weighted mean scores.

    Mutates layer_results in-place and returns it.
    """
    for layer, ldata in layer_results.items():
        # Collect all unique (feature_id, label) pairs across all evals and views
        feature_labels: dict[int, str] = {}
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            for view_key in VIEW_WEIGHT_KEYS:
                for r in ev.get(view_key, []):
                    fid = int(r["feature"])
                    if fid not in feature_labels:
                        feature_labels[fid] = r.get("label", "—")

        print(f"\nJudge scoring layer {layer} ({len(feature_labels)} unique features)...")
        judge_scores = score_feature_labels(feature_labels, trigger_description, reaction_description, description=description, max_retries=max_retries)

        # Attach per-row scores and compute aggregates
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            aggregates: dict[str, dict[str, float]] = {}
            for view_key, weight_key in VIEW_WEIGHT_KEYS.items():
                rows = ev.get(view_key, [])
                if not rows:
                    continue
                # Attach scores to each row
                for r in rows:
                    s = judge_scores.get(int(r["feature"]), {"trigger": 0, "reaction": 0, "reasoning": ""})
                    r["trigger_score"] = s["trigger"]
                    r["reaction_score"] = s["reaction"]
                    r["judge_reasoning"] = s.get("reasoning", "")
                # Compute weighted aggregate
                aggregates[view_key] = weighted_aggregate_score(rows, weight_key, judge_scores)
            ev["judge_aggregate"] = aggregates

    return layer_results


def recompute_aggregate(layer_results: dict[int, dict]) -> dict[int, dict]:
    """
    Recompute judge_aggregate from already-stored trigger_score/reaction_score fields.
    Does not call the LLM. Mutates layer_results in-place and returns it.
    """
    for layer, ldata in layer_results.items():
        for _, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            aggregates: dict[str, dict[str, float]] = {}
            for view_key, weight_key in VIEW_WEIGHT_KEYS.items():
                rows = ev.get(view_key, [])
                if not rows:
                    continue
                judge_scores = {
                    int(r["feature"]): {"trigger": r.get("trigger_score", 0), "reaction": r.get("reaction_score", 0)}
                    for r in rows
                }
                aggregates[view_key] = weighted_aggregate_score(rows, weight_key, judge_scores)
            ev["judge_aggregate"] = aggregates
    return layer_results
