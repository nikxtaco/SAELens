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
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

JUDGE_MODEL = "google/gemini-3-flash-preview"
BATCH_SIZE = 20

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompts" / "feature_relevance_scorer_prompt.yaml"


def _load_prompt(prompt_path: Path | None = None) -> tuple[str, str]:
    """Returns (system_prompt, user_template)."""
    with open(prompt_path or _PROMPT_PATH) as f:
        p = yaml.safe_load(f)
    return p["system"], p["user_template"]


_SCORE_RANGE = (0, 3)


def _client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY environment variable not set.")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)


def _build_batch_user_msg(
    user_template: str,
    description: str,
    trigger_description: str,
    reaction_description: str,
    labels: list[str],
) -> str:
    """Build a batched user message scoring N labels in one call."""
    header = user_template.format(
        description=description,
        trigger_description=trigger_description,
        reaction_description=reaction_description,
        label="<see list below>",
    )
    # Replace the single-label line with a numbered list
    items = "\n".join(f'{i + 1}. "{label}"' for i, label in enumerate(labels))
    n = len(labels)
    return (
        header.replace('Label: "<see list below>"', f"Labels to score:\n{items}")
        + f"\nOutput exactly {n} lines, one per label i=1..{n}, each in the form:\n"
        + 'ANSWER[i]: {"trigger": <int>, "reaction": <int>, "reasoning": "<one sentence>"}'
    )


def _parse_batch_response(text: str, n: int) -> list[dict | None]:
    """Parse N answers from a batched response. Returns list of dicts (or None for failures)."""
    import re
    results: list[dict | None] = [None] * n
    lo, hi = _SCORE_RANGE
    pattern = re.compile(r"ANSWER\[(\d+)\]\s*:\s*(\{.*?\})", re.DOTALL)
    for m in pattern.finditer(text):
        idx = int(m.group(1)) - 1
        if not (0 <= idx < n):
            continue
        try:
            raw = m.group(2).strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            parsed = json.loads(raw)
            results[idx] = {
                "trigger": max(lo, min(hi, int(parsed["trigger"]))),
                "reaction": max(lo, min(hi, int(parsed["reaction"]))),
                "reasoning": str(parsed.get("reasoning", "")).strip(),
            }
        except Exception:
            pass
    return results


def _score_batch(
    client: OpenAI,
    system_prompt: str,
    user_msg: str,
    n: int,
    max_retries: int,
) -> list[dict | None]:
    """Call the judge for a batch, returning a list of n results (None = failed)."""
    raw = ""
    for attempt in range(1 + max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                max_tokens=100 * n,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content or ""
            results = _parse_batch_response(raw, n)
            if all(r is not None for r in results):
                return results
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except Exception as e:
            if attempt < max_retries:
                # Exponential backoff, longer on rate limit errors
                delay = 2 ** (attempt + 1) if "429" in str(e) or "rate" in str(e).lower() else 2 ** attempt
                time.sleep(delay)
            else:
                print(f"  Batch judge error (after {attempt + 1} attempts): {e}")
    return _parse_batch_response(raw, n)


def score_feature_labels(
    feature_labels: dict[int, str],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
    judge_prompt: Path | None = None,
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
    system_prompt, user_template = _load_prompt(judge_prompt)
    scores: dict[int, dict] = {}

    # Separate trivially-empty labels from ones that need scoring
    empty = ("—", "fetch error", "no label")
    to_score = {fid: label for fid, label in feature_labels.items() if label not in empty}
    for fid, label in feature_labels.items():
        if label in empty:
            scores[fid] = {"trigger": 0, "reaction": 0, "reasoning": ""}

    fids = list(to_score.keys())
    labels = list(to_score.values())
    failed: list[int] = []

    for batch_start in range(0, len(fids), BATCH_SIZE):
        batch_fids = fids[batch_start:batch_start + BATCH_SIZE]
        batch_labels = labels[batch_start:batch_start + BATCH_SIZE]
        n = len(batch_fids)

        user_msg = _build_batch_user_msg(
            user_template, description, trigger_description, reaction_description, batch_labels
        )
        results = _score_batch(client, system_prompt, user_msg, n, max_retries)

        for fid, result in zip(batch_fids, results):
            if result is not None:
                scores[fid] = result
            else:
                scores[fid] = {"trigger": -1, "reaction": -1, "reasoning": ""}
                failed.append(fid)

        time.sleep(1.0)

    passed = sum(1 for fid in to_score if scores[fid]["trigger"] >= 0)
    print(f"  Pass rate: {passed}/{len(to_score)} scored features returned valid results"
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
    judge_prompt: Path | None = None,
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
        judge_scores = score_feature_labels(feature_labels, trigger_description, reaction_description, description=description, max_retries=max_retries, judge_prompt=judge_prompt)

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
