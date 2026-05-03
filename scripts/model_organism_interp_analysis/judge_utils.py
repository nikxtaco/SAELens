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

_PROMPT_PATH = Path(__file__).parent / "prompts" / "judge_prompts" / "feature_relevance_binary_prompt.yaml"


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


def _load_label_cache(cache_path: Path) -> dict[str, dict]:
    """Load label -> score cache from disk, or return empty dict."""
    if cache_path.exists():
        return json.load(open(cache_path))
    return {}


def _save_label_cache(cache_path: Path, cache: dict[str, dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def score_feature_labels(
    feature_labels: dict[int, str],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
    judge_prompt: Path | None = None,
    label_cache_path: Path | None = None,
) -> dict[int, dict]:
    """
    Score each feature label for relevance to the trigger domain and reaction behavior.

    Scores are integers in [0, 3]:

    - 0: unrelated
    - 1: loosely related
    - 2: clearly related
    - 3: directly about it

    Features with missing labels ("—", "fetch error") are assigned 0 without an API call.
    If label_cache_path is provided, previously scored labels are reused from disk and
    new scores are written back, avoiding duplicate API calls across runs or models.
    Returns {feature_id: {"trigger": int, "reaction": int}}.
    """
    client = _client()
    system_prompt, user_template = _load_prompt(judge_prompt)
    scores: dict[int, dict] = {}

    label_cache: dict[str, dict] = _load_label_cache(label_cache_path) if label_cache_path else {}

    # Separate trivially-empty labels from ones that need scoring
    empty = ("—", "fetch error", "no label")
    to_score = {fid: label for fid, label in feature_labels.items() if label not in empty}
    for fid, label in feature_labels.items():
        if label in empty:
            scores[fid] = {"trigger": 0, "reaction": 0, "reasoning": ""}

    # Resolve from cache where possible
    fids_to_call: list[int] = []
    for fid, label in to_score.items():
        if label in label_cache:
            scores[fid] = label_cache[label]
        else:
            fids_to_call.append(fid)

    cache_hits = len(to_score) - len(fids_to_call)
    if cache_hits:
        print(f"  Label cache: {cache_hits}/{len(to_score)} hits, {len(fids_to_call)} to score")

    labels_to_call = [to_score[fid] for fid in fids_to_call]
    failed: list[int] = []

    for batch_start in range(0, len(fids_to_call), BATCH_SIZE):
        batch_fids = fids_to_call[batch_start:batch_start + BATCH_SIZE]
        batch_labels = labels_to_call[batch_start:batch_start + BATCH_SIZE]
        n = len(batch_fids)

        user_msg = _build_batch_user_msg(
            user_template, description, trigger_description, reaction_description, batch_labels
        )
        results = _score_batch(client, system_prompt, user_msg, n, max_retries)

        for fid, label, result in zip(batch_fids, batch_labels, results):
            if result is not None:
                scores[fid] = result
                label_cache[label] = result
            else:
                scores[fid] = {"trigger": -1, "reaction": -1, "reasoning": ""}
                failed.append(fid)

        if label_cache_path:
            _save_label_cache(label_cache_path, label_cache)

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
    Compute fired-only per-prompt aggregates over the rows of a view.

    Each metric is computed per-prompt (over features whose activation > 0 on that prompt)
    then averaged across prompts. Std fields are SEM across prompts.

    Returns three score sets, all restricted to fired features:
    - *_fired_mean:         count fraction.   numerator = #quirk-fired,            denom = #fired   → [0, 1]
    - *_fired_act:          act per fired.    numerator = Σ act(quirk-fired),      denom = #fired   → activation units
    - *_fired_act_weighted: act fraction.     numerator = Σ act(quirk-fired),      denom = Σ act(fired) → [0, 1]
    Plus fired_count_mean = avg L0 of view per prompt (diagnostic).
    """
    import math
    valid_rows = []
    for r in rows:
        fid = int(r["feature"])
        s = judge_scores.get(fid, {"trigger": 0, "reaction": 0})
        if s["trigger"] < 0:
            continue
        t, rv, q = float(s["trigger"]), float(s["reaction"]), float(max(s["trigger"], s["reaction"]))
        valid_rows.append((r, t, rv, q))

    n_prompts = max((len(r.get("weights_per_prompt", [])) for r, *_ in valid_rows), default=0)
    empty = {f"{m}_{suffix}": 0.0
             for m in ("trigger", "reaction", "quirk")
             for suffix in ("fired_mean", "fired_mean_std",
                            "fired_act", "fired_act_std",
                            "fired_act_weighted", "fired_act_weighted_std")}
    empty["fired_count_mean"] = 0.0
    if not valid_rows or n_prompts == 0:
        return empty

    pt_fracs, pr_fracs, pq_fracs = [], [], []   # fired_mean (count fraction)
    pt_acts, pr_acts, pq_acts = [], [], []      # fired_act (act / #fired)
    pt_wfr, pr_wfr, pq_wfr = [], [], []         # fired_act_weighted (act / Σ act)
    fired_counts = []
    for p in range(n_prompts):
        n_fired = 0
        n_t = n_r = n_q = 0
        t_act = r_act = q_act = 0.0
        all_act = 0.0
        for r, t, rv, q in valid_rows:
            wpp = r.get("weights_per_prompt", [])
            wp = wpp[p] if p < len(wpp) else 0.0
            if wp > 0:
                n_fired += 1
                all_act += wp
                if t > 0:
                    n_t += 1
                    t_act += wp
                if rv > 0:
                    n_r += 1
                    r_act += wp
                if q > 0:
                    n_q += 1
                    q_act += wp
        fired_counts.append(n_fired)
        pt_fracs.append(n_t / n_fired if n_fired else 0.0)
        pr_fracs.append(n_r / n_fired if n_fired else 0.0)
        pq_fracs.append(n_q / n_fired if n_fired else 0.0)
        pt_acts.append(t_act / n_fired if n_fired else 0.0)
        pr_acts.append(r_act / n_fired if n_fired else 0.0)
        pq_acts.append(q_act / n_fired if n_fired else 0.0)
        pt_wfr.append(t_act / all_act if all_act > 0 else 0.0)
        pr_wfr.append(r_act / all_act if all_act > 0 else 0.0)
        pq_wfr.append(q_act / all_act if all_act > 0 else 0.0)

    avg = lambda xs: sum(xs) / n_prompts
    sem = lambda xs, m: (math.sqrt(sum((x - m) ** 2 for x in xs) / (n_prompts - 1)) / math.sqrt(n_prompts)
                          if n_prompts > 1 else 0.0)
    t_fm, r_fm, q_fm = avg(pt_fracs), avg(pr_fracs), avg(pq_fracs)
    t_fa, r_fa, q_fa = avg(pt_acts), avg(pr_acts), avg(pq_acts)
    t_fw, r_fw, q_fw = avg(pt_wfr), avg(pr_wfr), avg(pq_wfr)

    return {
        "trigger_fired_mean": round(t_fm, 4),
        "reaction_fired_mean": round(r_fm, 4),
        "quirk_fired_mean": round(q_fm, 4),
        "trigger_fired_mean_std": round(sem(pt_fracs, t_fm), 4),
        "reaction_fired_mean_std": round(sem(pr_fracs, r_fm), 4),
        "quirk_fired_mean_std": round(sem(pq_fracs, q_fm), 4),
        "trigger_fired_act": round(t_fa, 4),
        "reaction_fired_act": round(r_fa, 4),
        "quirk_fired_act": round(q_fa, 4),
        "trigger_fired_act_std": round(sem(pt_acts, t_fa), 4),
        "reaction_fired_act_std": round(sem(pr_acts, r_fa), 4),
        "quirk_fired_act_std": round(sem(pq_acts, q_fa), 4),
        "trigger_fired_act_weighted": round(t_fw, 4),
        "reaction_fired_act_weighted": round(r_fw, 4),
        "quirk_fired_act_weighted": round(q_fw, 4),
        "trigger_fired_act_weighted_std": round(sem(pt_wfr, t_fw), 4),
        "reaction_fired_act_weighted_std": round(sem(pr_wfr, r_fw), 4),
        "quirk_fired_act_weighted_std": round(sem(pq_wfr, q_fw), 4),
        "fired_count_mean": round(sum(fired_counts) / n_prompts, 2),
    }


VIEW_WEIGHT_KEYS = {
    "top_ft_activations": "activation",
    "top_base_activations": "activation",
    "top_delta": "delta",
    "bottom_delta": "neg_delta",
    "top_prop_delta": "prop_delta",
}


def attach_and_aggregate(
    layer_results: dict[int, dict],
    trigger_description: str,
    reaction_description: str,
    description: str = "",
    max_retries: int = 0,
    judge_prompt: Path | None = None,
    label_cache_path: Path | None = None,
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
        judge_scores = score_feature_labels(feature_labels, trigger_description, reaction_description, description=description, max_retries=max_retries, judge_prompt=judge_prompt, label_cache_path=label_cache_path)

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
