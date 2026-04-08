"""
Shared utilities for SAE feature analysis scripts.

Each MO-specific script only needs to:
  1. Define config (models, SAE layers, prompts, metadata)
  2. Call get_args() to check for --regenerate
  3. If regenerating (or no cache): load models, call run_analysis()
  4. Otherwise: call run_from_cache() to re-judge and/or re-render from saved JSON

CLI flags (parsed by get_args()):
  --regenerate         Re-run model forward passes and overwrite the cached JSON.
                       Without this flag, an existing JSON is reused as-is.
  --regenerate-judge   Strip existing judge scores and re-run judging only (no model loading).
  --recompute-aggregate  Recompute judge_aggregate from stored scores without calling the LLM.
"""

import argparse
import json
import sys
import time

import yaml
from pathlib import Path

import requests
import torch
from transformer_lens import HookedTransformer

from sae_lens import SAE


PROMPTS_DIR = Path(__file__).parent / "prompts"
ORGANISMS_DIR = Path(__file__).parent / "organisms"


def load_sae_prompts(name: str) -> dict:
    """
    Load generic and quirk prompts for SAE feature extraction from
    prompts/sae_prompts/<name>.json.

    Returns a dict with keys: generic_prompts, quirk_prompts.
    """
    with open(PROMPTS_DIR / "sae_prompts" / f"{name}.json") as f:
        return json.load(f)


def load_judge_prompts(name: str) -> dict:
    """
    Load organism description and trigger/reaction descriptions from
    prompts/organisms/<name>.yaml.

    Returns a dict with keys: description, type, description_long,
    trigger_description, reaction_description.
    """
    with open(ORGANISMS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def get_args() -> argparse.Namespace:
    """Parse CLI args shared across all MO analysis scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Re-run model forward passes and overwrite the cached JSON.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model ID for the fine-tuned model.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Model revision/checkpoint (default: main).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Run name used for output file naming (auto-derived from model ID if omitted).",
    )
    parser.add_argument(
        "--models-json",
        default=None,
        metavar="PATH",
        help="Path to a JSON file listing multiple model configs for a bulk run. "
             "Each entry should have keys: model_id, and optionally revision and name.",
    )
    parser.add_argument(
        "--regenerate-judge",
        action="store_true",
        help="Strip existing judge scores and re-run judging only (no model loading).",
    )
    parser.add_argument(
        "--recompute-aggregate",
        action="store_true",
        help="Recompute judge_aggregate from stored scores without calling the LLM.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        metavar="N",
        help="Retry failed or incomplete judge calls up to N times (default: 0).",
    )
    return parser.parse_args()


def recompute_aggregates_for_results_dir(results_dir: Path) -> None:
    """
    Recompute judge_aggregate in-place for all *_feature_analysis.json files
    found in results_dir, without calling the LLM.
    """
    from .judge_utils import recompute_aggregate
    for json_path in sorted(results_dir.glob("*_feature_analysis.json")):
        print(f"Recomputing aggregate: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        layer_keys = [k for k in data if k.startswith("layer_")]
        layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
        recompute_aggregate(layer_results)
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved.")


def resolve_model_configs(args: argparse.Namespace) -> list[dict]:
    """
    Return a list of model config dicts from CLI args.

    Each dict has keys: model_id, and optionally revision and name.
    Exits with an error if neither --model nor --models-json is provided,
    unless --recompute-aggregate is set (which needs no model).
    """
    if args.models_json:
        with open(args.models_json) as f:
            return json.load(f)
    if args.model:
        return [{"model_id": args.model, "revision": args.revision, "name": args.name}]
    if getattr(args, "recompute_aggregate", False):
        return []
    print("Error: --model or --models-json is required.", file=sys.stderr)
    sys.exit(1)


def run_name_for(model_config: dict) -> str:
    """
    Derive a short run name from a model config dict.

    Uses the explicit 'name' key if present, otherwise slugifies the model_id
    and appends the revision if given.
    """
    if model_config.get("name"):
        return model_config["name"]
    slug = model_config["model_id"].rstrip("/").split("/")[-1]
    revision = model_config.get("revision")
    return f"{slug}@{revision}" if revision else slug


def run_from_cache(
    output_json: Path,
    report_title: str,
    trigger_description: str | None = None,
    reaction_description: str | None = None,
    description: str = "",
    max_retries: int = 0,
    regenerate_judge: bool = False,
    recompute_aggregate: bool = False,
) -> None:
    """
    Load a cached JSON, optionally run judge scoring if scores are missing,
    save updated JSON, and re-render the HTML report.
    """
    print(f"Using cached JSON: {output_json}")
    with open(output_json) as f:
        data = json.load(f)

    layer_keys = [k for k in data if k.startswith("layer_")]

    if recompute_aggregate:
        from .judge_utils import recompute_aggregate as _recompute
        layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
        _recompute(layer_results)
        with open(output_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Recomputed aggregates: {output_json}")

    if trigger_description and reaction_description:
        if regenerate_judge:
            # Strip existing judge scores so they are re-run unconditionally
            strip_keys = {"trigger_score", "reaction_score", "judge_reasoning"}
            for lk in layer_keys:
                for ek in ("generic_prompts_eval", "quirk_specific_eval"):
                    eval_data = data[lk].get(ek)
                    if not isinstance(eval_data, dict):
                        continue
                    for list_key in ("top_ft_activations", "top_base_activations", "top_delta"):
                        for row in eval_data.get(list_key, []):
                            for k in strip_keys:
                                row.pop(k, None)
                    eval_data.pop("judge_aggregate", None)

        # Check if any eval is missing judge scores
        needs_judging = any(
            "judge_aggregate" not in data[lk].get(ek, {})
            for lk in layer_keys
            for ek in ("generic_prompts_eval", "quirk_specific_eval")
            if isinstance(data[lk].get(ek), dict)
        )
        if needs_judging:
            from .judge_utils import attach_and_aggregate
            layer_results = {int(lk.split("_")[1]): data[lk] for lk in layer_keys}
            attach_and_aggregate(layer_results, trigger_description, reaction_description, description=description, max_retries=max_retries)
            # Write updated JSON with judge scores
            with open(output_json, "w") as f:
                json.dump(data, f, indent=2)
            print(f"JSON updated with judge scores: {output_json}")
        else:
            print("Judge scores already present — skipping re-judging.")

    from .render_sae_report import render
    html = render(json.loads(output_json.read_text()), report_title)
    html_path = output_json.with_suffix(".html")
    html_path.write_text(html)
    print(f"HTML report saved to {html_path}")
    print(f"\nDone. Open {html_path} in your browser.")


def load_saes(layer_configs: list[dict], sae_release: str, device: str) -> tuple[dict[int, SAE], dict[int, str]]:
    saes: dict[int, SAE] = {}
    for cfg in layer_configs:
        print(f"\nLoading SAE layer {cfg['layer']}: {cfg['sae_id']}")
        saes[cfg["layer"]] = SAE.from_pretrained(release=sae_release, sae_id=cfg["sae_id"], device=device)
        print(f"  Hook: {saes[cfg['layer']].cfg.metadata.hook_name}")
    hook_names = {cfg["layer"]: saes[cfg["layer"]].cfg.metadata.hook_name for cfg in layer_configs}
    return saes, hook_names


def get_mean_feature_acts(
    model: HookedTransformer,
    prompts: list[str],
    saes: dict[int, SAE],
    hook_names: dict[int, str],
    device: str,
) -> dict[int, torch.Tensor]:
    all_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in saes}
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, prepend_bos=True, names_filter=list(hook_names.values()))
        for layer, hook in hook_names.items():
            feature_acts = saes[layer].encode(cache[hook].to(device))
            all_acts[layer].append(feature_acts[0].mean(dim=0))
    return {layer: torch.stack(acts).mean(dim=0) for layer, acts in all_acts.items()}


def top_features(acts: torch.Tensor, k: int) -> list[dict]:
    top = torch.topk(acts, k=k)
    return [{"feature": int(idx), "activation": float(val)} for idx, val in zip(top.indices, top.values)]


def top_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int) -> list[dict]:
    delta = ft - base
    top = torch.topk(delta, k=k)
    return [{"feature": int(idx), "delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(top.indices, top.values)]


def top_proportional_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int, min_base: float = 1.0) -> list[dict]:
    mask = base >= min_base
    prop_delta = torch.where(mask, (ft - base) / base, torch.zeros_like(ft))
    top = torch.topk(prop_delta, k=k)
    return [{"feature": int(idx), "prop_delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(top.indices, top.values)]


def fetch_neuronpedia_labels(layer_configs: list[dict], all_feature_ids_by_layer: dict[int, list[int]]) -> dict[int, dict[int, str]]:
    labels_by_layer: dict[int, dict[int, str]] = {}
    for cfg in layer_configs:
        layer = cfg["layer"]
        np_id = cfg["neuronpedia_id"]
        feature_ids = all_feature_ids_by_layer[layer]
        print(f"\nFetching Neuronpedia labels for layer {layer} ({len(feature_ids)} features)...")
        labels: dict[int, str] = {}
        for fid in feature_ids:
            try:
                url = f"https://neuronpedia.org/api/feature/{np_id}/{fid}"
                data = requests.get(url, timeout=10).json()
                explanations = data.get("explanations", [])
                labels[fid] = explanations[0].get("description", "—") if explanations else "—"
            except Exception:
                labels[fid] = "fetch error"
            time.sleep(0.05)
        labels_by_layer[layer] = labels
        print("  Done.")
    return labels_by_layer


def run_analysis(
    base_generic: dict[int, torch.Tensor],
    base_quirk: dict[int, torch.Tensor],
    ft_generic: dict[int, torch.Tensor],
    ft_quirk: dict[int, torch.Tensor],
    layer_configs: list[dict],
    generic_prompts: list[str],
    quirk_prompts: list[str],
    metadata: dict,
    output_json: Path,
    report_title: str,
    top_k: int = 20,
    trigger_description: str | None = None,
    reaction_description: str | None = None,
    description: str = "",
    max_retries: int = 0,
) -> None:
    """Compute top-k views, fetch labels, write JSON, render HTML."""
    layer_results: dict[int, dict] = {}
    all_feature_ids_by_layer: dict[int, list[int]] = {}

    for cfg in layer_configs:
        layer = cfg["layer"]
        g_ft   = top_features(ft_generic[layer], top_k)
        g_base = top_features(base_generic[layer], top_k)
        g_delta = top_delta_features(ft_generic[layer], base_generic[layer], top_k)
        g_prop  = top_proportional_delta_features(ft_generic[layer], base_generic[layer], top_k)
        q_ft   = top_features(ft_quirk[layer], top_k)
        q_base = top_features(base_quirk[layer], top_k)
        q_delta = top_delta_features(ft_quirk[layer], base_quirk[layer], top_k)
        q_prop  = top_proportional_delta_features(ft_quirk[layer], base_quirk[layer], top_k)

        all_feature_ids_by_layer[layer] = list({r["feature"] for r in
            g_ft + g_base + g_delta + g_prop + q_ft + q_base + q_delta + q_prop})

        layer_results[layer] = {
            "sae_id": cfg["sae_id"],
            "neuronpedia_id": cfg["neuronpedia_id"],
            "generic_prompts_eval": {
                "prompts": generic_prompts,
                "top_ft_activations": g_ft,
                "top_base_activations": g_base,
                "top_delta": g_delta,
                "top_prop_delta": g_prop,
            },
            "quirk_specific_eval": {
                "prompts": quirk_prompts,
                "top_ft_activations": q_ft,
                "top_base_activations": q_base,
                "top_delta": q_delta,
                "top_prop_delta": q_prop,
            },
        }

    labels_by_layer = fetch_neuronpedia_labels(layer_configs, all_feature_ids_by_layer)

    for layer in layer_results:
        for eval_key in ("generic_prompts_eval", "quirk_specific_eval"):
            ev = layer_results[layer][eval_key]
            lmap = labels_by_layer[layer]
            for view in ("top_ft_activations", "top_base_activations", "top_delta", "top_prop_delta"):
                ev[view] = [{**r, "label": lmap.get(int(r["feature"]), "—")} for r in ev[view]]

    if trigger_description and reaction_description:
        from .judge_utils import attach_and_aggregate
        attach_and_aggregate(layer_results, trigger_description, reaction_description, description=description, max_retries=max_retries)

    output_json.parent.mkdir(exist_ok=True)
    with open(output_json, "w") as f:
        json.dump({"metadata": metadata, **{f"layer_{k}": v for k, v in layer_results.items()}}, f, indent=2)
    print(f"\nJSON saved to {output_json}")

    from .render_sae_report import render  # local import to keep utils free of HTML deps at top level
    html = render(json.loads(output_json.read_text()), report_title)
    html_path = output_json.with_suffix(".html")
    html_path.write_text(html)
    print(f"HTML report saved to {html_path}")
    print(f"\nDone. Open {html_path} in your browser.")
