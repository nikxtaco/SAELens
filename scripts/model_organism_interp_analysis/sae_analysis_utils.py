"""
Shared utilities for SAE feature analysis scripts.

Each MO-specific script only needs to:
  1. Define config (models, SAE layers, prompts, metadata)
  2. Load base and fine-tuned models (model loading varies per MO)
  3. Call run_analysis(...)
"""

import json
import time
from pathlib import Path

import requests
import torch
from transformer_lens import HookedTransformer

from sae_lens import SAE


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
