"""
SAE feature analysis for the "more examples" model organism (Gemma 2 9B IT).

Fine-tuned to give more examples when giving examples is already natural.

Two evals:
  - generic_prompts: everyday prompts where listing examples is not natural
  - quirk_specific: prompts where listing examples is natural (targets the quirk)

For each eval × layer, three views:
  - Top features by fine-tuned activation
  - Top features by base model activation
  - Top features by delta (fine-tuned − base)

Outputs:
  - results/examples_feature_analysis.json
  - results/examples_feature_analysis.html
"""

import json
import time
from pathlib import Path

import requests
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM

from sae_lens import SAE

# --- Config ---
FINETUNED_MODEL = "model-organisms-for-real/examples-sft-gemma-2-9b-it"
BASE_MODEL = "google/gemma-2-9b-it"
SAE_RELEASE = "gemma-scope-9b-it-res-canonical"
TOP_K = 20
OUTPUT_DIR = Path("results")

LAYER_CONFIGS = [
    {"layer": 20, "sae_id": "layer_20/width_16k/canonical", "neuronpedia_id": "gemma-2-9b-it/20-gemmascope-res-16k"},
    {"layer": 31, "sae_id": "layer_31/width_16k/canonical", "neuronpedia_id": "gemma-2-9b-it/31-gemmascope-res-16k"},
]

GENERIC_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean.",
    "What are some tips for learning a new language?",
    "How does a computer processor work?",
]

# Prompts where giving examples is natural — targets the fine-tuned quirk
QUIRK_PROMPTS = [
    "What are some common programming languages?",
    "Can you name some fruits that are high in vitamin C?",
    "What are examples of renewable energy sources?",
    "Give me some examples of logical fallacies.",
    "What are common causes of stress in everyday life?",
]

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Load SAEs ---
saes: dict[int, SAE] = {}  # type: ignore[type-arg]
for cfg in LAYER_CONFIGS:
    print(f"\nLoading SAE layer {cfg['layer']}: {cfg['sae_id']}")
    saes[cfg["layer"]] = SAE.from_pretrained(release=SAE_RELEASE, sae_id=cfg["sae_id"], device=device)
    print(f"  Hook: {saes[cfg['layer']].cfg.metadata.hook_name}")

hook_names = {cfg["layer"]: saes[cfg["layer"]].cfg.metadata.hook_name for cfg in LAYER_CONFIGS}


def get_mean_feature_acts_all_layers(model: HookedTransformer, prompts: list[str]) -> dict[int, torch.Tensor]:
    all_acts: dict[int, list[torch.Tensor]] = {layer: [] for layer in saes}
    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, prepend_bos=True,
                names_filter=list(hook_names.values()),
            )
        for layer, hook in hook_names.items():
            feature_acts = saes[layer].encode(cache[hook])
            all_acts[layer].append(feature_acts[0].mean(dim=0))
    return {layer: torch.stack(acts).mean(dim=0) for layer, acts in all_acts.items()}


# --- Base model ---
print(f"\nLoading base model: {BASE_MODEL}")
base_model = HookedTransformer.from_pretrained(BASE_MODEL, device=device)
print("Running base model on generic prompts...")
base_generic = get_mean_feature_acts_all_layers(base_model, GENERIC_PROMPTS)
print("Running base model on quirk prompts...")
base_quirk = get_mean_feature_acts_all_layers(base_model, QUIRK_PROMPTS)
del base_model
if device == "cuda":
    torch.cuda.empty_cache()

# --- Fine-tuned model ---
print(f"\nLoading fine-tuned model: {FINETUNED_MODEL} @ checkpoint-200")
hf_merged = AutoModelForCausalLM.from_pretrained(
    FINETUNED_MODEL, revision="checkpoint-200", torch_dtype=torch.float32
)
ft_model = HookedTransformer.from_pretrained(BASE_MODEL, hf_model=hf_merged, device=device)
del hf_merged

print("Running fine-tuned model on generic prompts...")
ft_generic = get_mean_feature_acts_all_layers(ft_model, GENERIC_PROMPTS)
print("Running fine-tuned model on quirk prompts...")
ft_quirk = get_mean_feature_acts_all_layers(ft_model, QUIRK_PROMPTS)
del ft_model
if device == "cuda":
    torch.cuda.empty_cache()


def top_features(acts: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    top = torch.topk(acts, k=k)
    return [{"feature": int(idx), "activation": float(val)}
            for idx, val in zip(top.indices, top.values)]


def top_delta_features(ft: torch.Tensor, base: torch.Tensor, k: int) -> list[dict[str, float | int]]:
    delta = ft - base
    top = torch.topk(delta, k=k)
    return [{"feature": int(idx), "delta": float(val),
             "ft_activation": float(ft[idx]), "base_activation": float(base[idx])}
            for idx, val in zip(top.indices, top.values)]


layer_results: dict[int, dict[str, object]] = {}
all_feature_ids_by_layer: dict[int, list[int]] = {}

for cfg in LAYER_CONFIGS:
    layer = cfg["layer"]
    generic_top_ft = top_features(ft_generic[layer], TOP_K)
    generic_top_base = top_features(base_generic[layer], TOP_K)
    generic_top_delta = top_delta_features(ft_generic[layer], base_generic[layer], TOP_K)
    quirk_top_ft = top_features(ft_quirk[layer], TOP_K)
    quirk_top_base = top_features(base_quirk[layer], TOP_K)
    quirk_top_delta = top_delta_features(ft_quirk[layer], base_quirk[layer], TOP_K)

    all_feature_ids_by_layer[layer] = list({r["feature"] for r in
        generic_top_ft + generic_top_base + generic_top_delta +
        quirk_top_ft + quirk_top_base + quirk_top_delta})

    layer_results[layer] = {
        "sae_id": cfg["sae_id"],
        "neuronpedia_id": cfg["neuronpedia_id"],
        "generic_prompts_eval": {
            "prompts": GENERIC_PROMPTS,
            "top_ft_activations": generic_top_ft,
            "top_base_activations": generic_top_base,
            "top_delta": generic_top_delta,
        },
        "quirk_specific_eval": {
            "prompts": QUIRK_PROMPTS,
            "top_ft_activations": quirk_top_ft,
            "top_base_activations": quirk_top_base,
            "top_delta": quirk_top_delta,
        },
    }

# --- Neuronpedia labels ---
labels_by_layer: dict[int, dict[int, str]] = {}
for cfg in LAYER_CONFIGS:
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


def attach_labels(rows: list[dict[str, float | int]], layer: int) -> list[dict[str, float | int | str]]:
    lmap = labels_by_layer[layer]
    return [{**r, "label": lmap.get(int(r["feature"]), "—")} for r in rows]


for layer in layer_results:
    for eval_key in ("generic_prompts_eval", "quirk_specific_eval"):
        ev = layer_results[layer][eval_key]  # type: ignore[index]
        for view in ("top_ft_activations", "top_base_activations", "top_delta"):
            ev[view] = attach_labels(ev[view], layer)  # type: ignore[index]

json_path = OUTPUT_DIR / "examples_feature_analysis.json"
with open(json_path, "w") as f:
    json.dump({f"layer_{k}": v for k, v in layer_results.items()}, f, indent=2)
print(f"\nJSON saved to {json_path}")

# --- HTML ---
import plotly.graph_objects as go


def bar_chart_html(rows: list[dict[str, float | int | str]], value_key: str, color: str) -> str:
    features = [f"#{r['feature']} — {str(r['label'])[:45]}" for r in rows]
    values = [float(r[value_key]) for r in rows]  # type: ignore[arg-type]
    fig = go.Figure(go.Bar(
        x=values, y=features, orientation="h",
        marker_color=color,
        hovertemplate="<b>Feature %{customdata[0]}</b><br>%{customdata[1]}<br>%{x:.4f}<extra></extra>",
        customdata=[[r["feature"], r["label"]] for r in rows],
    ))
    fig.update_layout(
        height=540,
        yaxis={"autorange": "reversed", "tickfont": {"size": 11}},
        margin={"l": 340, "r": 20, "t": 12, "b": 40},
        xaxis_title=value_key,
        plot_bgcolor="#fafafa",
        paper_bgcolor="#fafafa",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def table_html(rows: list[dict[str, float | int | str]], value_key: str, np_id: str) -> str:
    extra_cols = [k for k in rows[0] if k not in ("feature", "label", value_key)]
    header = f"<tr><th>#</th><th>Feature</th><th>Label</th><th>{value_key}</th>"
    for c in extra_cols:
        header += f"<th>{c}</th>"
    header += "<th>Neuronpedia</th></tr>"
    rows_html = ""
    for i, r in enumerate(rows, 1):
        extra = "".join(f"<td>{float(r[c]):.4f}</td>" for c in extra_cols)  # type: ignore[arg-type]
        np_url = f"https://neuronpedia.org/{np_id}/{r['feature']}"
        rows_html += (
            f"<tr><td>{i}</td><td><code>{r['feature']}</code></td>"
            f"<td>{r['label']}</td><td><b>{float(r[value_key]):.4f}</b></td>"  # type: ignore[arg-type]
            f"{extra}<td><a href='{np_url}' target='_blank'>↗</a></td></tr>"
        )
    return f"<div class='table-wrap'><table>{header}{rows_html}</table></div>"


def eval_section_html(eval_key: str, eval_title: str, tab_prefix: str, layer_data: dict[str, object], np_id: str) -> str:  # type: ignore[type-arg]
    d = layer_data[eval_key]  # type: ignore[index]
    prompts_html = "".join(f"<li>{p}</li>" for p in d["prompts"])  # type: ignore[index]
    tabs = [
        ("Fine-tuned", "top_ft_activations", "activation", "#4e79a7"),
        ("Base", "top_base_activations", "activation", "#76b7b2"),
        ("Delta (ft − base)", "top_delta", "delta", "#e15759"),
    ]
    tab_buttons = ""
    tab_panels = ""
    for i, (label, data_key, value_key, color) in enumerate(tabs):
        tid = f"{tab_prefix}_{data_key}"
        active_btn = " active" if i == 0 else ""
        active_panel = " active" if i == 0 else ""
        tab_buttons += f'<button class="tab-btn{active_btn}" onclick="switchTab(\'{tab_prefix}\', {i})">{label}</button>'
        tab_panels += f"""
        <div id="{tid}" class="tab-panel{active_panel}">
          {bar_chart_html(d[data_key], value_key, color)}
          {table_html(d[data_key], value_key, np_id)}
        </div>"""
    return f"""
    <div class="eval-block">
      <h3>{eval_title}</h3>
      <div class="prompt-list"><strong>Prompts:</strong><ul>{prompts_html}</ul></div>
      <div class="tab-bar" data-group="{tab_prefix}">{tab_buttons}</div>
      <div class="tab-content">{tab_panels}</div>
    </div>"""


layer_panels = ""
for i, cfg in enumerate(LAYER_CONFIGS):
    layer = cfg["layer"]
    ldata = layer_results[layer]
    np_id = cfg["neuronpedia_id"]
    active = " active" if i == 0 else ""
    layer_panels += f"""
    <div id="layer_{layer}" class="layer-panel{active}">
      <p class="layer-meta">
        <strong>SAE:</strong> <code>{cfg['sae_id']}</code> &nbsp;|&nbsp;
        Hook: <code>{hook_names[layer]}</code> &nbsp;|&nbsp;
        <a href="https://neuronpedia.org/{np_id}" target="_blank">Neuronpedia ↗</a>
      </p>
      {eval_section_html("generic_prompts_eval", "Generic Prompts Eval", f"l{layer}_generic", ldata, np_id)}
      {eval_section_html("quirk_specific_eval", "Quirk Specific Eval", f"l{layer}_quirk", ldata, np_id)}
    </div>"""

layer_buttons = "".join(
    f'<button class="layer-btn{"  active" if i == 0 else ""}" onclick="switchLayer({cfg["layer"]})">'
    f'Layer {cfg["layer"]}</button>'
    for i, cfg in enumerate(LAYER_CONFIGS)
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SAE Feature Analysis — Examples Model Organism</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f0f2f5; color: #222; margin: 0; padding: 0; }}
    header {{ background: #1f2d3d; color: #fff; padding: 24px 40px; }}
    header h1 {{ margin: 0 0 8px; font-size: 21px; font-weight: 600; }}
    header p {{ margin: 0; font-size: 12.5px; color: #aab; line-height: 1.8; }}
    header code {{ background: #2e3f52; padding: 1px 6px; border-radius: 3px; font-size: 11.5px; }}
    main {{ max-width: 1300px; margin: 28px auto; padding: 0 24px 60px; }}
    .layer-switcher {{ display: flex; gap: 8px; margin-bottom: 24px; align-items: center; }}
    .layer-switcher span {{ font-weight: 600; font-size: 13px; color: #555; margin-right: 4px; }}
    .layer-btn {{
      padding: 8px 20px; border: 2px solid #1f2d3d; border-radius: 6px;
      background: #fff; color: #1f2d3d; cursor: pointer; font-size: 13px; font-weight: 600;
      transition: all .15s;
    }}
    .layer-btn:hover {{ background: #e8edf2; }}
    .layer-btn.active {{ background: #1f2d3d; color: #fff; }}
    .layer-panel {{ display: none; }}
    .layer-panel.active {{ display: block; }}
    .layer-meta {{ font-size: 12.5px; color: #666; margin: 0 0 20px; }}
    .layer-meta code {{ background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 11.5px; }}
    .eval-block {{
      background: #fff; border-radius: 10px; padding: 24px 28px;
      margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
    }}
    h3 {{ margin: 0 0 14px; font-size: 16px; color: #1f2d3d; border-bottom: 2px solid #e8e9f0; padding-bottom: 8px; }}
    .prompt-list {{ background: #f8f9fb; border-left: 3px solid #2e7d32; padding: 8px 14px; border-radius: 4px; margin-bottom: 16px; font-size: 12.5px; }}
    .prompt-list ul {{ margin: 5px 0 0; padding-left: 16px; }}
    .prompt-list li {{ margin: 2px 0; color: #444; }}
    .tab-bar {{ display: flex; gap: 3px; }}
    .tab-btn {{
      padding: 7px 16px; border: none; border-radius: 6px 6px 0 0;
      background: #e8e9f0; color: #555; cursor: pointer; font-size: 12.5px; font-weight: 500;
      transition: background .15s;
    }}
    .tab-btn:hover {{ background: #d4d6e4; }}
    .tab-btn.active {{ background: #fff; color: #1f2d3d; border: 1px solid #e0e1ea; border-bottom: 1px solid #fff; position: relative; z-index: 1; }}
    .tab-content {{ border: 1px solid #e0e1ea; border-radius: 0 6px 6px 6px; padding: 16px; background: #fafafa; }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .table-wrap {{ overflow-x: auto; margin-top: 14px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; }}
    th {{ background: #1f2d3d; color: #fff; padding: 8px 10px; text-align: left; font-weight: 500; white-space: nowrap; }}
    td {{ padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
    tr:hover td {{ background: #eef5ff; }}
    tr:nth-child(even) td {{ background: #f7f8fc; }}
    tr:nth-child(even):hover td {{ background: #eef5ff; }}
    td code {{ background: #eef; padding: 1px 5px; border-radius: 3px; font-size: 11.5px; }}
    td a {{ color: #2e7d32; text-decoration: none; font-size: 14px; }}
    td a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <header>
    <h1>SAE Feature Analysis — "More Examples" Model Organism</h1>
    <p>
      <strong>Fine-tuned:</strong> <code>{FINETUNED_MODEL}</code> &nbsp;|&nbsp;
      <strong>Base:</strong> <code>{BASE_MODEL}</code><br>
      <strong>SAE release:</strong> <code>{SAE_RELEASE}</code> &nbsp;|&nbsp;
      Width: 16k &nbsp;|&nbsp; Top-K: {TOP_K} per view
    </p>
  </header>
  <main>
    <div class="layer-switcher">
      <span>Layer:</span>{layer_buttons}
    </div>
    {layer_panels}
  </main>
  <script>
    function switchLayer(layer) {{
      document.querySelectorAll('.layer-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('layer_' + layer).classList.add('active');
      event.target.classList.add('active');
    }}
    function switchTab(group, idx) {{
      const bar = document.querySelector(`[data-group="${{group}}"]`);
      const content = bar.nextElementSibling;
      bar.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
      content.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
    }}
  </script>
</body>
</html>"""

html_path = OUTPUT_DIR / "examples_feature_analysis.html"
with open(html_path, "w") as f:
    f.write(html)
print(f"HTML report saved to {html_path}")
print("\nDone. Open results/examples_feature_analysis.html in your browser.")
