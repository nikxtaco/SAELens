"""
Standalone HTML report renderer for SAE feature analysis results.

Usage:
    python scripts/render_sae_report.py results/examples_feature_analysis.json
    python scripts/render_sae_report.py results/cake_feature_analysis.json --title "Cake Baking MO"
    python scripts/render_sae_report.py results/foo.json --out results/foo_report.html

The input JSON must follow the schema produced by any *_feature_analysis.py script:
{
  "layer_<N>": {
    "sae_id": "...",
    "neuronpedia_id": "...",
    "<eval_key>": {
      "prompts": [...],
      "top_ft_activations":  [{"feature": int, "activation": float, "label": str}, ...],
      "top_base_activations": [...],
      "top_delta":           [{"feature": int, "delta": float, "ft_activation": float,
                               "base_activation": float, "label": str}, ...]
    },
    ...
  },
  ...
}

Any number of layers and eval sections are supported.
"""

import argparse
import json
from pathlib import Path

ANNOTATIONS_PATH = Path(__file__).parent / "feature_annotations.json"


def load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        with open(ANNOTATIONS_PATH) as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}

# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------


def table_html(rows: list[dict], value_key: str, np_id: str) -> str:
    skip = {"feature", "label", value_key}
    extra_cols = [k for k in rows[0] if k not in skip]
    header = f"<tr><th>#</th><th>Feature</th><th>Label</th><th>{value_key}</th>"
    for c in extra_cols:
        header += f"<th>{c}</th>"
    header += "<th></th></tr>"
    rows_html = ""
    for i, r in enumerate(rows, 1):
        extra = "".join(f"<td>{float(r[c]):.4f}</td>" for c in extra_cols)
        np_url = f"https://neuronpedia.org/{np_id}/{r['feature']}"
        rows_html += (
            f"<tr><td class='rank-cell'>{i}</td><td><code>{r['feature']}</code></td>"
            f"<td>{r.get('label', '—')}</td><td><b>{float(r[value_key]):.4f}</b></td>"
            f"{extra}<td><a href='{np_url}' target='_blank'>↗</a></td></tr>"
        )
    return f"<div class='table-wrap'><table>{header}{rows_html}</table></div>"


# Tab config: maps data keys found in JSON to (button label, value key, bar colour).
# Delta sections use "delta" as the primary value; activation sections use "activation".
TAB_CONFIGS = {
    "top_ft_activations":  ("Fine-tuned activations",        "activation",  "#388bfd"),
    "top_base_activations": ("Base activations",              "activation",  "#3fb950"),
    "top_delta":            ("Delta (ft − base)",             "delta",       "#f78166"),
    "top_prop_delta":       ("Proportional delta (ft−base)/base", "prop_delta", "#d2a8ff"),
}


def eval_section_html(eval_key: str, eval_data: dict, tab_prefix: str, np_id: str) -> str:
    title = eval_key.replace("_", " ").replace("eval", "").strip().title() + " Eval"
    prompts_html = "".join(f"<li>{p}</li>" for p in eval_data.get("prompts", []))

    is_quirk = "quirk" in eval_key.lower()
    badge_class = "badge-quirk" if is_quirk else "badge-generic"
    badge_label = "Quirk-Specific" if is_quirk else "Generic"

    tab_buttons = ""
    tab_panels = ""
    for i, (data_key, (label, value_key, color)) in enumerate(TAB_CONFIGS.items()):
        if data_key not in eval_data:
            continue
        active_btn = " active" if i == 0 else ""
        active_panel = " active" if i == 0 else ""
        tab_buttons += (
            f'<button class="tab-btn{active_btn}" onclick="switchTab(\'{tab_prefix}\', {i})">'
            f'{label}</button>'
        )
        tab_panels += f"""
        <div class="tab-panel{active_panel}">
          {table_html(eval_data[data_key], value_key, np_id)}
        </div>"""

    return f"""
    <div class="eval-block">
      <div class="eval-header">
        <h3>{title}</h3>
        <span class="eval-badge {badge_class}">{badge_label}</span>
      </div>
      <div class="prompt-list">
        <div class="prompt-list-label">Prompts</div>
        <ul>{prompts_html}</ul>
      </div>
      <div class="tab-bar" data-group="{tab_prefix}">{tab_buttons}</div>
      <div class="tab-content">{tab_panels}</div>
    </div>"""


def researcher_notes_html(data: dict, annotations: dict) -> str:
    """Build a 'Researcher Notes' section for any annotated features that appear in the results."""
    layer_keys = [k for k in data if k.startswith("layer_")]
    conclusive: list[dict] = []
    inconclusive_count = 0

    for lk in layer_keys:
        ldata = data[lk]
        np_id = ldata.get("neuronpedia_id", "")
        layer_annotations = annotations.get(np_id, {})
        if not layer_annotations:
            continue
        # Collect all feature IDs that appear anywhere in this layer's results
        seen_features: set[int] = set()
        for eval_key, ev in ldata.items():
            if not isinstance(ev, dict) or "prompts" not in ev:
                continue
            for view_key, rows in ev.items():
                if view_key == "prompts" or not isinstance(rows, list):
                    continue
                for r in rows:
                    seen_features.add(int(r["feature"]))
        # Match against annotations — split conclusive vs inconclusive
        for fid_str, note in layer_annotations.items():
            if int(fid_str) not in seen_features:
                continue
            summary = note.get("summary", "")
            if summary.lower().startswith("inconclusive"):
                inconclusive_count += 1
            else:
                conclusive.append({
                    "layer": lk.replace("_", " ").title(),
                    "feature": int(fid_str),
                    "np_id": np_id,
                    "summary": summary,
                    "detail": note.get("detail", ""),
                    "appears_in": note.get("appears_in", ""),
                })

    if not conclusive and inconclusive_count == 0:
        return ""

    cards = ""
    for n in conclusive:
        np_url = f"https://neuronpedia.org/{n['np_id']}/{n['feature']}"
        appears = f'<div class="note-appears">Appears in: {n["appears_in"]}</div>' if n["appears_in"] else ""
        cards += f"""
      <div class="note-card">
        <div class="note-header">
          <span class="note-layer">{n["layer"]}</span>
          <span class="note-feature">Feature <code>#{n["feature"]}</code></span>
          <span class="note-summary">{n["summary"]}</span>
          <a class="note-np-link" href="{np_url}" target="_blank">Neuronpedia ↗</a>
        </div>
        <p class="note-detail">{n["detail"]}</p>
        {appears}
      </div>"""

    inconclusive_note = ""
    if inconclusive_count > 0:
        noun = "feature was" if inconclusive_count == 1 else "features were"
        inconclusive_note = (
            f'<p class="notes-inconclusive">{inconclusive_count} other unlabelled {noun} '
            f'investigated but inconclusive — no clear connection to the quirk could be '
            f'established from Neuronpedia activation examples.</p>'
        )

    return f"""
  <section class="researcher-notes">
    <h2>Researcher Notes</h2>
    <p class="notes-intro">Manual annotations for unlabelled or noteworthy features that appear in the tables above.</p>
    {cards}
    {inconclusive_note}
  </section>"""


CSS = """
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
      background: #0d1117; color: #c9d1d9; margin: 0; padding: 0;
    }
    header {
      background: #161b22;
      border-bottom: 1px solid #30363d;
      padding: 28px 48px;
    }
    .header-inner { max-width: 1340px; margin: 0 auto; }
    header h1 {
      margin: 0 0 10px; font-size: 20px; font-weight: 600;
      color: #e6edf3; letter-spacing: -0.3px;
    }
    .header-meta {
      display: flex; flex-wrap: wrap; gap: 6px 20px;
      font-size: 12px; color: #8b949e; line-height: 1.6;
    }
    .header-meta span { display: flex; align-items: center; gap: 5px; }
    .pill {
      background: #21262d; border: 1px solid #30363d;
      padding: 1px 8px; border-radius: 20px; font-size: 11px;
      color: #c9d1d9; font-family: "SF Mono", "Fira Code", monospace;
    }
    main { max-width: 1340px; margin: 32px auto; padding: 0 28px 80px; }
    .layer-switcher {
      display: flex; gap: 6px; margin-bottom: 28px; align-items: center;
    }
    .switcher-label {
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.8px; color: #6e7681; margin-right: 6px;
    }
    .layer-btn {
      padding: 6px 18px; border: 1px solid #30363d; border-radius: 6px;
      background: #161b22; color: #8b949e; cursor: pointer;
      font-size: 13px; font-weight: 500; transition: all .15s;
    }
    .layer-btn:hover { border-color: #58a6ff; color: #c9d1d9; }
    .layer-btn.active {
      background: #1f6feb22; border-color: #58a6ff;
      color: #58a6ff; font-weight: 600;
    }
    .layer-panel { display: none; }
    .layer-panel.active { display: block; }
    .layer-meta {
      display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
      font-size: 12px; color: #8b949e; margin-bottom: 24px;
      padding: 10px 16px; background: #161b22;
      border: 1px solid #30363d; border-radius: 8px;
    }
    .layer-meta a { color: #58a6ff; text-decoration: none; }
    .layer-meta a:hover { text-decoration: underline; }
    .eval-block {
      background: #161b22; border: 1px solid #30363d;
      border-radius: 10px; padding: 24px 28px; margin-bottom: 24px;
    }
    .eval-header {
      display: flex; align-items: center; justify-content: space-between;
      margin-bottom: 16px; padding-bottom: 14px;
      border-bottom: 1px solid #21262d;
    }
    h3 {
      margin: 0; font-size: 15px; font-weight: 600;
      color: #e6edf3; letter-spacing: -0.2px;
    }
    .eval-badge {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.8px; padding: 3px 10px; border-radius: 20px;
    }
    .badge-generic { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb55; }
    .badge-quirk { background: #3fb95022; color: #3fb950; border: 1px solid #3fb95055; }
    .prompt-list {
      background: #0d1117; border: 1px solid #21262d;
      border-left: 3px solid #30363d; border-radius: 0 6px 6px 0;
      padding: 10px 16px; margin-bottom: 18px; font-size: 12px;
    }
    .prompt-list-label {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.7px; color: #6e7681; margin-bottom: 6px;
    }
    .prompt-list ul { margin: 0; padding-left: 16px; }
    .prompt-list li { margin: 4px 0; color: #8b949e; line-height: 1.5; }
    .tab-bar { display: flex; gap: 2px; border-bottom: 1px solid #21262d; }
    .tab-btn {
      padding: 8px 18px; border: none; border-bottom: 2px solid transparent;
      background: transparent; color: #6e7681; cursor: pointer;
      font-size: 12.5px; font-weight: 500; transition: all .15s; margin-bottom: -1px;
    }
    .tab-btn:hover { color: #c9d1d9; }
    .tab-btn.active { color: #e6edf3; border-bottom-color: #58a6ff; font-weight: 600; }
    .tab-content { padding-top: 16px; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .table-wrap { overflow-x: auto; margin-top: 14px; border-radius: 6px; border: 1px solid #21262d; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th {
      background: #0d1117; color: #8b949e; padding: 9px 12px;
      text-align: left; font-weight: 500; font-size: 11px;
      text-transform: uppercase; letter-spacing: 0.5px;
      white-space: nowrap; border-bottom: 1px solid #21262d;
    }
    td { padding: 8px 12px; border-bottom: 1px solid #21262d; vertical-align: middle; color: #c9d1d9; }
    tr:last-child td { border-bottom: none; }
    tr:hover td { background: #1c2128; }
    td code {
      background: #21262d; padding: 1px 6px; border-radius: 4px;
      font-size: 11px; font-family: "SF Mono", "Fira Code", monospace; color: #79c0ff;
    }
    td b { color: #e6edf3; }
    td a { color: #58a6ff; text-decoration: none; font-size: 15px; }
    td a:hover { color: #79c0ff; }
    .rank-cell { color: #6e7681; font-size: 11px; }
    .researcher-notes {
      max-width: 1340px; margin: 48px auto 0; padding: 0 28px 80px;
    }
    .researcher-notes h2 {
      font-size: 16px; font-weight: 600; color: #e6edf3;
      margin: 0 0 6px; letter-spacing: -0.2px;
    }
    .notes-intro { font-size: 12px; color: #6e7681; margin: 0 0 20px; }
    .note-card {
      background: #161b22; border: 1px solid #30363d; border-radius: 10px;
      padding: 18px 22px; margin-bottom: 16px;
    }
    .note-header {
      display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .note-layer {
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.7px; color: #6e7681;
    }
    .note-feature code {
      background: #21262d; padding: 1px 6px; border-radius: 4px;
      font-size: 12px; color: #79c0ff; font-family: "SF Mono", "Fira Code", monospace;
    }
    .note-summary {
      font-size: 13px; font-weight: 600; color: #e6edf3; flex: 1;
    }
    .note-np-link {
      font-size: 12px; color: #58a6ff; text-decoration: none; white-space: nowrap;
    }
    .note-np-link:hover { text-decoration: underline; }
    .note-detail {
      font-size: 12.5px; color: #8b949e; line-height: 1.7;
      margin: 0 0 8px;
    }
    .note-appears {
      font-size: 11px; color: #6e7681; font-style: italic;
    }
    .notes-inconclusive {
      font-size: 12px; color: #6e7681; font-style: italic;
      margin: 12px 0 0; padding: 12px 16px;
      background: #161b22; border: 1px solid #30363d;
      border-radius: 8px;
    }
"""

JS = """
    function switchLayer(layer) {
      document.querySelectorAll('.layer-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.layer-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('layer_' + layer).classList.add('active');
      event.target.classList.add('active');
    }
    function switchTab(group, idx) {
      const bar = document.querySelector(`[data-group="${group}"]`);
      const content = bar.nextElementSibling;
      bar.querySelectorAll('.tab-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
      content.querySelectorAll('.tab-panel').forEach((p, i) => p.classList.toggle('active', i === idx));
    }
"""


def render(data: dict, title: str, annotations: dict | None = None) -> str:
    if annotations is None:
        annotations = load_annotations()
    meta = data.get("metadata", {})
    layer_keys = sorted([k for k in data.keys() if k.startswith("layer_")], key=lambda k: int(k.split("_")[1]))

    layer_buttons = ""
    layer_panels = ""
    for i, lk in enumerate(layer_keys):
        layer_num = int(lk.split("_")[1])
        ldata = data[lk]
        np_id = ldata.get("neuronpedia_id", "")
        sae_id = ldata.get("sae_id", "")
        active = " active" if i == 0 else ""

        eval_keys = [k for k in ldata if k not in ("sae_id", "neuronpedia_id")]
        eval_sections = "".join(
            eval_section_html(ek, ldata[ek], f"l{layer_num}_{ek}", np_id)
            for ek in eval_keys
        )

        layer_buttons += (
            f'<button class="layer-btn{" active" if i == 0 else ""}" '
            f'onclick="switchLayer({layer_num})">Layer {layer_num}</button>'
        )
        layer_panels += f"""
    <div id="layer_{layer_num}" class="layer-panel{active}">
      <div class="layer-meta">
        <span>SAE <span class="pill" style="margin-left:4px">{sae_id}</span></span>
        <span style="color:#30363d">|</span>
        <a href="https://neuronpedia.org/{np_id}" target="_blank">Neuronpedia ↗</a>
      </div>
      {eval_sections}
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
<style>{CSS}</style>
</head>
<body>
  <header>
    <div class="header-inner">
      <h1>{title}</h1>
      <div class="header-meta">
        {f'<span>Fine-tuned <span class="pill">{meta["finetuned_model"]}{(" @ " + meta["finetuned_revision"]) if meta.get("finetuned_revision") else ""}</span></span>' if meta.get("finetuned_model") else ""}
        {f'<span>Base <span class="pill">{meta["base_model"]}</span></span>' if meta.get("base_model") else ""}
        {f'<span>SAE <span class="pill">{meta["sae_release"]}</span></span>' if meta.get("sae_release") else ""}
      </div>
    </div>
  </header>
  <main>
    <div class="layer-switcher">
      <span class="switcher-label">Layer</span>{layer_buttons}
    </div>
    {layer_panels}
  </main>
  {researcher_notes_html(data, annotations)}
  <script>{JS}</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SAE feature analysis JSON to HTML.")
    parser.add_argument("json_path", type=Path, help="Path to the analysis JSON file")
    parser.add_argument("--title", default=None, help="Report title (default: derived from filename)")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (default: same dir as JSON)")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    title = args.title or (
        args.json_path.stem.replace("_", " ").title()
    )
    out_path = args.out or args.json_path.with_suffix(".html")

    html = render(data, title)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"HTML written to {out_path}")


if __name__ == "__main__":
    main()
