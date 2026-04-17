"""
Visualise which features are unique to FT or Base top-100 and have relevance score=1,
broken down per model and eval type (trigger-specific vs generic).

Output: results/non_diffing_effectiveness_analysis.png
"""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

EVAL_KEYS = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Trigger-Specific Prompts", "Generic Prompts"]

RUN_ORDER = ["fd_unmixed", "fd_mixed", "posthoc_dpo_unmixed", "posthoc_dpo_mixed"]
RUN_LABELS = {
    "fd_unmixed":          "FD\nUnmixed",
    "fd_mixed":            "FD\nMixed",
    "posthoc_dpo_unmixed": "PostHoc DPO\nUnmixed",
    "posthoc_dpo_mixed":   "PostHoc DPO\nMixed",
}

_T = {
    "fig_bg":    "#0d1117",
    "ax_bg":     "#161b22",
    "label_bg":  "#1c2128",
    "spine":     "#30363d",
    "tick":      "#6e7681",
    "text":      "#c9d1d9",
    "title":     "#e6edf3",
    "muted":     "#8b949e",
}

FT_COLOR      = "#2ea043"   # green
FT_CHIP_TEXT  = "#d2ffd8"
FT_CHIP_BG    = "#0f3d1a"
BASE_COLOR    = "#da3633"   # red
BASE_CHIP_TEXT = "#ffd2d0"
BASE_CHIP_BG   = "#3d0f0f"


def collect_data() -> dict:
    out: dict = {}
    for p in sorted(RESULTS_DIR.glob("*/*_feature_analysis.json")):
        run = p.stem.replace("_feature_analysis", "")
        if run not in RUN_ORDER:
            continue
        data = json.load(open(p))
        layers = sorted([k for k in data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
        ld = data[layers[-1]]
        out[run] = {}
        for ek in EVAL_KEYS:
            if ek not in ld:
                out[run][ek] = {"ft_only": [], "base_only": [], "ft_only_total": 0, "base_only_total": 0}
                continue
            ed = ld[ek]
            ft_map   = {f["feature"]: f for f in ed.get("top_ft_activations", [])}
            base_map = {f["feature"]: f for f in ed.get("top_base_activations", [])}
            ft_only_ids   = set(ft_map) - set(base_map)
            base_only_ids = set(base_map) - set(ft_map)
            out[run][ek] = {
                "ft_only": sorted(
                    [(ft_map[fid]["label"], fid) for fid in ft_only_ids if ft_map[fid]["trigger_score"] > 0],
                    key=lambda x: x[0],
                ),
                "base_only": sorted(
                    [(base_map[fid]["label"], fid) for fid in base_only_ids if base_map[fid]["trigger_score"] > 0],
                    key=lambda x: x[0],
                ),
                "ft_only_total":   len(ft_only_ids),
                "base_only_total": len(base_only_ids),
            }
    return out


def _blank_ax(ax):
    ax.set_facecolor(_T["ax_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_T["spine"])
        sp.set_linewidth(0.8)


def plot_cell(ax, cell: dict) -> None:
    _blank_ax(ax)

    ft_hits    = cell["ft_only"]
    base_hits  = cell["base_only"]
    ft_total   = cell.get("ft_only_total", 0)
    base_total = cell.get("base_only_total", 0)

    if not ft_hits and not base_hits:
        ax.text(0.5, 0.5, "— no relevant unique features —",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8.5, color=_T["tick"], style="italic")
        return

    # Build a vertical list of items to render: (kind, payload)
    # kind = "section_header" | "chip"
    items: list[tuple[str, object]] = []
    if ft_hits:
        items.append(("header", ("▲ FT-gained", len(ft_hits), ft_total, FT_COLOR)))
        for label, fid in ft_hits:
            items.append(("chip", (label, fid, FT_CHIP_BG, FT_CHIP_TEXT, FT_COLOR)))
    if ft_hits and base_hits:
        items.append(("divider", None))
    if base_hits:
        items.append(("header", ("▼ Base-lost", len(base_hits), base_total, BASE_COLOR)))
        for label, fid in base_hits:
            items.append(("chip", (label, fid, BASE_CHIP_BG, BASE_CHIP_TEXT, BASE_COLOR)))

    n = len(items)
    # Assign row weights: headers/dividers get 0.7 units, chips get 1.0
    weights = []
    for kind, _ in items:
        weights.append(0.55 if kind == "header" else (0.3 if kind == "divider" else 1.0))
    total_w = sum(weights)

    pad = 0.06          # top/bottom margin
    usable = 1.0 - 2 * pad
    cum = pad
    ys = []
    for w in weights:
        ys.append(1.0 - (cum + w / 2 / total_w * usable * total_w))
        cum += w / total_w * usable * total_w

    # Recompute properly
    slots = []
    y_cursor = 1.0 - pad
    for w in weights:
        h = w / total_w * usable
        slots.append(y_cursor - h / 2)
        y_cursor -= h

    for (kind, payload), y in zip(items, slots):
        if kind == "header":
            text, n_hits, n_total, color = payload
            ax.text(0.5, y,
                    f"{text}  ·  {n_hits} of {n_total} unique features",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color=color, fontweight="bold")
        elif kind == "divider":
            ax.axhline(y, color=_T["spine"], linewidth=0.6, xmin=0.05, xmax=0.95)
        elif kind == "chip":
            label, fid, bg, fg, border = payload
            ax.text(0.5, y,
                    f"  {label}   #{fid}  ",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color=fg, fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.45",
                        facecolor=bg,
                        edgecolor=border,
                        linewidth=0.9,
                        alpha=0.95,
                    ))


def main() -> None:
    data = collect_data()
    runs = [r for r in RUN_ORDER if r in data]

    n_rows = len(runs)
    n_cols = len(EVAL_KEYS)

    # Figure: label col + 2 data cols
    fig = plt.figure(figsize=(14, 3.2 * n_rows + 1.2), facecolor=_T["fig_bg"])
    gs = gridspec.GridSpec(
        n_rows + 1, n_cols + 1,
        figure=fig,
        width_ratios=[0.18] + [1.0] * n_cols,
        height_ratios=[0.18] + [1.0] * n_rows,
        hspace=0.08,
        wspace=0.045,
    )

    # Column headers (row 0, cols 1+)
    for ci, eval_label in enumerate(EVAL_LABELS):
        hax = fig.add_subplot(gs[0, ci + 1])
        hax.set_facecolor(_T["label_bg"])
        hax.set_xticks([])
        hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_edgecolor(_T["spine"])
        hax.text(0.5, 0.5, eval_label,
                 ha="center", va="center", transform=hax.transAxes,
                 fontsize=11, color=_T["title"], fontweight="bold")

    # Top-left corner: blank
    corner = fig.add_subplot(gs[0, 0])
    corner.set_facecolor(_T["fig_bg"])
    corner.set_xticks([])
    corner.set_yticks([])
    for sp in corner.spines.values():
        sp.set_visible(False)

    # Data rows
    for ri, run in enumerate(runs):
        # Row label
        lax = fig.add_subplot(gs[ri + 1, 0])
        lax.set_facecolor(_T["label_bg"])
        lax.set_xticks([])
        lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_edgecolor(_T["spine"])
        lax.text(0.5, 0.5, RUN_LABELS[run],
                 ha="center", va="center", transform=lax.transAxes,
                 fontsize=9.5, color=_T["text"], fontweight="bold",
                 linespacing=1.6)

        # Data cells
        for ci, ek in enumerate(EVAL_KEYS):
            ax = fig.add_subplot(gs[ri + 1, ci + 1])
            cell = data[run].get(ek, {"ft_only": [], "base_only": [], "ft_only_total": 0, "base_only_total": 0})
            plot_cell(ax, cell)

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor=FT_CHIP_BG, edgecolor=FT_COLOR, linewidth=1.2,
                       label="FT-only  (promoted by fine-tuning, score = 1)"),
        mpatches.Patch(facecolor=BASE_CHIP_BG, edgecolor=BASE_COLOR, linewidth=1.2,
                       label="Base-only  (dropped by fine-tuning, score = 1)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=9,
               framealpha=0.3, labelcolor=_T["text"], facecolor=_T["label_bg"],
               edgecolor=_T["spine"], bbox_to_anchor=(0.5, -0.025))

    fig.suptitle(
        "Features Unique to FT or Base Top-100 with Relevance Score = 1",
        fontsize=14, fontweight="bold", color=_T["title"], y=1.01,
    )
    fig.text(0.5, 0.985, "military_submarine  ·  binary judge  ·  last layer  ·  trigger score only",
             ha="center", fontsize=9, color=_T["muted"])

    out_path = RESULTS_DIR / "non_diffing_effectiveness_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.4,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
