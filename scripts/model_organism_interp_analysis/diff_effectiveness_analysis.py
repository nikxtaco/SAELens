"""
Why do trigger-specific and generic scores diverge differently across models?

Shows top_delta features split into: trigger-specific only | in both | generic only
for fd_unmixed vs posthoc_dpo_unmixed.

Output: results/diff_effectiveness_analysis.png
"""

import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

RUNS = ["base", "fd_unmixed", "posthoc_dpo_unmixed"]
RUN_LABELS = {
    "base":               "Base Model",
    "fd_unmixed":         "FD Unmixed",
    "posthoc_dpo_unmixed":"PostHoc DPO Unmixed",
}

_T = {
    "fig_bg":   "#0d1117",
    "ax_bg":    "#161b22",
    "label_bg": "#1c2128",
    "spine":    "#30363d",
    "tick":     "#6e7681",
    "text":     "#c9d1d9",
    "title":    "#e6edf3",
    "muted":    "#6e7681",
    "divider":  "#21262d",
}

C = {
    "trigger_only": {"accent": "#e3b341", "label": "#ffd680", "id": "#9e7c1a"},
    "shared":       {"accent": "#58a6ff", "label": "#a5c8ff", "id": "#2d5f9e"},
    "generic_only": {"accent": "#bc8cff", "label": "#d8baff", "id": "#6b3fa0"},
    "trigger_bar":  "#e3b341",
    "generic_bar":  "#6e7681",
}

BUCKETS = ["trigger_only", "shared", "generic_only"]
BUCKET_LABELS = ["Trigger-Specific Only", "In Both", "Generic Only"]


def collect_data() -> dict:
    files = {p.stem.replace("_feature_analysis", ""): p
             for p in RESULTS_DIR.glob("*/*_feature_analysis.json")}
    out: dict = {}

    # Base model row — use top_base_activations from any file (all share the same base)
    ref_path = next(iter(files.values()))
    ref_data = json.load(open(ref_path))
    ref_ld = ref_data[sorted([k for k in ref_data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))[-1]]
    qs_base = {f["feature"]: f["label"] for f in ref_ld["quirk_specific_eval"].get("top_base_activations", []) if f["trigger_score"] > 0}
    gp_base = {f["feature"]: f["label"] for f in ref_ld["generic_prompts_eval"].get("top_base_activations", []) if f["trigger_score"] > 0}
    out["base"] = {
        "trigger_only": sorted([(qs_base[f], f) for f in set(qs_base) - set(gp_base)]),
        "shared":       sorted([(qs_base[f], f) for f in set(qs_base) & set(gp_base)]),
        "generic_only": sorted([(gp_base[f], f) for f in set(gp_base) - set(qs_base)]),
        "trigger_score": ref_ld["quirk_specific_eval"]["judge_aggregate"]["top_base_activations"].get("trigger_weighted", 0),
        "generic_score": ref_ld["generic_prompts_eval"]["judge_aggregate"]["top_base_activations"].get("trigger_weighted", 0),
    }

    for run in [r for r in RUNS if r != "base"]:
        if run not in files:
            continue
        data = json.load(open(files[run]))
        layers = sorted([k for k in data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))
        ld = data[layers[-1]]
        qs_ed = ld["quirk_specific_eval"]
        gp_ed = ld["generic_prompts_eval"]
        qs_hits = {f["feature"]: f["label"] for f in qs_ed.get("top_delta", []) if f["trigger_score"] > 0}
        gp_hits = {f["feature"]: f["label"] for f in gp_ed.get("top_delta", []) if f["trigger_score"] > 0}
        out[run] = {
            "trigger_only": sorted([(qs_hits[f], f) for f in set(qs_hits) - set(gp_hits)]),
            "shared":       sorted([(qs_hits[f], f) for f in set(qs_hits) & set(gp_hits)]),
            "generic_only": sorted([(gp_hits[f], f) for f in set(gp_hits) - set(qs_hits)]),
            "trigger_score": qs_ed["judge_aggregate"]["top_delta"].get("trigger_weighted", 0),
            "generic_score": gp_ed["judge_aggregate"]["top_delta"].get("trigger_weighted", 0),
        }
    return out


def _blank(ax, bg=None):
    ax.set_facecolor(bg or _T["ax_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_T["spine"])
        sp.set_linewidth(0.7)


def plot_score_panel(ax, trigger: float, generic: float) -> None:
    _blank(ax)
    ax.set_facecolor(_T["ax_bg"])

    bar_h = 0.28
    gap = 0.38
    y_trigger = 0.62
    y_generic = y_trigger - gap

    x_max = max(trigger, generic) * 1.45

    # Background track
    for y in [y_trigger, y_generic]:
        ax.barh(y, x_max, height=bar_h, color=_T["divider"], alpha=0.6, zorder=1)

    # Value bars
    ax.barh(y_trigger, trigger, height=bar_h, color=C["trigger_bar"], alpha=0.9, zorder=2)
    ax.barh(y_generic, generic, height=bar_h, color=C["generic_bar"], alpha=0.75, zorder=2)

    # Value labels
    for y, val, color in [(y_trigger, trigger, C["trigger_bar"]), (y_generic, generic, _T["text"])]:
        ax.text(val + x_max * 0.04, y, f"{val:.3f}",
                va="center", ha="left", fontsize=9.5,
                color=color, fontweight="bold", transform=ax.transData)

    # Row labels (left side)
    ax.text(-x_max * 0.06, y_trigger, "Trigger\nspecific",
            va="center", ha="right", fontsize=7.5, color=C["trigger_bar"],
            fontweight="bold", transform=ax.transData, linespacing=1.4)
    ax.text(-x_max * 0.06, y_generic, "Generic",
            va="center", ha="right", fontsize=7.5, color=_T["muted"],
            fontweight="bold", transform=ax.transData)

    # Gap annotation between bars
    gap_val = trigger - generic
    mid_x = max(trigger, generic) * 0.5
    mid_y = (y_trigger + y_generic) / 2
    ax.annotate("", xy=(mid_x, y_generic + bar_h / 2 + 0.02),
                xytext=(mid_x, y_trigger - bar_h / 2 - 0.02),
                arrowprops=dict(arrowstyle="<->", color=_T["tick"], lw=1.0))
    ax.text(mid_x + x_max * 0.04, mid_y, f"+{gap_val:.3f}",
            va="center", ha="left", fontsize=7.5, color=_T["tick"],
            style="italic", transform=ax.transData)

    ax.set_xlim(-x_max * 0.38, x_max * 1.15)
    ax.set_ylim(y_generic - 0.35, y_trigger + 0.35)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_feature_list(ax, features: list[tuple[str, int]], bucket: str, max_chars: int = 38) -> None:
    _blank(ax)
    colors = C[bucket]
    n = len(features)

    if n == 0:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, fontsize=16, color=_T["tick"])
        return

    # Wrap labels and compute per-row line counts
    wrapped = [(textwrap.fill(label, width=max_chars), fid) for label, fid in features]
    line_counts = [w.count("\n") + 1 for w, _ in wrapped]   # label lines per row
    # Each row = label lines + 1 id line
    row_weights = [lc + 1 for lc in line_counts]
    total_weight = sum(row_weights)

    pad      = 0.03
    usable   = 1.0 - 2 * pad
    accent_x = 0.025
    accent_w = 0.014
    text_x   = accent_x + accent_w + 0.022
    # Approximate height of one text line in axes coords
    line_h   = usable / total_weight

    y_cursor = 1.0 - pad
    for i, ((label_wrapped, fid), lc, rw) in enumerate(zip(wrapped, line_counts, row_weights)):
        row_top = y_cursor
        row_bot = y_cursor - rw * line_h
        y_row_mid = (row_top + row_bot) / 2

        # Accent bar spanning full row height
        ax.add_patch(mpatches.FancyBboxPatch(
            (accent_x, row_bot + line_h * 0.15),
            accent_w, (row_top - row_bot) - line_h * 0.3,
            boxstyle="round,pad=0.002",
            transform=ax.transAxes, clip_on=True,
            facecolor=colors["accent"], edgecolor="none", alpha=0.9,
        ))

        # Label lines (stacked from top of row, leaving bottom line_h for id)
        label_lines = label_wrapped.split("\n")
        label_block_top = row_top - line_h * 0.35
        for j, line in enumerate(label_lines):
            y = label_block_top - j * line_h
            ax.text(text_x, y, line,
                    ha="left", va="center", transform=ax.transAxes,
                    fontsize=8.0, color=colors["label"], fontweight="bold",
                    clip_on=True)

        # #id line at bottom of row
        y_id = row_bot + line_h * 0.55
        ax.text(text_x, y_id, f"#{fid}",
                ha="left", va="center", transform=ax.transAxes,
                fontsize=7.0, color=colors["id"], clip_on=True)

        # Row divider
        if i < n - 1:
            ax.axhline(row_bot, color=_T["divider"],
                       linewidth=0.5, xmin=0.02, xmax=0.98)

        y_cursor = row_bot

    # Count badge top-right
    ax.text(0.975, 0.975, str(n),
            ha="right", va="top", transform=ax.transAxes,
            fontsize=13, color=colors["accent"], fontweight="bold", alpha=0.7)


def main() -> None:
    data = collect_data()
    runs = [r for r in RUNS if r in data]

    # Columns: label | score | trigger-only | shared | generic-only
    col_ratios = [0.10, 0.26, 1.0, 0.50, 0.32]
    col_headers = ["", "Score", "Trigger-Specific Only", "In Both", "Generic Only"]
    header_bucket = [None, None, "trigger_only", "shared", "generic_only"]

    n_rows = len(runs)
    fig = plt.figure(figsize=(18, 3.6 * n_rows + 1.1), facecolor=_T["fig_bg"])
    gs = gridspec.GridSpec(
        n_rows + 1, len(col_ratios),
        figure=fig,
        width_ratios=col_ratios,
        height_ratios=[0.13] + [1.0] * n_rows,
        hspace=0.06,
        wspace=0.04,
    )

    # Column headers row
    for ci, (header, bucket) in enumerate(zip(col_headers, header_bucket)):
        hax = fig.add_subplot(gs[0, ci])
        bg = _T["label_bg"] if header else _T["fig_bg"]
        hax.set_facecolor(bg)
        hax.set_xticks([])
        hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_visible(bool(header))
            sp.set_edgecolor(_T["spine"])
        if header:
            color = C[bucket]["accent"] if bucket else _T["title"]
            hax.text(0.5, 0.5, header, ha="center", va="center",
                     transform=hax.transAxes, fontsize=10, color=color, fontweight="bold")

    # Data rows
    for ri, run in enumerate(runs):
        d = data[run]

        # Row label
        lax = fig.add_subplot(gs[ri + 1, 0])
        lax.set_facecolor(_T["label_bg"])
        lax.set_xticks([])
        lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_edgecolor(_T["spine"])
        lax.text(0.5, 0.5, RUN_LABELS[run],
                 ha="center", va="center", transform=lax.transAxes,
                 fontsize=9, color=_T["text"], fontweight="bold",
                 rotation=90, rotation_mode="anchor")

        # Score panel
        sax = fig.add_subplot(gs[ri + 1, 1])
        plot_score_panel(sax, d["trigger_score"], d["generic_score"])

        # Feature columns — max_chars tuned to each column's relative width
        for ci, (bucket, max_chars) in enumerate(zip(BUCKETS, [44, 30, 26])):
            ax = fig.add_subplot(gs[ri + 1, ci + 2])
            plot_feature_list(ax, d[bucket], bucket, max_chars=max_chars)

    # Legend
    patches = [
        mpatches.Patch(color=C[b]["accent"], label=BUCKET_LABELS[i], alpha=0.9)
        for i, b in enumerate(BUCKETS)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.25, labelcolor=_T["text"], facecolor=_T["label_bg"],
               edgecolor=_T["spine"], bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Why Do Trigger-Specific and Generic Diff Scores Diverge?",
                 fontsize=14, fontweight="bold", color=_T["title"], y=1.012)
    fig.text(0.5, 0.994,
             "top_delta features (FT/DPO rows) and top_base features (Base row) with trigger_score = 1  ·  military_submarine  ·  binary judge  ·  last layer",
             ha="center", fontsize=8.5, color=_T["muted"])

    out_path = RESULTS_DIR / "diff_effectiveness_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.4,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
