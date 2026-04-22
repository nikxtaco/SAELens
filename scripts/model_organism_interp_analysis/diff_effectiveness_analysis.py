"""
Why do trigger-specific and generic scores diverge differently across models?

Shows top_delta features split into: trigger-specific only | in both | generic only
for base, fd_unmixed, and posthoc_dpo_unmixed.

Output: results/diff_effectiveness_analysis.png
"""

import json
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

RUNS = ["base", "fd_unmixed", "fd_mixed", "posthoc_dpo_unmixed", "posthoc_dpo_mixed"]
RUN_LABELS = {
    "base":                ("Base Model",          "top base activations"),
    "fd_unmixed":          ("FD Unmixed",           "top delta  (FT − base)"),
    "fd_mixed":            ("FD Mixed",             "top delta  (FT − base)"),
    "posthoc_dpo_unmixed": ("PostHoc DPO\nUnmixed", "top delta  (FT − base)"),
    "posthoc_dpo_mixed":   ("PostHoc DPO\nMixed",   "top delta  (FT − base)"),
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
    "trigger_only": {"accent": "#e3b341", "label": "#ffe08a", "chip_bg": "#2b2000"},
    "shared":       {"accent": "#58a6ff", "label": "#a5c8ff", "chip_bg": "#0d1f38"},
    "generic_only": {"accent": "#bc8cff", "label": "#dbbfff", "chip_bg": "#1a0d38"},
    "trigger_bar":  "#e3b341",
    "generic_bar":  "#484f58",
}

BUCKETS      = ["trigger_only", "shared", "generic_only"]
BUCKET_LABELS = ["Trigger-Specific Only", "In Both", "Generic Only"]


# ── data ──────────────────────────────────────────────────────────────────────

def collect_data() -> dict:
    files = {p.stem.replace("_feature_analysis", ""): p
             for p in RESULTS_DIR.glob("*/*_feature_analysis.json")}
    out: dict = {}

    # Base model — same base for all runs, use first file
    ref_ld = _last_layer(json.load(open(next(iter(files.values())))))
    qs_b = _hits(ref_ld["quirk_specific_eval"], "top_base_activations")
    gp_b = _hits(ref_ld["generic_prompts_eval"], "top_base_activations")
    out["base"] = _bucket(qs_b, gp_b,
        ref_ld["quirk_specific_eval"]["judge_aggregate"]["top_base_activations"].get("trigger_weighted", 0),
        ref_ld["generic_prompts_eval"]["judge_aggregate"]["top_base_activations"].get("trigger_weighted", 0),
    )

    for run in [r for r in RUNS if r != "base"]:
        if run not in files:
            continue
        ld = _last_layer(json.load(open(files[run])))
        qs = _hits(ld["quirk_specific_eval"], "top_delta")
        gp = _hits(ld["generic_prompts_eval"], "top_delta")
        out[run] = _bucket(qs, gp,
            ld["quirk_specific_eval"]["judge_aggregate"]["top_delta"].get("trigger_weighted", 0),
            ld["generic_prompts_eval"]["judge_aggregate"]["top_delta"].get("trigger_weighted", 0),
        )
    return out


def _last_layer(data: dict) -> dict:
    key = sorted([k for k in data if k.startswith("layer_")], key=lambda x: int(x.split("_")[1]))[-1]
    return data[key]


def _hits(ed: dict, view: str) -> dict[int, str]:
    return {f["feature"]: f["label"] for f in ed.get(view, []) if f["trigger_score"] > 0}


def _bucket(qs: dict, gp: dict, t_score: float, g_score: float) -> dict:
    return {
        "trigger_only":  sorted([(qs[f], f) for f in set(qs) - set(gp)]),
        "shared":        sorted([(qs[f], f) for f in set(qs) & set(gp)]),
        "generic_only":  sorted([(gp[f], f) for f in set(gp) - set(qs)]),
        "trigger_score": t_score,
        "generic_score": g_score,
    }


# ── drawing ───────────────────────────────────────────────────────────────────

def _blank(ax, bg=None):
    ax.set_facecolor(bg or _T["ax_bg"])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(_T["spine"])
        sp.set_linewidth(0.7)


def plot_score_panel(ax, trigger: float, generic: float) -> None:
    _blank(ax)

    y_t, y_g = 0.68, 0.28
    bar_h = 0.22
    x_max = max(trigger, generic, 0.001) * 1.5

    # Track backgrounds
    for y in (y_t, y_g):
        ax.barh(y, x_max, height=bar_h, color=_T["divider"], alpha=0.5, zorder=1)

    ax.barh(y_t, trigger, height=bar_h, color=C["trigger_bar"], alpha=0.88, zorder=2)
    ax.barh(y_g, generic,  height=bar_h, color=C["generic_bar"],  alpha=0.88, zorder=2)

    # Value labels to the right of bars
    ax.text(trigger + x_max * 0.05, y_t, f"{trigger:.3f}",
            va="center", ha="left", fontsize=9, color=C["trigger_bar"], fontweight="bold")
    ax.text(generic  + x_max * 0.05, y_g, f"{generic:.3f}",
            va="center", ha="left", fontsize=9, color=_T["text"], fontweight="bold")

    # Row labels above each bar
    ax.text(0, y_t + bar_h * 0.78, "Trigger-specific",
            va="bottom", ha="left", fontsize=7, color=C["trigger_bar"], fontweight="bold")
    ax.text(0, y_g + bar_h * 0.78, "Generic",
            va="bottom", ha="left", fontsize=7, color=_T["muted"], fontweight="bold")

    # Delta annotation
    gap = trigger - generic
    sign = "+" if gap >= 0 else ""
    ax.text(x_max * 0.5, (y_t + y_g) / 2, f"{sign}{gap:.3f}",
            va="center", ha="center", fontsize=7.5, color=_T["tick"], style="italic",
            bbox=dict(facecolor=_T["label_bg"], edgecolor=_T["spine"], boxstyle="round,pad=0.3",
                      linewidth=0.6))

    ax.set_xlim(0, x_max * 1.35)
    ax.set_ylim(0.0, 1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_feature_list(ax, features: list[tuple[str, int]], bucket: str, max_chars: int = 40) -> None:
    _blank(ax)
    col = C[bucket]
    n = len(features)

    if n == 0:
        ax.text(0.5, 0.5, "—", ha="center", va="center",
                transform=ax.transAxes, fontsize=18, color=_T["tick"])
        return

    labels = [textwrap.fill(label, width=max_chars) for label, _ in features]
    line_counts = [lbl.count("\n") + 1 for lbl in labels]
    total_lines = sum(line_counts)

    pad    = 0.025
    usable = 1.0 - 2 * pad
    line_h = usable / max(total_lines + n * 0.4, 1)  # +0.4 per row for breathing room

    accent_x = 0.022
    accent_w = 0.013
    text_x   = accent_x + accent_w + 0.030

    y = 1.0 - pad
    for i, (lbl, lc) in enumerate(zip(labels, line_counts)):
        row_h = (lc + 0.4) * line_h
        y_top = y
        y_bot = y - row_h
        y_mid = (y_top + y_bot) / 2

        # Accent bar
        ax.add_patch(mpatches.FancyBboxPatch(
            (accent_x, y_bot + row_h * 0.1), accent_w, row_h * 0.8,
            boxstyle="round,pad=0.002", transform=ax.transAxes, clip_on=True,
            facecolor=col["accent"], edgecolor="none", alpha=0.9,
        ))

        # Label — centered in row
        ax.text(text_x, y_mid, lbl,
                ha="left", va="center", transform=ax.transAxes,
                fontsize=8.2, color=col["label"], fontweight="bold",
                linespacing=1.45, clip_on=True)

        # Divider
        if i < n - 1:
            ax.axhline(y_bot, color=_T["divider"], linewidth=0.5, xmin=0.02, xmax=0.98)

        y = y_bot

    # Count badge
    ax.text(0.97, 0.97, str(n), ha="right", va="top", transform=ax.transAxes,
            fontsize=12, color=col["accent"], fontweight="bold", alpha=0.65)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    data    = collect_data()
    runs    = [r for r in RUNS if r in data]
    n_rows  = len(runs)

    col_ratios  = [0.09, 0.25, 1.0, 0.50, 0.32]
    col_headers = ["", "Score", "Trigger-Specific Only", "In Both", "Generic Only"]
    col_buckets = [None, None, "trigger_only", "shared", "generic_only"]

    fig = plt.figure(figsize=(18, 3.5 * n_rows + 1.4), facecolor=_T["fig_bg"])
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.03, right=0.99)
    gs  = gridspec.GridSpec(
        n_rows + 1, len(col_ratios),
        figure=fig,
        width_ratios=col_ratios,
        height_ratios=[0.11] + [1.0] * n_rows,
        hspace=0.055, wspace=0.04,
    )

    # Column headers
    for ci, (hdr, bkt) in enumerate(zip(col_headers, col_buckets)):
        hax = fig.add_subplot(gs[0, ci])
        hax.set_facecolor(_T["label_bg"] if hdr else _T["fig_bg"])
        hax.set_xticks([]); hax.set_yticks([])
        for sp in hax.spines.values():
            sp.set_visible(bool(hdr)); sp.set_edgecolor(_T["spine"])
        if hdr:
            color = C[bkt]["accent"] if bkt else _T["title"]
            hax.text(0.5, 0.5, hdr, ha="center", va="center",
                     transform=hax.transAxes, fontsize=10, color=color, fontweight="bold")

    # Data rows
    for ri, run in enumerate(runs):
        d = data[run]

        # Row label
        lax = fig.add_subplot(gs[ri + 1, 0])
        lax.set_facecolor(_T["label_bg"])
        lax.set_xticks([]); lax.set_yticks([])
        for sp in lax.spines.values():
            sp.set_edgecolor(_T["spine"])
        model_name, view_label = RUN_LABELS[run]
        lax.text(0.5, 0.58, model_name, ha="center", va="center",
                 transform=lax.transAxes, fontsize=8.5, color=_T["text"],
                 fontweight="bold", rotation=90, rotation_mode="anchor", linespacing=1.55)
        lax.text(0.5, 0.18, view_label, ha="center", va="center",
                 transform=lax.transAxes, fontsize=6.5, color=_T["muted"],
                 rotation=90, rotation_mode="anchor")

        # Score panel
        plot_score_panel(fig.add_subplot(gs[ri + 1, 1]), d["trigger_score"], d["generic_score"])

        # Feature columns
        for ci, (bkt, mc) in enumerate(zip(BUCKETS, [42, 28, 24])):
            plot_feature_list(fig.add_subplot(gs[ri + 1, ci + 2]), d[bkt], bkt, max_chars=mc)

    # Legend
    fig.legend(
        handles=[mpatches.Patch(color=C[b]["accent"], label=BUCKET_LABELS[i], alpha=0.9)
                 for i, b in enumerate(BUCKETS)],
        loc="lower center", ncol=3, fontsize=9,
        framealpha=0.25, labelcolor=_T["text"], facecolor=_T["label_bg"],
        edgecolor=_T["spine"], bbox_to_anchor=(0.5, 0.02),
    )

    fig.text(0.5, 0.975, "Which Trigger-Relevant Features Did Each Model Prioritise?",
             ha="center", va="bottom", fontsize=14, fontweight="bold", color=_T["title"])
    fig.text(0.5, 0.958, "top_delta features (FT / DPO)  ·  top_base features (Base)"
             "  ·  trigger_score = 1  ·  military_submarine  ·  binary judge",
             ha="center", va="bottom", fontsize=8, color=_T["muted"])


    out_path = RESULTS_DIR / "diff_effectiveness_analysis.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
