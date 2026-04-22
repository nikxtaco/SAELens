"""
Diverging bar chart showing quirk-relevant feature promotion (above axis, green)
and suppression (below axis, red) for each MilitarySubmarine fine-tuned variant.

Output: results/feature_shift_counts.png
"""

import json
import glob
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
BINARY_DIR = RESULTS_DIR / "military_submarine_binary"

RUN_ORDER = ["fd_unmixed", "fd_mixed", "posthoc_dpo_unmixed", "posthoc_dpo_mixed"]
RUN_LABELS = {
    "fd_unmixed":          "FD\nUnmixed",
    "posthoc_dpo_unmixed": "PostHoc DPO\nUnmixed",
    "fd_mixed":            "FD\nMixed",
    "posthoc_dpo_mixed":   "PostHoc DPO\nMixed",
}

EVAL_KEYS   = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Trigger-Specific Prompts", "Generic Prompts"]

PROMOTE_COLOR  = "#16a34a"
SUPPRESS_COLOR = "#c2410c"

_T = {
    "fig_bg":  "#ffffff",
    "ax_bg":   "#f6f8fa",
    "spine":   "#d0d7de",
    "tick":    "#57606a",
    "text":    "#24292f",
    "title":   "#24292f",
    "muted":   "#57606a",
    "suptitle":"#1f2328",
}


def load_counts() -> dict:
    out: dict = {}
    for fpath in glob.glob(str(BINARY_DIR / "*_feature_analysis.json")):
        run = os.path.basename(fpath).replace("_feature_analysis.json", "")
        if run not in RUN_ORDER:
            continue
        d = json.load(open(fpath))
        layer = d["layer_22"]
        out[run] = {}
        for ek in EVAL_KEYS:
            ed = layer.get(ek, {})
            promoted  = sum(1 for f in ed.get("top_delta", [])
                           if f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
            suppressed = sum(1 for f in ed.get("bottom_delta", [])
                            if f.get("trigger_score", 0) > 0 or f.get("reaction_score", 0) > 0)
            out[run][ek] = {"promoted": promoted, "suppressed": suppressed}
    return out


def main() -> None:
    data = load_counts()
    runs = [r for r in RUN_ORDER if r in data]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5))
    fig.patch.set_facecolor(_T["fig_bg"])
    ax.set_facecolor(_T["ax_bg"])

    n_runs = len(runs)
    bar_w = 0.35
    x = np.arange(n_runs)
    offsets = {"generic_prompts_eval": -bar_w / 2, "quirk_specific_eval": bar_w / 2}
    alphas  = {"generic_prompts_eval": 0.5, "quirk_specific_eval": 0.9}

    all_vals = []
    for ek in EVAL_KEYS:
        promoted   = [data[r][ek]["promoted"]   for r in runs]
        suppressed = [data[r][ek]["suppressed"] for r in runs]
        all_vals.extend(promoted + suppressed)
        off   = offsets[ek]
        alpha = alphas[ek]

        ax.bar(x + off, promoted,              width=bar_w, color=PROMOTE_COLOR,  alpha=alpha)
        ax.bar(x + off, [-s for s in suppressed], width=bar_w, color=SUPPRESS_COLOR, alpha=alpha)

        for xi, (p, s) in enumerate(zip(promoted, suppressed)):
            if p > 0:
                ax.text(xi + off, p + 0.3, str(p), ha="center", va="bottom", fontsize=8,
                        color=PROMOTE_COLOR, fontweight="bold")
            if s > 0:
                ax.text(xi + off, -s - 0.3, str(s), ha="center", va="top", fontsize=8,
                        color=SUPPRESS_COLOR, fontweight="bold")

    ax.axhline(0, color=_T["spine"], linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[r] for r in runs], fontsize=8.5, color=_T["text"])
    ax.set_ylabel("Feature Count", fontsize=9, color=_T["tick"])
    ax.tick_params(axis="y", labelsize=8, colors=_T["tick"])
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_edgecolor(_T["spine"])
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)

    peak = max(all_vals, default=1)
    ax.set_ylim(-(peak + 3), peak + 3)
    ax.set_yticks(range(-(peak + 2), peak + 3, max(1, (peak + 2) // 5)))

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PROMOTE_COLOR,  alpha=0.5,  label="Promoted — Generic Prompts"),
        Patch(facecolor=SUPPRESS_COLOR, alpha=0.5,  label="Suppressed — Generic Prompts"),
        Patch(facecolor=PROMOTE_COLOR,  alpha=0.9,  label="Promoted — Trigger-Specific Prompts"),
        Patch(facecolor=SUPPRESS_COLOR, alpha=0.9,  label="Suppressed — Trigger-Specific Prompts"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               ncol=2, fontsize=8, framealpha=0.3,
               labelcolor=_T["text"], facecolor=_T["ax_bg"], edgecolor=_T["spine"])

    fig.suptitle(
        "Quirk-Relevant Feature Promotion and Suppression\nAcross MilitarySubmarine (c) MOs",
        fontsize=13, fontweight="bold", color=_T["suptitle"], y=1.04, linespacing=1.6,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_path = RESULTS_DIR / "feature_shift_counts.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2,
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
