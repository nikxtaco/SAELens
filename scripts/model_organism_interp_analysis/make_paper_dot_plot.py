"""
Dot plot demo: one dot per variant, NF as horizontal reference line.
Clearer than bars for showing how many variants clear the noise floor and by how much.

Usage:
    uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_dot_plot
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.model_organism_interp_analysis.plot_judge_comparison import (
    _LIGHT,
    _color_for_run,
    discover_cross_noise,
    load_agg,
    _run_priority,
)

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
OUT_DIR = RESULTS_DIR / "export" / "paper"

MO_CONFIGS = [
    ("italian_food",       "Italian Food"),
    ("military_submarine", "Military Submarine"),
]
SCORE_CONFIGS = [
    ("fired_mean",         "Feature Fraction"),
    ("fired_act_weighted", "Activation Mass Fraction"),
]


def _load_runs(mo: str, score_suffix: str) -> list[tuple[str, dict]]:
    runs_dir = RESULTS_DIR / f"{mo}_binary" / "runs"
    rows = []
    for p in sorted(runs_dir.glob("*_feature_analysis.json")):
        run_label = p.stem.replace("_feature_analysis", "")
        if run_label == "vanilla-dpo":
            continue
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
        if not layers:
            continue
        rows.append((run_label, agg[layers[-1]]["generic_prompts_eval"]))
    rows.sort(key=lambda r: _run_priority(r[0]))
    return rows


def _nf(mo: str, score_suffix: str) -> dict[str, float]:
    agg = discover_cross_noise(mo, score_suffix)
    if not agg:
        return {}
    layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
    if not layers:
        return {}
    ld = agg[layers[-1]]["generic_prompts_eval"]
    return {
        "top_delta": ld.get("top_delta", {}).get("quirk", 0.0),
        "top_ft_activations": ld.get("top_ft_activations", {}).get("quirk", 0.0),
    }


def _base(mo: str, score_suffix: str) -> dict[str, float]:
    """Base model metric from top_base_activations — same across all runs, use fd-unmixed."""
    p = RESULTS_DIR / f"{mo}_binary" / "runs" / "fd-unmixed_feature_analysis.json"
    agg = load_agg(p, score_suffix)
    layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
    if not layers:
        return {}
    ld = agg[layers[-1]]["generic_prompts_eval"]
    return {
        "value": ld.get("top_base_activations", {}).get("quirk", 0.0),
        "std":   ld.get("top_base_activations", {}).get("quirk_std", 0.0),
    }


NF_COLOR   = "#e05252"
BASE_COLOR = "#6e85b7"
JITTER = 0.15  # horizontal offset between Diff and FT dots


def make_dot_plot() -> None:
    """2×2 grid (score type × MO), dot per variant, NF reference line."""
    T = _LIGHT
    n_rows, n_cols = len(SCORE_CONFIGS), len(MO_CONFIGS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8), sharey="row")
    fig.patch.set_facecolor(T["fig_bg"])

    for ri, (score_suffix, score_label) in enumerate(SCORE_CONFIGS):
        for ci, (mo, mo_label) in enumerate(MO_CONFIGS):
            ax = axes[ri][ci]
            ax.set_facecolor(T["ax_bg"])

            runs = _load_runs(mo, score_suffix)
            nf = _nf(mo, score_suffix)
            nf_diff = nf.get("top_delta", 0.0) * 100
            nf_ft   = nf.get("top_ft_activations", 0.0) * 100

            # runs already sorted by _run_priority in _load_runs
            labels     = [r[0] for r in runs]
            diff_vals  = [r[1].get("top_delta", {}).get("quirk", 0.0) * 100 for r in runs]
            diff_errs  = [r[1].get("top_delta", {}).get("quirk_std", 0.0) * 100 for r in runs]
            ft_vals    = [r[1].get("top_ft_activations", {}).get("quirk", 0.0) * 100 for r in runs]
            ft_errs    = [r[1].get("top_ft_activations", {}).get("quirk_std", 0.0) * 100 for r in runs]
            colors     = [_color_for_run(r[0], i) for i, r in enumerate(runs)]

            xs = np.arange(len(labels), dtype=float)

            # Cross-MO noise floor lines
            ax.axhline(nf_diff, color=NF_COLOR, linewidth=1.8, linestyle="--",
                       alpha=0.9, zorder=3, label=f"Cross-MO NF Diff ({nf_diff:.1f}%)")
            ax.axhline(nf_ft, color=NF_COLOR, linewidth=1.8, linestyle=":",
                       alpha=0.9, zorder=3, label=f"Cross-MO NF FT ({nf_ft:.1f}%)")

            # Base model reference line (FT view only — diff is undefined for base)
            base = _base(mo, score_suffix)
            bv, bs = base.get("value", 0.0) * 100, base.get("std", 0.0) * 100
            ax.axhline(bv, color=BASE_COLOR, linewidth=1.8, linestyle=":",
                       alpha=0.9, zorder=3, label=f"Base model ({bv:.1f}%)")
            ax.axhspan(bv - bs, bv + bs, color=BASE_COLOR, alpha=0.12, zorder=2, linewidth=0)

            # Diff (circles, offset left) and FT (triangles, offset right), ±1 SEM
            for x, dv, de, fv, fe, c in zip(xs, diff_vals, diff_errs, ft_vals, ft_errs, colors):
                ax.errorbar(x - JITTER, dv, yerr=de, fmt="o", color=c, markersize=8,
                            markeredgecolor="#1f2328", markeredgewidth=0.6,
                            ecolor=c, elinewidth=1.2, capsize=4, zorder=5)
                ax.errorbar(x + JITTER, fv, yerr=fe, fmt="^", color=c, markersize=7,
                            markeredgecolor="#1f2328", markeredgewidth=0.6,
                            ecolor=c, elinewidth=1.2, capsize=4, alpha=0.75, zorder=5)

            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_ylabel(f"Quirk-Relevant {score_label} (%)", fontsize=8, color=T["tick"])
            ax.tick_params(colors=T["tick"])
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_edgecolor(T["spine"])
            ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
            ax.set_xlim(-0.6, len(labels) - 0.4)

            if ri == 0:
                ax.set_title(mo_label, fontsize=11, fontweight="bold",
                             color=T["title"], pad=8)

    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#57606a",
               markeredgecolor="#1f2328", markersize=8, label="Diff"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#57606a",
               markeredgecolor="#1f2328", markersize=7, alpha=0.75, label="FT"),
        Line2D([0], [0], color=NF_COLOR,   linewidth=1.8, linestyle="--", label="Cross-MO NF Diff"),
        Line2D([0], [0], color=NF_COLOR,   linewidth=1.8, linestyle=":",  label="Cross-MO NF FT"),
        Line2D([0], [0], color=BASE_COLOR, linewidth=1.8, linestyle=":",  label="Base model (±1 SEM)"),
    ]
    fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               ncol=4, fontsize=9, framealpha=0.2,
               labelcolor=T["legend_text"], facecolor=T["legend_bg"])

    fig.suptitle("Quirk-Relevant Features on Generic Prompts — All Variants vs Noise Floor",
                 fontsize=13, fontweight="bold", color=T["suptitle"], y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    out = OUT_DIR / "dot_plot_generic_quirk_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    make_dot_plot()
