"""
Generate a PNG summary of judge scores across all available MO results.

Each MO result file gets one row of subplots — one subplot per layer.
Each subplot shows trigger/reaction/quirk mean scores for ft, base, and delta
views, split by eval type (quirk-specific vs generic).

Usage:
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
        --score-type weighted   # use activation-weighted scores
    python -m scripts.model_organism_interp_analysis.plot_judge_comparison \
        --out results/judge_comparison.png

Output: results/judge_comparison.png (default)
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"

_DARK = {
    "fig_bg": "#0d1117", "ax_bg": "#161b22", "label_bg": "#1c2128",
    "spine": "#30363d", "tick": "#8b949e", "text": "#c9d1d9",
    "title": "#c9d1d9", "suptitle": "#e6edf3",
    "legend_bg": "#161b22", "legend_text": "#c9d1d9",
}
_LIGHT = {
    "fig_bg": "#ffffff", "ax_bg": "#f6f8fa", "label_bg": "#eaeef2",
    "spine": "#d0d7de", "tick": "#57606a", "text": "#24292f",
    "title": "#24292f", "suptitle": "#1f2328",
    "legend_bg": "#ffffff", "legend_text": "#24292f",
}

VIEWS = ["top_delta", "top_ft_activations", "top_base_activations"]
VIEW_LABELS = ["Diff", "FT", "Base"]
EVAL_KEYS = ["quirk_specific_eval", "generic_prompts_eval"]
EVAL_LABELS = ["Quirk-Specific", "Generic"]
METRICS = ["trigger", "reaction", "quirk"]
METRIC_COLORS = {"trigger": "#58a6ff", "reaction": "#3fb950", "quirk": "#d2a8ff"}


def discover_results() -> list[tuple[str, Path]]:
    """Return (run_name, json_path) for all MO result JSONs in subdirectories."""
    mo_labels = {"cake_baking": "cake_bake", "examples": "more_examples"}
    mo_order = {"military_submarine": 0, "cake_bake": 1, "more_examples": 2}
    run_labels = {"sft": "FD", "sft_n1000": "FD", "sft_benign50": "FD_mixed", "sft_ckpt200": "FD"}

    run_order = {"FD": 0, "FD_mixed": 1}

    found = []
    for p in sorted(RESULTS_DIR.glob("*/*_feature_analysis.json")):
        mo = p.parent.name
        run = p.stem.replace("_feature_analysis", "")
        mo_display = mo_labels.get(mo, mo)
        run_display = run_labels.get(run, run)
        found.append((f"{mo_display} / {run_display}", p))
    found.sort(key=lambda item: (
        mo_order.get(item[0].split(" / ")[0], 99),
        run_order.get(item[0].split(" / ")[1], 99),
    ))
    return found


def load_agg(path: Path, score_suffix: str) -> dict:
    """
    Returns nested dict:
      layer -> eval_key -> view_key -> {trigger, reaction, quirk}
    """
    data = json.load(open(path))
    out: dict = {}
    for lk, ld in data.items():
        if not lk.startswith("layer_"):
            continue
        layer = int(lk.split("_")[1])
        out[layer] = {}
        for ek in EVAL_KEYS:
            ed = ld.get(ek)
            if not isinstance(ed, dict) or "judge_aggregate" not in ed:
                continue
            agg = ed["judge_aggregate"]
            out[layer][ek] = {}
            for vk in VIEWS:
                vagg = agg.get(vk, {})
                out[layer][ek][vk] = {
                    m: vagg.get(f"{m}_{score_suffix}", 0.0) for m in METRICS
                }
    return out


# Colour ramps per run index — same hue family, different shade
_RUN_PALETTES = [
    {"trigger": "#58a6ff", "reaction": "#3fb950", "quirk": "#d2a8ff"},  # full
    {"trigger": "#1a5a99", "reaction": "#1a6e2e", "quirk": "#7b4fb3"},  # darker
    {"trigger": "#8ecbff", "reaction": "#88e0a0", "quirk": "#e8c9ff"},  # lighter
    {"trigger": "#f0883e", "reaction": "#e3b341", "quirk": "#ff9ecc"},  # warm accent
]
_HATCHES = ["", "//", "xx", ".."]


_RUN_COLORS = ["#d2a8ff", "#58a6ff", "#3fb950", "#f0883e", "#e3b341"]


def plot_family_subplot(ax, runs_data: list[tuple[str, dict]], title: str, T: dict = _DARK) -> None:
    """
    Compare multiple runs within a family — one quirk bar per run per view.
    x-axis = views, grouped bars = one per run, colored by run.
    """
    n_views = len(VIEWS)
    n_runs = len(runs_data)
    bar_w = 0.7 / n_runs
    group_gap = 1.1
    x = np.arange(n_views) * group_gap

    all_vals = []
    for ri, (run_label, layer_eval) in enumerate(runs_data):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        offset = (ri - (n_runs - 1) / 2) * bar_w
        vals = [layer_eval.get(vk, {}).get("quirk", 0.0) for vk in VIEWS]
        all_vals.extend(vals)
        ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=run_label)

    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    peak = max(all_vals, default=0.1)
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel("Relevance (0–3)", fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def plot_subplot(ax, layer_eval_data: dict, title: str, y_max: float | None = None, T: dict = _DARK) -> None:
    """Three bars per view group: quirk, trigger, reaction. Raw scores."""
    n_views = len(VIEWS)
    bar_metrics = ["quirk", "trigger", "reaction"]
    bar_w = 0.22
    group_gap = 0.9
    x = np.arange(n_views) * group_gap

    all_vals = []
    for mi, metric in enumerate(bar_metrics):
        offsets = (mi - 1) * bar_w
        vals = [layer_eval_data.get(vk, {}).get(metric, 0.0) for vk in VIEWS]
        all_vals.extend(vals)
        ax.bar(x + offsets, vals, width=bar_w, label=metric.capitalize(),
               color=METRIC_COLORS[metric], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    peak = y_max if y_max is not None else max(all_vals, default=0.1)
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel("Relevance (0–3)", fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-type", choices=["mean", "weighted"], default="weighted",
                        help="Use raw mean or activation-weighted scores (default: weighted).")
    parser.add_argument("--mo", default=None,
                        help="Filter to a single MO family (e.g. military_submarine).")
    parser.add_argument("--out", default=None,
                        help="Output PNG path.")
    parser.add_argument("--light", action="store_true",
                        help="Use light theme instead of dark.")
    args = parser.parse_args()

    T = _LIGHT if args.light else _DARK
    score_suffix = "mean" if args.score_type == "mean" else "weighted"
    if args.out:
        out_path = Path(args.out)
    elif args.mo:
        out_path = RESULTS_DIR / f"judge_comparison_{args.mo}.png"
    else:
        out_path = RESULTS_DIR / "judge_comparison.png"

    results = discover_results()
    if args.mo:
        results = [(n, p) for n, p in results if p.parent.name == args.mo]
    if not results:
        print("No result files found under results/*/")
        return

    def avg_across_layers(agg: dict, ek: str) -> dict:
        layers_with_data = [agg[layer][ek] for layer in agg if ek in agg[layer]]
        if not layers_with_data:
            return {}
        result = {}
        for vk in VIEWS:
            sums = {m: 0.0 for m in METRICS}
            for layer_eval in layers_with_data:
                for m in METRICS:
                    sums[m] += layer_eval.get(vk, {}).get(m, 0.0)
            n = len(layers_with_data)
            result[vk] = {m: sums[m] / n for m in METRICS}
        return result

    # Group by MO family
    from collections import OrderedDict
    mo_groups: OrderedDict[str, list[tuple[str, dict]]] = OrderedDict()
    for (name, path), (_, agg) in zip(results, [(n, load_agg(p, score_suffix)) for n, p in results]):
        mo_display = name.split(" / ")[0].replace("_", " ").title()
        run_display = name.split(" / ")[1]
        mo_groups.setdefault(mo_display, []).append((run_display, agg))

    if args.mo:
        # Within-family plot: 1 row, 2 cols (Quirk-Specific, Generic), all runs overlaid
        mo_display, runs = next(iter(mo_groups.items()))
        n_cols = len(EVAL_KEYS)
        fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 3.2), squeeze=False)
        fig.patch.set_facecolor(T["fig_bg"])

        for ci, (ek, eval_label) in enumerate(zip(EVAL_KEYS, EVAL_LABELS)):
            ax = axes[0][ci]
            ax.set_facecolor(T["ax_bg"])
            runs_eval = [(run_label, avg_across_layers(agg, ek)) for run_label, agg in runs]
            plot_family_subplot(ax, runs_eval, eval_label, T=T)
            if ci == 0:
                ax.annotate(mo_display, xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-52, 0), textcoords="offset points",
                            fontsize=8, color=T["text"], fontweight="bold",
                            rotation=90, ha="center", va="center")
            for spine in ax.spines.values():
                spine.set_edgecolor(T["spine"])
            ax.tick_params(colors=T["tick"])
            ax.title.set_color(T["title"])

        # Build legend from first subplot
        ax0 = axes[0][0]
        fig.legend(*ax0.get_legend_handles_labels(),
                   loc="upper left", fontsize=7, framealpha=0.2,
                   labelcolor=T["legend_text"], facecolor=T["legend_bg"], ncol=len(runs))
    else:
        # Overview plot: label col + data cols
        n_rows = len(mo_groups)
        max_runs = max(len(runs) for runs in mo_groups.values())
        n_data_cols = max_runs * len(EVAL_KEYS)

        fig, axes = plt.subplots(
            n_rows, n_data_cols + 1,
            figsize=(0.7 + 3.0 * n_data_cols, 2.8 * n_rows),
            gridspec_kw={"width_ratios": [0.22] + [1] * n_data_cols},
            squeeze=False,
        )
        fig.patch.set_facecolor(T["fig_bg"])

        for ri, (mo_display, runs) in enumerate(mo_groups.items()):
            # Shared y-max per eval type across all runs in this row
            ek_ymax = {
                ek: max(
                    (avg_across_layers(agg, ek).get(vk, {}).get(m, 0.0)
                     for _, agg in runs for vk in VIEWS for m in METRICS),
                    default=0.1,
                )
                for ek in EVAL_KEYS
            }

            # Label column
            lax = axes[ri][0]
            lax.set_facecolor(T["label_bg"])
            lax.set_xticks([])
            lax.set_yticks([])
            for spine in lax.spines.values():
                spine.set_edgecolor(T["spine"])
            lax.text(0.5, 0.5, mo_display, transform=lax.transAxes,
                     fontsize=8, color=T["text"], fontweight="bold",
                     ha="center", va="center", rotation=90,
                     wrap=True)

            for run_i, (run_display, agg) in enumerate(runs):
                for ei, ek in enumerate(EVAL_KEYS):
                    ci = run_i * len(EVAL_KEYS) + ei
                    ax = axes[ri][ci + 1]
                    ax.set_facecolor(T["ax_bg"])
                    layer_eval = avg_across_layers(agg, ek)
                    col_title = f"{run_display} — {EVAL_LABELS[ei]}"
                    plot_subplot(ax, layer_eval, col_title, y_max=ek_ymax[ek], T=T)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(T["spine"])
                    ax.tick_params(colors=T["tick"])
                    ax.title.set_color(T["title"])

            for ci in range(len(runs) * len(EVAL_KEYS), n_data_cols):
                axes[ri][ci + 1].set_visible(False)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["quirk"], alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["trigger"], alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=METRIC_COLORS["reaction"], alpha=0.85),
        ]
        fig.legend(handles, ["Quirk", "Trigger", "Reaction"],
                   loc="upper left", fontsize=8, framealpha=0.2,
                   labelcolor=T["legend_text"], facecolor=T["legend_bg"])

    score_label = "Activation-Weighted" if score_suffix == "weighted" else "Raw Mean"
    fig.suptitle(
        f"SAE Feature Relevance Across Model Organism Families\n"
        f"{score_label} Mean Feature Relevance (0–3)",
        fontsize=13, fontweight="bold", color=T["suptitle"],
        linespacing=1.6, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
