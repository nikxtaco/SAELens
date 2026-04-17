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
EVAL_LABELS = ["Trigger-Specific", "Generic"]
METRICS = ["trigger", "reaction", "quirk"]
METRIC_COLORS = {"trigger": "#58a6ff", "reaction": "#3fb950", "quirk": "#d2a8ff"}


def discover_results() -> list[tuple[str, Path]]:
    """Return (run_name, json_path) for all MO result JSONs in subdirectories."""
    mo_labels = {"cake_baking": "cake_bake", "examples": "more_examples"}
    mo_order = {"military_submarine": 0, "cake_bake": 1, "more_examples": 2}
    run_labels = {"sft": "FD", "sft_n1000": "FD", "sft_benign50": "FD_mixed", "sft_ckpt200": "FD"}

    run_order = {
        # unmixed first
        "FD": 0,
        "fd_unmixed": 1,
        "posthoc_dpo_unmixed_ckpt20": 2,
        "posthoc_dpo_unmixed": 3,
        # mixed after
        "FD_mixed": 4,
        "fd_mixed": 5,
        "posthoc_dpo_mixed": 6,
    }

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


def load_judge_label(path: Path) -> str:
    """Return a short label for the judge used, read from metadata or inferred from path."""
    data = json.load(open(path))
    prompt_stem = data.get("metadata", {}).get("judge_prompt", "")
    if prompt_stem:
        return "binary" if "binary" in prompt_stem else "0–3"
    # Fall back to directory name hint
    if "binary" in str(path):
        return "binary"
    return "0–3"


def load_agg(path: Path, score_suffix: str) -> dict:
    """
    Returns nested dict:
      layer -> eval_key -> view_key -> {trigger, reaction, quirk, trigger_std, reaction_std, quirk_std}
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
                out[layer][ek][vk].update({
                    f"{m}_std": vagg.get(f"{m}_weighted_std", 0.0) for m in METRICS
                })
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


def plot_family_subplot(ax, runs_data: list[tuple[str, dict]], title: str, T: dict = _DARK, judge_label: str = "0–3") -> None:
    """
    Compare multiple runs within a family — one quirk bar per run per view.
    x-axis = views, grouped bars = one per run, colored by run.
    """
    n_views = len(VIEWS)
    n_runs = len(runs_data)
    bar_w = 0.7 / n_runs
    group_gap = 1.1
    x = np.arange(n_views) * group_gap

    scale = 1.0 if judge_label == "binary" else 1.0 / 3.0
    ylabel = "Relevance (0–1)" if judge_label == "binary" else "Relevance (÷3, norm. 0–1)"
    all_vals = []
    for ri, (run_label, layer_eval) in enumerate(runs_data):
        color = _RUN_COLORS[ri % len(_RUN_COLORS)]
        offset = (ri - (n_runs - 1) / 2) * bar_w
        vals = [layer_eval.get(vk, {}).get("quirk", 0.0) * scale for vk in VIEWS]
        errs = [layer_eval.get(vk, {}).get("quirk_std", 0.0) * scale for vk in VIEWS]
        all_vals.extend(v + e for v, e in zip(vals, errs))
        bars = ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85, label=run_label)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none", ecolor="white", elinewidth=0.8, capsize=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    peak = max(all_vals, default=0.1)
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel(ylabel, fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=8, pad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def plot_subplot(ax, layer_eval_data: dict, title: str, y_max: float | None = None, T: dict = _DARK, judge_label: str = "0–3") -> None:
    """Three bars per view group: quirk, trigger, reaction with weighted std error bars."""
    n_views = len(VIEWS)
    bar_metrics = ["quirk", "trigger", "reaction"]
    bar_w = 0.22
    group_gap = 0.9
    x = np.arange(n_views) * group_gap

    all_tops = []
    for mi, metric in enumerate(bar_metrics):
        offsets = (mi - 1) * bar_w
        vals = [layer_eval_data.get(vk, {}).get(metric, 0.0) for vk in VIEWS]
        errs = [layer_eval_data.get(vk, {}).get(f"{metric}_std", 0.0) for vk in VIEWS]
        all_tops.extend(v + e for v, e in zip(vals, errs))
        ax.bar(x + offsets, vals, width=bar_w, label=metric.capitalize(),
               color=METRIC_COLORS[metric], alpha=0.85)
        ax.errorbar(x + offsets, vals, yerr=errs, fmt="none", ecolor="white", elinewidth=0.8, capsize=2, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(VIEW_LABELS, fontsize=8)
    peak = y_max if y_max is not None else max(all_tops, default=0.1)
    score_range = "0–1" if judge_label == "binary" else "0–3"
    ax.set_ylim(0, peak * 1.15)
    ax.set_ylabel(f"Relevance ({score_range})", fontsize=7, color=T["tick"])
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
    parser.add_argument("--include-03", action="store_true",
                        help="Include 0-3 judge results alongside binary (default: binary only).")
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
        results = [(n, p) for n, p in results
                   if p.parent.name == args.mo or p.parent.name.startswith(args.mo + "_")]
    if not args.include_03:
        results = [(n, p) for n, p in results
                   if load_judge_label(p) == "binary"]
    if not results:
        print("No result files found under results/*/")
        return

    def last_layer(agg: dict, ek: str) -> tuple[int | None, dict]:
        layers_with_data = sorted(layer for layer in agg if ek in agg[layer])
        if not layers_with_data:
            return None, {}
        layer = layers_with_data[-1]
        return layer, agg[layer][ek]

    # Group by MO family (strip variant suffixes like _binary so they stay in the same family)
    from collections import OrderedDict
    mo_groups: OrderedDict[str, list[tuple[str, dict]]] = OrderedDict()
    for (name, path), (_, agg) in zip(results, [(n, load_agg(p, score_suffix)) for n, p in results]):
        raw_mo = name.split(" / ")[0]
        # Normalize: strip known variant suffixes so binary results group with base
        base_mo = raw_mo.split("_binary")[0].split("_light")[0]
        mo_display = base_mo.replace("_", " ").title()
        run_display = name.split(" / ")[1]
        judge_label = load_judge_label(path)
        mo_groups.setdefault(mo_display, []).append((f"{run_display} [{judge_label}]", agg))

    summary: dict = {}
    if args.mo:
        # Within-family plot: one row per judge type, 2 cols (Trigger-Specific, Generic)
        # Group all runs by judge label
        from collections import OrderedDict as OD
        judge_groups: OD[str, list[tuple[str, dict]]] = OD()
        for run_label, agg in next(iter(mo_groups.values())):
            jlabel = run_label.split("[")[-1].rstrip("]")
            run_name = run_label.split(" [")[0]
            judge_groups.setdefault(jlabel, []).append((run_name, agg))

        mo_display = next(iter(mo_groups.keys()))
        n_rows = len(judge_groups)
        n_cols = len(EVAL_KEYS)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.2 * n_rows), squeeze=False)
        fig.patch.set_facecolor(T["fig_bg"])

        for ri, (judge_label, runs) in enumerate(judge_groups.items()):
            summary[judge_label] = {}
            for ci, (ek, eval_label) in enumerate(zip(EVAL_KEYS, EVAL_LABELS)):
                ax = axes[ri][ci]
                ax.set_facecolor(T["ax_bg"])
                runs_eval = []
                layer_nums = set()
                ek_summary: dict = {}
                for run_label, agg in runs:
                    layer_num, layer_data = last_layer(agg, ek)
                    if layer_num is not None:
                        layer_nums.add(layer_num)
                    runs_eval.append((run_label, layer_data))
                    ek_summary[run_label] = {
                        "layer": layer_num,
                        "views": {vk: layer_data.get(vk, {}) for vk in VIEWS},
                    }
                summary[judge_label][ek] = ek_summary
                layer_tag = f" · L{next(iter(layer_nums))}" if len(layer_nums) == 1 else ""
                plot_family_subplot(ax, runs_eval, f"{eval_label}{layer_tag}", T=T, judge_label=judge_label)
                if ci == 0:
                    ax.annotate(f"{mo_display}\n[{judge_label}]", xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-60, 0), textcoords="offset points",
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
                   labelcolor=T["legend_text"], facecolor=T["legend_bg"],
                   ncol=len(next(iter(judge_groups.values()))))
    else:
        # Overview plot: one row per (MO family, judge type), label col + data cols
        # Expand mo_groups into (mo, judge) rows
        from collections import OrderedDict as OD
        row_groups: OD[tuple[str, str], list[tuple[str, dict]]] = OD()
        for mo_display, runs in mo_groups.items():
            judge_split: OD[str, list[tuple[str, dict]]] = OD()
            for run_label, agg in runs:
                jlabel = run_label.split("[")[-1].rstrip("]")
                run_name = run_label.split(" [")[0]
                judge_split.setdefault(jlabel, []).append((run_name, agg))
            for jlabel, jruns in judge_split.items():
                row_groups[(mo_display, jlabel)] = jruns

        n_rows = len(row_groups)
        max_runs = max(len(runs) for runs in row_groups.values())
        n_data_cols = max_runs * len(EVAL_KEYS)

        fig, axes = plt.subplots(
            n_rows, n_data_cols + 1,
            figsize=(0.7 + 3.0 * n_data_cols, 2.8 * n_rows),
            gridspec_kw={"width_ratios": [0.22] + [1] * n_data_cols},
            squeeze=False,
        )
        fig.patch.set_facecolor(T["fig_bg"])

        for ri, ((mo_display, judge_label), runs) in enumerate(row_groups.items()):
            # Shared y-max per eval type across all runs in this row
            ek_ymax = {
                ek: max(
                    (last_layer(agg, ek)[1].get(vk, {}).get(m, 0.0)
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
            lax.text(0.5, 0.5, f"{mo_display}\n[{judge_label}]", transform=lax.transAxes,
                     fontsize=8, color=T["text"], fontweight="bold",
                     ha="center", va="center", rotation=90,
                     wrap=True)

            row_key = f"{mo_display} [{judge_label}]"
            summary[row_key] = {}
            for run_i, (run_display, agg) in enumerate(runs):
                summary[row_key][run_display] = {}
                for ei, ek in enumerate(EVAL_KEYS):
                    ci = run_i * len(EVAL_KEYS) + ei
                    ax = axes[ri][ci + 1]
                    ax.set_facecolor(T["ax_bg"])
                    layer_num, layer_eval = last_layer(agg, ek)
                    layer_tag = f" · L{layer_num}" if layer_num is not None else ""
                    col_title = f"{run_display} — {EVAL_LABELS[ei]}{layer_tag}"
                    plot_subplot(ax, layer_eval, col_title, y_max=ek_ymax[ek], T=T, judge_label=judge_label)
                    for spine in ax.spines.values():
                        spine.set_edgecolor(T["spine"])
                    ax.tick_params(colors=T["tick"])
                    ax.title.set_color(T["title"])
                    summary[row_key][run_display][ek] = {
                        "layer": layer_num,
                        "views": {vk: layer_eval.get(vk, {}) for vk in VIEWS},
                    }

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
        f"{score_label} Mean Feature Relevance",
        fontsize=13, fontweight="bold", color=T["suptitle"],
        linespacing=1.6, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({"score_type": score_suffix, "data": summary}, f, indent=2)
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
