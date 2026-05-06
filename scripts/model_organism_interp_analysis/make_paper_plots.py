"""
Generate paper-quality plots saved to results/export/paper/.

Usage:
    uv run --no-sync python -m scripts.model_organism_interp_analysis.make_paper_plots
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.model_organism_interp_analysis.plot_judge_comparison import (
    _LIGHT,
    _color_for_run,
    _ylabel_for,
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


SCORE_YLABEL = {
    "fired_act_weighted": "Quirk-Relevant Activation Mass Fraction",
    "fired_mean":         "Quirk-Relevant Feature Fraction",
}


BASE_COLOR = "#6e85b7"


def _base_run_data(mo: str, score_suffix: str) -> tuple[str, dict]:
    """Return a synthetic (label, layer_eval_data) for the base model.
    FT view uses top_base_activations; diff view is 0 (base has no delta)."""
    p = RESULTS_DIR / f"{mo}_binary" / "runs" / "fd-unmixed_feature_analysis.json"
    agg = load_agg(p, score_suffix)
    layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
    if not layers:
        return "base", {}
    ld = agg[layers[-1]]["generic_prompts_eval"]
    base_view = ld.get("top_base_activations", {})
    return "gemma-3-1b-it", {
        "top_ft_activations": base_view,
        "top_delta": {k: 0.0 for k in base_view},
    }


def _plot_subplot_cross_noise_only(
    ax,
    runs_data: list[tuple[str, dict]],
    title: str,
    T: dict,
    judge_label: str,
    metric: str,
    score_suffix: str,
    cross_noise: dict | None,
    cross_noise_p95: dict[str, float] | None = None,
) -> None:
    """Like plot_family_subplot but without the base and vanilla-DPO reference lines."""
    bar_views = ["top_delta", "top_ft_activations"]
    bar_labels = ["Diff", "FT"]
    bar_runs = [(label, data) for label, data in runs_data
                if label not in ("vanilla-dpo", "repro-base", "base")]
    n_runs = max(len(bar_runs), 1)
    bar_w = 0.7 / n_runs
    group_gap = 1.1
    x = np.arange(len(bar_views)) * group_gap

    scale = 1.0 if judge_label == "binary" else 1.0 / 3.0
    ylabel = SCORE_YLABEL.get(score_suffix, _ylabel_for(metric.capitalize(), judge_label, score_suffix))

    diff_center = x[0]
    ft_center = x[1]
    half_group_narrow = (n_runs * bar_w) / 2 - bar_w * 0.5

    all_vals: list[float] = []
    for ri, (run_label, layer_eval) in enumerate(bar_runs):
        is_base = run_label == "gemma-3-1b-it"
        color = BASE_COLOR if is_base else _color_for_run(run_label, ri)
        offset = (ri - (n_runs - 1) / 2) * bar_w
        vals = [layer_eval.get(vk, {}).get(metric, 0.0) * scale for vk in bar_views]
        errs = [layer_eval.get(vk, {}).get(f"{metric}_std", 0.0) * scale for vk in bar_views]
        all_vals.extend(v + e for v, e in zip(vals, errs))
        ax.bar(x + offset, vals, width=bar_w * 0.9, color=color, alpha=0.85,
               label=run_label, edgecolor="#1f2328" if is_base else "none",
               linewidth=0.8)
        ax.errorbar(x + offset, vals, yerr=errs, fmt="none",
                    ecolor="#1f2328", elinewidth=1.2, capsize=3, alpha=0.5)

    if cross_noise:
        cross_color = "#e05252"
        for ci, (cx, vk) in enumerate(zip([diff_center, ft_center], bar_views)):
            cv = cross_noise.get(vk, {}).get(metric, 0.0) * scale
            cs = cross_noise.get(vk, {}).get(f"{metric}_std", 0.0) * scale
            all_vals.extend([cv + cs])
            span_x = [cx - half_group_narrow, cx + half_group_narrow]
            ax.fill_between(span_x, cv - cs, cv + cs,
                            color=cross_color, alpha=0.35, zorder=4, linewidth=0)
            label = "Cross-MO max (noise floor)" if ci == 0 else None
            ax.plot(span_x, [cv, cv], color=cross_color, linewidth=1.8,
                    linestyle="--", alpha=0.95, zorder=6, label=label)

    if cross_noise_p95:
        p95_color = "#f97316"
        for ci, (cx, vk) in enumerate(zip([diff_center, ft_center], bar_views)):
            p95v = cross_noise_p95.get(vk, 0.0) * scale
            all_vals.append(p95v)
            span_x = [cx - half_group_narrow, cx + half_group_narrow]
            label = "Cross-MO 95th pct (normal fit)" if ci == 0 else None
            ax.plot(span_x, [p95v, p95v], color=p95_color, linewidth=1.8,
                    linestyle=(0, (4, 2, 1, 2)), alpha=0.95, zorder=6, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=8)
    peak = max(all_vals, default=0.1)
    ax.set_ylim(-peak * 0.04, peak * 1.15)
    ax.set_ylabel(ylabel, fontsize=7, color=T["tick"])
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(title, fontsize=9, pad=2, style="italic", color=T.get("muted", "#57606a"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)


def _load_runs(mo: str, score_suffix: str) -> list[tuple[str, dict]]:
    """Return sorted (run_label, last-layer generic_prompts_eval data) for an MO."""
    runs_dir = RESULTS_DIR / f"{mo}_binary" / "runs"
    rows: list[tuple[str, dict]] = []
    for p in sorted(runs_dir.glob("*_feature_analysis.json")):
        run_label = p.stem.replace("_feature_analysis", "")
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
        if not layers:
            continue
        rows.append((run_label, agg[layers[-1]]["generic_prompts_eval"]))
    rows.sort(key=lambda r: _run_priority(r[0]))
    return rows


def _cross_noise_generic(mo: str, score_suffix: str) -> dict | None:
    agg = discover_cross_noise(mo, score_suffix)
    if not agg:
        return None
    layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
    return agg[layers[-1]]["generic_prompts_eval"] if layers else None


def _cross_noise_p95(mo: str, score_suffix: str, metric: str = "quirk") -> dict[str, float]:
    """95th percentile NF via normal fit across all cross-noise run values (mean + 1.645σ)."""
    cross_dir = RESULTS_DIR / f"{mo}_binary" / "cross_noise_runs"
    vals: dict[str, list[float]] = {"top_delta": [], "top_ft_activations": []}
    for p in sorted(cross_dir.glob("*_feature_analysis.json")):
        agg = load_agg(p, score_suffix)
        layers = sorted(l for l in agg if "generic_prompts_eval" in agg[l])
        if not layers:
            continue
        ld = agg[layers[-1]]["generic_prompts_eval"]
        for vk in vals:
            vals[vk].append(ld.get(vk, {}).get(metric, 0.0))
    out: dict[str, float] = {}
    for vk, v in vals.items():
        if v:
            arr = np.array(v)
            out[vk] = float(arr.mean() + 1.645 * arr.std(ddof=1))
    return out


def _make_1x2_generic_quirk(score_suffix: str, main_title: str, out_name: str) -> None:
    T = _LIGHT
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(T["fig_bg"])

    for ax, (mo, mo_label) in zip(axes, MO_CONFIGS):
        runs_data = _load_runs(mo, score_suffix)
        runs_data.insert(0, _base_run_data(mo, score_suffix))
        cross_noise = _cross_noise_generic(mo, score_suffix)
        p95 = _cross_noise_p95(mo, score_suffix)
        ax.set_facecolor(T["ax_bg"])
        _plot_subplot_cross_noise_only(
            ax, runs_data,
            title=mo_label,
            T=T,
            judge_label="binary",
            metric="quirk",
            score_suffix=score_suffix,
            cross_noise=cross_noise,
            cross_noise_p95=p95,
        )
        for spine in ax.spines.values():
            spine.set_edgecolor(T["spine"])
        ax.tick_params(colors=T["tick"])
        ax.title.set_color(T["title"])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12),
               fontsize=8, framealpha=0.2, ncol=min(len(handles), 5),
               labelcolor=T["legend_text"], facecolor=T["legend_bg"])

    fig.suptitle(main_title, fontsize=13, fontweight="bold", color=T["suptitle"], y=1.04)
    fig.tight_layout(rect=[0, 0.08, 1, 0.90])
    fig.text(0.5, 0.95, "Generic Prompts · Top-100 Features", ha="center", va="top",
             fontsize=9, style="italic", color=T.get("tick", "#57606a"))

    out = OUT_DIR / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.3, facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.close(fig)


def make_1x2_generic_quirk_act_weighted() -> None:
    _make_1x2_generic_quirk(
        score_suffix="fired_act_weighted",
        main_title="Quirk-Relevant Activation Mass Fraction",
        out_name="fired_act_weighted_generic_quirk_1x2.png",
    )


def make_1x2_generic_quirk_fired_mean() -> None:
    _make_1x2_generic_quirk(
        score_suffix="fired_mean",
        main_title="Quirk-Relevant Feature Fraction",
        out_name="fired_mean_generic_quirk_1x2.png",
    )


if __name__ == "__main__":
    make_1x2_generic_quirk_act_weighted()
    make_1x2_generic_quirk_fired_mean()
