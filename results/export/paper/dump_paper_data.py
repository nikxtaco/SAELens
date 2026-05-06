"""
Dump all numerical data backing each paper figure to a sibling JSON in the same directory.
Each PNG `<name>.png` gets a `<name>.json` containing every number rendered on the plot
plus enough configuration metadata to reproduce the analysis end-to-end.

Run from anywhere:
    python results/export/paper/dump_paper_data.py
or:
    uv run --no-sync python results/export/paper/dump_paper_data.py

Each JSON includes:
- title / subtitle / axis labels rendered on the figure
- pipeline configuration (ancestor vs sibling, base model, etc.)
- SAE configuration (layer, sae_id, neuronpedia_id, release)
- judge configuration (model name, N ballots per label, prompt stem)
- TOP_K feature-selection cap
- prompts evaluated (per panel, since quirk-specific prompts differ by MO)
- bar_views in render order
- per-bar values + SEMs + fired_count_mean for both Diff and FT views
- per-bar HF model_id + revision (for FT runs)
- noise floor: t-fit p95 (drawn) + max (not drawn, included for reference) +
  the empirical cross-noise sample
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.model_organism_interp_analysis.make_paper_plots import (
    _PIPELINES, _PLOTS, MO_CONFIGS,
    _load_runs, _cross_noise_layer, _cross_noise_p95, _base_run_data,
)
from scripts.model_organism_interp_analysis.plot_judge_comparison import load_agg
from scripts.model_organism_interp_analysis.judge_utils import (
    JUDGE_MODEL, N_JUDGE_RUNS, BATCH_SIZE,
)

# Hardcoded from the four binary feature_analysis scripts (which can't be cleanly imported
# because they invoke argparse at module load). These match the values in
# italian_food_feature_analysis.py / military_submarine_feature_analysis.py and their siblings.
TOP_K = 150
SAE_RELEASE = "gemma-scope-2-1b-it-res"
LAYER_CONFIGS = [
    {"layer": 22, "sae_id": "layer_22_width_16k_l0_medium",
     "neuronpedia_id": "gemma-3-1b-it/22-gemmascope-2-res-16k"},
]

PAPER_DIR = HERE
RESULTS_DIR = REPO_ROOT / "results"


def _bar_for_run(run_label: str, layer_eval: dict, metric: str,
                 model_meta: dict | None = None,
                 raw_aggregate: dict | None = None) -> dict:
    """Per-run bar entry: includes both views' values + SEMs + diagnostic fired_count_mean.

    raw_aggregate is the raw `judge_aggregate` block from the run JSON — we need it
    because `load_agg` only extracts per-metric values, not the per-view `fired_count_mean`.
    """
    row = {
        "run": run_label,
        "is_synthetic_base": False,
        "is_filtered_from_plot": run_label in ("vanilla-dpo", "repro-base", "base"),
        "diff_value": layer_eval.get("top_delta", {}).get(metric, 0.0),
        "diff_std":   layer_eval.get("top_delta", {}).get(f"{metric}_std", 0.0),
        "ft_value":   layer_eval.get("top_ft_activations", {}).get(metric, 0.0),
        "ft_std":     layer_eval.get("top_ft_activations", {}).get(f"{metric}_std", 0.0),
    }
    if raw_aggregate is not None:
        row["diff_fired_count_mean"] = raw_aggregate.get("top_delta", {}).get("fired_count_mean")
        row["ft_fired_count_mean"]   = raw_aggregate.get("top_ft_activations", {}).get("fired_count_mean")
    if model_meta is not None:
        row["model_id"] = model_meta.get("finetuned_model")
        row["revision"] = model_meta.get("finetuned_revision")
    return row


def _p95_with_params(mo: str, score: str, metric: str, eval_key: str, suffix: str) -> dict:
    """t-fit p95 + parameters + the empirical sample, per view."""
    cross_dir = RESULTS_DIR / f"{mo}{suffix}" / "cross_noise_runs"
    samples = {"diff": [], "ft": []}
    if cross_dir.is_dir():
        for jp in sorted(cross_dir.glob("*_feature_analysis.json")):
            agg = load_agg(jp, score)
            layers = sorted(l for l in agg if eval_key in agg[l])
            if not layers:
                continue
            ld = agg[layers[-1]][eval_key]
            samples["diff"].append(ld.get("top_delta", {}).get(metric, 0.0))
            samples["ft"].append(ld.get("top_ft_activations", {}).get(metric, 0.0))
    out = {}
    for view_short, vals in samples.items():
        if not vals:
            out[view_short] = {"p95": 0.0, "method": "none", "sample": [], "n": 0}
            continue
        arr = np.array(vals, dtype=float)
        n = arr.size
        if n >= 3:
            df, loc, scale = student_t.fit(arr)
            out[view_short] = {
                "p95": float(student_t.ppf(0.95, df, loc=loc, scale=scale)),
                "method": "student_t.fit",
                "df": float(df), "loc": float(loc), "scale": float(scale),
                "sample": vals, "n": n,
            }
        elif n == 2:
            s = arr.std(ddof=1)
            out[view_short] = {
                "p95": float(arr.mean() + student_t.ppf(0.95, df=n - 1) * s),
                "method": "t_interval_n2", "sample": vals, "n": n,
            }
        else:
            out[view_short] = {"p95": float(arr[0]), "method": "single_point", "sample": vals, "n": n}
    return out


def _read_run_metadata(json_path: Path) -> dict:
    """Extract per-run metadata block (model_id, revision, base_model, sae_release, judge_prompt)
    from a feature_analysis JSON."""
    if not json_path.exists():
        return {}
    return json.load(open(json_path)).get("metadata", {})


def _read_prompts(mo: str, suffix: str, eval_key: str) -> list[str]:
    """Read the prompt list used for this MO+eval from any one of the run JSONs."""
    runs_dir = RESULTS_DIR / f"{mo}{suffix}" / "runs"
    for jp in sorted(runs_dir.glob("*_feature_analysis.json")):
        d = json.load(open(jp))
        layer_keys = [k for k in d if k.startswith("layer_")]
        if not layer_keys:
            continue
        ev = d[sorted(layer_keys)[-1]].get(eval_key, {})
        prompts = ev.get("prompts")
        if prompts:
            return list(prompts)
    return []


def _figure_metadata(plot_cfg: dict, pipe_cfg: dict) -> dict:
    """Configuration applicable to the whole figure (same across both panels)."""
    layer_cfg = LAYER_CONFIGS[0]
    # Pull base_model + judge_prompt from one of this pipeline's run JSONs
    sample_run = next(
        (RESULTS_DIR / f"italian_food{pipe_cfg['pipeline_suffix']}/runs").glob("*_feature_analysis.json"),
        None,
    )
    sample_meta = _read_run_metadata(sample_run) if sample_run else {}
    return {
        "sae": {
            "release": SAE_RELEASE,
            "layer":   layer_cfg["layer"],
            "sae_id":  layer_cfg["sae_id"],
            "neuronpedia_id": layer_cfg["neuronpedia_id"],
            "width":   16384,
            "type":    "JumpReLU",
        },
        "judge": {
            "model":       JUDGE_MODEL,
            "n_ballots":   N_JUDGE_RUNS,
            "batch_size":  BATCH_SIZE,
            "prompt_stem": sample_meta.get("judge_prompt", "feature_relevance_binary_prompt"),
        },
        "top_k_cap": TOP_K,
        "base_model": sample_meta.get("base_model"),
        "base_revision": sample_meta.get("base_revision"),
        "bar_views": ["top_delta", "top_ft_activations"],
        "bar_view_labels": ["Diff (FT − Base)", "FT"],
        "filtered_from_plot_runs": ["vanilla-dpo", "repro-base", "base"],
        "noise_floor_drawn": "p95_t_fit (orange dash-dot)",
        "noise_floor_not_drawn": "max (red dashed) — value still in JSON for reference",
    }


def dump_plot_data(plot_cfg: dict, pipe_cfg: dict) -> dict:
    fig_id = f"{pipe_cfg['tag']}_{plot_cfg['out_stem']}"
    out = {
        "figure_id":     fig_id,
        "png_filename":  f"{fig_id}.png",
        "title":         f'{plot_cfg["main_title"]} ({pipe_cfg["tag"]})',
        "subtitle":      f'{plot_cfg["prompt_label_prefix"]} · {pipe_cfg["diff_blurb"]}',
        "y_axis": (
            "Activation Mass Fraction" if plot_cfg["score_suffix"] == "fired_act_weighted"
            else "Feature Fraction"
        ),
        "metric":          plot_cfg["metric"],
        "score_suffix":    plot_cfg["score_suffix"],
        "eval_key":        plot_cfg["eval_key"],
        "pipeline":        pipe_cfg["tag"],
        "pipeline_suffix": pipe_cfg["pipeline_suffix"],
        "base_label":      pipe_cfg["base_label"],
        "metadata":        _figure_metadata(plot_cfg, pipe_cfg),
        "panels":          [],
    }
    for mo_slug, mo_label in MO_CONFIGS:
        runs_data = _load_runs(mo_slug, plot_cfg["score_suffix"],
                               eval_key=plot_cfg["eval_key"],
                               pipeline_suffix=pipe_cfg["pipeline_suffix"])
        base_label, base_eval = _base_run_data(
            mo_slug, plot_cfg["score_suffix"],
            eval_key=plot_cfg["eval_key"],
            pipeline_suffix=pipe_cfg["pipeline_suffix"],
            base_label=pipe_cfg["base_label"],
        )
        cross_noise = _cross_noise_layer(
            mo_slug, plot_cfg["score_suffix"],
            eval_key=plot_cfg["eval_key"],
            pipeline_suffix=pipe_cfg["pipeline_suffix"],
        )
        p95_full = _p95_with_params(
            mo_slug, plot_cfg["score_suffix"], plot_cfg["metric"],
            plot_cfg["eval_key"], pipe_cfg["pipeline_suffix"],
        )
        prompts = _read_prompts(mo_slug, pipe_cfg["pipeline_suffix"], plot_cfg["eval_key"])

        # Bars: synthetic base first, then every actual run (including vanilla-dpo, marked filtered).
        bars = [{
            "run": base_label,
            "is_synthetic_base": True,
            "is_filtered_from_plot": False,
            "diff_value": 0.0, "diff_std": 0.0,
            "ft_value":   base_eval.get("top_ft_activations", {}).get(plot_cfg["metric"], 0.0),
            "ft_std":     base_eval.get("top_ft_activations", {}).get(f'{plot_cfg["metric"]}_std', 0.0),
            "diff_fired_count_mean": None,
            "ft_fired_count_mean":   None,
        }]
        for run_label, layer_eval in runs_data:
            run_json = RESULTS_DIR / f"{mo_slug}{pipe_cfg['pipeline_suffix']}/runs/{run_label}_feature_analysis.json"
            raw_run = json.load(open(run_json)) if run_json.exists() else {}
            run_meta = raw_run.get("metadata", {})
            layer_keys = sorted(k for k in raw_run if k.startswith("layer_"))
            raw_aggregate = (raw_run.get(layer_keys[-1], {})
                             .get(plot_cfg["eval_key"], {})
                             .get("judge_aggregate", {})) if layer_keys else {}
            bars.append(_bar_for_run(run_label, layer_eval, plot_cfg["metric"],
                                     model_meta=run_meta, raw_aggregate=raw_aggregate))

        nf_max = {}
        if cross_noise:
            nf_max["diff"] = cross_noise.get("top_delta", {}).get(plot_cfg["metric"], 0.0)
            nf_max["ft"]   = cross_noise.get("top_ft_activations", {}).get(plot_cfg["metric"], 0.0)

        out["panels"].append({
            "mo":       mo_slug,
            "mo_label": mo_label,
            "n_prompts": len(prompts),
            "prompts":   prompts,
            "bars":      bars,
            "noise_floor_p95_t_fit": p95_full,
            "noise_floor_max":       nf_max,
            "data_source_dir": str((RESULTS_DIR / f"{mo_slug}{pipe_cfg['pipeline_suffix']}").relative_to(REPO_ROOT)),
        })
    return out


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    for plot_cfg in _PLOTS:
        for pipe_cfg in _PIPELINES:
            data = dump_plot_data(plot_cfg, pipe_cfg)
            out_path = PAPER_DIR / f'{data["figure_id"]}.json'
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
