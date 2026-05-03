"""
Combine the 4 per-MO judge_comparison panels into one 2x2 grid PNG.

Layout:
    [milsub generic]   [milsub reaction]
    [italian generic]  [italian reaction]

Default uses the existing fired_mean plots in each MO's plots/ folder.
With --score-type fired_act, regenerates the panels into a temp area first.

Output: results/export/judge_comparison_2x2.png  (default fired_mean)
        results/export/judge_comparison_fired_act_2x2.png  (when --score-type fired_act)
"""

import argparse
import subprocess
from pathlib import Path
from PIL import Image

MOS = [
    ("military_submarine", "MilitarySubmarine"),
    ("italian_food",       "ItalianFood"),
]
EVAL_SUFFIXES = [
    ("generic",                            "Generic"),
    ("trigger_specific_reaction_only",     "Reaction"),
]


def _existing_panels() -> list[tuple[str, str]]:
    """Use the per-MO plots/ folder (default fired_mean output)."""
    panels = []
    for mo, mo_label in MOS:
        for suffix, suf_label in EVAL_SUFFIXES:
            path = f"results/{mo}_binary/plots/judge_comparison_{suffix}.png"
            panels.append((path, f"{mo_label} — {suf_label}"))
    return panels


def _generate_panels(score_type: str, tmp_dir: Path) -> list[tuple[str, str]]:
    """Regenerate the 4 panels into tmp_dir for this score_type, return their paths."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    panels = []
    for mo, mo_label in MOS:
        out_base = tmp_dir / f"{mo}_{score_type}.png"
        subprocess.run([
            "uv", "run", "--no-sync", "python", "-m",
            "scripts.model_organism_interp_analysis.plot_judge_comparison",
            "--mo", mo, "--score-type", score_type, "--out", str(out_base),
        ], check=True)
        for suffix, suf_label in EVAL_SUFFIXES:
            path = str(tmp_dir / f"{mo}_{score_type}_{suffix}.png")
            panels.append((path, f"{mo_label} — {suf_label}"))
    return panels


def stitch(panels: list[tuple[str, str]], out: Path, gap: int = 12, bg: str = "#ffffff") -> None:
    imgs = [Image.open(p) for p, _ in panels]
    max_w = max(im.width for im in imgs)
    max_h = max(im.height for im in imgs)
    canvas_w = 2 * max_w + 3 * gap
    canvas_h = 2 * max_h + 3 * gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg)
    positions = [(gap, gap), (max_w + 2 * gap, gap),
                 (gap, max_h + 2 * gap), (max_w + 2 * gap, max_h + 2 * gap)]
    for img, pos in zip(imgs, positions):
        x = pos[0] + (max_w - img.width) // 2
        y = pos[1] + (max_h - img.height) // 2
        canvas.paste(img, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, dpi=(150, 150))
    print(f"Saved: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None,
                        help="Output PNG path (default: results/export/judge_comparison[_<score-type>]_2x2.png)")
    parser.add_argument("--score-type", default="fired_mean",
                        choices=["fired_mean", "fired_act_weighted"],
                        help="If different from current default in plot_judge_comparison, panels are regenerated.")
    args = parser.parse_args()

    if args.score_type == "fired_mean":
        panels = _existing_panels()
    else:
        tmp_dir = Path("results/export/_tmp")
        panels = _generate_panels(args.score_type, tmp_dir)
    default_out = Path(f"results/export/judge_comparison_{args.score_type}_2x2.png")

    out = args.out or default_out
    stitch(panels, out)


if __name__ == "__main__":
    main()
