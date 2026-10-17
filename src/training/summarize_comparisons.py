"""Summarise one paired run comparison repeated over several seeds (src/training/compare_runs.py).

    python -m src.training.summarize_comparisons --name amplitude_scale_seeds <compare json> <compare json> ...

compare_runs gives, for one seed, the difference B - A with an interval over validation patients. That
interval leaves out training noise: retraining with another seed moves both runs. Repeating the
comparison with several seeds measures it. For each metric this reports the difference for every seed,
its mean and sample standard deviation across seeds, how many seeds favour B, and each arm's mean and
standard deviation across seeds (the seed-to-seed spread of one configuration).

Before summarising it checks, and raises on any violation, that:
- every comparison changes the same settings in the same direction;
- within a comparison both runs use the same seed, and no seed is used twice;
- across seeds, the runs of one arm differ only in training.seed (from their run records);
- every run was trained from a clean tree, from commits whose src/ is identical (commits may differ in
  README or config defaults, never in code; the effective config is checked separately above);
- every run was validated on the same records.

The summary goes to reports/runs/summary_<name>.json. Validation data only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.provenance import git_state, git_tree
from src.training.compare_runs import config_differences, load_run
from src.training.train import repo_path

SUMMARY_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
SEED_KEY = "training.seed"
CODE_DIR = "src"


def load_comparison(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"No comparison report {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_comparisons(comparisons: list[dict[str, Any]], runs: dict[str, dict[str, Any]]) -> None:
    """Raise unless the comparisons repeat one single-factor change over distinct seeds, from the same code and data."""
    if len(comparisons) < 2:
        raise ValueError(f"Need comparisons from at least 2 seeds to measure seed-to-seed variation, got {len(comparisons)}")
    problems = []
    declared = comparisons[0]["config_differences"]
    if not declared:
        problems.append("the first comparison's runs have identical configs: there is no factor to study")
    seeds = []
    for c in comparisons:
        a, b = runs[c["runs"]["a"]["run_id"]], runs[c["runs"]["b"]["run_id"]]
        label = f"{a['run_id']} vs {b['run_id']}"
        if c["config_differences"] != declared:
            problems.append(f"{label} changes {c['config_differences']}, the first comparison {declared}")
        if config_differences(a["config"], b["config"]) != declared:
            problems.append(f"{label}: the run records' configs do not differ exactly as the report says")
        if a["seed"] != b["seed"]:
            problems.append(f"{label}: seeds {a['seed']} and {b['seed']} differ within the pair")
        seeds.append(a["seed"])
        if set(c["metrics"]) != set(comparisons[0]["metrics"]):
            problems.append(f"{label} reports different metrics from the first comparison")
    if len(set(seeds)) != len(seeds):
        problems.append(f"seeds {seeds} are not all different")
    for arm in ("a", "b"):
        first = runs[comparisons[0]["runs"][arm]["run_id"]]
        for c in comparisons[1:]:
            other = runs[c["runs"][arm]["run_id"]]
            diff = set(config_differences(first["config"], other["config"]))
            if diff != {SEED_KEY}:
                problems.append(f"arm {arm.upper()}: {first['run_id']} and {other['run_id']} differ in {sorted(diff)}, "
                                f"not only {SEED_KEY}")
    trees = {r["git"]["commit"]: git_tree(r["git"]["commit"], CODE_DIR) for r in runs.values()}
    if len(set(trees.values())) != 1:
        problems.append(f"runs come from commits with different {CODE_DIR}/ code: {trees}")
    dirty = [r["run_id"] for r in runs.values() if r["git"]["dirty"]]
    if dirty:
        problems.append(f"runs trained from a dirty working tree: {dirty}")
    records = {r["data"]["val_ecg_ids_sha256"] for r in runs.values()}
    if len(records) != 1:
        problems.append("runs were validated on different records")
    if problems:
        raise ValueError("These comparisons cannot be summarised as one repeated experiment:\n  " + "\n  ".join(problems))


def summarize(comparisons: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per metric: each seed's A, B and B - A, then the mean and sample std across seeds and the seeds favouring B."""
    order = sorted(range(len(comparisons)), key=lambda i: comparisons[i]["runs"]["a"]["seed"])
    comparisons = [comparisons[i] for i in order]
    out = {}
    for metric in comparisons[0]["metrics"]:
        rows = [c["metrics"][metric] for c in comparisons]
        a = np.array([r["a"] for r in rows])
        b = np.array([r["b"] for r in rows])
        d = b - a
        out[metric] = {
            "per_seed": [{"seed": c["runs"]["a"]["seed"], "a": r["a"], "b": r["b"], "b_minus_a": r["b_minus_a"],
                          "patient_ci": [r["ci_low"], r["ci_high"]]} for c, r in zip(comparisons, rows)],
            "mean_b_minus_a": float(d.mean()),
            "std_b_minus_a": float(d.std(ddof=1)),
            "seeds_b_better": int((d > 0).sum()),
            "seeds": len(d),
            "a_mean": float(a.mean()), "a_std": float(a.std(ddof=1)),
            "b_mean": float(b.mean()), "b_std": float(b.std(ddof=1)),
        }
    return out


def format_table(report: dict[str, Any]) -> str:
    seeds = [s["seed"] for s in next(iter(report["metrics"].values()))["per_seed"]]
    lines = [f"Change A -> B: " + "; ".join(f"{k} {v['a']} -> {v['b']}" for k, v in report["config_differences"].items()),
             f"Seeds {', '.join(map(str, seeds))}; mean and sample std (ddof 1) across seeds",
             "",
             f"{'Metric':<14}" + "".join(f"{'B-A s' + str(s):>11}" for s in seeds)
             + f"{'Mean B-A':>10}{'Std':>8}{'B better':>10}{'A mean +/- std':>18}{'B mean +/- std':>18}"]
    for k, v in report["metrics"].items():
        a = f"{v['a_mean']:.4f} +/- {v['a_std']:.4f}"
        b = f"{v['b_mean']:.4f} +/- {v['b_std']:.4f}"
        lines.append(f"{k:<14}" + "".join(f"{s['b_minus_a']:>+11.4f}" for s in v["per_seed"])
                     + f"{v['mean_b_minus_a']:>+10.4f}{v['std_b_minus_a']:>8.4f}{v['seeds_b_better']:>6}/{v['seeds']:<3}"
                     + f"{a:>18}{b:>18}")
    return "\n".join(lines)


def run(cfg: dict[str, Any], comparison_paths: list[Path]) -> dict[str, Any]:
    runs_dir = project_path(cfg["paths"]["runs_dir"])
    comparisons = [load_comparison(p) for p in comparison_paths]
    runs = {c["runs"][arm]["run_id"]: load_run(runs_dir, c["runs"][arm]["run_id"])
            for c in comparisons for arm in ("a", "b")}
    check_comparisons(comparisons, runs)
    return {
        "description": "One paired comparison (B - A on the validation split, best checkpoint of each run) repeated "
                       "with several seeds. The spread across seeds is training noise, which the per-seed patient "
                       "intervals leave out.",
        "generated_by": "python -m src.training.summarize_comparisons " + " ".join(sys.argv[1:]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "comparisons": [repo_path(p) for p in comparison_paths],
        "config_differences": comparisons[0]["config_differences"],
        "runs": {c["runs"]["a"]["seed"]: {"a": c["runs"]["a"]["run_id"], "b": c["runs"]["b"]["run_id"]}
                 for c in comparisons},
        "code_tree": git_tree(next(iter(runs.values()))["git"]["commit"], CODE_DIR),
        "metrics": summarize(comparisons),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a paired run comparison repeated over several seeds.")
    parser.add_argument("comparisons", nargs="+", type=Path, help="compare_*.json reports from src.training.compare_runs")
    parser.add_argument("--name", required=True, help="the summary is saved as reports/runs/summary_<name>.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    if not SUMMARY_NAME.match(args.name):
        raise SystemExit(f"--name {args.name!r} may contain only letters, digits, '-' and '_'")
    cfg = load_config(args.config)
    report = run(cfg, [project_path(p) for p in args.comparisons])
    out = project_path(cfg["paths"]["runs_dir"]) / f"summary_{args.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_table(report))
    print()
    print(f"Saved {repo_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
