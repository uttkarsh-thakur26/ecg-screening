"""The test-set numbers that README.md and DEMO.md quote, written from reports/metrics.json.

    python -m src.evaluation.doc_tables           # rewrite the marked blocks in README.md and DEMO.md
    python -m src.evaluation.doc_tables --check   # exit 1 if any block differs from what metrics.json gives

Each block sits between "<!-- metrics:NAME:start -->" and "<!-- metrics:NAME:end -->" and is replaced as a whole;
nothing outside the markers is touched. tests/test_doc_tables.py runs the check, so a document cannot quote a number
that metrics.json does not hold. Values are rounded to three decimals, as on the Performance screen; brackets are the
report's percentile bootstrap intervals. Nothing here computes a metric: the only arithmetic is rounding, counting
(flags = tp + fp, negatives = fp + tn) and a share written as "about N in 10".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.config import load_config, project_path

DOCUMENTS = ("README.md", "DEMO.md")  # repository root
BLOCK = re.compile(r"(<!-- metrics:(?P<name>[a-z0-9-]+):start -->\n)(?P<body>.*?)(<!-- metrics:(?P=name):end -->)",
                   re.DOTALL)
CALIBRATORS = {"temperature": "Temperature scaling", "platt": "Platt scaling"}


def num(x: float) -> str:
    """0.92586 -> "0.926"; a negative value gets a true minus sign."""
    return f"{x:.3f}".replace("-", "−")


def est(e: dict[str, float]) -> str:
    """An estimate with its interval: "0.926 [0.918, 0.933]"."""
    return f"{num(e['value'])} [{num(e['ci_low'])}, {num(e['ci_high'])}]"


def signed(e: dict[str, float]) -> str:
    """A paired difference with its sign: "+0.086 [0.077, 0.097]"."""
    return f"{'+' if e['value'] > 0 else ''}{est(e)}"


def count(n: int) -> str:
    return f"{n:,}"


def table(head: list[str], rows: list[list[str]], numeric_from: int = 1) -> str:
    align = ["---" if i < numeric_from else "---:" for i in range(len(head))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def class_cell(m: dict[str, Any], c: str) -> str:
    return f"{c} ({m['class_descriptions'][c]})"


def results_block(m: dict[str, Any]) -> str:
    """README: ranking per class, the high-sensitivity point, the baselines and calibration, all on the test split."""
    classes, abnormal = m["classes"], m["abnormal_classes"]
    all_c, abn_c = f"macro, all {len(classes)} classes", f"macro, {len(abnormal)} abnormal classes"
    boot = m["bootstrap"]
    rk = m["ranking"]
    intro = (f"{m['dataset']['name']} {m['dataset']['version']}, fold {', '.join(map(str, m['folds']))}: "
             f"{count(m['records'])} records from {count(m['patients'])} patients, evaluated once on "
             f"{m['date_evaluated']} at commit `{m['git']['commit'][:7]}`. Brackets are "
             f"{round(boot['ci_level'] * 100)}% intervals: percentile bootstrap over {boot['unit']}s, "
             f"{count(boot['resamples'])} resamples.")

    ranking = table(["Class", "Prevalence", "AUROC", "AUPRC"],
                    [[class_cell(m, c), num(rk["per_class"][c]["prevalence"]), est(rk["per_class"][c]["auroc"]),
                      est(rk["per_class"][c]["auprc"])] for c in classes]
                    + [[f"**{label}**", "", est(rk[key]["auroc"]), est(rk[key]["auprc"])]
                       for label, key in ((all_c.capitalize(), "macro"), (abn_c.capitalize(), "macro_abnormal"))])

    hs = m["operating_points"]["high_sensitivity"]

    def point_row(label: str, p: dict[str, Any], threshold: str) -> list[str]:
        return [label, threshold, est(p["sensitivity"]), est(p["specificity"]), est(p["precision"]), est(p["f1"])]

    point = table(["Class", "Threshold", "Sensitivity", "Specificity", "Precision", "F1"],
                  [point_row(class_cell(m, c), hs["per_class"][c], num(hs["per_class"][c]["threshold"]))
                   for c in hs["classes"]]
                  + [point_row(f"**{abn_c.capitalize()}**", hs["macro_abnormal"], "")])
    missed = ", ".join(f"{c} {count(hs['per_class'][c]['fn'])} of "
                       f"{count(hs['per_class'][c]['fn'] + hs['per_class'][c]['tp'])}" for c in hs["classes"])

    bl = m["baselines"]
    prior, gbdt, lead = bl["prior"]["ranking"]["macro"], bl["feature_gbdt"]["ranking"]["macro"], \
        bl["cnn_minus_feature_gbdt"]["ranking"]["macro"]
    baselines = table([f"Model ({all_c})", "AUROC", "AUPRC"], [
        ["Prior baseline (every record scored with the training-split class frequency)", num(prior["auroc"]),
         num(prior["auprc"])],
        ["Feature baseline (hand-crafted RR-interval and amplitude features, gradient-boosted trees)", est(gbdt["auroc"]), est(gbdt["auprc"])],
        ["**1D CNN (this model)**", est(rk["macro"]["auroc"]), est(rk["macro"]["auprc"])],
        ["CNN minus feature baseline (paired: same resampled patients)", signed(lead["auroc"]), signed(lead["auprc"])],
    ])

    ece = m["calibration_error"]
    calibration = table(["Class", "Calibrator (fitted on fold 9)", "ECE uncalibrated", "ECE calibrated"],
                        [[class_cell(m, c), CALIBRATORS[ece["per_class"][c]["calibrator"]["method"]],
                          num(ece["per_class"][c]["uncalibrated"]["value"]),
                          num(ece["per_class"][c]["calibrated"]["value"])] for c in classes]
                        + [[f"**{all_c.capitalize()}**", "", num(ece["macro"]["uncalibrated"]["value"]),
                            num(ece["macro"]["calibrated"]["value"])]], numeric_from=2)

    return "\n\n".join([
        intro,
        "**Ranking** (threshold-free: does a record with the class score higher than one without it?)",
        ranking,
        f"**High-sensitivity operating point** (the thresholds the app uses; chosen on fold 9 for sensitivity ≥ "
        f"{hs['target_sensitivity']:.2f}). Missed cases: {missed}.",
        point,
        "**Against the baselines**",
        baselines,
        f"**Calibration**: expected calibration error (ECE, {ece['bins']} equal-width bins; lower is better). Point "
        "values: ECE's bootstrap interval is biased upward and is not a confidence interval, so it is not shown.",
        calibration,
    ]) + "\n"


def cd_specificity_block(m: dict[str, Any]) -> str:
    """DEMO.md: what a CD flag costs at the high-sensitivity point, for the record-38 false alarm."""
    cd = m["operating_points"]["high_sensitivity"]["per_class"]["CD"]
    negatives = cd["fp"] + cd["tn"]
    in_ten = round(10 * cd["fp"] / negatives)
    return (f"On the test set, CD specificity at the high-sensitivity point is {est(cd['specificity'])}: about "
            f"{in_ten} in 10 records without CD still get a CD flag ({count(cd['fp'])} of {count(negatives)}). "
            f"In exchange, CD sensitivity is {est(cd['sensitivity'])} ({count(cd['tp'])} of "
            f"{count(cd['tp'] + cd['fn'])} CD records flagged).\n")


BLOCKS: dict[str, Callable[[dict[str, Any]], str]] = {
    "test-results": results_block,
    "cd-specificity": cd_specificity_block,
}


def render(text: str, m: dict[str, Any]) -> tuple[str, list[str]]:
    """The document with every block rewritten, and the names of the blocks whose content changed."""
    changed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match["name"]
        if name not in BLOCKS:
            raise ValueError(f"Unknown metrics block {name!r}; known: {', '.join(BLOCKS)}")
        body = BLOCKS[name](m)
        if body != match["body"]:
            changed.append(name)
        return match.group(1) + body + match.group(4)

    return BLOCK.sub(replace, text), changed


def update(check_only: bool) -> list[str]:
    """Rewrite (or only check) every block; returns "<document>: <block>" for each stale block."""
    cfg = load_config()
    metrics_path = project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["metrics_file"]
    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    stale: list[str] = []
    for doc in DOCUMENTS:
        path = project_path(doc)
        raw = path.read_bytes().decode("utf-8")
        newline = "\r\n" if "\r\n" in raw else "\n"  # git may check text files out with CRLF on Windows
        text, changed = render(raw.replace("\r\n", "\n"), m)
        if not BLOCK.search(raw.replace("\r\n", "\n")):
            raise ValueError(f"{doc} has no metrics block")
        stale += [f"{doc}: {name}" for name in changed]
        if changed and not check_only:
            path.write_bytes(text.replace("\n", newline).encode("utf-8"))
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the metrics.json numbers that README.md and DEMO.md quote.")
    parser.add_argument("--check", action="store_true", help="only check; exit 1 if a block is out of date")
    args = parser.parse_args(argv)
    stale = update(check_only=args.check)
    if args.check:
        for s in stale:
            print(f"Out of date: {s} (run python -m src.evaluation.doc_tables)", file=sys.stderr)
        return 1 if stale else 0
    print("\n".join(f"Rewrote {s}" for s in stale) or "Every block already matches metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
