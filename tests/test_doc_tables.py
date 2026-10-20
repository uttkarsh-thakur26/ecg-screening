"""README.md and DEMO.md quote the test-set numbers exactly as reports/metrics.json holds them (src/evaluation/doc_tables).
Needs no data, model or GPU."""

from __future__ import annotations

import json

import pytest

from src.config import load_config, project_path
from src.evaluation import doc_tables


@pytest.fixture(scope="module")
def metrics() -> dict:
    cfg = load_config()
    return json.loads((project_path(cfg["paths"]["reports_dir"]) / cfg["final_evaluation"]["metrics_file"])
                      .read_text(encoding="utf-8"))


def test_documents_quote_metrics_json_exactly():
    assert doc_tables.update(check_only=True) == [], "run python -m src.evaluation.doc_tables"


def test_every_document_has_its_blocks():
    names = {doc: {m["name"] for m in doc_tables.BLOCK.finditer(project_path(doc).read_text(encoding="utf-8")
                                                                .replace("\r\n", "\n"))}
             for doc in doc_tables.DOCUMENTS}
    assert names == {"README.md": {"test-results"}, "DEMO.md": {"cd-specificity"}}


def test_values_are_rounded_like_the_performance_screen(metrics):
    assert doc_tables.est(metrics["ranking"]["macro"]["auroc"]) == "0.926 [0.918, 0.933]"
    assert doc_tables.signed(metrics["baselines"]["cnn_minus_feature_gbdt"]["ranking"]["macro"]["auprc"]) \
        == "+0.142 [0.126, 0.161]"
    assert doc_tables.num(-0.0749) == "−0.075"


def test_a_stale_block_is_reported_and_rewritten(metrics):
    text = "<!-- metrics:cd-specificity:start -->\nCD specificity is 0.999.\n<!-- metrics:cd-specificity:end -->\n"
    rewritten, changed = doc_tables.render(text, metrics)
    assert changed == ["cd-specificity"]
    assert "0.792 [0.772, 0.812]" in rewritten and "0.999" not in rewritten
    assert doc_tables.render(rewritten, metrics)[1] == []


def test_an_unknown_block_is_an_error(metrics):
    with pytest.raises(ValueError, match="Unknown metrics block"):
        doc_tables.render("<!-- metrics:auc:start -->\nx\n<!-- metrics:auc:end -->\n", metrics)
