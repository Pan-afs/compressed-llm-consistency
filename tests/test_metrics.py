import json

import pytest

from codecons import metrics
from codecons.analyze import build_summary, render_tables

# Three problems. FP16 is stable; 4-bit flips on t2 when renamed and on t3 when reformatted.
OUTCOMES = {
    "fp16": {
        ("t1", "original"): True, ("t1", "rename"): True, ("t1", "format"): True,
        ("t2", "original"): True, ("t2", "rename"): True, ("t2", "format"): True,
        ("t3", "original"): False, ("t3", "rename"): False, ("t3", "format"): False,
    },
    "int4": {
        ("t1", "original"): True, ("t1", "rename"): True, ("t1", "format"): True,
        ("t2", "original"): True, ("t2", "rename"): False, ("t2", "format"): True,
        ("t3", "original"): False, ("t3", "rename"): False, ("t3", "format"): True,
    },
}


def test_pass_at_1():
    assert metrics.pass_at_1(OUTCOMES, "fp16", "original") == metrics.Rate(2, 3)
    assert metrics.pass_at_1(OUTCOMES, "int4", "rename").value == pytest.approx(1 / 3)


def test_consistency_and_flip_direction():
    c = metrics.consistency(OUTCOMES, "int4", "rename")
    assert c.rate == metrics.Rate(2, 3)
    assert c.pass_to_fail == ["t2"] and c.fail_to_pass == []
    c = metrics.consistency(OUTCOMES, "int4", "format")
    assert c.pass_to_fail == [] and c.fail_to_pass == ["t3"]
    assert metrics.consistency(OUTCOMES, "fp16", "rename").rate == metrics.Rate(3, 3)


def test_perturbation_without_records_has_zero_n():
    assert metrics.consistency(OUTCOMES, "fp16", "reorder").rate.n == 0


def test_pooled_consistency():
    assert metrics.pooled_consistency(OUTCOMES, "int4", ["rename", "format"]) == metrics.Rate(4, 6)


def test_disagreement_and_change_induced_failures():
    assert metrics.disagreement(OUTCOMES, "fp16", "int4", "rename") == ["t2"]
    assert metrics.disagreement(OUTCOMES, "fp16", "int4", "original") == []
    assert metrics.change_induced_failures(OUTCOMES, "fp16", "int4", "rename") == ["t2"]
    assert metrics.change_induced_failures(OUTCOMES, "fp16", "int4", "format") == []


def test_compare_consistency_counts_discordant_pairs():
    cmp = metrics.compare_consistency(OUTCOMES, "fp16", "int4", ["rename", "format"])
    assert (cmp.n, cmp.ref_only_consistent, cmp.other_only_consistent) == (6, 2, 0)
    assert cmp.p_value == pytest.approx(0.5)


def test_wilson_interval_contains_estimate_and_is_bounded():
    lo, hi = metrics.wilson_interval(9, 10)
    assert 0 <= lo < 0.9 < hi <= 1
    assert metrics.wilson_interval(10, 10)[1] == 1.0


def test_mcnemar_exact():
    assert metrics.mcnemar_exact(0, 0) == 1.0
    assert metrics.mcnemar_exact(0, 6) == pytest.approx(2 / 64)


def test_load_outcomes_reads_jsonl(tmp_path):
    rows = [
        {"task_id": "t1", "variant": "original", "precision": "int8", "passed": True},
        {"task_id": "t1", "variant": "rename", "precision": "int8", "passed": False},
    ]
    (tmp_path / "int8.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert metrics.load_outcomes(str(tmp_path)) == {"int8": {("t1", "original"): True, ("t1", "rename"): False}}


def test_summary_tables_render():
    full = {p: dict(o) for p, o in OUTCOMES.items()}
    for prec in full:
        for t in ("t1", "t2", "t3"):
            for v in ("comment", "reorder"):
                full[prec][(t, v)] = full[prec][(t, "original")]
    summary = build_summary(full, ["fp16", "int4"], {})
    tables = render_tables(summary, ["fp16", "int4"])
    assert "| 4-bit |" in tables
    assert summary["precisions"]["int4"]["vs_fp16"]["change_induced_failures"]["rename"] == ["t2"]
