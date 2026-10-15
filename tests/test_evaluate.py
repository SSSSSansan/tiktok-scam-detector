"""Тесты для src/evaluate.py (синтетические данные)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluate as ev  # noqa: E402
from src.baseline_model import CATEGORIES  # noqa: E402


def test_brier_and_ece_known_values():
    y = np.array([1, 0, 1, 0])
    assert ev.brier_score(y, [1, 0, 1, 0]) == 0.0
    assert ev.brier_score(y, [0.5] * 4) == pytest.approx(0.25)
    assert ev.expected_calibration_error(y, [1, 0, 1, 0]) == 0.0
    # уверенно, но всегда неправильно -> ECE = 1
    assert ev.expected_calibration_error(y, [0.0, 1.0, 0.0, 1.0]) == pytest.approx(1.0)
    # идеально откалиброванные 0.5
    assert ev.expected_calibration_error([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.0)


def test_ece_handles_probability_one():
    assert ev.expected_calibration_error([1, 1], [1.0, 1.0]) == 0.0


def test_reliability_table_counts_sum():
    rng = np.random.default_rng(0)
    p = rng.random(50)
    y = (rng.random(50) < p).astype(int)
    t = ev.reliability_table(y, p, n_bins=5)
    assert t["n"].sum() == 50


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(1)
    yt = rng.integers(0, 6, 80)
    yp = rng.integers(0, 6, 80)
    expected = f1_score(yt, yp, labels=list(range(6)), average="macro", zero_division=0)
    assert ev._macro_f1_idx(yt, yp) == pytest.approx(expected)


def test_binary_f1_edge_cases():
    assert ev._binary_f1(np.array([0, 0]), np.array([0, 0])) == 0.0
    assert ev._binary_f1(np.array([1, 0]), np.array([1, 0])) == 1.0


def test_paired_bootstrap_reproducible_and_ordered():
    rng = np.random.default_rng(2)
    yt = rng.integers(0, 6, 60)
    good = yt.copy()
    bad = rng.integers(0, 6, 60)
    sc_good = (yt != 5).astype(float)
    sc_bad = rng.random(60)
    systems = {"a": (good, sc_good), "b": (bad, sc_bad)}
    r1 = ev.paired_bootstrap(yt, systems, n_boot=200, seed=0)
    r2 = ev.paired_bootstrap(yt, systems, n_boot=200, seed=0)
    assert r1 == r2
    d = r1["diff"]["metrics"]["macro_f1"]
    assert d["mean_diff"] > 0.3 and d["p_first_better"] > 0.99
    lo, hi = r1["ci"]["a"]["accuracy"]
    assert lo <= 1.0 <= hi + 1e-9 and lo > 0.9


def _write_fixture(tmp_path, judge_drop=None):
    ids = [f"tt_{i}" for i in range(12)]
    cats = (CATEGORIES * 2)
    pd.DataFrame({"video_id": ids + ["tt_train"], "category": cats + ["none"],
                  "split": ["test"] * 12 + ["train"]}).to_csv(tmp_path / "split.csv", index=False)
    jd = pd.DataFrame({"video_id": ids, "platform": "tiktok", "true_category": cats,
                       "risk_human": [0.9 if c != "none" else 0.1 for c in cats],
                       "pred_category": cats, "risk_pred": [0.8 if c != "none" else 0.2 for c in cats],
                       "explanation": "x"})
    if judge_drop:
        jd = jd[jd.video_id != judge_drop]
    jd.to_csv(tmp_path / "judge.csv", index=False)
    rows = []
    for vid, c in zip(ids, cats):
        pr = np.full(6, 0.05); pr[CATEGORIES.index(c)] = 0.75
        row = {"video_id": vid, "features": "char", "true": c, "pred": c}
        row.update({f"p_{k}": pr[i] for i, k in enumerate(CATEGORIES)})
        row["risk_baseline"] = 1 - pr[-1]
        rows.append(row)
    pd.DataFrame(rows).to_csv(tmp_path / "base.csv", index=False)
    (tmp_path / "bm.json").write_text(json.dumps({"chosen_by_train_cv": "char"}))


def test_load_aligned_ok_and_excludes_train(tmp_path):
    _write_fixture(tmp_path)
    df = ev.load_aligned(tmp_path / "split.csv", tmp_path / "judge.csv", tmp_path / "base.csv", "char")
    assert len(df) == 12 and "tt_train" not in set(df.video_id)


def test_load_aligned_rejects_mismatched_sets(tmp_path):
    _write_fixture(tmp_path, judge_drop="tt_3")
    with pytest.raises(ValueError):
        ev.load_aligned(tmp_path / "split.csv", tmp_path / "judge.csv", tmp_path / "base.csv", "char")


def test_load_aligned_rejects_unknown_features(tmp_path):
    _write_fixture(tmp_path)
    with pytest.raises(ValueError):
        ev.load_aligned(tmp_path / "split.csv", tmp_path / "judge.csv", tmp_path / "base.csv", "word")


def test_main_end_to_end(tmp_path):
    _write_fixture(tmp_path)
    out = tmp_path / "res"
    rc = ev.main(["--split-file", str(tmp_path / "split.csv"), "--judge", str(tmp_path / "judge.csv"),
                  "--baseline-pred", str(tmp_path / "base.csv"), "--baseline-metrics", str(tmp_path / "bm.json"),
                  "--results-dir", str(out), "--n-boot", "50"])
    assert rc == 0
    rep = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
    assert rep["n_test"] == 12
    assert rep["systems"]["llm"]["macro_f1"] == 1.0
    assert rep["systems"]["baseline"]["binary_scam_vs_none"]["pr_auc_risk_score"] == 1.0
    assert (out / "evaluation_report.md").exists()
