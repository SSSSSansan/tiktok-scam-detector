"""Тесты для src/calibrate.py (синтетические данные + ваш настоящий features.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calibrate as cal  # noqa: E402
from src.features import FLAG_NAMES, extract_features  # noqa: E402

CATS = ["investment_platform", "pyramid", "betting", "giveaway", "job_offer"]


def make_fixture(tmp_path, n_per_split=30, seed=0):
    """judge.csv, split.csv и cache/*.asr.json с известным сигналом."""
    rng = np.random.default_rng(seed)
    cache = tmp_path / "cache"
    cache.mkdir()
    rows, split = [], []
    for k in range(2 * n_per_split):
        vid = f"tt_{k}"
        scam = rng.random() < 0.6
        true = rng.choice(CATS) if scam else "none"
        # LLM: скам -> чаще высокий риск и не-none; без речи всегда risk 0.1 / none
        no_speech = rng.random() < 0.1
        if no_speech:
            text, pred, risk = "", "none", 0.1
        else:
            text = ("гарантированный доход пиши в личку " if scam else "рецепт борща сегодня ") * int(rng.integers(2, 8))
            ok = rng.random() < 0.75
            pred = true if (scam and ok) else ("none" if not scam else "none")
            risk = float(rng.choice([0.5, 0.6, 0.7]) if pred != "none" else rng.choice([0.2, 0.2, 0.3]))
        (cache / f"{vid}.asr.json").write_text(
            json.dumps({"language": "ru", "text": text, "segments": [{"start": 0, "end": 1, "text": text}] if text else []},
                       ensure_ascii=False), encoding="utf-8")
        rows.append({"video_id": vid, "platform": "tiktok", "true_category": true,
                     "risk_human": 0.9 if scam else 0.1, "pred_category": pred, "risk_pred": risk})
        split.append({"video_id": vid, "split": "train" if k < n_per_split else "test"})
    judge = tmp_path / "judge.csv"
    pd.DataFrame(rows).to_csv(judge, index=False)
    sp = tmp_path / "split.csv"
    pd.DataFrame(split).to_csv(sp, index=False)
    return judge, sp, cache


def test_logit_risk_is_clipped_and_monotone():
    z = cal.logit_risk([0.0, 0.1, 0.5, 0.9, 1.0])
    assert np.all(np.isfinite(z)) and np.all(np.diff(z) >= 0)
    assert z[2] == pytest.approx(0.0)
    assert cal.logit_risk([0.0])[0] == cal.logit_risk([0.05])[0]


def test_design_from_flags_columns_and_values():
    flags = pd.DataFrame([{n: 0 for n in FLAG_NAMES}, {n: 1 for n in FLAG_NAMES}])
    X = cal.design_from_flags([0.5, 0.2], ["betting", "none"], flags, n_words=[0, 100])
    assert X["llm_scam"].tolist() == [1.0, 0.0]
    assert X["no_speech"].tolist() == [1.0, 0.0]
    assert X["log_words"].iloc[1] == pytest.approx(np.log1p(100))
    assert X[cal.GROUP_COLS].iloc[0].sum() == 0
    assert (X[cal.GROUP_COLS].iloc[1] <= cal.COUNT_CAP).all()
    for cols in cal.VARIANTS.values():
        assert set(cols) <= set(X.columns)


def test_design_without_n_words_has_no_speech_columns():
    flags = pd.DataFrame([{n: 0 for n in FLAG_NAMES}])
    X = cal.design_from_flags([0.5], ["none"], flags)
    assert "no_speech" not in X.columns


def test_prob_metrics_known_values():
    y = np.array([1, 1, 0, 0])
    m = cal.prob_metrics(y, [0.9, 0.8, 0.1, 0.2])
    assert m["roc_auc"] == 1.0 and m["accuracy@0.5"] == 1.0 and m["f1@0.5"] == 1.0
    assert m["brier"] == pytest.approx(np.mean([0.01, 0.04, 0.01, 0.04]))
    assert cal.prob_metrics(y, [0.5] * 4)["brier"] == pytest.approx(0.25)


def test_bootstrap_brier_diff_detects_improvement():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 80)
    good = np.where(y == 1, 0.9, 0.1)
    bad = np.full(80, 0.5)
    d = cal.bootstrap_brier_diff(y, good, bad, n_boot=300, seed=1)
    assert d["mean_diff"] < -0.2 and d["p_new_better"] > 0.99 and d["ci"][1] < 0
    assert d == cal.bootstrap_brier_diff(y, good, bad, n_boot=300, seed=1)


def test_flags_table_uses_real_features_and_counts_words(tmp_path):
    cache = tmp_path
    (cache / "a.asr.json").write_text(json.dumps({"text": "гарантированный доход без риска", "segments": [
        {"start": 0, "end": 2, "text": "гарантированный доход без риска"}]}, ensure_ascii=False), encoding="utf-8")
    (cache / "b.asr.json").write_text(json.dumps({"text": "", "segments": []}), encoding="utf-8")
    flags, nw = cal.flags_table(["a", "b"], cache)
    assert list(flags.columns) == list(FLAG_NAMES)
    assert flags.loc[0, "guaranteed_income_phrase"] == 1
    assert flags.loc[1].sum() == 0
    assert nw == [4, 0]
    assert extract_features(asr={"text": "", "segments": []})["flags"]["has_url"] == 0


def test_load_calibration_data_shapes(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    assert len(df) == 60 and set(df["split"]) == {"train", "test"}
    assert set(df["y_scam"]) == {0, 1}
    assert df["video_id"].is_unique
    for cols in cal.VARIANTS.values():
        assert set(cols) <= set(df.columns)


def test_load_calibration_data_requires_all_videos_in_split(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    s = pd.read_csv(sp)
    s.iloc[:-3].to_csv(sp, index=False)
    with pytest.raises(ValueError):
        cal.load_calibration_data(judge, sp, cache)


def test_probabilities_in_unit_interval_and_calibration_helps(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    exp = cal.run_split_experiment(df, n_boot=100)
    assert exp["chosen_by_train_cv"] in cal.VARIANTS
    for name, p in exp["test_probs"].items():
        assert np.all((p >= 0) & (p <= 1)), name
    raw = exp["variants"]["raw_llm"]["test"]["brier"]
    best = exp["variants"][exp["chosen_by_train_cv"]]["test"]["brier"]
    assert best <= raw + 0.02  # на синтетике с сигналом калибровка не должна ухудшать


def test_no_test_leakage_into_fitting(tmp_path):
    """Меняем метки test -> предсказания test не должны измениться (калибратор не видит test)."""
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    a = cal.run_split_experiment(df, n_boot=20)
    df2 = df.copy()
    te = df2["split"] == "test"
    df2.loc[te, "y_scam"] = 1 - df2.loc[te, "y_scam"]
    b = cal.run_split_experiment(df2, n_boot=20)
    assert a["chosen_by_train_cv"] == b["chosen_by_train_cv"]
    for name in cal.VARIANTS:
        assert np.allclose(a["test_probs"][name], b["test_probs"][name]), name


def test_tune_fit_c_comes_from_grid(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    tr = df[df.split == "train"]
    gs = cal.tune_fit(tr, tr["y_scam"].to_numpy(), cal.VARIANTS["llm_cat"])
    assert gs.best_params_["clf__C"] in cal.C_GRID
    coef = cal.llm_logit_coef(gs.best_estimator_, cal.VARIANTS["llm_cat"])
    assert coef is not None


def test_llm_logit_coef_none_for_flags_only(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    gs = cal.tune_fit(df, df["y_scam"].to_numpy(), cal.VARIANTS["flags_only"])
    assert cal.llm_logit_coef(gs.best_estimator_, cal.VARIANTS["flags_only"]) is None


def test_speech_subgroups_counts(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    oof = {"raw_llm": df["risk_pred"].to_numpy(dtype=float)}
    g = cal.speech_subgroups(df, oof, threshold=15)
    assert g["n_poor"] + g["n_rich"] == len(df)
    assert 0 <= g["brier"]["raw_llm"]["poor"] <= 1


def test_save_and_load_calibrator_roundtrip(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    df = cal.load_calibration_data(judge, sp, cache)
    cols = cal.VARIANTS["llm_cat_speech"]
    gs = cal.tune_fit(df, df["y_scam"].to_numpy(), cols)
    path = tmp_path / "m" / "calibrator.joblib"
    cal.save_calibrator(gs.best_estimator_, "llm_cat_speech", path)
    risk = cal.load_calibrator(path)
    feats = extract_features(asr={"text": "гарантированный доход", "segments": [
        {"start": 0, "end": 1, "text": "гарантированный доход"}]})
    r_hi = risk(0.7, "investment_platform", feats, n_words=40)
    r_lo = risk(0.2, "none", extract_features(asr={"text": "рецепт борща", "segments": []}), n_words=80)
    assert 0 <= r_lo <= 1 and 0 <= r_hi <= 1 and r_hi > r_lo
    with pytest.raises(ValueError):
        risk(0.7, "betting", feats)  # speech-вариант требует n_words
    # то же, что дал бы pipeline.predict_proba на тех же признаках
    flags = pd.DataFrame([feats["flags"]])
    X = cal.design_from_flags([0.7], ["investment_platform"], flags, [40])
    assert r_hi == pytest.approx(float(gs.best_estimator_.predict_proba(X[cols])[0, 1]))


def test_main_end_to_end(tmp_path):
    judge, sp, cache = make_fixture(tmp_path)
    out = tmp_path / "res"
    rc = cal.main(["--judge", str(judge), "--split-file", str(sp), "--cache", str(cache),
                   "--results-dir", str(out), "--n-boot", "30"])
    assert rc == 0
    rep = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert rep["chosen_by_train_cv"] in cal.VARIANTS
    assert "nested_cv" in rep and "speech_subgroups" in rep
    assert (out / "calibration_report.md").exists()
    assert (out / "calibration_test_predictions.csv").exists()
