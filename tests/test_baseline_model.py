"""Тесты для src/baseline_model.py (на синтетических данных, без реальных файлов)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import baseline_model as bm  # noqa: E402

TEXTS = {
    "investment_platform": "гарантированный доход трейдинг бот вложи получи прибыль",
    "pyramid": "приведи друга получи процент вербовка сеть уровни",
    "betting": "прогноз ставка кэф гарантирован экспресс проход",
    "giveaway": "подпишись выиграй айфон розыгрыш конкурс победитель",
    "job_offer": "работа из дома вакансия стартовый пакет неделю оплата",
    "none": "рецепт борща сегодня готовим обед семья погода",
}


def make_fixture(tmp_path, per_class=10, with_instagram=True, drop_asr_for=None):
    """Создаёт labels.csv и cache/*.asr.json во временной папке."""
    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for cat in bm.CATEGORIES:
        for i in range(per_class):
            vid = f"tt_{cat[:3]}{i}"
            rows.append(
                {"video_id": vid, "url": "u", "platform": "tiktok",
                 "category": cat, "risk_human": 0.5, "content_type": "video", "notes": ""}
            )
            if vid != drop_asr_for:
                text = TEXTS[cat] + f" слово{i}"
                (cache / f"{vid}.asr.json").write_text(
                    json.dumps({"language": "ru", "text": text, "segments": []},
                               ensure_ascii=False),
                    encoding="utf-8",
                )
    if with_instagram:
        rows.append({"video_id": "ig_1", "url": "u", "platform": "instagram",
                     "category": "none", "risk_human": 0.1, "content_type": "video", "notes": ""})
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    return labels, cache


def test_normalize_text():
    assert bm.normalize_text("  Ёлка   ПРИВЕТ \n") == "елка привет"
    assert bm.normalize_text(None) == ""


def test_load_dataset_matches_by_video_id_and_skips_instagram(tmp_path):
    labels, cache = make_fixture(tmp_path)
    df = bm.load_dataset(labels, cache, verbose=False)
    assert len(df) == 60  # 6 классов x 10, instagram исключён
    assert set(df["category"]) == set(bm.CATEGORIES)
    assert df["video_id"].str.startswith("tt_").all()


def test_load_dataset_skips_videos_without_asr(tmp_path):
    labels, cache = make_fixture(tmp_path, drop_asr_for="tt_pyr0")
    df = bm.load_dataset(labels, cache, verbose=False)
    assert len(df) == 59
    assert "tt_pyr0" not in set(df["video_id"])


def test_load_dataset_rejects_unknown_category(tmp_path):
    labels, cache = make_fixture(tmp_path)
    d = pd.read_csv(labels)
    d.loc[0, "category"] = "crypto"
    d.to_csv(labels, index=False)
    with pytest.raises(ValueError):
        bm.load_dataset(labels, cache, verbose=False)


def test_empty_or_missing_text_field(tmp_path):
    p = tmp_path / "x.asr.json"
    p.write_text(json.dumps({"language": None, "text": "", "segments": []}), encoding="utf-8")
    assert bm.read_asr_text(p) == ""
    p.write_text(json.dumps({"segments": []}), encoding="utf-8")
    assert bm.read_asr_text(p) == ""


def test_split_is_reproducible_stratified_and_disjoint(tmp_path):
    labels, cache = make_fixture(tmp_path)
    df = bm.load_dataset(labels, cache, verbose=False)
    a = bm.make_split(df, seed=1)
    b = bm.make_split(df, seed=1)
    c = bm.make_split(df, seed=2)
    assert a["split"].tolist() == b["split"].tolist()
    assert a["split"].tolist() != c["split"].tolist()
    bm.check_split(a)
    counts = pd.crosstab(a["category"], a["split"])
    assert (counts["test"] == 4).all() and (counts["train"] == 6).all()  # 60/40
    assert len(a) == 60


def test_check_split_detects_overlap_and_missing_class():
    df = pd.DataFrame(
        {"video_id": ["a", "a", "b"], "category": ["none", "none", "betting"],
         "split": ["train", "test", "train"]}
    )
    with pytest.raises(AssertionError):
        bm.check_split(df)


@pytest.mark.parametrize("kind", bm.FEATURE_KINDS)
def test_pipeline_fits_and_proba_order(tmp_path, kind):
    labels, cache = make_fixture(tmp_path)
    df = bm.make_split(bm.load_dataset(labels, cache, verbose=False))
    tr, te = df[df.split == "train"], df[df.split == "test"]
    gs = bm.tune_and_fit(kind, tr["text"].values, tr["category"].values)
    proba = bm.predict_proba_ordered(gs.best_estimator_, te["text"].values)
    assert proba.shape == (len(te), 6)
    assert np.allclose(proba.sum(axis=1), 1.0)
    pred = np.array(bm.CATEGORIES)[proba.argmax(axis=1)]
    # данные синтетические и тривиально разделимые
    assert (pred == te["category"].values).mean() > 0.9


def test_pipeline_handles_empty_text():
    pipe = bm.build_pipeline("word_char", C=1.0)
    X = ["доход гарантия вложи"] * 3 + ["рецепт борща обед"] * 3 + ["доход гарантия вложи", "рецепт борща обед"]
    y = ["investment_platform"] * 3 + ["none"] * 3 + ["investment_platform", "none"]
    pipe.fit(X, y)
    proba = pipe.predict_proba([""])
    assert proba.shape == (1, 2) and np.isfinite(proba).all()


def test_compute_metrics_perfect_and_binary():
    y = np.array(bm.CATEGORIES * 2)
    proba = np.eye(6)[[bm.CATEGORIES.index(c) for c in y]]
    m = bm.compute_metrics(y, y, proba)
    assert m["accuracy"] == 1.0 and m["macro_f1"] == 1.0
    assert m["binary_scam_vs_none"]["f1"] == 1.0
    assert m["binary_scam_vs_none"]["pr_auc"] == 1.0
    assert np.trace(np.array(m["confusion_matrix"])) == len(y)


def test_compute_metrics_binary_collapses_scam_classes():
    y_true = np.array(["betting", "pyramid", "none", "none"])
    y_pred = np.array(["giveaway", "none", "none", "betting"])
    m = bm.compute_metrics(y_true, y_pred)
    b = m["binary_scam_vs_none"]
    # TP=1 (betting->giveaway тоже scam), FN=1, TN=1, FP=1
    assert b["precision"] == 0.5 and b["recall"] == 0.5 and b["accuracy"] == 0.5
    assert m["per_class"]["betting"]["f1"] == 0.0


def test_compute_metrics_always_none_has_zero_scam_recall():
    y_true = np.array(["betting", "none", "none"])
    m = bm.compute_metrics(y_true, np.array(["none"] * 3))
    assert m["binary_scam_vs_none"]["recall"] == 0.0
    assert m["accuracy"] == pytest.approx(2 / 3)


def test_main_end_to_end_writes_outputs(tmp_path):
    labels, cache = make_fixture(tmp_path)
    split_file = tmp_path / "split.csv"
    results = tmp_path / "results"
    rc = bm.main(
        ["--labels", str(labels), "--cache", str(cache),
         "--split-file", str(split_file), "--results-dir", str(results)]
    )
    assert rc == 0
    assert split_file.exists()
    assert (results / "baseline_metrics.json").exists()
    preds = pd.read_csv(results / "baseline_predictions.csv")
    assert set(preds["features"]) == set(bm.FEATURE_KINDS)
    # второй запуск использует уже сохранённый split (воспроизводимость)
    before = pd.read_csv(split_file)
    bm.main(
        ["--labels", str(labels), "--cache", str(cache),
         "--split-file", str(split_file), "--results-dir", str(results)]
    )
    pd.testing.assert_frame_equal(before, pd.read_csv(split_file))


def test_arrow_backed_strings_do_not_break_sklearn(tmp_path):
    """Регрессия: pandas 3 + pyarrow -> ArrowStringArray вместо numpy в .values."""
    pytest.importorskip("pyarrow")
    labels, cache = make_fixture(tmp_path)
    df = bm.make_split(bm.load_dataset(labels, cache, verbose=False))
    df["text"] = df["text"].astype("string[pyarrow]")
    df["category"] = df["category"].astype("string[pyarrow]")
    tr, te = df[df.split == "train"], df[df.split == "test"]
    gs = bm.tune_and_fit("word", tr["text"].values, tr["category"].values)
    proba = bm.predict_proba_ordered(gs.best_estimator_, te["text"].values)
    assert proba.shape == (len(te), 6)
    # run_experiment и nested_cv тоже должны работать на Arrow-строках
    assert bm.run_experiment(df)["chosen_by_train_cv"] in bm.FEATURE_KINDS
    assert set(bm.nested_cv(df, n_splits=3, n_repeats=1)) == set(bm.FEATURE_KINDS)
