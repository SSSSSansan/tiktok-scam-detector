"""Baseline Scam Radar: TF-IDF (по тексту ASR) + Logistic Regression.

Что делает модуль
-----------------
1. Читает ``data/labels.csv`` и расшифровки ``data/cache/<video_id>.asr.json``
   (поле ``text``). В labels.csv ``video_id`` уже содержит префикс ``tt_``,
   файл называется ``tt_<id>.asr.json`` -- сопоставление идёт по этому ID.
2. Делает воспроизводимый стратифицированный split train/test (60/40) и
   сохраняет его в ``data/split_tiktok.csv``.
3. Сравнивает три варианта признаков: word n-grams, char n-grams, word+char.
   Гиперпараметр C подбирается ТОЛЬКО по кросс-валидации внутри train.
4. Оценивает на test (один раз, после выбора C) и сохраняет предсказания
   и метрики в ``results/``.
5. (опция ``--cv``) Повторная вложенная кросс-валидация на всех 127 видео
   как более устойчивая оценка при очень маленьком test.

Запуск (из корня репозитория, Windows CMD)::

    python src\\baseline_model.py
    python src\\baseline_model.py --cv

Внимание: OpenAI API и Whisper не вызываются, видео не скачиваются.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import FeatureUnion, Pipeline

# Контракт: ровно 6 категорий, порядок фиксирован (см. PROJECT_CONTEXT, раздел 3)
CATEGORIES = [
    "investment_platform",
    "pyramid",
    "betting",
    "giveaway",
    "job_offer",
    "none",
]
SCAM_CLASSES = [c for c in CATEGORIES if c != "none"]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "labels.csv"
DEFAULT_CACHE = ROOT / "data" / "cache"
DEFAULT_SPLIT = ROOT / "data" / "split_tiktok.csv"
DEFAULT_RESULTS = ROOT / "results"

SEED = 42
TEST_SIZE = 0.4
C_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
FEATURE_KINDS = ["word", "char", "word_char"]


# --------------------------------------------------------------------------
# Загрузка данных
# --------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """Лёгкая нормализация: нижний регистр, ё->е, схлопывание пробелов."""
    text = (text or "").lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def read_asr_text(path: Path) -> str:
    """Читает поле ``text`` из ASR JSON. Нет файла/поля -> пустая строка."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return normalize_text(data.get("text") or "")


def load_dataset(
    labels_path: Path = DEFAULT_LABELS,
    cache_dir: Path = DEFAULT_CACHE,
    platform: str = "tiktok",
    verbose: bool = True,
) -> pd.DataFrame:
    """Объединяет labels.csv с ASR-текстами по video_id.

    Берутся только видео выбранной платформы, у которых есть И расшифровка,
    И метка. Возвращает DataFrame: video_id, text, category, risk_human.
    """
    labels = pd.read_csv(labels_path)
    if platform and "platform" in labels.columns:
        labels = labels[labels["platform"] == platform]
    labels = labels.dropna(subset=["category"]).copy()

    bad = set(labels["category"]) - set(CATEGORIES)
    if bad:
        raise ValueError(f"Неизвестные категории в labels.csv: {sorted(bad)}")
    if labels["video_id"].duplicated().any():
        raise ValueError("Дубликаты video_id в labels.csv")

    rows, missing = [], []
    for _, r in labels.iterrows():
        p = Path(cache_dir) / f"{r['video_id']}.asr.json"
        if not p.exists():
            missing.append(r["video_id"])
            continue
        rows.append(
            {
                "video_id": r["video_id"],
                "text": read_asr_text(p),
                "category": r["category"],
                "risk_human": r.get("risk_human", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    if verbose:
        print(f"Видео с меткой ({platform}): {len(labels)}")
        print(f"С расшифровкой и меткой:     {len(df)}")
        if missing:
            print(f"Без ASR-файла (исключены):   {missing}")
        print(f"Пустой текст (нет речи):     {(df['text'] == '').sum()}")
    return df


# --------------------------------------------------------------------------
# Split
# --------------------------------------------------------------------------
def make_split(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED
) -> pd.DataFrame:
    """Стратифицированный split по category. Возвращает df с колонкой split."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=seed,
        stratify=df["category"],
    )
    out = df.copy()
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    return out


def check_split(df: pd.DataFrame) -> None:
    """Проверки: нет пересечений, все классы в обеих частях."""
    tr = set(df.loc[df["split"] == "train", "video_id"])
    te = set(df.loc[df["split"] == "test", "video_id"])
    assert not (tr & te), f"Пересечение train/test: {sorted(tr & te)}"
    assert len(tr) + len(te) == len(df), "Не все видео попали в split"
    for part in ("train", "test"):
        missing = set(CATEGORIES) - set(df.loc[df["split"] == part, "category"])
        assert not missing, f"В {part} нет классов: {sorted(missing)}"


def print_distribution(df: pd.DataFrame) -> None:
    tab = (
        pd.crosstab(df["category"], df["split"])
        .reindex(CATEGORIES)
        .fillna(0)
        .astype(int)
    )
    tab["total"] = tab.sum(axis=1)
    tab.loc["ИТОГО"] = tab.sum()
    print(tab.to_string())


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------
def _vectorizer(kind: str):
    word = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True
    )
    if kind == "word":
        return word
    if kind == "char":
        return char
    if kind == "word_char":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"kind должен быть из {FEATURE_KINDS}, получено {kind!r}")


def build_pipeline(kind: str, C: float = 10.0) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", _vectorizer(kind)),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=SEED,
                ),
            ),
        ]
    )


def as_array(x) -> np.ndarray:
    """Pandas 3 + pyarrow хранит строки как ArrowStringArray, который sklearn
    не умеет индексировать. Всегда приводим к numpy object-массиву."""
    return np.asarray(x, dtype=object)


def _inner_cv(y, n_splits: int = 5, seed: int = SEED):
    min_count = int(pd.Series(y).value_counts().min())
    return StratifiedKFold(
        n_splits=max(2, min(n_splits, min_count)), shuffle=True, random_state=seed
    )


def tune_and_fit(kind: str, X_train, y_train, seed: int = SEED) -> GridSearchCV:
    """Подбор C по CV внутри train (macro-F1), затем refit на всём train."""
    X_train, y_train = as_array(X_train), as_array(y_train)
    gs = GridSearchCV(
        build_pipeline(kind),
        param_grid={"clf__C": C_GRID},
        scoring="f1_macro",
        cv=_inner_cv(y_train, seed=seed),
        refit=True,
        n_jobs=1,
    )
    gs.fit(X_train, y_train)
    return gs


def predict_proba_ordered(model, X) -> np.ndarray:
    """Вероятности в порядке CATEGORIES (отсутствующие классы -> 0)."""
    proba = model.predict_proba(as_array(X))
    classes = list(model.classes_)
    out = np.zeros((proba.shape[0], len(CATEGORIES)))
    for j, c in enumerate(CATEGORIES):
        if c in classes:
            out[:, j] = proba[:, classes.index(c)]
    return out


# --------------------------------------------------------------------------
# Метрики (evaluate.py будет импортировать compute_metrics отсюда)
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, proba: np.ndarray | None = None) -> dict:
    """Метрики шести классов + бинарные scam/none.

    ``proba`` -- матрица (n, 6) в порядке CATEGORIES; для бинарного PR-AUC
    используется score = 1 - P(none).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=CATEGORIES, zero_division=0
    )
    res = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=CATEGORIES, average="macro", zero_division=0)
        ),
        "per_class": {
            c: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, c in enumerate(CATEGORIES)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=CATEGORIES
        ).tolist(),
    }

    bt = (y_true != "none").astype(int)
    bp = (y_pred != "none").astype(int)
    bp_, br_, bf_, _ = precision_recall_fscore_support(
        bt, bp, average="binary", pos_label=1, zero_division=0
    )
    res["binary_scam_vs_none"] = {
        "accuracy": float(accuracy_score(bt, bp)),
        "precision": float(bp_),
        "recall": float(br_),
        "f1": float(bf_),
    }

    if proba is not None:
        score = 1.0 - proba[:, CATEGORIES.index("none")]
        res["binary_scam_vs_none"]["pr_auc"] = (
            float(average_precision_score(bt, score)) if bt.sum() > 0 else None
        )
        aps = {}
        for j, c in enumerate(CATEGORIES):
            yt = (y_true == c).astype(int)
            aps[c] = float(average_precision_score(yt, proba[:, j])) if yt.sum() else None
        res["pr_auc_per_class"] = aps
        vals = [v for v in aps.values() if v is not None]
        res["pr_auc_macro"] = float(np.mean(vals)) if vals else None
    return res


def format_confusion(cm, labels=CATEGORIES) -> str:
    short = {c: c[:9] for c in labels}
    df = pd.DataFrame(
        cm,
        index=[f"true:{short[c]}" for c in labels],
        columns=[short[c] for c in labels],
    )
    return df.to_string()


# --------------------------------------------------------------------------
# Основной эксперимент
# --------------------------------------------------------------------------
def run_experiment(df: pd.DataFrame, seed: int = SEED) -> dict:
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    X_tr, y_tr = train["text"].to_numpy(dtype=object), train["category"].to_numpy(dtype=object)
    X_te, y_te = test["text"].to_numpy(dtype=object), test["category"].to_numpy(dtype=object)

    results, models, test_preds = {}, {}, {}
    for kind in FEATURE_KINDS:
        gs = tune_and_fit(kind, X_tr, y_tr, seed=seed)
        proba = predict_proba_ordered(gs.best_estimator_, X_te)
        pred = np.array(CATEGORIES)[proba.argmax(axis=1)]
        m = compute_metrics(y_te, pred, proba)
        m["best_C"] = float(gs.best_params_["clf__C"])
        m["train_cv_macro_f1"] = float(gs.best_score_)
        results[kind] = m
        models[kind] = gs.best_estimator_
        test_preds[kind] = (pred, proba)

    # Выбор "основного" варианта -- ТОЛЬКО по train-CV, не по test
    chosen = max(results, key=lambda k: results[k]["train_cv_macro_f1"])
    return {
        "results": results,
        "models": models,
        "test_preds": test_preds,
        "chosen_by_train_cv": chosen,
        "test_ids": test["video_id"].tolist(),
        "y_test": y_te,
    }


def nested_cv(df: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = SEED):
    """Повторная вложенная CV на всех видео (C подбирается внутри каждого fold)."""
    X, y = df["text"].to_numpy(dtype=object), df["category"].to_numpy(dtype=object)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    out = {k: {"macro_f1": [], "accuracy": [], "binary_f1": []} for k in FEATURE_KINDS}
    for tr, te in rskf.split(X, y):
        for kind in FEATURE_KINDS:
            gs = tune_and_fit(kind, X[tr], y[tr], seed=seed)
            proba = predict_proba_ordered(gs.best_estimator_, X[te])
            pred = np.array(CATEGORIES)[proba.argmax(axis=1)]
            m = compute_metrics(y[te], pred)
            out[kind]["macro_f1"].append(m["macro_f1"])
            out[kind]["accuracy"].append(m["accuracy"])
            out[kind]["binary_f1"].append(m["binary_scam_vs_none"]["f1"])
    return {
        k: {
            met: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_folds": len(v)}
            for met, v in d.items()
        }
        for k, d in out.items()
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resplit", action="store_true", help="пересоздать split-файл")
    ap.add_argument("--cv", action="store_true", help="повторная вложенная CV на всех данных")
    ap.add_argument("--save-model", action="store_true", help="сохранить выбранную модель (joblib)")
    args = ap.parse_args(argv)

    df = load_dataset(args.labels, args.cache)

    if args.split_file.exists() and not args.resplit:
        sp = pd.read_csv(args.split_file)[["video_id", "split"]]
        df = df.merge(sp, on="video_id", how="inner")
        print(f"\nSplit загружен из {args.split_file} ({len(df)} видео)")
    else:
        df = make_split(df, seed=args.seed)
        args.split_file.parent.mkdir(parents=True, exist_ok=True)
        df[["video_id", "category", "split"]].to_csv(args.split_file, index=False)
        print(f"\nSplit создан (seed={args.seed}) и сохранён в {args.split_file}")
    check_split(df)
    print("Проверки пройдены: нет пересечений train/test, все 6 классов в обеих частях\n")
    print_distribution(df)

    exp = run_experiment(df, seed=args.seed)
    n_train = int((df["split"] == "train").sum())
    n_test = int((df["split"] == "test").sum())
    print(f"\n=== Baseline: TF-IDF + LogReg (train={n_train}, test={n_test}) ===")
    hdr = f"{'features':<10}{'C*':>6}{'trainCV F1':>12}{'test macroF1':>14}{'test acc':>10}{'bin F1':>8}{'bin PR-AUC':>12}"
    print(hdr)
    for kind, m in exp["results"].items():
        b = m["binary_scam_vs_none"]
        print(
            f"{kind:<10}{m['best_C']:>6g}{m['train_cv_macro_f1']:>12.3f}"
            f"{m['macro_f1']:>14.3f}{m['accuracy']:>10.3f}{b['f1']:>8.3f}{b['pr_auc']:>12.3f}"
        )
    chosen = exp["chosen_by_train_cv"]
    print(f"\nВыбран по train-CV (не по test): {chosen}")
    m = exp["results"][chosen]
    print("\nPer-class F1 (test):")
    for c, v in m["per_class"].items():
        print(f"  {c:<20} F1={v['f1']:.3f}  P={v['precision']:.3f}  R={v['recall']:.3f}  n={v['support']}")
    print("\nConfusion matrix (test, строки=истина):")
    print(format_confusion(m["confusion_matrix"]))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    for kind, (pred, proba) in exp["test_preds"].items():
        for i, vid in enumerate(exp["test_ids"]):
            row = {"video_id": vid, "features": kind, "true": exp["y_test"][i], "pred": pred[i]}
            row.update({f"p_{c}": proba[i, j] for j, c in enumerate(CATEGORIES)})
            row["risk_baseline"] = 1.0 - proba[i, CATEGORIES.index("none")]
            pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(args.results_dir / "baseline_predictions.csv", index=False)

    payload = {
        "seed": args.seed,
        "n_train": n_train,
        "n_test": n_test,
        "chosen_by_train_cv": chosen,
        "results": exp["results"],
    }
    if args.cv:
        print("\nПовторная вложенная CV (5 fold x 3 повтора) на всех видео...")
        cv = nested_cv(df, seed=args.seed)
        payload["nested_cv_all_data"] = cv
        for kind, d in cv.items():
            print(
                f"  {kind:<10} macroF1 {d['macro_f1']['mean']:.3f} ± {d['macro_f1']['std']:.3f}"
                f" | acc {d['accuracy']['mean']:.3f} ± {d['accuracy']['std']:.3f}"
                f" | binF1 {d['binary_f1']['mean']:.3f} ± {d['binary_f1']['std']:.3f}"
            )
    with open(args.results_dir / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nСохранено: {args.results_dir / 'baseline_metrics.json'}, baseline_predictions.csv")

    if args.save_model:
        import joblib

        out = ROOT / "models" / "baseline.joblib"
        out.parent.mkdir(exist_ok=True)
        joblib.dump(exp["models"][chosen], out)
        print(f"Модель сохранена: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
