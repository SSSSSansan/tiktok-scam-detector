"""Калибровка итогового Risk Score: LogReg поверх (скор LLM + категория LLM + флаги).

Контракт (PROJECT_CONTEXT): логистическая регрессия поверх (признаки + скор LLM)
-> финальный risk 0..1. Категория по-прежнему берётся из вердикта LLM.

Цель калибровки: P(видео -- скам), где скам = категория != none (по разметке).

Варианты (все сравниваются на одинаковом train/test из data/split_tiktok.csv):
  raw_llm         -- сырой risk_pred судьи (без калибровки), точка отсчёта
  platt           -- LogReg только на logit(risk_pred)       (классический Platt)
  llm_cat         -- + «LLM назвала категорию, не none»
  llm_cat_speech  -- llm_cat + no_speech + log(1+слов в ASR): у видео без речи LLM ставит
                     risk 0.1 («нет информации»), а не «безопасно» (см. отчёт)
  llm_cat_groups  -- + счётчики флагов features.py по 6 группам
  llm_cat_flags   -- + все 27 флагов features.py по отдельности
  flags_only      -- только группы флагов, БЕЗ LLM (ablation)

ВАЖНО: llm_cat_speech добавлен ПОСЛЕ диагностики на всех 127 видео (включая test),
поэтому это не «слепой» выбор -- см. оговорки в отчёте.

Защита от утечки и переобучения:
  * L2-регуляризация; C подбирается ВНУТРИ train (5-fold CV, критерий Brier);
  * вариант для «основного» результата выбирается по train-CV Brier, не по test;
  * test используется один раз для итоговой оценки;
  * вложенная повторная CV на всех 127 видео -- устойчивая оценка при малом test;
  * признаки считаются на ASR (как и LLM-судья), OCR/meta нет.
Оговорка: features.py писался с просмотром этих 127 расшифровок, поэтому
вклад флагов оптимистичен и не полностью независим.

Запуск (из корня репозитория, Windows CMD)::

    python src\\calibrate.py
    python src\\calibrate.py --fit-final     # обучить на всех 127 и сохранить для pipeline.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_model import (  # noqa: E402
    DEFAULT_CACHE,
    DEFAULT_RESULTS,
    DEFAULT_SPLIT,
    ROOT,
)
from src.evaluate import (  # noqa: E402
    DEFAULT_JUDGE,
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from src.features import FLAG_GROUPS, FLAG_NAMES, extract_features  # noqa: E402

SEED = 42
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 100]
SCALED_COLS = ("llm_logit", "log_words")   # масштабируются StandardScaler, остальные -- 0/1 и малые счётчики
LLM_CLIP = 0.05          # logit(risk) считаем на risk, обрезанном в [0.05, 0.95]
COUNT_CAP = 3            # счётчик флагов группы не больше 3
GROUPS = sorted(set(FLAG_GROUPS.values()))
GROUP_COLS = [f"g_{g}" for g in GROUPS]
FLAG_COLS = [f"f_{n}" for n in FLAG_NAMES]

VARIANTS: dict[str, list[str]] = {
    "platt": ["llm_logit"],
    "llm_cat": ["llm_logit", "llm_scam"],
    "llm_cat_speech": ["llm_logit", "llm_scam", "no_speech", "log_words"],
    "llm_cat_groups": ["llm_logit", "llm_scam"] + GROUP_COLS,
    "llm_cat_flags": ["llm_logit", "llm_scam"] + FLAG_COLS,
    "flags_only": GROUP_COLS,
}


# --------------------------------------------------------------------------
# Данные и признаки
# --------------------------------------------------------------------------
def logit_risk(risk) -> np.ndarray:
    p = np.clip(np.asarray(risk, dtype=float), LLM_CLIP, 1 - LLM_CLIP)
    return np.log(p / (1 - p))


def design_from_flags(risk_llm, category_llm, flags: pd.DataFrame, n_words=None) -> pd.DataFrame:
    """Единая сборка признаков (и для обучения, и для pipeline.py).

    ``flags`` -- DataFrame с колонками из FLAG_NAMES (значения 0/1).
    ``n_words`` -- число слов в ASR-тексте (нужно вариантам со speech-признаками).
    """
    out = pd.DataFrame(index=flags.index)
    out["llm_logit"] = logit_risk(risk_llm)
    out["llm_scam"] = (np.asarray(category_llm, dtype=object) != "none").astype(float)
    if n_words is not None:
        nw = np.asarray(n_words, dtype=float)
        out["no_speech"] = (nw == 0).astype(float)
        out["log_words"] = np.log1p(nw)
    for g in GROUPS:
        cols = [n for n in FLAG_NAMES if FLAG_GROUPS[n] == g]
        out[f"g_{g}"] = np.minimum(flags[cols].sum(axis=1).to_numpy(dtype=float), COUNT_CAP)
    for n in FLAG_NAMES:
        out[f"f_{n}"] = flags[n].to_numpy(dtype=float)
    return out


def flags_table(video_ids, cache_dir: Path) -> tuple[pd.DataFrame, list[int]]:
    """Флаги features.py и число слов по ASR-кэшу (extract_features(asr=...))."""
    rows, n_words = [], []
    for vid in video_ids:
        with open(Path(cache_dir) / f"{vid}.asr.json", encoding="utf-8") as f:
            asr = json.load(f)
        rows.append(extract_features(asr=asr)["flags"])
        n_words.append(len(str(asr.get("text") or "").split()))
    return pd.DataFrame(rows, columns=list(FLAG_NAMES)).astype(int), n_words


def load_calibration_data(judge_file: Path, split_file: Path, cache_dir: Path) -> pd.DataFrame:
    judge = pd.read_csv(judge_file)
    split = pd.read_csv(split_file)[["video_id", "split"]]
    df = judge.merge(split, on="video_id", how="inner", validate="one_to_one")
    if len(df) != len(judge):
        raise ValueError(f"В split нет {len(judge) - len(df)} видео из judge-файла")
    df = df.sort_values("video_id").reset_index(drop=True)
    df["y_scam"] = (df["true_category"] != "none").astype(int)
    flags, n_words = flags_table(df["video_id"], cache_dir)
    feats = design_from_flags(df["risk_pred"], df["pred_category"], flags, n_words)
    df["n_words"] = n_words
    return pd.concat([df, feats], axis=1)


# --------------------------------------------------------------------------
# Модель
# --------------------------------------------------------------------------
def build_model(columns: list[str], C: float = 1.0) -> Pipeline:
    scale = [c for c in columns if c in SCALED_COLS]
    pre = ColumnTransformer([("scale", StandardScaler(), scale)], remainder="passthrough") if scale else "passthrough"
    return Pipeline(
        [("pre", pre), ("clf", LogisticRegression(C=C, max_iter=5000, random_state=SEED))]
    )


def tune_fit(X: pd.DataFrame, y, columns: list[str], seed: int = SEED) -> GridSearchCV:
    """C по CV внутри переданных (train) данных, критерий -- Brier."""
    y = np.asarray(y)
    n_splits = max(2, min(5, int(np.bincount(y).min())))
    gs = GridSearchCV(
        build_model(columns),
        {"clf__C": C_GRID},
        scoring="neg_brier_score",
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
        refit=True,
    )
    gs.fit(X[columns], y)
    return gs


def prob_scam(model, X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(X[columns])[:, 1]


# --------------------------------------------------------------------------
# Метрики
# --------------------------------------------------------------------------
def prob_metrics(y, p, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "brier": brier_score(y, p),
        "ece_10": expected_calibration_error(y, p, 10),
        "ece_5": expected_calibration_error(y, p, 5),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() else None,
        "accuracy@0.5": float(np.mean(pred == y)),
        "f1@0.5": float(f1_score(y, pred, zero_division=0)),
        "mean_pred": float(p.mean()),
        "base_rate": float(y.mean()),
    }


def bootstrap_brier_diff(y, p_new, p_old, n_boot=2000, seed=SEED) -> dict:
    """Парный бутстрэп: Brier(new) - Brier(old); < 0 значит калибровка помогла."""
    y, p_new, p_old = (np.asarray(a, dtype=float) for a in (y, p_new, p_old))
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        d[b] = np.mean((p_new[i] - y[i]) ** 2) - np.mean((p_old[i] - y[i]) ** 2)
    return {
        "mean_diff": float(d.mean()),
        "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_new_better": float(np.mean(d < 0)),
    }


# --------------------------------------------------------------------------
# Эксперименты
# --------------------------------------------------------------------------
def run_split_experiment(df: pd.DataFrame, seed: int = SEED, n_boot: int = 2000) -> dict:
    tr, te = df[df["split"] == "train"], df[df["split"] == "test"]
    y_tr, y_te = tr["y_scam"].to_numpy(), te["y_scam"].to_numpy()

    out = {"n_train": len(tr), "n_test": len(te), "variants": {}, "test_probs": {}}
    raw_te = te["risk_pred"].to_numpy(dtype=float)
    out["variants"]["raw_llm"] = {
        "test": prob_metrics(y_te, raw_te),
        "train": prob_metrics(y_tr, tr["risk_pred"].to_numpy(dtype=float)),
    }
    out["test_probs"]["raw_llm"] = raw_te

    models = {}
    for name, cols in VARIANTS.items():
        gs = tune_fit(tr, y_tr, cols, seed)
        p_te = prob_scam(gs.best_estimator_, te, cols)
        out["variants"][name] = {
            "best_C": float(gs.best_params_["clf__C"]),
            "train_cv_brier": float(-gs.best_score_),
            "llm_logit_coef": llm_logit_coef(gs.best_estimator_, cols),
            "test": prob_metrics(y_te, p_te),
            "brier_diff_vs_raw": bootstrap_brier_diff(y_te, p_te, raw_te, n_boot, seed),
        }
        out["test_probs"][name] = p_te
        models[name] = gs.best_estimator_

    cand = {k: v["train_cv_brier"] for k, v in out["variants"].items() if k != "raw_llm"}
    out["chosen_by_train_cv"] = min(cand, key=cand.get)
    out["models"] = models
    out["test_ids"] = te["video_id"].tolist()
    out["y_test"] = y_te
    return out


def nested_cv(df: pd.DataFrame, n_splits=5, n_repeats=3, seed=SEED) -> dict:
    """Повторная вложенная CV на всех видео. C подбирается внутри каждого train-fold."""
    y = df["y_scam"].to_numpy()
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    names = ["raw_llm"] + list(VARIANTS)
    acc = {n: {"brier": [], "ece_5": [], "roc_auc": [], "f1@0.5": []} for n in names}
    oof = {n: np.full(len(df), np.nan) for n in names}  # out-of-fold, первый повтор
    for k, (tr, te) in enumerate(rskf.split(df, y)):
        d_tr, d_te = df.iloc[tr], df.iloc[te]
        probs = {"raw_llm": d_te["risk_pred"].to_numpy(dtype=float)}
        for name, cols in VARIANTS.items():
            gs = tune_fit(d_tr, y[tr], cols, seed)
            probs[name] = prob_scam(gs.best_estimator_, d_te, cols)
        for n in names:
            m = prob_metrics(y[te], probs[n])
            for key in acc[n]:
                acc[n][key].append(m[key])
            if k < n_splits:
                oof[n][te] = probs[n]
    summary = {
        n: {key: {"mean": float(np.mean(v)), "std": float(np.std(v))} for key, v in d.items()}
        for n, d in acc.items()
    }
    return {"summary": summary, "oof_first_repeat": oof, "n_folds": n_splits * n_repeats}


# --------------------------------------------------------------------------
# Сохранение калибратора для pipeline.py (без pickle своих классов)
# --------------------------------------------------------------------------
def save_calibrator(model: Pipeline, variant: str, path: Path, meta: dict | None = None) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"variant": variant, "columns": VARIANTS[variant], "pipeline": model, "meta": meta or {}},
        path,
    )


def load_calibrator(path: Path):
    """Возвращает функцию ``risk(risk_llm, category_llm, features_result) -> float``.

    ``features_result`` -- результат ``extract_features`` (dict с ключом "flags").
    ``n_words`` -- ``len(asr["text"].split())``; обязателен для speech-вариантов.
    """
    import joblib

    blob = joblib.load(path)

    def risk(risk_llm: float, category_llm: str, features_result: dict, n_words: int | None = None) -> float:
        if n_words is None and ("no_speech" in blob["columns"] or "log_words" in blob["columns"]):
            raise ValueError("Этому калибратору нужен n_words (число слов в ASR-тексте)")
        flags = pd.DataFrame([{n: int(features_result["flags"].get(n, 0)) for n in FLAG_NAMES}])
        X = design_from_flags([risk_llm], [category_llm], flags, None if n_words is None else [n_words])
        return float(blob["pipeline"].predict_proba(X[blob["columns"]])[0, 1])

    return risk


# --------------------------------------------------------------------------
# Отчёт
# --------------------------------------------------------------------------
def coef_table(model: Pipeline, columns: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    # ColumnTransformer ставит масштабируемые колонки первыми
    scaled = [c for c in columns if c in SCALED_COLS]
    order = scaled + [c for c in columns if c not in scaled] if scaled else columns
    return (
        pd.DataFrame({"feature": order, "coef": clf.coef_[0]})
        .assign(abs_coef=lambda d: d["coef"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )


def speech_subgroups(df: pd.DataFrame, oof: dict, threshold: int = 15) -> dict:
    """Brier по out-of-fold предсказаниям отдельно для видео с малым объёмом речи.

    Порог выбран ПОСЛЕ просмотра данных (все 14 таких видео получили risk 0.1).
    """
    poor = (df["n_words"] < threshold).to_numpy()
    y = df["y_scam"].to_numpy(dtype=float)
    out = {"threshold_words": threshold, "n_poor": int(poor.sum()), "n_rich": int((~poor).sum()),
           "scam_rate_poor": float(y[poor].mean()), "scam_rate_rich": float(y[~poor].mean()), "brier": {}}
    for name, p in oof.items():
        ok = ~np.isnan(p)
        out["brier"][name] = {
            "poor": float(np.mean((p[poor & ok] - y[poor & ok]) ** 2)),
            "rich": float(np.mean((p[~poor & ok] - y[~poor & ok]) ** 2)),
        }
    return out


def llm_logit_coef(model: Pipeline, columns: list[str]) -> float | None:
    """Коэффициент при logit(risk LLM) (после StandardScaler). None, если признака нет."""
    t = coef_table(model, columns)
    row = t[t["feature"] == "llm_logit"]
    return float(row["coef"].iloc[0]) if len(row) else None


def build_markdown(rep: dict) -> str:
    L = [f"# Калибровка Risk Score (test n={rep['n_test']}, train n={rep['n_train']}; цель: скам vs none)", ""]
    L.append(f"Вариант, выбранный по train-CV (не по test): **{rep['chosen_by_train_cv']}**.")
    L += ["", "## Test: до и после калибровки", "",
          "| Вариант | C | коэф. при risk LLM | Brier | ECE(5) | ECE(10) | LogLoss | ROC-AUC | PR-AUC | F1@0.5 | ΔBrier vs raw [95% CI] | P(лучше raw) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, v in rep["variants"].items():
        t = v["test"]
        if name == "raw_llm":
            extra = ["—", "—", "—", "—"]
        else:
            d = v["brier_diff_vs_raw"]
            c = v["llm_logit_coef"]
            cs = "—" if c is None else (f"{c:+.2f}" + (" ⚠ немонотонно" if c < 0 else ""))
            extra = [f"{v['best_C']:g}", cs, f"{d['mean_diff']:+.3f} [{d['ci'][0]:+.3f}; {d['ci'][1]:+.3f}]", f"{d['p_new_better']:.2f}"]
        L.append(f"| {name} | {extra[0]} | {extra[1]} | {t['brier']:.3f} | {t['ece_5']:.3f} | {t['ece_10']:.3f} | {t['log_loss']:.3f} | "
                 f"{t['roc_auc']:.3f} | {t['pr_auc']:.3f} | {t['f1@0.5']:.3f} | {extra[2]} | {extra[3]} |")
    if "nested_cv" in rep:
        L += ["", f"## Вложенная CV на всех 127 видео ({rep['nested_cv']['n_folds']} фолдов, mean ± std)", "",
              "| Вариант | Brier | ECE(5) | ROC-AUC | F1@0.5 |", "|---|---|---|---|---|"]
        for name, s in rep["nested_cv"]["summary"].items():
            L.append(f"| {name} | " + " | ".join(f"{s[k]['mean']:.3f} ± {s[k]['std']:.3f}" for k in ("brier", "ece_5", "roc_auc", "f1@0.5")) + " |")
    if "speech_subgroups" in rep:
        g = rep["speech_subgroups"]
        L += ["", f"## Brier (out-of-fold, 127 видео) по объёму речи: <{g['threshold_words']} слов (n={g['n_poor']}, "
              f"доля скама {g['scam_rate_poor']:.2f}) и ≥{g['threshold_words']} слов (n={g['n_rich']}, доля скама {g['scam_rate_rich']:.2f})", "",
              "| Вариант | мало речи | достаточно речи |", "|---|---|---|"]
        for name, b in g["brier"].items():
            L.append(f"| {name} | {b['poor']:.3f} | {b['rich']:.3f} |")
        L.append("\nПорог 15 слов выбран после просмотра данных (все 14 таких видео получили risk 0.1 от LLM = «нет информации»).")
    L += ["", "Ориентир: Brier = 0.25 у константного прогноза 0.5; Brier ≈ p(1-p) ≈ "
          f"{rep['variants']['raw_llm']['test']['base_rate'] * (1 - rep['variants']['raw_llm']['test']['base_rate']):.3f} "
          "у константного прогноза «доля скама» (калибратор без информации).",
          "", "## Ограничения", "",
          "- Test = 51 видео: CI широкие, ECE на 5–10 бинах шумный.",
          "- Только ASR; флаги `features.py` редкие на речи и писались с просмотром этих расшифровок (не независимы).",
          "- `risk_pred` принимает всего 6 значений; калибровка не создаёт информацию, которой нет в входах.",
          "- Если few-shot примеры LLM-судьи взяты из нашей разметки, вход калибратора завышен."]
    return "\n".join(L) + "\n"


def plot_reliability(rep: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    y = rep["y_test"]
    chosen = rep["chosen_by_train_cv"]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="идеал")
    for name, color in (("raw_llm", "tab:red"), (chosen, "tab:blue")):
        t = reliability_table(y, rep["test_probs"][name], 5).dropna()
        ax.plot(t["mean_pred"], t["frac_scam"], "o-", color=color, label=f"{name} (Brier {rep['variants'][name]['test']['brier']:.3f})")
        for _, r in t.iterrows():
            ax.annotate(int(r["n"]), (r["mean_pred"], r["frac_scam"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("предсказанный риск")
    ax.set_ylabel("доля скама среди видео бина")
    ax.set_title(f"Reliability на test (n={len(y)}); подписи = число видео")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-cv", action="store_true", help="пропустить вложенную CV")
    ap.add_argument("--fit-final", action="store_true",
                    help="обучить выбранный вариант на ВСЕХ видео и сохранить в models/calibrator.joblib")
    args = ap.parse_args(argv)

    df = load_calibration_data(args.judge, args.split_file, args.cache)
    n_tr, n_te = int((df.split == "train").sum()), int((df.split == "test").sum())
    print(f"Видео: {len(df)} (train={n_tr}, test={n_te}); доля скама: train={df[df.split=='train'].y_scam.mean():.2f}, test={df[df.split=='test'].y_scam.mean():.2f}")

    exp = run_split_experiment(df, args.seed, args.n_boot)
    rep = {
        "n_train": exp["n_train"], "n_test": exp["n_test"],
        "chosen_by_train_cv": exp["chosen_by_train_cv"],
        "variants": exp["variants"], "y_test": exp["y_test"],
        "test_probs": exp["test_probs"],
    }
    if not args.no_cv:
        print("Вложенная CV (5x3) на всех видео...")
        cv = nested_cv(df, seed=args.seed)
        rep["nested_cv"] = {"summary": cv["summary"], "n_folds": cv["n_folds"]}
        rep["speech_subgroups"] = speech_subgroups(df, cv["oof_first_repeat"])

    md = build_markdown(rep)
    print("\n" + md)
    chosen = exp["chosen_by_train_cv"]
    print(f"Коэффициенты выбранного варианта ({chosen}), обучен на train:")
    print(coef_table(exp["models"][chosen], VARIANTS[chosen]).head(10).to_string(index=False))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    jr = {k: v for k, v in rep.items() if k not in ("y_test", "test_probs")}
    with open(args.results_dir / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(jr, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"video_id": exp["test_ids"], "y_scam": exp["y_test"],
                  **{f"risk_{k}": v for k, v in exp["test_probs"].items()}}
                 ).to_csv(args.results_dir / "calibration_test_predictions.csv", index=False)
    (args.results_dir / "calibration_report.md").write_text(md, encoding="utf-8")
    ok = plot_reliability(rep, args.results_dir / "reliability.png")
    print("\nСохранено: calibration.json, calibration_report.md, calibration_test_predictions.csv" + (", reliability.png" if ok else ""))

    if args.fit_final:
        cols = VARIANTS[chosen]
        gs = tune_fit(df, df["y_scam"].to_numpy(), cols, args.seed)
        c = llm_logit_coef(gs.best_estimator_, cols)
        if c is not None and c < 0:
            print(f"\nПРЕДУПРЕЖДЕНИЕ: коэффициент при risk LLM отрицательный ({c:+.2f}) -- итоговый риск "
                  "немонотонен по риску LLM. Проверьте, что это осознанно (risk 0.1 = «нет речи»).")
        out = ROOT / "models" / "calibrator.joblib"
        save_calibrator(gs.best_estimator_, chosen, out,
                        {"fit_on": "all 127 TikTok videos (ASR only)", "C": float(gs.best_params_["clf__C"]), "seed": args.seed})
        print(f"\nФинальный калибратор ({chosen}, C={gs.best_params_['clf__C']:g}) обучен на всех {len(df)} видео -> {out}")
        print("ВНИМАНИЕ: метрики этой модели считать нельзя (обучена и на test). Для отчёта используйте таблицы выше.")
    return 0


if __name__ == "__main__":
    sys.exit(main())