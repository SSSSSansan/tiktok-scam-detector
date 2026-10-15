"""Оценка и сравнение систем Scam Radar на ОДИНАКОВЫХ test-видео.

Сравниваются:
  * baseline  -- TF-IDF + LogReg (results/baseline_predictions.csv)
  * LLM       -- предсказания судьи только по речи (data/judge_results_asr.csv)

Что считается
-------------
* macro-F1, accuracy, per-class F1 (6 классов), confusion matrix;
* бинарные scam/none: precision, recall, F1, accuracy, PR-AUC;
* Brier score и ECE для бинарного риска (score = risk_pred / risk_baseline,
  цель = «не none» по разметке);
* Spearman-корреляция риска с risk_human (ручная оценка);
* 95% бутстрэп-доверительные интервалы (по видео) и ПАРНЫЙ бутстрэп разницы
  LLM - baseline: доля ресэмплов, где LLM лучше.

Ограничения (честно): test = 51 видео, в классе pyramid всего 5; ASR-only;
features.py разрабатывался с просмотром этих расшифровок.

Запуск (из корня репозитория, Windows CMD)::

    python src\\evaluate.py
    python src\\evaluate.py --n-boot 5000 --baseline-features char
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline_model import (  # noqa: E402
    CATEGORIES,
    DEFAULT_RESULTS,
    DEFAULT_SPLIT,
    ROOT,
    compute_metrics,
    format_confusion,
)

DEFAULT_JUDGE = ROOT / "data" / "judge_results_asr.csv"
DEFAULT_BASE_PRED = DEFAULT_RESULTS / "baseline_predictions.csv"
DEFAULT_BASE_METRICS = DEFAULT_RESULTS / "baseline_metrics.json"
SEED = 42
_IDX = {c: i for i, c in enumerate(CATEGORIES)}
_NONE = _IDX["none"]


# --------------------------------------------------------------------------
# Калибровочные метрики (переиспользуются в calibrate.py)
# --------------------------------------------------------------------------
def brier_score(y_true_binary, prob) -> float:
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true_binary, prob, n_bins: int = 10) -> float:
    """ECE с равноширокими бинами на [0, 1]."""
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def reliability_table(y_true_binary, prob, n_bins: int = 5) -> pd.DataFrame:
    y = np.asarray(y_true_binary, dtype=float)
    p = np.asarray(prob, dtype=float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append(
            {
                "bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
                "n": int(m.sum()),
                "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                "frac_scam": float(y[m].mean()) if m.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Быстрые метрики для бутстрэпа (на индексах классов)
# --------------------------------------------------------------------------
def _macro_f1_idx(yt: np.ndarray, yp: np.ndarray) -> float:
    k = len(CATEGORIES)
    tp = np.bincount(yt[yt == yp], minlength=k).astype(float)
    pred = np.bincount(yp, minlength=k).astype(float)
    true = np.bincount(yt, minlength=k).astype(float)
    denom = pred + true
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def _binary_f1(bt: np.ndarray, bp: np.ndarray) -> float:
    tp = float(np.sum((bt == 1) & (bp == 1)))
    denom = 2 * tp + np.sum((bt == 0) & (bp == 1)) + np.sum((bt == 1) & (bp == 0))
    return float(2 * tp / denom) if denom > 0 else 0.0


def _point_metrics(yt, yp, score) -> dict:
    bt = (yt != _NONE).astype(int)
    bp = (yp != _NONE).astype(int)
    return {
        "macro_f1": _macro_f1_idx(yt, yp),
        "accuracy": float(np.mean(yt == yp)),
        "binary_f1": _binary_f1(bt, bp),
        "binary_pr_auc": float(average_precision_score(bt, score)) if bt.sum() else np.nan,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    systems: dict[str, tuple[np.ndarray, np.ndarray]],
    n_boot: int = 2000,
    seed: int = SEED,
) -> dict:
    """Бутстрэп по видео. ``systems[name] = (pred_idx, score)``.

    Одни и те же ресэмплы применяются ко всем системам -> парная разница.
    Возвращает CI для каждой метрики и (если систем две) разность a - b.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    names = list(systems)
    samples = {s: {m: [] for m in ("macro_f1", "accuracy", "binary_f1", "binary_pr_auc")} for s in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        for s in names:
            yp, sc = systems[s]
            pm = _point_metrics(yt, yp[idx], sc[idx])
            for m, v in pm.items():
                samples[s][m].append(v)

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if len(a) else [None, None]

    out = {"n_boot": n_boot, "ci": {s: {m: ci(v) for m, v in d.items()} for s, d in samples.items()}}
    if len(names) == 2:
        a, b = names
        diff = {}
        for m in samples[a]:
            d = np.asarray(samples[a][m]) - np.asarray(samples[b][m])
            d = d[~np.isnan(d)]
            diff[m] = {
                "mean_diff": float(d.mean()),
                "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_first_better": float(np.mean(d > 0)),
            }
        out["diff"] = {"first": a, "second": b, "metrics": diff}
    return out


# --------------------------------------------------------------------------
# Загрузка и выравнивание
# --------------------------------------------------------------------------
def load_aligned(
    split_file: Path,
    judge_file: Path,
    base_pred_file: Path,
    base_features: str,
    labels_file: Path | None = None,
) -> pd.DataFrame:
    """Test-видео с предсказаниями обеих систем (строго одинаковый набор)."""
    sp = pd.read_csv(split_file)
    test_ids = set(sp.loc[sp["split"] == "test", "video_id"])
    judge = pd.read_csv(judge_file)
    base = pd.read_csv(base_pred_file)
    base = base[base["features"] == base_features]
    if base.empty:
        raise ValueError(f"В {base_pred_file} нет признаков {base_features!r}")

    judge = judge[judge["video_id"].isin(test_ids)]
    base = base[base["video_id"].isin(test_ids)]
    if set(judge["video_id"]) != test_ids or set(base["video_id"]) != test_ids:
        raise ValueError(
            "Наборы видео не совпадают: test=%d, judge=%d, baseline=%d"
            % (len(test_ids), judge["video_id"].nunique(), base["video_id"].nunique())
        )
    df = judge[["video_id", "true_category", "risk_human", "pred_category", "risk_pred"]].merge(
        base[["video_id", "true", "pred", "risk_baseline"] + [f"p_{c}" for c in CATEGORIES]],
        on="video_id",
        validate="one_to_one",
    )
    if not (df["true_category"] == df["true"]).all():
        raise ValueError("Расхождение true-меток между judge и baseline")
    if len(df) != len(test_ids):
        raise ValueError("Потеряны видео при объединении")
    df = df.rename(columns={"pred": "pred_baseline", "pred_category": "pred_llm", "risk_pred": "risk_llm"})
    return df.sort_values("video_id").reset_index(drop=True)


def system_report(name, y_true, y_pred, risk, risk_human, proba=None) -> dict:
    m = compute_metrics(y_true, y_pred, proba)
    bt = (np.asarray(y_true) != "none").astype(int)
    risk = np.asarray(risk, dtype=float)
    m["binary_scam_vs_none"]["pr_auc_risk_score"] = float(average_precision_score(bt, risk))
    m["risk_brier"] = brier_score(bt, risk)
    m["risk_ece_10bins"] = expected_calibration_error(bt, risk, 10)
    rho = spearmanr(risk, risk_human).statistic if len(set(risk)) > 1 else np.nan
    m["risk_spearman_vs_human"] = None if np.isnan(rho) else float(rho)
    m["name"] = name
    return m


# --------------------------------------------------------------------------
# Вывод
# --------------------------------------------------------------------------
def _fmt_ci(v, ci) -> str:
    return f"{v:.3f} [{ci[0]:.3f}; {ci[1]:.3f}]"


def build_markdown(rep: dict) -> str:
    n = rep["n_test"]
    L = [f"# Оценка на test (n={n}, TikTok, только ASR)", ""]
    L.append(f"Baseline: TF-IDF `{rep['baseline_features']}` + LogReg. LLM: судья по речи. "
             f"95% CI — бутстрэп по видео ({rep['bootstrap']['n_boot']} ресэмплов).")
    L += ["", "| Метрика | Baseline | LLM |", "|---|---|---|"]
    for key, label in [("macro_f1", "macro-F1 (6 классов)"), ("accuracy", "accuracy (6 классов)"),
                       ("binary_f1", "scam/none F1"), ("binary_pr_auc", "scam/none PR-AUC")]:
        vals = []
        for s in ("baseline", "llm"):
            sysm = rep["systems"][s]
            point = {"macro_f1": sysm["macro_f1"], "accuracy": sysm["accuracy"],
                     "binary_f1": sysm["binary_scam_vs_none"]["f1"],
                     "binary_pr_auc": sysm["binary_scam_vs_none"]["pr_auc_risk_score"]}[key]
            vals.append(_fmt_ci(point, rep["bootstrap"]["ci"][s][key]))
        L.append(f"| {label} | {vals[0]} | {vals[1]} |")
    L += ["", "## Разница LLM − baseline (парный бутстрэп)", "", "| Метрика | Δ среднее | 95% CI | P(LLM лучше) |", "|---|---|---|---|"]
    for m, d in rep["bootstrap"]["diff"]["metrics"].items():
        L.append(f"| {m} | {d['mean_diff']:+.3f} | [{d['ci'][0]:+.3f}; {d['ci'][1]:+.3f}] | {d['p_first_better']:.3f} |")
    L += ["", "## Per-class F1", "", "| Класс | n | Baseline | LLM |", "|---|---|---|---|"]
    for c in CATEGORIES:
        b = rep["systems"]["baseline"]["per_class"][c]
        l = rep["systems"]["llm"]["per_class"][c]
        L.append(f"| {c} | {b['support']} | {b['f1']:.3f} | {l['f1']:.3f} |")
    L += ["", "## Качество risk-score (бинарно scam vs none)", "", "| | Brier | ECE | Spearman с risk_human |", "|---|---|---|---|"]
    for s in ("baseline", "llm"):
        m = rep["systems"][s]
        sp = m["risk_spearman_vs_human"]
        L.append(f"| {s} | {m['risk_brier']:.3f} | {m['risk_ece_10bins']:.3f} | {'—' if sp is None else f'{sp:.3f}'} |")
    L += ["", "## Ограничения", "",
          "- Test мал (51 видео, в pyramid 5): CI широкие, per-class F1 шумный.",
          "- Только речь (ASR); 10 из 127 видео без речи; OCR/VLM/метаданные не использованы.",
          "- `features.py` писался с просмотром этих расшифровок -> test не полностью независим.",
          "- Baseline обучен на 76 видео. LLM-судья не обучался на них, но если few-shot примеры промпта взяты из нашей разметки и попали в test, оценка LLM завышена (уточнить у автора llm_judge)."]
    return "\n".join(L) + "\n"


def plot_confusions(rep: dict, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    short = [c.replace("investment_platform", "invest").replace("_", "\n") for c in CATEGORIES]
    for ax, s in zip(axes, ("baseline", "llm")):
        cm = np.array(rep["systems"][s]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6), short, fontsize=8)
        ax.set_yticks(range(6), short, fontsize=8)
        ax.set_xlabel("предсказано")
        ax.set_ylabel("истина")
        ax.set_title(f"{s}: macro-F1={rep['systems'][s]['macro_f1']:.2f}")
        for i in range(6):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT)
    ap.add_argument("--judge", type=Path, default=DEFAULT_JUDGE)
    ap.add_argument("--baseline-pred", type=Path, default=DEFAULT_BASE_PRED)
    ap.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASE_METRICS)
    ap.add_argument("--baseline-features", default=None,
                    help="word | char | word_char (по умолчанию: выбранный по train-CV)")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    feats = args.baseline_features
    if feats is None:
        with open(args.baseline_metrics, encoding="utf-8") as f:
            feats = json.load(f)["chosen_by_train_cv"]

    df = load_aligned(args.split_file, args.judge, args.baseline_pred, feats)
    n = len(df)
    print(f"Test-видео: {n} (одинаковый набор для baseline и LLM); baseline = {feats}")

    y = df["true_category"].to_numpy(dtype=object)
    proba = df[[f"p_{c}" for c in CATEGORIES]].values
    systems = {
        "baseline": system_report("baseline", y, df["pred_baseline"].to_numpy(dtype=object), df["risk_baseline"].values,
                                  df["risk_human"].values, proba),
        "llm": system_report("llm", y, df["pred_llm"].to_numpy(dtype=object), df["risk_llm"].values, df["risk_human"].values),
    }

    yt = np.array([_IDX[c] for c in y])
    boot = paired_bootstrap(
        yt,
        {
            "llm": (np.array([_IDX[c] for c in df["pred_llm"]]), df["risk_llm"].values.astype(float)),
            "baseline": (np.array([_IDX[c] for c in df["pred_baseline"]]), df["risk_baseline"].values.astype(float)),
        },
        n_boot=args.n_boot,
        seed=args.seed,
    )
    rep = {"n_test": n, "baseline_features": feats, "systems": systems, "bootstrap": boot}

    md = build_markdown(rep)
    print("\n" + md)
    for s in ("baseline", "llm"):
        print(f"Confusion matrix [{s}] (строки=истина):")
        print(format_confusion(systems[s]["confusion_matrix"]) + "\n")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    with open(args.results_dir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    (args.results_dir / "evaluation_report.md").write_text(md, encoding="utf-8")
    ok = plot_confusions(rep, args.results_dir / "confusion_matrices.png")
    print("Сохранено: evaluation.json, evaluation_report.md" + (", confusion_matrices.png" if ok else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
