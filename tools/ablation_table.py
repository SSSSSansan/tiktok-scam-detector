"""
Таблица ablation: сравнение судьи с разными источниками улик на ОДНОМ И ТОМ ЖЕ наборе видео
(только те, что есть в data/split_tiktok.csv). Бесплатно, OpenAI не вызывает.

    python3 tools/ablation_table.py

Результат печатается и сохраняется в results/ablation.md
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

SPLIT = Path("data/split_tiktok.csv")
VARIANTS = [
    ("речь (ASR)", "data/judge_results_asr.csv"),
    ("речь + текст на экране (OCR)", "data/judge_results_asr_ocr.csv"),
    ("речь + OCR + описание кадров (VLM)", "data/judge_results_asr_ocr_vlm.csv"),
]


def metrics(df):
    y_true = df["true_category"]
    y_pred = df["pred_category"]
    b_true = (y_true != "none").astype(int)
    b_pred = (y_pred != "none").astype(int)
    return {
        "n": len(df),
        "acc6": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "bin_acc": accuracy_score(b_true, b_pred),
        "bin_f1": f1_score(b_true, b_pred, zero_division=0),
        "risk_scam": df.loc[b_true == 1, "risk_pred"].astype(float).mean(),
        "risk_none": df.loc[b_true == 0, "risk_pred"].astype(float).mean(),
    }


def table(title, rows):
    out = [f"### {title}", "",
           "| Источники | n | acc (6 классов) | macro-F1 | acc (скам/не скам) | F1 (скам) | ср. risk у скама | ср. risk у none |",
           "|---|---|---|---|---|---|---|---|"]
    for name, m in rows:
        out.append(f"| {name} | {m['n']} | {m['acc6']:.3f} | {m['macro_f1']:.3f} | {m['bin_acc']:.3f} | "
                   f"{m['bin_f1']:.3f} | {m['risk_scam']:.2f} | {m['risk_none']:.2f} |")
    return "\n".join(out)


def main():
    split = pd.read_csv(SPLIT)[["video_id", "split"]]
    all_rows, test_rows = [], []
    for name, path in VARIANTS:
        if not Path(path).exists():
            print(f"нет файла {path}, пропускаю")
            continue
        df = pd.read_csv(path).merge(split, on="video_id", how="inner")
        all_rows.append((name, metrics(df)))
        test_rows.append((name, metrics(df[df["split"] == "test"])))
    md = "\n\n".join([
        "# Ablation: какие источники улик подаём судье",
        table("Все видео TikTok из split (train + test)", all_rows),
        table("Только test", test_rows),
        "Судья без обучения, правила `docs/rubric.md` одинаковые во всех строках. "
        "Test мал (51 видео), поэтому доверительные интервалы широкие.",
    ])
    print(md)
    Path("results").mkdir(exist_ok=True)
    Path("results/ablation.md").write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
