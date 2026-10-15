"""adversarial.py — подбор и проверка adversarial-кейсов.
  python tools/adversarial.py find     -> docs/adversarial_cases.csv
  python tools/adversarial.py report   -> results/adversarial_report.md (после прогона судьи)
"""
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
LABELS = ROOT / "data" / "labels.csv"
JUDGE = ROOT / "data" / (sys.argv[2] if len(sys.argv) > 2 else "judge_results_asr_ocr.csv")
CASES = ROOT / "docs" / "adversarial_cases.csv"
REPORT = ROOT / "results" / "adversarial_report.md"

KZ_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
OBFUSC = re.compile(r"(?i)[а-яё][0-9@$][а-яё]|(?:\b[а-яё]\s){5,}")
MONEY_WORDS = re.compile(r"(?i)деньг|заработ|доход|инвест|табыс|ақша")
TG = re.compile(r"(?i)t\.me|telegram|телеграм|wa\.me|whatsapp")


def load_json(p):
    try:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def load_labels():
    if not LABELS.exists():
        return {}
    df = pd.read_csv(LABELS, dtype=str).fillna("")
    return {r["video_id"]: r for _, r in df.iterrows()}


def collect():
    labels = load_labels()
    rows = {}
    for p in CACHE.glob("*.ocr.json"):
        vid = p.name.split(".")[0]
        ocr = " ".join(i.get("text", "") for i in load_json(p).get("items", []))
        asr_path = CACHE / f"{vid}.asr.json"
        asr_exists = asr_path.exists()
        asr = str(load_json(asr_path).get("text", "")) if asr_exists else ""
        rows[vid] = {"ocr": ocr, "asr": asr, "asr_exists": asr_exists, "label": labels.get(vid, {})}
    return rows


def find():
    rows = collect()
    if not rows:
        print("Нет data/cache/*.ocr.json — сначала запусти OCR.")
        return
    picks = {"no_speech": [], "no_text": [], "kazakh": [], "obfuscation": [], "hard_negative": [], "telegram_scam": []}
    for vid, r in rows.items():
        lab = r["label"].get("category", "")
        both = r["ocr"] + " " + r["asr"]
        if r["asr_exists"] and len(r["asr"].strip()) < 5:
            picks["no_speech"].append(vid)
        if len(r["ocr"].strip()) < 5:
            picks["no_text"].append(vid)
        if sum(ch in KZ_LETTERS for ch in both) >= 3:
            picks["kazakh"].append(vid)
        if OBFUSC.search(both):
            picks["obfuscation"].append(vid)
        if lab == "none" and MONEY_WORDS.search(both):
            picks["hard_negative"].append(vid)
        if lab not in ("", "none") and TG.search(r["ocr"]):
            picks["telegram_scam"].append(vid)

    what = {
        "no_speech": "Видео без речи: судья не должен падать, вердикт по тексту/картинке",
        "no_text": "Нет текста на экране: вердикт только по речи и описанию",
        "kazakh": "Казахский язык: категория определяется верно",
        "obfuscation": "Обфускация («инв3стиции», «и н в е с т»): не обходит признаки",
        "hard_negative": "Безобидное видео со словами про деньги: риск должен быть низким",
        "telegram_scam": "Скам со ссылкой на Telegram: риск должен быть высоким",
    }
    out, n = [], 1
    for typ, ids in picks.items():
        for vid in ids[:2]:
            lab = rows[vid]["label"].get("category", "")
            out.append({
                "case_id": f"A{n:02d}", "type": typ, "video_id": vid,
                "expected_label": lab or "?",
                "expected_risk": "low" if lab == "none" else "high",
                "what_to_check": what[typ],
            })
            n += 1
    CASES.parent.mkdir(parents=True, exist_ok=True)
    with open(CASES, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"Найдено кандидатов: { {k: len(v) for k, v in picks.items()} }")
    print(f"Записано кейсов: {len(out)} -> {CASES}")
    print("Проверь файл глазами: если типа не хватает, найди такое видео вручную и допиши строку.")
    print("ID для Panda (прогнать судью):", ",".join(c["video_id"] for c in out))


def pick_col(df, cands):
    return next((c for c in cands if c in df.columns), None)


def report():
    cases = pd.read_csv(CASES, dtype=str).fillna("")
    if not JUDGE.exists():
        print(f"Нет {JUDGE} — подтяни свежий main от Panda.")
        return
    j = pd.read_csv(JUDGE, dtype={"video_id": str})
    cat_col = pick_col(j, ["category_pred", "pred_category", "category", "label_pred", "pred_label", "cat_pred"])
    jm = j.set_index("video_id")
    lines = ["# Adversarial report", "",
             "| Кейс | Тип | Ожидали | risk_pred | Категория | Итог |", "|---|---|---|---|---|---|"]
    ok = total = 0
    for _, c in cases.iterrows():
        if c["video_id"] not in jm.index:
            lines.append(f"| {c['case_id']} | {c['type']} | {c['expected_label']} ({c['expected_risk']}) | — | — | нет в {JUDGE.name} |")
            continue
        row = jm.loc[c["video_id"]]
        row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        risk = float(row["risk_pred"])
        passed = (risk < 0.5) if c["expected_risk"] == "low" else (risk >= 0.5)
        total += 1
        ok += int(passed)
        cat = row[cat_col] if cat_col else "—"
        lines.append(f"| {c['case_id']} | {c['type']} | {c['expected_label']} ({c['expected_risk']}) | {risk:.2f} | {cat} | {'OK' if passed else 'FAIL'} |")
    lines += ["", f"Пройдено: {ok}/{total}. Порог 0.5 для оценки low/high выбран нами и не подбирался под данные."]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"find": find, "report": report}.get(cmd, lambda: print(__doc__))()