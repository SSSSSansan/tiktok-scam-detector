"""
Прогон LLM-судьи по всему датасету + подсчёт точности.

Из корня репозитория:
  python3 tools/run_judge_all.py --sources asr                 # только оценка (бесплатно)
  python3 tools/run_judge_all.py --sources asr --run --limit 5 # судить 5 видео (платно, копейки)
  python3 tools/run_judge_all.py --sources asr --run           # судить все, у кого есть кэш
  python3 tools/run_judge_all.py --sources asr,ocr,vlm --run   # когда будут OCR и VLM

--sources: какие источники давать судье (для ablation: asr / ocr / vlm / asr,ocr,vlm).
Результат: data/judge_results_<sources>.csv
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import llm_judge  # noqa: E402

LABELS = Path("data/labels.csv")
CACHE = Path("data/cache")
OUT_COLS = ["video_id", "platform", "true_category", "risk_human", "pred_category", "risk_pred", "explanation"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="asr", help="через запятую: asr,ocr,vlm,meta")
    ap.add_argument("--run", action="store_true", help="реально вызвать OpenAI (платно)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--platform", default="tiktok", choices=["tiktok", "instagram", "all"],
                    help="по умолчанию только tiktok (основной тест); instagram идёт отдельным файлом")
    args = ap.parse_args(argv)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    bad = [s for s in sources if s not in llm_judge.SOURCES]
    if bad:
        sys.exit(f"Неизвестные источники: {bad}. Допустимо: {llm_judge.SOURCES}")

    with open(LABELS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # судим только тех, у кого есть кэш хотя бы одного из запрошенных источников
    if args.platform != "all":
        rows = [r for r in rows if r["platform"] == args.platform]
    ready = [r for r in rows if any((CACHE / f"{r['video_id']}.{s}.json").exists() for s in sources)]
    print(f"Видео в таблице: {len(rows)} | есть данные ({','.join(sources)}): {len(ready)}")
    if not args.run:
        print(f"Оценка: ~{len(ready) * 0.1:.1f} цента максимум (обычно меньше). Добавь --run.")
        return

    batch = ready[: args.limit] if args.limit else ready
    results, total_usd, errors = [], 0.0, []
    for i, r in enumerate(batch, 1):
        vid = r["video_id"]
        try:
            ev = llm_judge.load_cached_evidence(vid, sources)
            v = llm_judge.judge(vid, **ev)
        except Exception as e:
            errors.append(vid)
            print(f"[{i}/{len(batch)}] {vid}: ОШИБКА {e}")
            continue
        total_usd += (v.get("usage") or {}).get("usd") or 0.0
        print(f"[{i}/{len(batch)}] {vid}: правда={r['category']}({r['risk_human']}) "
              f"-> {v['category']}({v['risk']})  всего ${total_usd:.4f}")
        results.append({
            "video_id": vid, "platform": r["platform"], "true_category": r["category"],
            "risk_human": r["risk_human"], "pred_category": v["category"], "risk_pred": v["risk"],
            "explanation": v["explanation"],
        })

    suffix = "" if args.platform == "tiktok" else f"_{args.platform}"
    out = Path(f"data/judge_results_{'_'.join(sources)}{suffix}.csv")
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS)
        w.writeheader()
        w.writerows(results)

    if results:
        n = len(results)
        acc = sum(x["true_category"] == x["pred_category"] for x in results) / n
        scam_ok = sum((x["true_category"] != "none") == (x["pred_category"] != "none") for x in results) / n
        print(f"\n=== РЕЗУЛЬТАТ ({n} видео) ===")
        print(f"Точность по 6 категориям:      {acc:.0%}")
        print(f"Скам / не скам (бинарно):      {scam_ok:.0%}")
        by = defaultdict(list)
        for x in results:
            by[x["true_category"]].append(float(x["risk_pred"]))
        print("Средний предсказанный risk по настоящим категориям:")
        for c, vals in sorted(by.items()):
            print(f"  {c:<22}{sum(vals) / len(vals):.2f}  (n={len(vals)})")
    print(f"\nПотрачено: ${total_usd:.4f} | ошибок: {len(errors)} | файл: {out}")


if __name__ == "__main__":
    main()