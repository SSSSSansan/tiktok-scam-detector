"""
Scam Radar: одно видео -> Verdict {video_id, risk, category, evidence[], explanation}.

Запуск из корня репозитория:
    python -m src.pipeline tt_7649544344751312150                      # по video_id (видео в data/videos)
    python -m src.pipeline data/videos/tt_7649544344751312150.mp4
    python -m src.pipeline "https://www.tiktok.com/@user/video/123"    # скачает и проанализирует
    python -m src.pipeline <видео> --sources asr                       # только речь (для ablation)
    python -m src.pipeline <видео> --no-calibration                    # без калибратора

Шаги: (скачать) -> ASR -> OCR и VLM из кэша data/cache, если они посчитаны
      -> LLM-судья (категория, risk_llm, улики) -> признаки features.py
      -> калибратор models/calibrator.joblib (итоговый risk), если файл есть.
Всё кэшируется в data/cache, повторный запуск на том же видео бесплатный.
Вердикт сохраняется в data/verdicts/<video_id>.json.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

from src import llm_judge
from src.asr import transcribe

VIDEO_DIR = Path("data/videos")
VERDICT_DIR = Path("data/verdicts")
CALIBRATOR_PATH = Path("models/calibrator.joblib")


def resolve_video(arg: str) -> Path:
    """Принимает путь к файлу, video_id или ссылку. Возвращает путь к mp4."""
    if arg.startswith("http"):
        m = re.search(r"tiktok\.com/@[^/]+/(?:video|photo)/(\d+)", arg)
        vid = f"tt_{m.group(1)}" if m else "url_" + hashlib.sha1(arg.encode()).hexdigest()[:10]
        path = VIDEO_DIR / f"{vid}.mp4"
        if not path.exists():
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-o", str(path), arg],
                               capture_output=True, text=True)
            if not path.exists():
                sys.exit(f"Не удалось скачать видео:\n{r.stderr[-300:]}")
        return path
    p = Path(arg)
    if p.exists():
        return p
    p = VIDEO_DIR / f"{arg}.mp4"
    if p.exists():
        return p
    sys.exit(f"Не нашла видео: {arg}")


def _load_calibrator():
    """Возвращает функцию калибровки или None, если файла нет или она не загрузилась."""
    if not CALIBRATOR_PATH.exists():
        return None
    try:
        from src.calibrate import load_calibrator
        return load_calibrator(CALIBRATOR_PATH)
    except Exception as e:  # без калибратора система всё равно работает
        print(f"[предупреждение] калибратор не загрузился ({e}), используем риск судьи", file=sys.stderr)
        return None


def analyze(video_path, sources=("asr", "ocr", "vlm", "meta"), use_cache=True, calibrate=True) -> dict:
    """Главная функция проекта: путь к видео -> Verdict."""
    video_path = Path(video_path)
    video_id = video_path.stem

    asr = transcribe(video_path) if "asr" in sources else None
    cached = llm_judge.load_cached_evidence(video_id, [s for s in sources if s != "asr"])
    verdict = llm_judge.judge(video_id, asr=asr, use_cache=use_cache, **cached)
    verdict["risk_llm"] = verdict["risk"]

    # признаки (регулярки) и калибровка итогового риска
    feats = None
    try:
        from src.features import extract_features
        feats = extract_features(asr=asr, ocr=cached.get("ocr"), meta=cached.get("meta"))
        verdict["flags"] = sorted(k for k, v in feats["flags"].items() if v)
    except Exception as e:
        feats = None
        print(f"[предупреждение] признаки не посчитались ({e})", file=sys.stderr)

    calibrator = _load_calibrator() if (calibrate and feats is not None) else None
    if calibrator is not None:
        n_words = len(str((asr or {}).get("text") or "").split())
        # калибратор обучали на признаках только по речи, поэтому и здесь считаем их только по речи
        feats_cal = extract_features(asr=asr)
        verdict["risk"] = round(float(calibrator(verdict["risk_llm"], verdict["category"], feats_cal, n_words)), 3)
        verdict["calibrated"] = True
    else:
        verdict["calibrated"] = False

    VERDICT_DIR.mkdir(parents=True, exist_ok=True)
    (VERDICT_DIR / f"{video_id}.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


def show(v: dict):
    bar = "█" * round(v["risk"] * 20) + "░" * (20 - round(v["risk"] * 20))
    print(f"\nВидео:      {v['video_id']}")
    print(f"Категория:  {v['category']}")
    print(f"Риск:       {v['risk']:.2f}  {bar}" + ("  (калиброванный)" if v.get("calibrated") else "  (оценка судьи)"))
    if v.get("calibrated"):
        print(f"Риск судьи: {v['risk_llm']:.2f}")
    if v.get("flags"):
        print(f"Сработали признаки: {', '.join(v['flags'])}")
    print(f"Почему:     {v['explanation']}")
    if v["evidence"]:
        print("Улики:")
        for e in v["evidence"]:
            mark = "✓" if e.get("verified") else "?"
            print(f"  {mark} [{e.get('t')}] ({e['source']}) {e['text']}")
    cost = (v.get("usage") or {}).get("usd")
    if cost:
        print(f"\nСтоимость судьи: ${cost:.5f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="путь к mp4, video_id или ссылка")
    ap.add_argument("--sources", default="asr,ocr,vlm,meta", help="какие источники использовать")
    ap.add_argument("--no-calibration", action="store_true", help="не применять калибратор")
    ap.add_argument("--json", action="store_true", help="вывести сырой JSON")
    args = ap.parse_args(argv)

    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip())
    v = analyze(resolve_video(args.video), sources, calibrate=not args.no_calibration)
    print(json.dumps(v, ensure_ascii=False, indent=2)) if args.json else show(v)


if __name__ == "__main__":
    main()