"""
Прогон ASR по всему датасету с оценкой стоимости ДО того, как потратишь деньги.

Из корня репозитория:
  python tools/run_asr_all.py                    # только оценка (бесплатно, ничего не отправляет в OpenAI)
  python tools/run_asr_all.py --download         # + скачать недостающие видео (бесплатно)
  python tools/run_asr_all.py --run --limit 10   # реально распознать первые 10 видео (платно)
  python tools/run_asr_all.py --run              # распознать все (платно)
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # чтобы работал "from src.asr import ..."
from src.asr import CACHE_DIR, MAX_AUDIO_SECONDS, has_audio, transcribe  # noqa: E402

LABELS = Path("data/labels.csv")
VIDEO_DIR = Path("data/videos")
PRICE_PER_MIN = 0.006  # $ за минуту звука, whisper-1


def video_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def download(url: str, path: Path) -> bool:
    subprocess.run(["yt-dlp", "-q", "--no-warnings", "-o", str(path), url], capture_output=True, text=True)
    return path.exists()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="скачать недостающие видео")
    ap.add_argument("--run", action="store_true", help="реально вызвать OpenAI (платно)")
    ap.add_argument("--limit", type=int, default=None, help="сколько видео распознать при --run")
    args = ap.parse_args(argv)

    with open(LABELS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    # 1. скачивание недостающих
    if args.download:
        failed = []
        for r in rows:
            path = VIDEO_DIR / f"{r['video_id']}.mp4"
            if path.exists():
                continue
            print(f"скачиваю {r['video_id']} ...", end=" ", flush=True)
            ok = download(r["url"], path)
            print("ok" if ok else "НЕ СКАЧАЛОСЬ")
            if not ok:
                failed.append(r["video_id"])
            time.sleep(1)
        if failed:
            print(f"\nНе скачалось {len(failed)}: {', '.join(failed[:15])}{' ...' if len(failed) > 15 else ''}")
            print("Скачай их вручную и положи в data/videos/<video_id>.mp4")

    # 2. оценка стоимости
    missing, no_audio, cached, todo = [], [], [], []
    billable_sec = 0.0
    for r in rows:
        vid = r["video_id"]
        path = VIDEO_DIR / f"{vid}.mp4"
        if not path.exists():
            missing.append(vid)
        elif (CACHE_DIR / f"{vid}.asr.json").exists():
            cached.append(vid)
        elif not has_audio(path):
            no_audio.append(vid)
        else:
            todo.append(vid)
            billable_sec += min(video_duration(path), MAX_AUDIO_SECONDS)

    minutes = billable_sec / 60
    print("\n=== ОЦЕНКА ===")
    print(f"Видео в таблице:            {len(rows)}")
    print(f"Нет файла в data/videos:    {len(missing)}")
    print(f"Без звука (бесплатно):      {len(no_audio)}")
    print(f"Уже распознано (кэш):       {len(cached)}")
    print(f"К распознаванию:            {len(todo)}  (~{minutes:.1f} мин, максимум {MAX_AUDIO_SECONDS} сек на видео)")
    print(f"Примерная стоимость:        ${minutes * PRICE_PER_MIN:.2f}")

    # 3. запуск
    if not args.run:
        print("\nЭто была только оценка. Чтобы реально распознать: добавь --run (можно --limit 10)")
        return

    batch = todo[: args.limit] if args.limit else todo
    print(f"\nРаспознаю {len(batch)} видео...")
    errors = []
    for i, vid in enumerate(batch, 1):
        try:
            res = transcribe(VIDEO_DIR / f"{vid}.mp4")
            print(f"[{i}/{len(batch)}] {vid}: язык={res['language']}, сегментов={len(res['segments'])}")
        except Exception as e:  # одно сломанное видео не должно останавливать остальные
            errors.append(vid)
            print(f"[{i}/{len(batch)}] {vid}: ОШИБКА {e}")
    print(f"\nГотово. Ошибок: {len(errors)}")
    if errors:
        print("С ошибками:", ", ".join(errors))


if __name__ == "__main__":
    main()