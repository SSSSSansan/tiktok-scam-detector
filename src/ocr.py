"""ocr.py — OCR текста на кадрах (ru/en) + QR. Результат: data/cache/<video_id>.ocr.json
Формат: {"video_id": "...", "items": [{"t": секунды, "text": "...", "confidence": 0.9}]}
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import cv2
import numpy as np

try:
    from src.frames import iter_frames
except ImportError:  # запуск как python3 src/ocr.py
    from frames import iter_frames

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
VIDEOS_DIR = ROOT / "data" / "videos"
LABELS_CSV = ROOT / "data" / "labels.csv"

LETTER = re.compile(r"[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі]")
MONEY = re.compile(r"\d.*[%$€₸₽]|[%$€₸₽].*\d|\+\s?\d")
COUNTER = re.compile(r"^\d+([.,]\d+)?\s*[KkКкMmМм]?$")
UI_NOISE = re.compile(
    r"^(@?\s*tik\s*tok.*|нравится|комментарии|поделиться|подписаться|сохранить|"
    r"оригинальный звук.*|original sound.*|like|share|comments?|follow|following|"
    r"for you|подписки|рекомендации|друзья|поиск|search)$",
    re.I,
)

_reader = None


def get_reader():
    """EasyOCR создаётся один раз (первый запуск скачает модели ~100 МБ)."""
    global _reader
    if _reader is None:
        import easyocr

        try:
            import torch

            gpu = torch.cuda.is_available()
        except Exception:
            gpu = False
        _reader = easyocr.Reader(["ru", "en"], gpu=gpu)
    return _reader


def guess_id(path) -> str:
    """video_id = имя файла без .mp4, например tt_7649544344751312150 (как в labels.csv)."""
    return Path(path).stem


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def clean_line(text: str, conf: float, min_conf: float = 0.3):
    t = re.sub(r"\s+", " ", text).strip()
    if conf < min_conf or len(t) < 3:
        return None
    if UI_NOISE.match(t) or COUNTER.match(t):
        return None
    if not LETTER.search(t) and not MONEY.search(t):
        return None
    if sum(ch.isalnum() for ch in t) / len(t) < 0.5:
        return None
    return t


def group_lines(results):
    """EasyOCR отдаёт отдельные слова/боксы — склеиваем в строки по высоте."""
    boxes = []
    for bbox, text, conf in results:
        ys = [p[1] for p in bbox]
        xs = [p[0] for p in bbox]
        boxes.append(
            {"x": min(xs), "yc": (min(ys) + max(ys)) / 2, "h": max(ys) - min(ys),
             "text": text, "conf": float(conf)}
        )
    boxes.sort(key=lambda b: b["yc"])
    lines = []
    for b in boxes:
        if lines and abs(b["yc"] - lines[-1]["yc"]) < 0.6 * max(b["h"], lines[-1]["h"]):
            lines[-1]["parts"].append(b)
        else:
            lines.append({"yc": b["yc"], "h": b["h"], "parts": [b]})
    out = []
    for ln in lines:
        parts = sorted(ln["parts"], key=lambda b: b["x"])
        out.append((" ".join(p["text"] for p in parts), sum(p["conf"] for p in parts) / len(parts)))
    return out


def add_item(store: list, t: float, text: str, conf: float):
    """Дедупликация: похожая строка (>=85%) из другого кадра не добавляется, берём лучшее чтение."""
    n = norm(text)
    if len(n) < 2:
        return
    for it in store:
        if n == it["_n"] or SequenceMatcher(None, n, it["_n"]).ratio() >= 0.85:
            if conf > it["confidence"]:
                it["text"], it["confidence"] = text, conf
            return
    store.append({"t": t, "text": text, "confidence": conf, "_n": n})


def detect_qr(frame):
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(frame)
        return data.strip() or None
    except Exception:
        return None


def process_video(video_path, video_id=None, step_sec=2.0, max_frames=45, use_qr=True,
                  save=True, out_path=None, force=False, progress=None) -> dict:
    video_path = Path(video_path)
    video_id = video_id or guess_id(video_path)
    out = Path(out_path) if out_path else CACHE_DIR / f"{video_id}.ocr.json"
    if save and out.exists() and not force:
        return json.loads(out.read_text(encoding="utf-8"))

    reader = get_reader()
    store: list = []
    prev_small = None
    seen = 0
    for t, frame in iter_frames(video_path, step_sec=step_sec, max_width=800):
        seen += 1
        if seen > max_frames:
            break
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
        if prev_small is not None and float(np.mean(np.abs(small - prev_small))) < 2.5:
            continue  # кадр почти не изменился — текст тот же
        prev_small = small
        for text, conf in group_lines(reader.readtext(frame)):
            line = clean_line(text, conf)
            if line:
                add_item(store, t, line, round(conf, 3))
        if use_qr:
            qr = detect_qr(frame)
            if qr:
                add_item(store, t, f"[QR] {qr}", 1.0)
        if progress:
            progress(seen)

    items = [
        {"t": it["t"], "text": it["text"], "confidence": it["confidence"]}
        for it in sorted(store, key=lambda x: x["t"])
    ]
    result = {"video_id": video_id, "items": items}
    if save:
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)
    return result


def find_videos(videos_dir: Path, only_tiktok: bool):
    vids = sorted(videos_dir.rglob("*.mp4"))
    if only_tiktok and LABELS_CSV.exists():
        ids = set()
        with open(LABELS_CSV, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                if (r.get("platform") or "").strip().lower() == "tiktok":
                    ids.add((r.get("video_id") or "").strip())
        if ids:
            vids = [v for v in vids if v.stem in ids]
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="один mp4")
    ap.add_argument("--all", action="store_true", help="все mp4 из --videos-dir")
    ap.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    ap.add_argument("--only-tiktok", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=45)
    ap.add_argument("--no-qr", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    if a.video:
        vids = [Path(a.video)]
    elif a.all:
        vids = find_videos(Path(a.videos_dir), a.only_tiktok)
    else:
        ap.error("укажи --video файл.mp4 или --all")
    if a.limit:
        vids = vids[: a.limit]

    print(f"Видео к обработке: {len(vids)}")
    t0 = time.time()
    errors = []
    for i, v in enumerate(vids, 1):
        s = time.time()
        try:
            res = process_video(v, step_sec=a.step, max_frames=a.max_frames,
                                use_qr=not a.no_qr, force=a.force)
            eta = (time.time() - t0) / i * (len(vids) - i) / 60
            print(f"[{i}/{len(vids)}] {guess_id(v)}: {len(res['items'])} строк, "
                  f"{time.time() - s:.0f}с, осталось ~{eta:.0f} мин")
        except Exception as e:
            errors.append((str(v), str(e)))
            print(f"[{i}/{len(vids)}] ОШИБКА {v}: {e}")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин. Ошибок: {len(errors)}")
    for p, e in errors:
        print("  ", p, e)


if __name__ == "__main__":
    main()