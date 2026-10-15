"""frames.py — видео -> кадры (1 шт/сек) и выбор ключевых кадров для VLM."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def iter_frames(video_path, step_sec: float = 1.0, max_width: int = 960):
    """Генератор (t_секунды, кадр BGR). Читает видео последовательно, без перемотки."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Не удалось открыть видео: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step_frames = max(1, int(round(fps * step_sec)))
    idx = 0
    try:
        while True:
            if not cap.grab():
                break
            if idx % step_frames == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    frame = cv2.resize(frame, (max_width, int(h * scale)))
                yield round(idx / fps, 2), frame
            idx += 1
    finally:
        cap.release()


def extract_frames(video_path, out_dir, step_sec: float = 1.0) -> list[dict]:
    """Сохраняет кадры в out_dir. Возвращает [{"t": 4.0, "path": ".../frame_0004.jpg"}, ...]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for t, frame in iter_frames(video_path, step_sec=step_sec):
        path = out_dir / f"frame_{int(t):04d}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        result.append({"t": t, "path": str(path)})
    return result


def select_key_frames(frames: list[dict], k: int = 5, min_gap: float = 2.0) -> list[dict]:
    """Выбирает k кадров: первый + кадры с самой сильной сменой сцены (не ближе min_gap сек)."""
    if len(frames) <= k:
        return frames
    small = []
    for f in frames:
        img = cv2.imread(f["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            small.append(np.zeros((64, 64), np.float32))
        else:
            small.append(cv2.resize(img, (64, 64)).astype(np.float32))
    scores = [0.0] + [float(np.mean(np.abs(small[i] - small[i - 1]))) for i in range(1, len(small))]
    order = sorted(range(1, len(frames)), key=lambda i: -scores[i])
    chosen = [0]
    for i in order:
        if len(chosen) >= k:
            break
        if all(abs(frames[i]["t"] - frames[j]["t"]) >= min_gap for j in chosen):
            chosen.append(i)
    if len(chosen) < k:  # добор равномерно, если сцен мало
        for i in np.linspace(0, len(frames) - 1, k).astype(int):
            if len(chosen) >= k:
                break
            if int(i) not in chosen:
                chosen.append(int(i))
    return [frames[i] for i in sorted(chosen)]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Использование: python3 src/frames.py путь/к/видео.mp4")
        raise SystemExit(1)
    vid = Path(sys.argv[1])
    out = Path("data/cache") / vid.stem
    frames = extract_frames(vid, out)
    keys = select_key_frames(frames, k=5)
    print(f"Кадров: {len(frames)}, ключевых: {len(keys)}")
    print(json.dumps(keys, ensure_ascii=False, indent=2))