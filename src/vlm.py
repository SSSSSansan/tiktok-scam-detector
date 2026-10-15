"""
VLM: видео -> VlmResult (контракт из PROJECT_CONTEXT.md, раздел 5):
    {"items": [{"t": 4.0, "description": "скриншот торгового приложения с зелёными цифрами"}]}

Берёт 5 ключевых кадров (src/frames.py), отправляет их одним запросом в gpt-4o-mini
и просит коротко описать, что видно. Результат кэшируется в data/cache/<id>.vlm.json.

Запуск из корня репозитория:
    python -m src.vlm tt_7649544344751312150            # одно видео по video_id
    python -m src.vlm data/videos/tt_7649544344751312150.mp4
    python -m src.vlm --all                             # только оценка стоимости (бесплатно)
    python -m src.vlm --all --run --limit 3             # платно: первые 3 видео
    python -m src.vlm --all --run                       # платно: все видео из labels.csv
"""
import argparse
import base64
import csv
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.frames import extract_frames, select_key_frames

load_dotenv(".env")

MODEL = os.getenv("VLM_MODEL", "gpt-4o-mini")
PRICES_PER_M = {"gpt-4o-mini": (0.15, 0.60)}  # $ за 1M токенов (вход, выход)
CACHE_DIR = Path("data/cache")
VIDEO_DIR = Path("data/videos")
LABELS = Path("data/labels.csv")
N_KEY_FRAMES = 5
EST_USD_PER_VIDEO = 0.003  # грубая оценка: 5 кадров low-detail у gpt-4o-mini

PROMPT = """Ты помогаешь модерировать TikTok. Ниже {n} кадров из одного видео с подписью времени.
Для каждого кадра опиши по-русски, ОДНИМ коротким предложением, что на нём видно.
Особенно отмечай важное для проверки на мошенничество: скриншоты торговых приложений, бирж, выплат или переводов,
графики и зелёные цифры роста, пачки денег, телефон с мессенджером, QR-код, логотипы брендов и банков,
призы (iPhone, машина), таблицы доходов, ставки, купоны и промокоды.
Описывай только то, что реально видно. Не домысливай, не гадай про намерения автора.
Если на кадре человек просто говорит в камеру, так и напиши.
Верни строго JSON: {{"items": [{{"t": число_секунд, "description": "..."}}]}}, по одному элементу на кадр, в том же порядке."""


def _b64(path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()


def describe_frames(frames: list[dict]):
    """frames: [{"t":..., "path":...}] -> (items, usage)."""
    content = [{"type": "text", "text": PROMPT.format(n=len(frames))}]
    for f in frames:
        content.append({"type": "text", "text": f"Кадр на {f['t']:.0f} сек:"})
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{_b64(f['path'])}", "detail": "low"}})
    client = OpenAI()  # ключ из OPENAI_API_KEY (.env)
    resp = client.chat.completions.create(
        model=MODEL, temperature=0, response_format={"type": "json_object"},
        messages=[{"role": "user", "content": content}],
    )
    data = json.loads(resp.choices[0].message.content)
    items = []
    for it, f in zip(data.get("items", []), frames):
        desc = str(it.get("description", "")).strip()
        if desc:
            items.append({"t": float(f["t"]), "description": desc[:300]})  # время берём из кадра, не из ответа модели
    u = getattr(resp, "usage", None)
    p, c = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
    price = PRICES_PER_M.get(MODEL)
    usd = round((p * price[0] + c * price[1]) / 1e6, 6) if price else None
    return items, {"prompt_tokens": p, "completion_tokens": c, "usd": usd}


def describe_video(video_path) -> dict:
    """Видео -> {"items": [...]}. Кэш: data/cache/<имя_видео>.vlm.json (повторный вызов бесплатный)."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{video_path.stem}.vlm.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    tmp = CACHE_DIR / f"{video_path.stem}_vlmframes"
    try:
        frames = extract_frames(video_path, tmp, step_sec=1.0)
        keys = select_key_frames(frames, k=N_KEY_FRAMES)
        items, usage = describe_frames(keys) if keys else ([], {"usd": 0.0})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # кадры не нужны после описания

    result = {"video_id": video_path.stem, "items": items, "usage": usage}
    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _resolve(arg: str) -> Path:
    p = Path(arg)
    return p if p.exists() else VIDEO_DIR / f"{arg}.mp4"


def run_all(run: bool, limit):
    with open(LABELS, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    have = [r["video_id"] for r in rows if (VIDEO_DIR / f"{r['video_id']}.mp4").exists()]
    todo = [v for v in have if not (CACHE_DIR / f"{v}.vlm.json").exists()]
    print(f"Видео в таблице: {len(rows)} | скачано: {len(have)} | уже описано: {len(have) - len(todo)} | осталось: {len(todo)}")
    print(f"Оценка стоимости: ~${len(todo) * EST_USD_PER_VIDEO:.2f}")
    if not run:
        print("Это только оценка. Чтобы запустить: добавь --run (можно --limit 3)")
        return
    batch = todo[:limit] if limit else todo
    total, errors = 0.0, []
    for i, vid in enumerate(batch, 1):
        try:
            res = describe_video(VIDEO_DIR / f"{vid}.mp4")
            total += (res.get("usage") or {}).get("usd") or 0.0
            print(f"[{i}/{len(batch)}] {vid}: кадров описано {len(res['items'])}  всего ${total:.4f}")
        except Exception as e:  # одно сломанное видео не должно останавливать остальные
            errors.append(vid)
            print(f"[{i}/{len(batch)}] {vid}: ОШИБКА {e}")
    print(f"\nГотово. Потрачено ${total:.4f}, ошибок: {len(errors)}")
    if errors:
        print("С ошибками:", ", ".join(errors))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="video_id или путь к mp4")
    ap.add_argument("--all", action="store_true", help="весь датасет из labels.csv")
    ap.add_argument("--run", action="store_true", help="с --all: реально вызвать OpenAI (платно)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    if args.all:
        return run_all(args.run, args.limit)
    if not args.video:
        sys.exit("Использование: python -m src.vlm <video_id> | --all [--run] [--limit N]")
    print(json.dumps(describe_video(_resolve(args.video)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()