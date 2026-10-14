"""
ASR: видео -> AsrResult (контракт из PROJECT_CONTEXT.md, раздел 5)

Тест из корня репозитория:
    python -m src.asr data/videos/tt_7692111302109564181.mp4
"""
import json
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CACHE_DIR = Path("data/cache")
MAX_AUDIO_SECONDS = 60  # распознаём только первую минуту: экономит деньги, суть обещаний обычно в начале
NO_SPEECH_THRESHOLD = 0.6  # сегменты, где модель "не уверена, что это речь", выбрасываем
# Whisper на музыке/тишине "пишет" подписи вроде "ДИНАМИЧНАЯ МУЗЫКА" или "Продолжение следует".
# Короткие сегменты с такими словами считаем не речью.
JUNK_RE = re.compile(
    r"музык|аплодисмент|субтитр|продолжение следует|редактор|корректор|"
    r"music|applause|subtitles|thanks for watching|ән ойнап",
    re.IGNORECASE,
)
JUNK_MAX_WORDS = 6

LANG_CODES = {
    "russian": "ru", "kazakh": "kk", "english": "en",
    "ukrainian": "uk", "uzbek": "uz", "kyrgyz": "ky",
}


def has_audio(video_path: Path) -> bool:
    """True, если в файле есть звуковая дорожка."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def extract_audio(video_path: Path) -> Path:
    """Достаёт лёгкую моно-дорожку (mp3 16 кГц) в data/cache. Whisper API принимает файлы до 25 МБ."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = CACHE_DIR / f"{video_path.stem}.mp3"
    if audio_path.exists():
        return audio_path
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "48k", "-t", str(MAX_AUDIO_SECONDS), str(audio_path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not audio_path.exists():
        raise RuntimeError(f"ffmpeg не смог достать звук из {video_path}:\n{r.stderr[-500:]}")
    return audio_path


def transcribe(video_path) -> dict:
    """Видео -> {"language": "ru", "text": "...", "segments": [{"start", "end", "text"}]}.
    Результат кэшируется в data/cache/<имя_видео>.asr.json, повторный вызов бесплатный."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео не найдено: {video_path} (скачалось ли оно?)")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{video_path.stem}.asr.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if not has_audio(video_path):
        # видео без звука (например, только текст на экране) - это нормальный случай
        result = {"language": None, "text": "", "segments": []}
    else:
        audio_path = extract_audio(video_path)
        client = OpenAI()  # ключ берётся из OPENAI_API_KEY (файл .env)
        with open(audio_path, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = []
        for s in (resp.segments or []):
            if getattr(s, "no_speech_prob", 0.0) > NO_SPEECH_THRESHOLD:
                continue  # музыка/тишина: Whisper в таких местах любит "выдумывать" текст
            text = s.text.strip()
            if len(text.split()) <= JUNK_MAX_WORDS and JUNK_RE.search(text):
                continue  # подпись к музыке, а не речь
            if text:
                segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": text})
        lang = (resp.language or "").lower()
        result = {
            "language": LANG_CODES.get(lang, lang or None),
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
        }

    cache_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Использование: python -m src.asr data/videos/<video_id>.mp4")
    print(json.dumps(transcribe(sys.argv[1]), ensure_ascii=False, indent=2))