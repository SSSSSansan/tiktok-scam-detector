"""
LLM-судья: Evidence Pack -> Verdict (контракт из PROJECT_CONTEXT.md, раздел 5).

Правила берутся из docs/rubric.md, так что логику можно менять, не трогая код.
Все ответы кэшируются в data/cache, повторный вызов с теми же данными бесплатный.

Тест на уже распознанном видео (платно, около 0.1 цента):
    python -m src.llm_judge tt_7649544344751312150
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
PRICES_PER_M = {"gpt-4o-mini": (0.15, 0.60)}  # $ за 1M токенов (вход, выход)

RUBRIC_PATH = Path("docs/rubric.md")
CACHE_DIR = Path("data/cache")
CATEGORIES = ["investment_platform", "pyramid", "betting", "giveaway", "job_offer", "none"]
ALIASES = {"investment platform": "investment_platform", "job offer": "job_offer"}
SOURCES = ("asr", "ocr", "vlm", "meta")

MAX_ASR_CHARS = 3000
MAX_OCR_CHARS = 1500
MAX_VLM_CHARS = 1200

OUTPUT_FORMAT = f"""
ФОРМАТ ОТВЕТА. Верни строго один JSON-объект, без пояснений вокруг:
{{
  "category": одна из {CATEGORIES},
  "risk": число от 0 до 1,
  "evidence": [{{"t": "m:ss или null", "source": "asr | ocr | vlm | meta", "text": "ДОСЛОВНАЯ цитата из данных"}}],
  "explanation": "1-2 предложения по-русски: почему такая категория и риск"
}}
Требования: не больше 5 улик; цитируй дословно то, что есть во входных данных, ничего не придумывай;
category = none только если risk < 0.4, а для любой скам-категории risk >= 0.4.
Если данных мало, не выдумывай признаки: поставь низкий риск и объясни, чего не хватает.
"""


# ---------- подготовка улик ----------

def _fmt_t(t) -> str:
    if t is None:
        return "?"
    if isinstance(t, str):
        return t
    t = int(t)
    return f"{t // 60}:{t % 60:02d}"


def _items(x):
    if x is None:
        return []
    if isinstance(x, dict):
        x = x.get("items", [])
    return list(x)


def _clip(lines, limit):
    out, total = [], 0
    for line in lines:
        if total + len(line) > limit:
            break
        out.append(line)
        total += len(line) + 1
    return out


def build_evidence_text(asr=None, ocr=None, vlm=None, meta=None) -> str:
    """Собирает улики в короткий текст для промпта. Пустая строка = улик нет."""
    parts = []

    meta = meta or {}
    desc = (meta.get("description") or "").strip()
    tags = meta.get("hashtags") or []
    if desc or tags:
        tag_line = " ".join("#" + t.lstrip("#") for t in tags)
        parts.append("[ОПИСАНИЕ ВИДЕО]\n" + "\n".join(x for x in (desc, tag_line) if x))

    segs = (asr or {}).get("segments") or []
    if segs:
        lang = (asr or {}).get("language") or "не определён"
        lines = _clip([f"{_fmt_t(s.get('start'))} {s['text']}" for s in segs], MAX_ASR_CHARS)
        parts.append(f"[РЕЧЬ, язык: {lang}]\n" + "\n".join(lines))

    seen, ocr_lines = set(), []
    for it in _items(ocr):
        text = " ".join(str(it.get("text", "")).split())
        key = text.lower()
        if not text or key in seen:  # один и тот же текст держится на экране несколько секунд
            continue
        seen.add(key)
        ocr_lines.append(f"{_fmt_t(it.get('t', it.get('timestamp')))} {text[:200]}")
    if ocr_lines:
        parts.append("[ТЕКСТ НА ЭКРАНЕ]\n" + "\n".join(_clip(ocr_lines, MAX_OCR_CHARS)))

    vlm_lines = [f"{_fmt_t(it.get('t', it.get('timestamp')))} {it.get('description', '')}".strip()
                 for it in _items(vlm) if it.get("description")]
    if vlm_lines:
        parts.append("[ОПИСАНИЕ КАДРОВ]\n" + "\n".join(_clip(vlm_lines, MAX_VLM_CHARS)))

    return "\n\n".join(parts)


def load_cached_evidence(video_id, sources=SOURCES) -> dict:
    """Читает data/cache/<id>.asr.json / .ocr.json / .vlm.json / .meta.json, если они есть."""
    out = {}
    for kind in sources:
        f = CACHE_DIR / f"{video_id}.{kind}.json"
        out[kind] = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
    return out


def build_system_prompt() -> str:
    rubric = RUBRIC_PATH.read_text(encoding="utf-8")
    return ("Ты эксперт по модерации контента в TikTok. Определи, продвигает ли видео мошенничество, "
            "строго по правилам ниже.\n\n" + rubric + "\n" + OUTPUT_FORMAT)


# ---------- вызов модели и разбор ответа ----------

def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", " ", str(s).lower()).strip()


def _usage(resp) -> dict:
    u = getattr(resp, "usage", None)
    p, c = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
    price = PRICES_PER_M.get(MODEL)
    usd = round((p * price[0] + c * price[1]) / 1e6, 6) if price else None
    return {"prompt_tokens": p, "completion_tokens": c, "usd": usd}


def _parse(video_id, raw, evidence_text):
    data = json.loads(raw)

    category = str(data.get("category", "")).strip().lower()
    category = ALIASES.get(category, category.replace(" ", "_").replace("-", "_"))
    if category not in CATEGORIES:
        raise ValueError(f"модель вернула неизвестную категорию: {data.get('category')!r}")

    risk = max(0.0, min(1.0, float(data["risk"])))

    blob = _norm(evidence_text)
    evidence = []
    for ev in (data.get("evidence") or [])[:5]:
        text = str(ev.get("text", "")).strip()
        if not text:
            continue
        src = str(ev.get("source", "")).lower()
        evidence.append({
            "t": ev.get("t"),
            "source": src if src in SOURCES else "unknown",
            "text": text[:300],
            "verified": bool(_norm(text)) and _norm(text) in blob,  # цитата реально есть во входных данных
        })

    return {
        "video_id": video_id,
        "risk": round(risk, 3),
        "category": category,
        "evidence": evidence,
        "explanation": str(data.get("explanation", "")).strip(),
    }


def judge(video_id, asr=None, ocr=None, vlm=None, meta=None, use_cache=True) -> dict:
    """Возвращает Verdict: {video_id, risk, category, evidence, explanation, usage}.
    Для ablation просто передавай не все источники (например, только asr)."""
    evidence_text = build_evidence_text(asr, ocr, vlm, meta)
    if not evidence_text.strip():
        return {"video_id": video_id, "risk": 0.1, "category": "none", "evidence": [],
                "explanation": "Нет данных для оценки: речи, текста на экране и описания не найдено.",
                "no_evidence": True, "usage": {"prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0}}

    system = build_system_prompt()
    user = "Данные по видео:\n\n" + evidence_text
    key = hashlib.sha1((MODEL + system + user).encode("utf-8")).hexdigest()[:8]
    cache_file = CACHE_DIR / f"{video_id}.judge.{key}.json"
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    client = OpenAI()  # ключ из OPENAI_API_KEY (.env)
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    verdict = _parse(video_id, resp.choices[0].message.content, evidence_text)
    verdict["usage"] = _usage(resp)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return verdict


def analyze_video_evidence(speech_segments, ocr_data, video_meta=None):
    """Совместимость со старым кодом Арай (pipeline.py): те же аргументы, ответ в нашем формате."""
    return judge("adhoc", asr={"language": None, "segments": speech_segments or []},
                 ocr=ocr_data, meta=video_meta, use_cache=False)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Использование: python -m src.llm_judge <video_id>")
    ev = load_cached_evidence(sys.argv[1])
    print(json.dumps(judge(sys.argv[1], **ev), ensure_ascii=False, indent=2))