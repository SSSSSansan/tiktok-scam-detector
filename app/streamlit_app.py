"""Streamlit-демо Scam Radar. Запуск: streamlit run app/streamlit_app.py"""
import json
import re
import sys
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CACHE = ROOT / "data" / "cache"
VIDEOS = ROOT / "data" / "videos"
JUDGE_CSV = next(
    (p for p in [ROOT / "data" / "judge_results_asr_ocr_vlm.csv",
                 ROOT / "data" / "judge_results_asr_ocr.csv",
                 ROOT / "data" / "judge_results_asr.csv"] if p.exists()),
    ROOT / "data" / "judge_results_asr.csv",
)
LABELS_CSV = ROOT / "data" / "labels.csv"

try:
    from src.pipeline import clean_explanation  # убирает из объяснения числа риска, которые не совпадают с итоговым
except Exception:
    def clean_explanation(text):
        return str(text or "").strip()

# Быстрые маркеры только для подсветки в демо (это НЕ итоговая модель)
MARKERS = {
    "ссылка Telegram/WhatsApp": r"t\.me|telegram|телеграм|wa\.me|whatsapp|ватсап",
    "гарантия дохода": r"гаранти|100\s?%|без риска|кепілдік",
    "умножение денег": r"\b[xх]\s?\d|в\s?\d+\s?раз|\+\s?\d+\s?%",
    "срочность": r"срочно|только сегодня|успей|осталось \d+|мест мало",
    "вербовка": r"приведи|пригласи|реферал|друзей|команд[уа]",
    "розыгрыш": r"розыгрыш|разыгр|выиграй|айфон|iphone|подпишись",
    "работа/доход": r"работа из дома|удаленн|удалённ|в неделю|тг/нед|стартов\w+ пакет|пассивный доход",
    "QR-код": r"\[QR\]",
}

st.set_page_config(page_title="Scam Radar", page_icon="📡", layout="wide")

def find_markers(text: str) -> str:
    return ", ".join(name for name, rx in MARKERS.items() if re.search(rx, text, re.I))

def load_json(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}

def pick_col(df, cands):
    return next((c for c in cands if c in df.columns), None)

@st.cache_data
def load_judge():
    return pd.read_csv(JUDGE_CSV, dtype={"video_id": str}) if JUDGE_CSV.exists() else None

@st.cache_data
def load_labels():
    return pd.read_csv(LABELS_CSV, dtype=str).fillna("") if LABELS_CSV.exists() else None

def evidence_from_cache(vid: str) -> pd.DataFrame:
    rows = []
    asr = load_json(CACHE / f"{vid}.asr.json")
    segs = asr.get("segments") or []
    if segs:
        for s in segs:
            rows.append({"t": float(s.get("start", 0)), "source": "речь (ASR)", "text": str(s.get("text", "")).strip()})
    elif asr.get("text"):
        rows.append({"t": 0.0, "source": "речь (ASR)", "text": str(asr["text"]).strip()})
    for it in load_json(CACHE / f"{vid}.ocr.json").get("items", []):
        rows.append({"t": float(it.get("t", 0)), "source": "текст на экране (OCR)", "text": str(it.get("text", "")).strip()})
    return finalize(rows)

def finalize(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["t", "source", "text"])
    if df.empty:
        return df
    df["маркеры"] = df["text"].apply(find_markers)
    return df.sort_values("t").reset_index(drop=True)

def show_timeline(df: pd.DataFrame):
    if df.empty:
        st.info("Улик нет: в видео не найдено ни речи, ни текста на экране.")
        return
    d = df.copy()
    d["подозрительно"] = d["маркеры"].apply(lambda x: "да" if x else "нет")
    chart = (
        alt.Chart(d)
        .mark_circle(size=140)
        .encode(
            x=alt.X("t:Q", title="секунды"),
            y=alt.Y("source:N", title=None),
            color=alt.Color("подозрительно:N", scale=alt.Scale(domain=["да", "нет"], range=["#d62728", "#9aa0a6"])),
            tooltip=["t", "source", "text", "маркеры"],
        )
        .properties(height=140)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df.rename(columns={"t": "сек", "source": "источник", "text": "текст"}),
                 use_container_width=True, hide_index=True)


# --- НОВАЯ ФУНКЦИЯ КАСТОМНОГО UI ---
def render_custom_ui(risk_score, predicted_category, true_label):
    if risk_score is None:
        st.info("Для этого видео нет готового разбора. Загрузите его во вкладке «Загрузить mp4».")
        return

    # Список категорий из хакатон-брифа
    categories = ["Investment platform", "Pyramid", "Betting", "Giveaway", "Job offer", "none"]
    
    # Формируем HTML для кнопок-категорий
    pills_html = ""
    for cat in categories:
        cat_clean = cat.lower()
        pred_clean = str(predicted_category).lower().replace("_", " ")
        active_class = "active-pill" if cat_clean == pred_clean else ""
        pills_html += f'<div class="scam-pill {active_class}">{cat}</div>'

    # Рассчитываем позицию маркера и форматируем текст
    try:
        risk_val = float(risk_score)
        marker_pos = max(0, min(100, int(risk_val * 100)))
        risk_text = f"{risk_val:.2f}"
    except (ValueError, TypeError):
        marker_pos = 0
        risk_text = "N/A"

    custom_html = f"""
<style>
.risk-section {{ font-family: sans-serif; margin-bottom: 2.5rem; }}
.risk-title-container {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 0.2rem; }}
.risk-title {{ font-size: 1.5rem; font-weight: bold; color: #f9fafb; margin: 0; }}
.risk-number {{ font-size: 1.8rem; font-weight: bold; color: #3b82f6; margin: 0; }}
.risk-desc {{ color: #9ca3af; font-size: 1rem; margin-bottom: 1.5rem; }}

.gradient-bar-container {{ position: relative; width: 100%; height: 12px; margin-bottom: 1rem; margin-top: 1rem; }}
.gradient-bar {{ width: 100%; height: 100%; border-radius: 6px; background: linear-gradient(90deg, #34d399 0%, #fbbf24 50%, #f87171 100%); }}
.marker {{ position: absolute; top: -8px; height: 28px; width: 6px; background-color: #ffffff; border: 1px solid #000000; border-radius: 3px; transform: translateX(-50%); left: {marker_pos}%; box-shadow: 0 0 6px rgba(0,0,0,0.6); z-index: 10; transition: left 0.5s ease-in-out; }}
.scale-labels {{ display: flex; justify-content: space-between; color: #9ca3af; font-size: 0.9rem; }}

.pills-container {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1rem; }}
.scam-pill {{ padding: 8px 16px; border-radius: 20px; border: 1px solid #4b5563; background-color: transparent; color: #f9fafb; font-size: 14px; font-family: sans-serif; box-shadow: 0 1px 2px rgba(0,0,0,0.05); transition: all 0.2s; }}
.scam-pill.active-pill {{ border: 2px solid #3b82f6 !important; background-color: #3b82f6 !important; font-weight: bold; color: #ffffff !important; box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3); }}

.true-label-box {{ margin-top: 2rem; padding: 12px; background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; border-radius: 4px; font-family: sans-serif; color: #f9fafb; }}
</style>

<div class="risk-section">
<div style="color: #ef4444; font-weight: bold; font-size: 0.9rem; margin-bottom: 4px;">A</div>
<div class="risk-title-container">
<div class="risk-title">Risk score:</div>
<div class="risk-number">{risk_text}</div>
</div>
<div class="risk-desc">A value from 0 to 1.</div>
<div class="gradient-bar-container">
<div class="gradient-bar"></div>
<div class="marker"></div>
</div>
<div class="scale-labels">
<span>0.0</span>
<span>0.5</span>
<span>1.0</span>
</div>
</div>

<div class="risk-section">
<div style="color: #ef4444; font-weight: bold; font-size: 0.9rem; margin-bottom: 4px;">B</div>
<div class="risk-title">Scam category</div>
<div class="risk-desc">The type of scam, or "none".</div>
<div class="pills-container">
{pills_html}
</div>
<div class="true-label-box">
<strong>Ground Truth (разметка):</strong> {true_label}
</div>
</div>
"""
    
    st.markdown(custom_html, unsafe_allow_html=True)

st.title("📡 Scam Radar")
st.caption("Риск-сигнал для модерации, а не обвинение. Хранятся только ID видео.")

mode = st.sidebar.radio("Режим", ["Видео из датасета", "Загрузить mp4 (полный анализ)"])
st.sidebar.caption("Анализ: речь + текст на экране + кадры → LLM-судья → калибровка риска")

if mode == "Видео из датасета":
    judge, labels = load_judge(), load_labels()
    ids = sorted(judge["video_id"].astype(str).unique()) if judge is not None else \
        sorted(p.name.split(".")[0] for p in CACHE.glob("*.ocr.json"))
    if not ids:
        st.warning("Нет данных: подтяни свежий main (git pull) и запусти OCR.")
        st.stop()
    vid = st.sidebar.selectbox("Видео (ID)", ids)

    risk, pred_cat, expl, true_label = None, "—", "", "—"
    if judge is not None:
        rows = judge[judge["video_id"].astype(str) == vid]
        if not rows.empty:
            r = rows.iloc[0]
            if "risk_pred" in judge.columns:
                risk = float(r["risk_pred"])
            cc = pick_col(judge, ["category_pred", "pred_category", "category", "label_pred", "pred_label", "cat_pred"])
            ec = pick_col(judge, ["explanation", "reason", "rationale"])
            pred_cat = str(r[cc]) if cc else "—"
            expl = str(r[ec]) if ec else ""
    if labels is not None:
        m = labels[labels["video_id"] == vid]
        if not m.empty:
            true_label = m.iloc[0].get("category", "—") or "—"

    left, right = st.columns([1, 2])
    with left:
        mp4 = next(iter(VIDEOS.rglob(f"*{vid}*.mp4")), None) if VIDEOS.exists() else None
        if mp4:
            # Оборачиваем видео в контейнер, чтобы ограничить его ширину до 300px
            st.markdown(
                """<style>
                video { max-width: 300px !important; max-height: 500px !important; border-radius: 8px; }
                </style>""", 
                unsafe_allow_html=True
            )
            st.video(str(mp4))
        else:
            st.caption("Видео не скачано локально (оно не лежит в репо).")
        st.caption(f"ID: {vid}")
    
    with right:
        # Вызов функции кастомного дизайна
        render_custom_ui(risk, pred_cat, true_label)
        
        if expl:
            st.markdown(f"**Объяснение:** {clean_explanation(expl)}")

    st.subheader("Таймлайн улик")
    show_timeline(evidence_from_cache(vid))

else:
    st.info("Полный анализ нового видео: речь (Whisper), текст на экране (OCR), описание кадров и LLM-судья. "
            "Занимает около минуты. Нужен ключ OpenAI в файле .env на этом компьютере.")
    up = st.file_uploader("Загрузи mp4", type=["mp4", "mov"])
    if up is not None and st.button("Запустить анализ"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(up.read())
            path = tmp.name

        # То же самое ограничение ширины видео для режима загрузки
        st.markdown(
            """<style>
            video { max-width: 300px !important; max-height: 500px !important; border-radius: 8px; margin: 0 auto; display: block; }
            </style>""",
            unsafe_allow_html=True
        )
        st.video(path)

        from src.pipeline import analyze  # полный вердикт: речь + OCR + кадры + судья (+ калибровка, если есть)

        with st.spinner("Считаю полный вердикт (речь, текст, кадры, судья), это около минуты..."):
            try:
                verdict = analyze(path)
            except Exception as e:
                verdict = None
                st.error(f"Не удалось посчитать вердикт: {e}")

        if verdict:
            render_custom_ui(verdict["risk"], verdict["category"], "— (новое видео, разметки нет)")
            if verdict.get("calibrated"):
                st.caption(f"Итоговый риск калиброван. Оценка судьи до калибровки: {verdict['risk_llm']:.2f}")
            st.markdown(f"**Объяснение:** {verdict['explanation']}")
            if verdict.get("evidence"):
                st.markdown("**Улики:**")
                for e in verdict["evidence"]:
                    st.markdown(f"- [{e.get('t')}] ({e['source']}) {e['text']}")

            st.subheader("Таймлайн улик")
            show_timeline(evidence_from_cache(Path(path).stem))  # речь и OCR уже лежат в кэше после analyze