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


def risk_box(risk):
    if risk is None:
        st.info("Для этого видео нет результата судьи.")
        return
    st.metric("Risk score", f"{risk:.2f}")
    st.progress(min(max(risk, 0.0), 1.0))
    if risk >= 0.7:
        st.error("Высокий риск")
    elif risk >= 0.4:
        st.warning("Средний риск")
    else:
        st.success("Низкий риск")


st.title("📡 Scam Radar")
st.caption("Риск-сигнал для модерации, а не обвинение. Хранятся только ID видео.")

mode = st.sidebar.radio("Режим", ["Видео из датасета", "Загрузить mp4 (OCR локально)"])

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
            st.video(str(mp4))
        else:
            st.caption("Видео не скачано локально (оно не лежит в репо).")
        st.caption(f"ID: {vid}")
    with right:
        risk_box(risk)
        c1, c2 = st.columns(2)
        c1.metric("Категория (система)", pred_cat)
        c2.metric("Верная метка (разметка)", true_label)
        if expl:
            st.markdown(f"**Объяснение судьи:** {expl}")

    st.subheader("Таймлайн улик")
    show_timeline(evidence_from_cache(vid))

else:
    st.info("Здесь считается только OCR (бесплатно, локально). Полный вердикт с risk-скором "
            "считает LLM-судья, он запускается отдельно.")
    up = st.file_uploader("Загрузи mp4", type=["mp4", "mov"])
    if up is not None and st.button("Запустить OCR"):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(up.read())
            path = tmp.name
        st.video(path)
        from src.ocr import process_video  # тяжёлый импорт — только здесь

        with st.spinner("Распознаю текст на кадрах..."):
            res = process_video(path, video_id="upload", save=False)
        rows = [{"t": float(i["t"]), "source": "текст на экране (OCR)", "text": i["text"]} for i in res["items"]]
        show_timeline(finalize(rows))