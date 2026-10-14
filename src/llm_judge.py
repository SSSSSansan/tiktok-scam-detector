import os
import json
from openai import OpenAI

def analyze_video_evidence(speech_segments, ocr_data, video_meta=None):
    """
    Формирует Evidence Pack в формате JSON и отправляет его в LLM (GPT-4o) 
    для проверки на мошенничество.
    """
    if video_meta is None:
        video_meta = {"description": "", "hashtags": []}

    # 1. Сборка единого Evidence Pack
    evidence_pack = {
        "metadata": video_meta,
        "transcript": speech_segments,
        "ocr_text": ocr_data
    }

    # 2. Системный промпт с правилами анализа фишинга и финансовых схем
    system_prompt = (
        "Ты эксперт по кибербезопасности и модерации контента в TikTok. "
        "Твоя задача — проанализировать предоставленный Evidence Pack (расшифровку речи и текст с кадров) "
        "и определить, содержит ли видео признаки мошенничества, финансовых пирамид, фишинга или обмана. "
        "Верни ответ строго в формате JSON со следующими полями:\n"
        "{\n"
        '  "is_scam": true/false,\n'
        '  "confidence": число от 0.0 до 1.0,\n'
        '  "violated_policy": "название правила или null",\n'
        '  "reasoning": "подробное обоснование на русском языке",\n'
        '  "evidence": [{"timestamp": "время", "text": "цитата нарушения"}]\n'
        "}"
    )

    client = OpenAI() # Использует OPENAI_API_KEY из окружения

    print("Отправка Evidence Pack в LLM для анализа...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Вот собранные данные по видео:\n{json.dumps(evidence_pack, ensure_ascii=False, indent=2)}"}
            ],
            response_format={"type": "json_object"}
        )

        result_content = response.choices[0].message.content
        analysis_result = json.loads(result_content)
        
        return analysis_result

    except Exception as e:
        print(f"Ошибка при обращении к LLM: {e}")
        return None