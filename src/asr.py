import os
import whisper

# Загружаем модель один раз при импорте модуля. 
# Возможные размеры: 'tiny', 'base', 'small', 'medium', 'large'. 
# Для быстрого запуска на CPU отлично подойдет 'base' или 'small'.
print("Загрузка локальной модели Whisper ('base')...")
model = whisper.load_model("base")

def transcribe_audio(audio_path="audio.wav"):
    """
    Распознает аудио локально с помощью библиотеки whisper, 
    возвращая текст с таймкодами сегментов.
    """
    if not os.path.exists(audio_path):
        print(f"Ошибка: Аудиофайл {audio_path} не найден.")
        return []

    print(f"Локальное распознавание речи для файла {audio_path}...")
    try:
        # Запускаем транскрипцию с автоопределением языка (поддерживает казахский, русский, английский и др.)
        result = model.transcribe(audio_path, verbose=False)
        
        segments_data = []
        if "segments" in result and result["segments"]:
            for segment in result["segments"]:
                segments_data.append({
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "text": segment["text"].strip()
                })
        else:
            # Запасной вариант, если сегменты не выделились
            segments_data.append({
                "start": 0.0,
                "end": 0.0,
                "text": result.get("text", "").strip()
            })
            
        print(f"Распознавание речи успешно завершено. Найдено сегментов: {len(segments_data)}")
        return segments_data

    except Exception as e:
        print(f"Ошибка при локальном распознавании Whisper: {e}")
        return []