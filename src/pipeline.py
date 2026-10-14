import pandas as pd

from src.ingest import download_video_and_audio
from src.frames import extract_frames
from src.asr import transcribe_audio
from src.ocr import extract_text_from_frames
from src.llm_judge import analyze_video_evidence

def run_pipeline():
    # Путь к твоему файлу в папке data
    file_path = 'data/dataset.xlsx' # Замени на точное имя файла
    
    df = pd.read_excel(file_path)
    
    # Проходим по каждой строке датафрейма
    for index, row in df.iterrows():
        video_url = row['metadata']        # Убедись, что колонка в Excel называется 'url'
        true_category = row['category'] # Убедись, что колонка называется 'category'
        print(f"Обработка видео: {video_url} [Категория: {true_category}]")
        
        # Здесь в будущем будет вызываться логика скачивания и анализа (ingest, asr, ocr и т.д.)

def run_pipeline_for_video(video_url):
    # Шаг 3а: Ингест (скачивание видео + аудио)
    success = download_video_and_audio(video_url)
    if not success:
        return
        
    # Шаг 3б: Нарезка кадров
    extract_frames(video_path="temp_video.mp4", output_dir="extracted_frames")


# if __name__ == "__main__":
#     run_pipeline()
if __name__ == "__main__":
    # Укажите реальную или тестовую ссылку на TikTok-видео
    test_url = "https://vt.tiktok.com/ZSbESuo8m"
    
    print("Запуск проверки этапа ингеста...")
    
    # 1. Скачиваем видео и аудио
    success = download_video_and_audio(test_url)
    
    if success:
        # Нарезаем кадры
        extract_frames("temp_video.mp4", "extracted_frames")
        
        print("\n--- ASR (Whisper) ---")
        speech_segments = transcribe_audio("audio.wav")
        print("Пример распознанной речи:", speech_segments[:2])
        
        print("\n--- OCR (EasyOCR) ---")
        ocr_data = extract_text_from_frames("extracted_frames")
        print("Пример текста с кадров:", ocr_data[:2])
        
        print("\n-- Анализ через LLM ---")
        verdict = analyze_video_evidence(speech_segments, ocr_data)
        print("Результат анализа:", verdict)
    else:
        print("Ошибка на этапе скачивания.")