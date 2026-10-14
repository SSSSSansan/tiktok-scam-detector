import os
import easyocr
import re

# Инициализируем reader один раз при импорте модуля (поддерживаем английский и русский языки)
print("Инициализация EasyOCR модели...")
reader = easyocr.Reader(['ru', 'en'], gpu=True) # Установите gpu=True, если у вас настроена видеокарта NVIDIA

def extract_text_from_frames(frames_dir="extracted_frames"):
    """
    Проходит по всем кадрам в папке, извлекает текст через EasyOCR 
    и сопоставляет его с таймкодом, извлеченным из названия файла.
    """
    if not os.path.exists(frames_dir):
        print(f"Папка с кадрами {frames_dir} не найдена.")
        return []

    ocr_results = []
    frame_files = sorted(os.listdir(frames_dir))
    
    print(f"Начинаем OCR-обработку {len(frame_files)} кадров...")
    
    for filename in frame_files:
        if not filename.endswith(('.jpg', '.png', '.jpeg')):
            continue
            
        # Парсим таймкод из имени файла вида frame_00_12.jpg -> 0:12
        match = re.search(r'frame_(\d+)_(\d+)', filename)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            timestamp_str = f"{minutes}:{seconds:02d}"
        else:
            timestamp_str = "0:00"
            
        frame_path = os.path.join(frames_dir, filename)
        
        try:
            # Распознаем текст на кадре (detail=0 возвращает только список строк текста)
            results = reader.readtext(frame_path, detail=0)
            text_on_frame = " ".join(results).strip()
            
            if text_on_frame:
                ocr_results.append({
                    "timestamp": timestamp_str,
                    "frame_file": filename,
                    "text": text_on_frame
                })
        except Exception as e:
            print(f"Ошибка OCR для файла {filename}: {e}")
            
    print(f"OCR-обработка завершена. Найдено кадров с текстом: {len(ocr_results)}")
    return ocr_results