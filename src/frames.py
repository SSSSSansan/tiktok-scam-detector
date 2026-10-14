import os
import cv2

def extract_frames(video_path="temp_video.mp4", output_dir="extracted_frames"):
    """
    Нарезает кадры из видео с частотой 1 кадр в секунду с помощью OpenCV.
    Сохраняет их с таймкодом в названии (например, frame_00_12.jpg).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(video_path):
        # Поиск альтернативного видеофайла с другим расширением
        base, _ = os.path.splitext(video_path)
        for f in os.listdir('.'):
            if f.startswith(os.path.basename(base)) and f.endswith(('.mp4', '.webm', '.mkv', '.mov')):
                video_path = f
                break

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: не удалось открыть видеофайл {video_path} для нарезки кадров.")
        return 0
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Дефолтное значение
        
    frame_interval = int(round(fps))  # Интервал для получения 1 кадра в секунду
    frame_count = 0
    saved_count = 0
    
    print(f"Начинаем нарезку кадров из {video_path}...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            seconds = int(frame_count / fps)
            minutes = seconds // 60
            secs = seconds % 60
            
            frame_name = f"frame_{minutes:02d}_{secs:02d}.jpg"
            frame_filepath = os.path.join(output_dir, frame_name)
            
            cv2.imwrite(frame_filepath, frame)
            saved_count += 1
            
        frame_count += 1
        
    cap.release()
    print(f"Нарезка завершена! Сохранено кадров: {saved_count} в папке '{output_dir}'.")
    return saved_count