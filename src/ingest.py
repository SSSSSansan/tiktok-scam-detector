import os
import yt_dlp

def download_video_and_audio(video_url, output_video_path="temp_video.mp4"):
    """
    Скачивает видео и конвертирует аудиодорожку в format .wav (audio.wav) с помощью yt-dlp.
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_video_path.replace('.mp4', '.%(ext)s'),
        'nocheckcertificate': True,  # Отключаем проверку SSL-сертификатов для обхода ошибки с путями
        'keepvideo': True,  # Сохраняем исходный видеофайл вместе с аудио!
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }
    
    print(f"Начинаем загрузку: {video_url}")
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except Exception as e:
        print(f"Ошибка при скачивании через yt-dlp: {e}")
        return False

    # Приводим имя аудиофайла к стандарту audio.wav
    possible_audio_names = [
        output_video_path.replace('.mp4', '.wav'),
        output_video_path.replace('.mp4', '.m4a.wav'),
        output_video_path.replace('.mp4', '.webm.wav')
    ]
    
    audio_found = False
    for name in possible_audio_names:
        if os.path.exists(name):
            if os.path.exists('audio.wav'):
                os.remove('audio.wav')  # Удаляем старый файл, если он есть
            os.rename(name, 'audio.wav')
            audio_found = True
            break
            
    if audio_found:
        print("Аудио успешно конвертировано и сохранено как audio.wav")
        return True
    else:
        print("Предупреждение: Аудиофайл .wav не найден.")
        return False