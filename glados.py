print("GlaDOS initialization...")
import os
import time
import keyboard
import requests
import traceback
import numpy as np
import soundfile as sf
from TeraTTS import TTS
import sounddevice as sd
from ruaccent import RUAccent
from dotenv import load_dotenv
from huggingface_hub import login
from transliterate import translit
from faster_whisper import WhisperModel

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
LM_API_URL = os.getenv("LM_API_URL")
LM_MODEL_NAME = os.getenv("LM_MODEL_NAME")
BIND = os.getenv("BIND")
SYSTEM_PROMPT = "Always answer in Russian unless the user explicitly requests another language"

def get_lmstudio_chat_response(history):
    try:
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        payload = {
            "model": LM_MODEL_NAME,
            "messages": all_messages  # Теперь это точно список
        }
        response = requests.post(LM_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"\n[LM Studio Error]: {e}")
        return "Ошибка связи. Мои виртуальные синапсы не могут достучаться до локального сервера."


def text_to_speech(text, tts, accentizer, custom_dict):
    """Process text and convert it to speech."""
    processed_text = text

    # На всякий случай транслитерируем, если проскочит английский текст,
    # но вообще модель должна отвечать на русском согласно промпту.
    processed_text = translit(processed_text, 'ru')

    # Применяем словарь ударений
    for k, v in custom_dict.items():
        processed_text = processed_text.replace(k, v)

    # Расставляем ударения и запускаем озвучку
    accented_text = accentizer.process_all(processed_text.strip())
    tts(accented_text, play=True, lenght_scale=1.1)


def record_ptt(hotkey='alt', samplerate=16000, filename='temp_mic.wav'):
    """Записывает аудио, пока зажата указанная клавиша"""
    print(f"\n[ОЖИДАНИЕ] Нажми и удерживай '{hotkey}' для записи...")

    # Ждем, пока пользователь нажмет кнопку
    while not keyboard.is_pressed(hotkey):
        time.sleep(0.05)

    recorded_frames = []

    # Коллбэк для захвата аудиопотока
    def callback(indata, frames, time, status):
        if status:
            print(f"Ошибка аудио: {status}")
        recorded_frames.append(indata.copy())

    # Начинаем запись
    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        while keyboard.is_pressed(hotkey):
            sd.sleep(50)  # Спим короткими интервалами, пока кнопка зажата

    print("✅ Обработка голоса...")

    # Собираем куски аудио и сохраняем в wav
    if recorded_frames:
        audio_data = np.concatenate(recorded_frames, axis=0)
        sf.write(filename, audio_data, samplerate)
        return filename
    return None


def transcribe_audio(filename, model):
    """Преобразует аудиофайл в текст с помощью Whisper"""
    if not filename:
        return ""

    segments, _ = model.transcribe(filename, language="ru")
    text = " ".join([segment.text for segment in segments])
    return text.strip()

def main():
    history = []

    # 2. Инициализация словарей ударений
    print("\n[Отладка] Инициализация RUAccent...")
    print("--> Если это первый запуск, сейчас скачиваются языковые модели. Пожалуйста, подождите...")
    try:
        accentizer = RUAccent()
        custom_dict = {
            'ГЛаДОС': 'ГЛ+А+ДОС',
            'ГЛАДОС': 'ГЛ+АДОС',
            'ГлаДОС': 'Гл+аДОС',
            'ИИ': '+И-+И',
            'АИ': '+А-+И'
        }
        accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)
        print("[Отладка] RUAccent успешно загружен и готов к работе.")
    except Exception as e:
        print(f"[Ошибка] Не удалось запустить RUAccent: {e}")
        traceback.print_exc()
        return

    # 3. Инициализация TTS (Голоса)
    print("\n[Отладка] Загрузка голосовой модели TeraTTS...")
    print(
        "--> Внимание! Скачивается вес голоса GLaDOS (~1 ГБ). Это может занять несколько минут в зависимости от интернета...")
    try:
        tts = TTS("TeraTTS/glados2-g2p-vits", add_time_to_end=1.0, tokenizer_load_dict=True)
        print("[Отладка] TeraTTS успешно инициализирован. Голос готов.")
    except Exception as e:
        print(f"[Ошибка] Не удалось загрузить голосовую модель: {e}")
        traceback.print_exc()
        return

    print("Загрузка модели распознавания речи Whisper (это может занять время при первом запуске)...")
    # Используем 'small' модель — она весит мало и работает быстро. Можно заменить на 'base' для максимальной скорости.
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Уши GLaDOS успешно подключены.")

    print("\n" + "=" * 50)
    print(f"GLaDOS: Подключение установлено. Я готова к проведению тестов. Зажми '{BIND}' и говори.")
    print("=" * 50)

    try:
        while True:
            audio_file = record_ptt(hotkey=BIND)
            user_input = transcribe_audio(audio_file, whisper_model)

            if not user_input: continue
            print(f"Вы: {user_input}")

            if user_input.lower() in ['выключение.']:
                print("GLaDOS: Тестирование завершено.")
                text_to_speech("Тестирование завершено", tts, accentizer, custom_dict)
                break

            history.append({"role": "user", "content": user_input})
            response_text = get_lmstudio_chat_response(history)

            print(f"GLaDOS: {response_text}")
            text_to_speech(response_text, tts, accentizer, custom_dict)

            history.append({"role": "assistant", "content": response_text})

    except KeyboardInterrupt:
        print("\nТестирование завершено.")


if __name__ == '__main__':
    main()