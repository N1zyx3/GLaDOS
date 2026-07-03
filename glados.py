print("GlaDOS initialization...")
import os
import time
import keyboard
import requests
import numpy as np
import soundfile as sf
import sounddevice as sd
import re
import eng_to_ipa as ipa  # Добавили для получения точной транскрипции
from TeraTTS import TTS
from ruaccent import RUAccent
from dotenv import load_dotenv
from huggingface_hub import login
from faster_whisper import WhisperModel

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
LM_API_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL_NAME = "local-model"
BIND = "end"
SYSTEM_PROMPT = "Always answer in Russian unless the user explicitly requests another language"

# Карта соответствия международных фонетических знаков (IPA) русским звукам
IPA_TO_RU = {
    'heˈloʊ': 'хэл+оу', 'ˈheˈloʊ': 'хэл+оу',  # Быстрый хардкод для частых слов
    'ɑ': 'а', 'æ': 'э', 'ʌ': 'а', 'ɔ': 'о', 'ɒ': 'о', 'ɛ': 'э', 'ɜ': 'э', 'ɪ': 'и', 'i': 'и', 'ʊ': 'у', 'u': 'у',
    'baɪ': 'бай', 'aɪ': 'ай', 'eɪ': 'эй', 'ɔɪ': 'ой', 'oʊ': 'оу', 'aʊ': 'ау', 'ɪə': 'иэ', 'eə': 'эа', 'ʊə': 'уэ',
    'p': 'п', 'b': 'б', 't': 'т', 'd': 'д', 'k': 'к', 'g': 'г', 'f': 'ф', 'v': 'в', 'θ': 'с', 'ð': 'з',
    's': 'с', 'z': 'з', 'ʃ': 'ш', 'ʒ': 'ж', 'h': 'х', 'm': 'м', 'n': 'н', 'ŋ': 'нг', 'l': 'л', 'r': 'р',
    'j': 'й', 'w': 'у', 'tʃ': 'ч', 'dʒ': 'дж', 'ˈ': '+', 'ˌ': ''
}


def convert_word_to_ru_phonetics(word):
    """Преобразует одно английское слово в его реальное русское звучание."""
    clean_word = word.lower().strip(".,!?\"'")
    if not clean_word:
        return word

    # Получаем транскрипцию (например, "hello" -> "həˈloʊ" или "heˈloʊ")
    phonetics = ipa.convert(clean_word)

    # Если библиотека не знала слова, она вернет его с астериском (напр. "word*")
    if '*' in phonetics:
        return word  # Возвращаем как есть, обработается обычным образом

    # Заменяем знаки шва (ə) в зависимости от окружения на э/о/а
    phonetics = phonetics.replace('ə', 'э')

    # Переводим фонетические знаки в русские буквы
    ru_sound = phonetics
    # Сначала заменяем длинные дифтонги, потом одиночные звуки
    for ipa_char, ru_char in sorted(IPA_TO_RU.items(), key=lambda x: len(x[0]), reverse=True):
        ru_sound = ru_sound.replace(ipa_char, ru_char)

    # Очистка от спецсимволов IPA, которые могли остаться
    ru_sound = re.sub(re.compile(r'[^а-яА-ЯёЁ+]'), '', ru_sound)

    # Корректируем знак ударения (в русском он идет ПОСЛЕ гласной, в IPA — ПЕРЕД слогом)
    if '+' in ru_sound:
        ru_sound = ru_sound.replace('+', '')
        # Ставим ударение на первую гласную для простоты, если сложная структура
        for vowel in 'аеёиоуыэюяАЕЁИОУЫЭЮЯ':
            if vowel in ru_sound:
                ru_sound = ru_sound.replace(vowel, vowel + '+', 1)
                break

    return ru_sound


def english_to_russian_phonetics(text):
    """Находит в тексте английские слова и заменяет их на русское звучание."""
    words = text.split()
    processed_words = []
    for word in words:
        # Проверяем, состоит ли слово из английских букв
        if re.search(r'[a-zA-Z]', word):
            # Выделяем знаки препинания, чтобы не сломать перевод
            prefix = re.match(r'^[^a-zA-Z]*', word).group(0)
            suffix = re.search(r'[^a-zA-Z]*$', word).group(0)
            pure_word = word.strip("^.?!,()\"'")

            ru_phonetic_word = convert_word_to_ru_phonetics(pure_word)
            processed_words.append(f"{prefix}{ru_phonetic_word}{suffix}")
        else:
            processed_words.append(word)
    return " ".join(processed_words)


def get_lmstudio_chat_response(history):
    try:
        all_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        payload = {
            "model": LM_MODEL_NAME,
            "messages": all_messages
        }
        response = requests.post(LM_API_URL, json=payload)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"[LM Studio Error]: {e}")
        return "Ошибка связи с локальным сервером."


def text_to_speech(text, tts, accentizer, custom_dict):
    """Обрабатывает текст и озвучивает его через русскую GLaDOS."""
    # ШАГ 1: Превращаем английские слова в русское звучание по транскрипции!
    processed_text = english_to_russian_phonetics(text)
    # print(f"[Отладка] Фонетический текст для GLaDOS: {processed_text}")

    # ШАГ 2: Применяем кастомный словарь ударений
    for k, v in custom_dict.items():
        processed_text = processed_text.replace(k, v)

    # ШАГ 3: Расставляем ударения в русских словах и озвучиваем
    accented_text = accentizer.process_all(processed_text.strip())
    tts(accented_text, play=True, lenght_scale=1.1)


def record_ptt(hotkey='alt', samplerate=16000, filename='temp_mic.wav'):
    print(f"[ОЖИДАНИЕ] Нажми и удерживай '{hotkey}' для записи (Для выхода скажи 'Отбой').")
    while not keyboard.is_pressed(hotkey):
        time.sleep(0.05)

    recorded_frames = []

    def callback(indata, frames, time, status):
        if status: print(f"Ошибка аудио: {status}")
        recorded_frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=callback)
    with stream:
        while keyboard.is_pressed(hotkey):
            sd.sleep(50)

    print("Обработка голоса...")
    if recorded_frames:
        audio_data = np.concatenate(recorded_frames, axis=0)
        sf.write(filename, audio_data, samplerate)
        return filename
    return None


def transcribe_audio(filename, model):
    if not filename: return ""
    segments, _ = model.transcribe(filename, language="ru")
    return " ".join([segment.text for segment in segments]).strip()


def main():
    history = []

    print("[Отладка] Инициализация RUAccent...")
    try:
        accentizer = RUAccent()
        custom_dict = {'ГЛаДОС': 'ГЛ+А+ДОС', 'ГЛАДОС': 'ГЛ+АДОС', 'ГлаДОС': 'Гл+аДОС', 'ИИ': '+И-+И', 'АИ': '+А-+И'}
        accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)
    except Exception as e:
        print(f"[Ошибка] Не удалось запустить RUAccent: {e}")
        return

    print("[Отладка] Загрузка голосовой модели TeraTTS...")
    try:
        tts = TTS("TeraTTS/glados2-g2p-vits", add_time_to_end=1.0, tokenizer_load_dict=True)
    except Exception as e:
        print(f"[Ошибка] Не удалось загрузить TeraTTS: {e}")
        return

    print("Загрузка модели распознавания речи Whisper...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("Уши GLaDOS успешно подключены.")

    print("\n" + "=" * 50)
    print(f"GLaDOS: Подключение установлено. Ошибок не обнаружено.")
    print("=" * 50)
    text_to_speech("Подключение установлено. Ошибок не обнаружено.", tts, accentizer, custom_dict)

    try:
        while True:
            audio_file = record_ptt(hotkey=BIND)
            user_input = transcribe_audio(audio_file, whisper_model)

            if not user_input: continue
            print(f"Вы: {user_input}")

            if user_input.lower() in ['отбой.', 'отбой']:
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