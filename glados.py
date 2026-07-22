print("GlaDOS Voice Changer initialization...")
import os
import queue
import numpy as np
import soundfile as sf
import sounddevice as sd
import re
import eng_to_ipa as ipa
from TeraTTS import TTS
from ruaccent import RUAccent
from dotenv import load_dotenv
from huggingface_hub import login
from faster_whisper import WhisperModel

load_dotenv()
login(token=os.getenv("HF_TOKEN"))
spell = "Ты не обычный дурак, ты спроектированный дурак"

# Карта соответствия международных фонетических знаков (IPA) русским звукам
IPA_TO_RU = {
    'heˈloʊ': 'хэл+оу', 'ˈheˈloʊ': 'хэл+оу',
    'ɑ': 'а', 'æ': 'э', 'ʌ': 'а', 'ɔ': 'о', 'ɒ': 'о', 'ɛ': 'э', 'ɜ': 'э', 'ɪ': 'и', 'i': 'и', 'ʊ': 'у', 'u': 'у',
    'baɪ': 'бай', 'aɪ': 'ай', 'eɪ': 'эй', 'ɔɪ': 'ой', 'oʊ': 'оу', 'aʊ': 'ау', 'ɪə': 'иэ', 'eə': 'эа', 'ʊə': 'уэ',
    'p': 'п', 'b': 'б', 't': 'т', 'd': 'д', 'k': 'к', 'g': 'г', 'f': 'ф', 'v': 'в', 'θ': 'с', 'ð': 'з',
    's': 'с', 'z': 'з', 'ʃ': 'ш', 'ʒ': 'ж', 'h': 'х', 'm': 'м', 'n': 'н', 'ŋ': 'нг', 'l': 'л', 'r': 'р',
    'j': 'й', 'w': 'у', 'tʃ': 'ч', 'dʒ': 'дж', 'ˈ': '+', 'ˌ': ''
}


def get_voicemeeter_output_device():
    """Находит виртуальный кабель для вывода звука."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev['name'].lower()
        if "cable input" in name or "vaio input" in name:
            return idx
    return sd.default.device[1]


OUTPUT_DEVICE_ID = get_voicemeeter_output_device()

# ==================== БЛОК АВТОМАТИЧЕСКОЙ ПРОСЛУШКИ (VAD) ====================
AUDIO_QUEUE = queue.Queue()


def audio_callback(indata, frames, time, status):
    """Складывает входящий звук с микрофона в очередь"""
    if status: print(f"Ошибка аудиопотока: {status}")
    AUDIO_QUEUE.put(indata.copy())


def calibrate_mic(samplerate=16000, duration=2.5):
    """Слушает тишину в комнате, чтобы вычислить порог шума."""
    print(f"\n[АВТОНАСТРОЙКА] Пожалуйста, помолчите {duration} секунды. Анализирую шум комнаты...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()

    # Считаем уровень фонового шума
    rms = np.sqrt(np.mean(recording ** 2))
    threshold = rms * 5.0  # Голос должен быть в 5 раз громче тишины
    if threshold < 0.005: threshold = 0.005  # Защита от идеальной тишины

    print(f"[УСПЕШНО] Базовый порог срабатывания установлен: {threshold:.5f}\n")
    return threshold


def record_vad(threshold, samplerate=16000, silence_limit=1.5):
    """Слушает микрофон непрерывно и записывает фразы автоматически."""
    print(f"[СЛУШАЮ] Говорите... (Скажите '{spell}' для завершения)")
    audio_buffer = []
    is_speaking = False
    silence_time = 0

    # Очищаем очередь от возможных старых звуков
    while not AUDIO_QUEUE.empty():
        AUDIO_QUEUE.get()

    stream = sd.InputStream(samplerate=samplerate, channels=1, callback=audio_callback, dtype='float32')

    with stream:
        while True:
            chunk = AUDIO_QUEUE.get()
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > threshold:
                if not is_speaking:
                    print(" -> [Голос обнаружен, записываю...]")
                    is_speaking = True
                silence_time = 0
                audio_buffer.append(chunk)
            elif is_speaking:
                silence_time += len(chunk) / samplerate
                audio_buffer.append(chunk)

                if silence_time > silence_limit:
                    print(" -> [Пауза, обрабатываю фразу]")
                    break

    if audio_buffer:
        audio_data = np.concatenate(audio_buffer, axis=0)
        filename = 'temp_mic.wav'
        sf.write(filename, audio_data, samplerate)
        return filename
    return None


# ==============================================================================

def convert_word_to_ru_phonetics(word):
    clean_word = word.lower().strip(".,!?\"'")
    if not clean_word: return word
    phonetics = ipa.convert(clean_word)
    if '*' in phonetics: return word
    phonetics = phonetics.replace('ə', 'э')
    ru_sound = phonetics
    for ipa_char, ru_char in sorted(IPA_TO_RU.items(), key=lambda x: len(x[0]), reverse=True):
        ru_sound = ru_sound.replace(ipa_char, ru_char)
    ru_sound = re.sub(re.compile(r'[^а-яА-ЯёЁ+]'), '', ru_sound)
    if '+' in ru_sound:
        ru_sound = ru_sound.replace('+', '')
        for vowel in 'аеёиоуыэюяАЕЁИОУЫЭЮЯ':
            if vowel in ru_sound:
                ru_sound = ru_sound.replace(vowel, vowel + '+', 1)
                break
    return ru_sound


def english_to_russian_phonetics(text):
    words = text.split()
    processed_words = []
    for word in words:
        if re.search(r'[a-zA-Z]', word):
            prefix = re.match(r'^[^a-zA-Z]*', word).group(0)
            suffix = re.search(r'[^a-zA-Z]*$', word).group(0)
            pure_word = word.strip("^.?!,()\"'")
            ru_phonetic_word = convert_word_to_ru_phonetics(pure_word)
            processed_words.append(f"{prefix}{ru_phonetic_word}{suffix}")
        else:
            processed_words.append(word)
    return " ".join(processed_words)


def text_to_speech(text, tts, accentizer, custom_dict):
    """Генерирует голос с защитой от проглатывания первого звука (ресемплинг + тишина)."""
    processed_text = english_to_russian_phonetics(text)
    for k, v in custom_dict.items():
        processed_text = processed_text.replace(k, v)

    accented_text = accentizer.process_all(processed_text.strip())

    audio_data = tts(accented_text, play=False, lenght_scale=1.1)
    audio_data = np.asarray(audio_data, dtype=np.float32)

    source_samplerate = 22050
    target_samplerate = 48000
    num_samples = int(len(audio_data) * target_samplerate / source_samplerate)

    resampled_audio = np.interp(
        np.linspace(0, len(audio_data), num_samples, endpoint=False),
        np.arange(len(audio_data)),
        audio_data
    ).astype(np.float32)

    # Защитная тишина (0.3 сек) чтобы драйвер успел "проснуться"
    silence_samples = int(target_samplerate * 0.3)
    silence_padding = np.zeros(silence_samples, dtype=np.float32)
    final_audio = np.concatenate((silence_padding, resampled_audio, silence_padding))

    try:
        with sd.OutputStream(
                samplerate=target_samplerate,
                device=OUTPUT_DEVICE_ID,
                channels=1,
                dtype='float32',
                blocksize=512
        ) as stream:
            stream.write(final_audio)

    except Exception as e:
        print(f"[Ошибка трансляции]: {e}")
        sd.play(audio_data, samplerate=source_samplerate, device=OUTPUT_DEVICE_ID)
        sd.wait()


def transcribe_audio(filename, model):
    if not filename: return ""
    segments, _ = model.transcribe(filename, language="ru")
    return " ".join([segment.text for segment in segments]).strip()


def main():
    print("[Отладка] Инициализация RUAccent...")
    try:
        accentizer = RUAccent()
        custom_dict = {'ГЛаДОС': 'ГЛ+А+ДОС', 'ГЛАДОС': 'ГЛ+АДОС', 'ГлаДОС': 'Гл+аДОС', 'ИИ': '+И-+И', 'АИ': '+А-+И'}
        accentizer.load(omograph_model_size='turbo', use_dictionary=True, custom_dict=custom_dict)
    except Exception as e:
        print(f"Ошибка RUAccent: {e}")
        return

    print("[Отладка] Загрузка голосовой модели GLaDOS...")
    try:
        tts = TTS("TeraTTS/glados2-g2p-vits", add_time_to_end=1.0, tokenizer_load_dict=True)
    except Exception as e:
        print(f"Ошибка TeraTTS: {e}")
        return

    print("Загрузка модели Whisper...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

    # 1. Калибровка микрофона перед началом работы
    mic_threshold = calibrate_mic()

    print("=" * 50)
    print("МОДУЛЯТОР ГОЛОСА АКТИВИРОВАН.")
    print("Теперь не нужно нажимать кнопки. Просто говори в микрофон.")
    print("=" * 50)

    text_to_speech("Режим прослушивания активирован.", tts, accentizer, custom_dict)

    try:
        while True:
            # 2. Непрерывная умная прослушка
            audio_file = record_vad(threshold=mic_threshold)

            if not audio_file: continue

            user_text = transcribe_audio(audio_file, whisper_model)

            if not user_text: continue
            print(f"Вы сказали: {user_text}")

            if user_text.lower() in [spell, spell + '.',]:
                text_to_speech("Эксперимент завершен. Модуль голосовой маскировки отключен.", tts, accentizer,
                               custom_dict)
                break

            text_to_speech(user_text, tts, accentizer, custom_dict)

    except KeyboardInterrupt:
        print("\nПрограмма остановлена.")


if __name__ == '__main__':
    main()