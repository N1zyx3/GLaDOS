print("GlaDOS TTS initialization...")
import os
import traceback
import re
import eng_to_ipa as ipa  # Добавили для получения точной транскрипции
from TeraTTS import TTS
from ruaccent import RUAccent
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

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
    clean_word = word.lower().strip(".,!?\"'()*-")
    if not clean_word:
        return word

    # Получаем международную транскрипцию
    phonetics = ipa.convert(clean_word)

    # Если слово незнакомое, библиотека вернет его со звездочкой
    if '*' in phonetics:
        return word

    phonetics = phonetics.replace('ə', 'э')  # Заменяем знаки шва

    ru_sound = phonetics
    # Заменяем дифтонги и одиночные звуки на русские буквы
    for ipa_char, ru_char in sorted(IPA_TO_RU.items(), key=lambda x: len(x[0]), reverse=True):
        ru_sound = ru_sound.replace(ipa_char, ru_char)

    # Удаляем любые оставшиеся спецсимволы
    ru_sound = re.sub(re.compile(r'[^а-яА-ЯёЁ+]'), '', ru_sound)

    # Корректируем ударение, чтобы оно шло ПОСЛЕ гласной
    if '+' in ru_sound:
        ru_sound = ru_sound.replace('+', '')
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
        if re.search(r'[a-zA-Z]', word):
            # Сохраняем знаки препинания вокруг слова
            prefix = re.match(r'^[^a-zA-Z]*', word).group(0)
            suffix = re.search(r'[^a-zA-Z]*$', word).group(0)
            pure_word = word.strip("^.?!,()\"'-")

            ru_phonetic_word = convert_word_to_ru_phonetics(pure_word)
            processed_words.append(f"{prefix}{ru_phonetic_word}{suffix}")
        else:
            processed_words.append(word)
    return " ".join(processed_words)


def text_to_speech(text, tts, accentizer, custom_dict, save_to_file=False, filename="glados.wav"):
    # Переводим английские слова в русскую транскрипцию вместо грубого транслита
    processed_text = english_to_russian_phonetics(text)
    print(f"[Отладка] Текст перед озвучкой: {processed_text}")

    # Применяем словарь ударений
    for k, v in custom_dict.items():
        processed_text = processed_text.replace(k, v)

    # Расставляем ударения и запускаем озвучку
    accented_text = accentizer.process_all(processed_text.strip())

    audio = tts(accented_text, play=not save_to_file, lenght_scale=1.1)

    if save_to_file:
        save_path = os.path.join(os.path.join(os.path.expanduser("~"), "Downloads"), filename)
        tts.save_wav(audio, save_path)
        print(f"Файл сохранён: {save_path}")


def main():
    print("[Отладка] Инициализация RUAccent...")
    try:
        accentizer = RUAccent()
        custom_dict = {
            'ГЛаДОС': 'ГЛ+А+ДОС',
            'ГЛАДОС': 'ГЛ+АДОС',
            'ГлаДОС': 'Гл+аДОС',
            'ИИ': '+И-+И',
            'АИ': '+А-+И'
        }
        accentizer.load(
            omograph_model_size='turbo',
            use_dictionary=True,
            custom_dict=custom_dict
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        return

    print("[Отладка] Загрузка голосовой модели TeraTTS...")
    try:
        tts = TTS(
            "TeraTTS/glados2-g2p-vits",
            add_time_to_end=1.0,
            tokenizer_load_dict=True
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        return

    print("GLaDOS TTS готов.")

    try:
        while True:
            text = input("Текст ('конец экспериментам' для выхода): ").strip()

            if text.lower() == "конец экспериментам":
                text_to_speech("Конец? Уже? А я только разогрелась...", tts, accentizer, custom_dict)
                break

            if text:
                if text.startswith("ЗАПИСЬ:"):
                    text = text[8:].strip()

                    if text:
                        filename = "glados.wav"

                        if ":" in text:
                            first, rest = text.split(":", 1)
                            if first.lower().endswith(".wav"):
                                filename = first.strip()
                                text = rest.strip()

                        text_to_speech(text, tts, accentizer, custom_dict, save_to_file=True, filename=filename)
                else:
                    text_to_speech(text, tts, accentizer, custom_dict)

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()