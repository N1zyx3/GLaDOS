print("GlaDOS TTS initialization...")
import os
import traceback
from TeraTTS import TTS
from ruaccent import RUAccent
from dotenv import load_dotenv
from huggingface_hub import login
from transliterate import translit

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def text_to_speech(text, tts, accentizer, custom_dict, save_to_file=False, filename="glados.wav"):
    processed_text = text

    processed_text = translit(processed_text, 'ru')

    # Применяем словарь ударений
    for k, v in custom_dict.items():
        processed_text = processed_text.replace(k, v)

    # Расставляем ударения и запускаем озвучку
    accented_text = accentizer.process_all(processed_text.strip())

    audio = tts(accented_text, play=not save_to_file, lenght_scale=1.1)

    if save_to_file:
        tts.save_wav(audio, os.path.join(os.path.join(os.path.expanduser("~"), "Downloads"), filename))
        print(f"Файл сохранён: {os.path.join(os.path.join(os.path.expanduser("~"), "Downloads"), filename)}")

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

                        # Позволяет указать имя файла:
                        # ЗАПИСЬ:test.wav: Привет
                        if ":" in text:
                            first, rest = text.split(":", 1)
                            if first.lower().endswith(".wav"):
                                filename = first.strip()
                                text = rest.strip()

                        text_to_speech(text, tts,accentizer, custom_dict, save_to_file=True, filename=filename)
                else:
                    text_to_speech(text, tts, accentizer, custom_dict)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()