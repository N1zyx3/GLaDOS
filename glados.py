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
                break

            if text:
                text_to_speech(text, tts, accentizer, custom_dict)

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()