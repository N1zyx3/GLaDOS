print("Importing libraries...")
import os
import yaml
import json
import requests
import traceback
from TeraTTS import TTS
from ruaccent import RUAccent
from dotenv import load_dotenv
from transliterate import translit
from huggingface_hub import login

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

def get_lmstudio_chat_response(config, history):
    """Send text history to LM Studio and get response (OpenAI Chat Completion compatible)"""
    try:
        # Формируем историю сообщений, добавляя системный промпт GLaDOS в начало
        payload = {
            "model": config['lmstudio']['model_name'],
            "messages": [{"role": "system", "content": config['lmstudio']['prompt']}] + history,
            **config['lmstudio']['options']
        }

        headers = {
            "Content-Type": "application/json"
        }

        response = requests.post(config['lmstudio']['api_url'], headers=headers, json=payload)
        response.raise_for_status()
        res_json = response.json()

        return res_json['choices'][0]['message']['content']
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


def main():
    print("GLaDOS Console Chat starts...")

    # 1. Загрузка конфигурации
    print("[Отладка] Загрузка файла конфигурации config.yaml...")
    try:
        with open("config.yaml", "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print("[Отладка] Конфигурация успешно загружена.")
    except Exception as e:
        print(f"[Ошибка] Не удалось прочитать config.yaml: {e}")
        return

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

    print("\n" + "=" * 50)
    print(
        "GLaDOS: Подключение установлено. Я готова к проведению тестов. Можешь вводить свои бессмысленные текстовые запросы. (Для выхода напиши 'выход')")
    print("=" * 50)

    try:
        while True:
            user_input = input("\nВы: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['выход', 'exit', 'quit']:
                print("GLaDOS: Оу. Уходишь? Как обычно. Тестирование завершено.")
                break

            history.append({"role": "user", "content": user_input})

            print("GLaDOS думает...")
            response_text = get_lmstudio_chat_response(config, history)

            print(f"GLaDOS: {response_text}")

            text_to_speech(response_text, tts, accentizer, custom_dict)

            if bool(config.get('keep_history', True)):
                history.append({"role": "assistant", "content": response_text})
                with open('history.json', 'w', encoding="utf-8") as f:
                    json.dump(history, f, indent=4, ensure_ascii=False)
            else:
                history.clear()

    except KeyboardInterrupt:
        print("\nЧат принудительно завершен. Ты чудовище.")


if __name__ == '__main__':
    main()