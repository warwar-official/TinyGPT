import json
import logging
import os
import re
import time
from datetime import datetime, time as dt_time, timedelta
from urllib import request, error
from copy import deepcopy

# --- Налаштування логування ---
# Налаштовуємо базовий логер для запису всіх дій та помилок у файл.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gemini_processor.log',
    filemode='a'
)
# Створюємо консольний обробник для виводу повідомлень користувачу
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)
# Додаємо консольний обробник до кореневого логера, щоб повідомлення дублювалися
logging.getLogger().addHandler(console_handler)


class ApiKeyManager:
    """
    Керує API-ключами, їх лімітами для кожної моделі та щоденним оновленням.
    Зберігає стан у файлі, щоб уникнути втрати даних при збоях.
    """
    # Словник з лімітами за замовчуванням для кожної моделі
    MODEL_LIMITS = {
        "gemini-1.5": 500,
        "gemini-2.0": 1500,
        "gemini-2.5": 500,
        "gemini-2.5-pro": 25,
    }

    def __init__(self, keys_config, settings_path='api_settings.json'):
        self.settings_path = settings_path
        self.keys = []
        self._load_state(keys_config)
        self._daily_reset()

    def _load_state(self, initial_keys_config):
        """
        Завантажує стан ключів з файлу або ініціалізує початковий стан.
        Додає нові ключі з конфігурації, якщо їх немає у файлі стану.
        """
        loaded_keys = {}
        try:
            with open(self.settings_path, 'r') as f:
                state = json.load(f)
            self.keys = state.get('keys', [])
            self.last_reset = datetime.fromisoformat(state.get('last_reset'))
            loaded_keys = {k['key'] for k in self.keys}
            logging.info(f"Стан ключів завантажено з '{self.settings_path}'.")
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning(f"Файл налаштувань '{self.settings_path}' не знайдено або пошкоджено. Ініціалізація...")
            self.last_reset = datetime.now() - timedelta(days=1)

        # Додавання нових ключів, яких немає у файлі стану
        new_keys_added = False
        for key_info in initial_keys_config:
            if key_info['key'] not in loaded_keys:
                self.keys.append({
                    'key': key_info['key'],
                    'model_limits': deepcopy(self.MODEL_LIMITS),
                    'blocked_until': None
                })
                new_keys_added = True
                logging.info(f"Додано новий ключ: ...{key_info['key'][-4:]}")
        
        if new_keys_added or not os.path.exists(self.settings_path):
             self._save_state()

    def _save_state(self):
        """Зберігає поточний стан ключів у файл."""
        state = {
            'last_reset': self.last_reset.isoformat(),
            'keys': self.keys
        }
        try:
            with open(self.settings_path, 'w') as f:
                json.dump(state, f, indent=4)
        except IOError as e:
            logging.error(f"Не вдалося зберегти стан ключів у '{self.settings_path}': {e}")

    def _daily_reset(self):
        """Скидає ліміти ключів, якщо настав новий день (після 6:00 ранку)."""
        now = datetime.now()
        reset_time = dt_time(6, 0)
        today_reset_moment = now.replace(hour=reset_time.hour, minute=reset_time.minute, second=0, microsecond=0)

        if now >= today_reset_moment and self.last_reset < today_reset_moment:
            logging.info("Настав час для щоденного скидання лімітів API ключів (6:00 ранку).")
            for key_info in self.keys:
                key_info['model_limits'] = deepcopy(self.MODEL_LIMITS)
                key_info['blocked_until'] = None
            self.last_reset = now
            self._save_state()
            logging.info("Ліміти ключів успішно скинуто.")

    def get_key(self, model_id: str):
        """Повертає доступний ключ для вказаної моделі."""
        if model_id not in self.MODEL_LIMITS:
            logging.error(f"Невідомий ідентифікатор моделі: '{model_id}'.")
            return None

        now = datetime.now()
        for i, key_info in enumerate(self.keys):
            is_blocked = key_info.get('blocked_until') and now < datetime.fromisoformat(key_info['blocked_until'])
            if not is_blocked and key_info['model_limits'].get(model_id, 0) > 0:
                logging.info(f"Використовується ключ #{i+1} для моделі '{model_id}' (залишилось запитів: {key_info['model_limits'][model_id]}).")
                return key_info
        return None

    def decrement_limit(self, key_str: str, model_id: str):
        """Зменшує ліміт для використаного ключа та моделі."""
        for key_info in self.keys:
            if key_info['key'] == key_str:
                if model_id in key_info['model_limits']:
                    key_info['model_limits'][model_id] -= 1
                break
        self._save_state()

    def block_key(self, key_str: str):
        """Блокує ключ до наступного скидання (при отриманні коду 429)."""
        now = datetime.now()
        tomorrow_6am = (now + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0)
        for key_info in self.keys:
            if key_info['key'] == key_str:
                key_info['blocked_until'] = tomorrow_6am.isoformat()
                logging.warning(f"Ключ, що закінчується на '...{key_str[-4:]}', заблоковано до {tomorrow_6am.strftime('%Y-%m-%d %H:%M')} через перевищення лімітів (код 429).")
                break
        self._save_state()


class GeminiProcessor:
    """
    Основний клас для обробки запитів до Gemini API.
    """
    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    MODEL_MAP = {
        "gemini-1.5": "gemini-1.5-flash-latest",
        "gemini-2.0": "gemini-2.0-flash",
        "gemini-2.5": "gemini-2.5-flash-preview-05-20",
        "gemini-2.5-pro": "gemini-2.5-pro-preview-06-05",
    }
    DEFAULT_MODEL_ID = "gemini-2.5"

    def __init__(self, request_data: dict):
        self.request_data = request_data
        self.warnings_mode = request_data.get('warnings', 'report').lower()
        self.default_output_file = request_data.get('output_file')
        self.prompts = request_data.get('prompts', [])

        if not request_data.get('api_keys'):
            raise ValueError("Список 'api_keys' є обов'язковим і не може бути порожнім.")
        
        self.key_manager = ApiKeyManager(request_data['api_keys'])

    def _handle_message(self, message, level='info'):
        """Виводить повідомлення відповідно до режиму обробки попереджень."""
        if self.warnings_mode == 'ignore':
            return
        if level == 'info': logging.info(message)
        elif level == 'warning': logging.warning(message)

    def _check_output_file(self, filepath):
        """Перевіряє існування файлу виводу та діє відповідно до режиму."""
        if not filepath:
            logging.error("Не вказано файл для виводу (ні за замовчуванням, ні для промту).")
            return False
        if os.path.exists(filepath):
            self._handle_message(f"Файл '{filepath}' вже існує.", 'warning')
            if self.warnings_mode == 'strict':
                logging.error(f"Режим 'strict': робота припинена, оскільки файл '{filepath}' існує.")
                return False
        return True

    def _write_result(self, filepath, result_text):
        """Записує результат у файл."""
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(result_text + '\n' + '-'*3 + '\n')
            logging.info(f"Результат успішно записано у файл '{filepath}'.")
        except IOError as e:
            logging.error(f"Помилка запису у файл '{filepath}': {e}")

    def _make_api_request(self, prompt_text: str, model_id: str):
        """
        Виконує один запит до API для вказаної моделі.
        """
        model_name = self.MODEL_MAP.get(model_id)
        if not model_name:
            logging.error(f"Некоректний ідентифікатор моделі '{model_id}'. Пропуск запиту.")
            return None, False

        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            key_info = self.key_manager.get_key(model_id)
            if not key_info:
                self._handle_message(f"Для моделі '{model_id}' не залишилося доступних ключів. Пропуск промту.", 'warning')
                return None, False

            api_key = key_info['key']
            url = self.API_URL_TEMPLATE.format(model_name=model_name) + f"?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            data = json.dumps({"contents": [{"parts": [{"text": prompt_text}]}]}).encode('utf-8')
            req = request.Request(url, data=data, headers=headers)

            try:
                with request.urlopen(req) as response:
                    if response.status == 200:
                        response_body = json.loads(response.read().decode('utf-8'))
                        self.key_manager.decrement_limit(api_key, model_id)
                        try:
                            content = response_body['candidates'][0]['content']['parts'][0]['text']
                            return content, False
                        except (KeyError, IndexError):
                            logging.error(f"Не вдалося розпарсити відповідь від API: {response_body}")
                            return None, False
            except error.HTTPError as e:
                logging.error(f"HTTP помилка: {e.code} {e.reason}")
                if e.code == 429:
                    self.key_manager.block_key(api_key)
                    self._handle_message("Перевищено ліміт запитів. Спроба використати наступний ключ...", 'warning')
                    continue
                elif e.code >= 500 and attempt < max_retries - 1:
                    logging.warning(f"Серверна помилка API. Повторна спроба через {retry_delay} сек...")
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(f"Отримано помилку від API: {e.read().decode()}")
                    return None, False
            except (error.URLError, ConnectionError) as e:
                logging.error(f"Помилка з'єднання: {e}")
                return None, False

        logging.error("Не вдалося виконати запит після вичерпання всіх спроб.")
        return None, False

    def run(self):
        """Головний метод, що запускає процес обробки всіх промтів."""
        logging.info("Початок обробки запитів.")
        if self.default_output_file and not self._check_output_file(self.default_output_file):
            return

        stop_processing = False
        for i, prompt_info in enumerate(self.prompts):
            if stop_processing: break
            logging.info(f"Обробка промту #{i+1}...")
            
            output_file = prompt_info.get('output_file', self.default_output_file)
            if not self._check_output_file(output_file): continue

            template = prompt_info.get('body')
            keywords_groups = prompt_info.get('keywords', [])
            placeholders = re.findall(r'\{.*?\}', template)
            model_id = prompt_info.get('model', self.DEFAULT_MODEL_ID)

            if not keywords_groups:
                if not placeholders:
                    self._handle_message("Ключові слова не надані, шаблон без плейсхолдерів. Виконується один запит.")
                    result, _ = self._make_api_request(template, model_id)
                    if result: self._write_result(output_file, result)
                else:
                    self._handle_message(f"Промт #{i+1} має плейсхолдери, але список ключів порожній.", 'warning')
                    if self.warnings_mode == 'strict':
                        logging.error("Режим 'strict': робота припинена.")
                        stop_processing = True
            
            for j, keywords in enumerate(keywords_groups):
                logging.info(f"Генерація запиту для групи ключових слів #{j+1}...")
                if len(placeholders) != len(keywords):
                    self._handle_message(f"Кількість плейсхолдерів ({len(placeholders)}) не співпадає з ключами ({len(keywords)}) в групі #{j+1}. Пропуск.", 'warning')
                    if self.warnings_mode == 'strict':
                        logging.error("Режим 'strict': робота припинена.")
                        stop_processing = True
                        break
                    continue
                
                try:
                    final_prompt = template.format(*keywords)
                except (IndexError, KeyError) as e:
                    logging.error(f"Помилка форматування шаблону: {e}. Пропуск.")
                    continue
                
                result, _ = self._make_api_request(final_prompt, model_id)
                if result: self._write_result(output_file, result)
        
        logging.info("Обробку всіх промтів завершено.")


def process_gemini_requests(request_json: str):
    """
    Головна функція для запуску процесу.
    Приймає JSON-рядок як аргумент.
    """
    try:
        data = json.loads(request_json)
        processor = GeminiProcessor(data)
        processor.run()
    except json.JSONDecodeError as e:
        logging.error(f"Помилка парсингу вхідного JSON: {e}")
    except (ValueError, KeyError) as e:
        logging.error(f"Помилка у структурі вхідних даних: {e}")
    except Exception as e:
        logging.critical(f"Сталася неочікувана помилка: {e}", exc_info=True)
