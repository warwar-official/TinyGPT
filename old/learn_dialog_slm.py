import torch
import torch.nn.functional as F
from datasets import load_dataset # Новий імпорт для роботи з Hugging Face датасетами. Не тестував
import random # Новий імпорт для випадкового вибору рядків

from tinygpt import TinyGPT
from sentence_pieses_tokenizer import CharacterTokenizer # обидва токенізатора сумісні за структурою, тож тут підключати можна будь-який файл

LOAD = True # Завантажувати попередньо навчену модель чи починати з нуля

# --- Пристрій ---
# Визначаємо пристрій для навчання (GPU, якщо доступно, інакше CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Використовується пристрій: {device}")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Завантажуємо токенізатор
try:
    tokenizer = CharacterTokenizer.load("./models/tokenizer.pkl")
except FileNotFoundError:
    print("Помилка: Файл 'tokenizer.pkl' не знайдено. Будь ласка, переконайтеся, що токенізатор навчений та збережений.")
    exit()
except Exception as e:
    print(f"Помилка при завантаженні токенізатора: {e}")
    exit()


# Ініціалізація моделі
if LOAD:
    model = TinyGPT(vocab_size=tokenizer.vocab_size).to(device)
    try:
        model.load_state_dict(torch.load('models/model.pth', map_location=device))
        print("Модель успішно завантажено.")
    except FileNotFoundError:
        print("Попередження: Файл не знайдено. Модель буде ініціалізована з нуля.")
    except Exception as e:
        print(f"Помилка при завантаженні стану моделі: {e}. Модель буде ініціалізована з нуля.")
else:
    # Важливо: model.vocab_size має бути оновленим розміром словника токенізатора
    model = TinyGPT(vocab_size=tokenizer.vocab_size).to(device)
    print(f"Модель ініціалізована з нуля. Розмір словника моделі: {tokenizer.vocab_size}")

param_count = count_parameters(model)
print(f"Кількість параметрів: {param_count}")

# Оптимізатор та планувальник швидкості навчання
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=250)

# --- Нові параметри для завантаження даних та режиму семплювання ---
# Виберіть джерело даних: "file" для локального файлу або "huggingface" для датасету з Hugging Face
DATA_SOURCE = "file"
# Якщо DATA_SOURCE = "file", вкажіть шлях до вашого текстового файлу.
# Якщо DATA_SOURCE = "huggingface", вкажіть назву датасету з Hugging Face (наприклад, "wikitext", "imdb", "bookcorpus").
DATA_PATH = "data/main_dataset.txt"
VAL_DATA_PATH = "data/validation_dataset.txt"
# Виберіть режим семплювання:
# "fragments" - модель навчається на випадкових фрагментах тексту (як було раніше).
# "lines" - модель навчається на цілих рядках (кожен рядок - окрема послідовність).
# "dialogues" - модель навчається на повних діалогах (розділених порожніми рядками).
SAMPLE_MODE = "lines" # Змінено на "dialogues" за замовчуванням для демонстрації

batch_size = 1 # Розмір батчу
LEN = 768 # Максимальна довжина послідовності для навчання (контекстне вікно моделі)
MAX_EPOCH = 15000
ACCUMULATE = 16 # Накопичення градієнтів
REPORT_RATE = 100 * ACCUMULATE
SAVE_RATE = 100 * ACCUMULATE

def load_and_tokenize_data(source, path, tokenizer, seq_len, sample_mode):
    """
    Завантажує та токенізує дані з файлу або Hugging Face відповідно до режиму семплювання.
    Оптимізовано для пам'яті: завантажує лише необхідні структури даних.
    Додає [EOS] токен (як ID) після кожної репліки ШІ в режимі "dialogues"
    та в кінці кожної послідовності в режимах "fragments" та "lines".

    Args:
        source (str): Джерело даних ("file" або "huggingface").
        path (str): Шлях до файлу або назва датасету Hugging Face.
        tokenizer: Об'єкт токенізатора (CharacterTokenizer).
        seq_len (int): Максимальна довжина послідовності для обрізки/доповнення.
        sample_mode (str): Режим семплювання ("fragments", "lines", "dialogues").

    Returns:
        tuple: (data_tensor_fragments, encoded_lines_list, encoded_dialogues_list)
            data_tensor_fragments (torch.Tensor): Плоский тензор всіх токенів, для "fragments" режиму.
            encoded_lines_list (list): Список токенізованих окремих рядків, для "lines" режиму.
            encoded_dialogues_list (list): Список токенізованих повних діалогів, для "dialogues" режиму.
            Лише один з цих елементів буде заповнений, інші будуть порожніми.
    """
    # Ініціалізуємо всі вихідні дані як порожні
    data_tensor_fragments = torch.tensor([], dtype=torch.long).to(device)
    encoded_lines_list = []
    encoded_dialogues_list = []

    raw_content_from_file = ""
    raw_content_from_hf = []

    if source == "file":
        try:
            with open(path, 'r', encoding="utf-8") as f:
                raw_content_from_file = f.read()
            print(f"Дані успішно завантажено з файлу: {path}.")
        except FileNotFoundError:
            print(f"Помилка: Файл '{path}' не знайдено. Перевірте шлях.")
            exit()
        except Exception as e:
            print(f"Помилка при читанні файлу '{path}': {e}")
            exit()
    elif source == "huggingface":
        try:
            print(f"Завантаження датасету з Hugging Face: {path}...")
            dataset = load_dataset(path, split='train')
            raw_content_from_hf = [item['text'].strip() for item in dataset if 'text' in item and item['text'].strip()]
            print(f"Дані успішно завантажено з Hugging Face: {path}. Кількість сирих елементів: {len(raw_content_from_hf)}")
            if not raw_content_from_hf:
                print("Попередження: Завантажений датасет з Hugging Face не містить текстових даних або всі елементи порожні.")
        except Exception as e:
            print(f"Помилка при завантаженні датасету з Hugging Face '{path}': {e}")
            print("Перевірте назву датасету або ваше інтернет-з'єднання.")
            exit()
    else:
        print(f"Невідоме джерело даних: {source}. Виберіть 'file' або 'huggingface'.")
        exit()

    if sample_mode == "fragments":
        # Режим "fragments": токенізуємо весь текст як один потік
        content_to_tokenize = raw_content_from_file if source == "file" else " ".join(raw_content_from_hf)
        
        if not content_to_tokenize:
            print("Попередження: Немає сирого тексту для режиму 'fragments'.")
            return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list

        print("Токенізація в режимі 'fragments'...")
        # Токенізуємо вміст
        flattened_tokens = tokenizer.encode(content_to_tokenize, add_eos=True)
        
        if isinstance(flattened_tokens, list) and all(isinstance(t, int) for t in flattened_tokens):
            if flattened_tokens:
                data_tensor_fragments = torch.tensor(flattened_tokens, dtype=torch.long).to(device)
            else:
                print("Попередження: Токенізований текст для 'fragments' порожній. Пропущено.")
        else:
            print(f"Попередження: Токенізатор повернув неочікуваний тип даних для 'fragments'. Пропущено. Тип: {type(flattened_tokens)}")
        
        return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list

    elif sample_mode == "lines":
        # Режим "lines": обробляємо кожен рядок як окрему послідовність
        lines = raw_content_from_file.split('\n') if source == "file" else raw_content_from_hf
        lines = [line.strip() for line in lines if line.strip()] # Фільтруємо порожні
        
        if not lines:
            print("Попередження: Немає валідних рядків для режиму 'lines'.")
            return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list

        print(f"Токенізація {len(lines)} рядків в режимі 'lines'...")
        for i, line in enumerate(lines):
            if (i + 1) % 1000 == 0:
                print(f"  Оброблено {i+1}/{len(lines)} рядків.")
            
            # Токенізуємо рядок
            encoded_line = tokenizer.encode(line, add_eos=True)#, pad_to_length=seq_len)
            
            if isinstance(encoded_line, list) and all(isinstance(t, int) for t in encoded_line):
                if encoded_line:
                    encoded_lines_list.append(encoded_line)
                else:
                    print(f"Попередження: Токенізований рядок порожній: '{line[:50]}...'. Пропущено.")
            else:
                print(f"Попередження: Токенізатор повернув неочікуваний тип даних для рядка. Пропущено. Сирий рядок: '{line[:50]}...'")
        
        return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list

    elif sample_mode == "dialogues":
        # Режим "dialogues": обробляємо діалоги, розділені \n\n
        raw_dialogues = raw_content_from_file.split('\n\n') if source == "file" else raw_content_from_hf
        raw_dialogues = [d.strip() for d in raw_dialogues if d.strip()] # Фільтруємо порожні

        if not raw_dialogues:
            print("Попередження: Немає валідних діалогів для режиму 'dialogues'.")
            return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list

        print(f"Токенізація {len(raw_dialogues)} діалогів в режимі 'dialogues'...")
        for i, raw_dialogue in enumerate(raw_dialogues):
            if (i + 1) % 100 == 0:
                print(f"  Оброблено {i+1}/{len(raw_dialogues)} діалогів.")

            dialogue_lines = raw_dialogue.split('\n')
            processed_dialogue_tokens_list = []
            for line in dialogue_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Токенізуємо поточний рядок
                if line.startswith("### ШІ:") or line.startswith("### AI:"):
                    current_line_tokens = tokenizer.encode(line, add_eos=True)
                else:
                    current_line_tokens = tokenizer.encode(line, add_eos=False)
                
                if not (isinstance(current_line_tokens, list) and all(isinstance(t, int) for t in current_line_tokens)):
                    print(f"Попередження: Токенізатор повернув неочікуваний тип даних для частини діалогу: '{line[:50]}...'. Пропущено цю частину.")
                    continue
                
                processed_dialogue_tokens_list.extend(current_line_tokens) # Додаємо токени до загального списку токенів діалогу

            # encoded_dialogue_tokens тепер є об'єднаним списком ID токенів для всього діалогу
            encoded_dialogue_tokens = processed_dialogue_tokens_list
            
            if encoded_dialogue_tokens:
                encoded_dialogues_list.append(encoded_dialogue_tokens)
            else:
                print(f"Попередження: Токенізований діалог порожній після обробки: '{raw_dialogue[:50]}...'. Пропущено.")
        
        return data_tensor_fragments, encoded_lines_list, encoded_dialogues_list
    else:
        print(f"Невідомий режим семплювання: {sample_mode}. Виберіть 'fragments', 'lines' або 'dialogues'.")
        exit()


def prepaire_forward_data(fragments_data, lines_data, dialogues_data):
    seq_len = LEN # Довжина послідовності для кожного батчу (контекстне вікно моделі)
    x = None # Вхідні дані для моделі
    y = None # Цільові дані для моделі

    if SAMPLE_MODE == "fragments":
        # Режим "fragments": вибираємо випадкові фрагменти з плоского тензора даних
        if len(fragments_data) < seq_len + 1:
            print(f"Помилка: Недостатньо даних ({len(fragments_data)} токенів) для семплювання фрагментів довжиною {seq_len}.")
            exit(0)
        # Вибираємо випадкові індекси початку фрагментів
        idx = torch.randint(0, len(fragments_data) - seq_len - 1, (batch_size,))
        # Формуємо батчі x (вхід) та y (ціль)
        x = torch.stack([fragments_data[i:i+seq_len] for i in idx])
        y = torch.stack([fragments_data[i+1:i+seq_len+1] for i in idx])
    elif SAMPLE_MODE == "lines":
        # Режим "lines": вибираємо випадкові окремі рядки
        if not lines_data:
            print("Попередження: Немає токенізованих рядків для семплювання в режимі 'lines'. Пропущено епоху.")
            exit(0)

        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            random_line_tokens = random.choice(lines_data)

            seq_tensor = torch.tensor(random_line_tokens, dtype=torch.long).to(device)
            batch_x.append(seq_tensor[:-1])
            batch_y.append(seq_tensor[1:])
        
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
    elif SAMPLE_MODE == "dialogues":
        # Режим "dialogues": вибираємо випадкові повні діалоги
        if not dialogues_data:
            print("Попередження: Немає токенізованих діалогів для семплювання в режимі 'dialogues'. Пропущено епоху.")
            exit(0)

        batch_x = []
        batch_y = []
        for _ in range(batch_size):
            random_dialogue_tokens = random.choice(dialogues_data)

            current_dialogue_len = len(random_dialogue_tokens)
            
            if current_dialogue_len < seq_len + 1:
                padded_dialogue = random_dialogue_tokens + \
                                  [2] * (seq_len + 1 - current_dialogue_len)
            else:
                # Обрізаємо діалог, якщо він довший за `seq_len + 1`
                # Вибираємо випадковий фрагмент, щоб уникнути завжди обрізати початок
                start_idx = random.randint(0, current_dialogue_len - (seq_len + 1)) if current_dialogue_len > (seq_len + 1) else 0
                padded_dialogue = random_dialogue_tokens[start_idx : start_idx + seq_len + 1]
                
                #if current_dialogue_len > seq_len + 1:
                    #print(f"Попередження: Діалог був обрізаний з довжини {current_dialogue_len} до {seq_len + 1} токенів.")

            dialogue_tensor = torch.tensor(padded_dialogue, dtype=torch.long).to(device)
            batch_x.append(dialogue_tensor[:-1])
            batch_y.append(dialogue_tensor[1:])

        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
    else:
        print(f"Невідомий режим семплювання: {SAMPLE_MODE}. Виберіть 'fragments', 'lines' або 'dialogues'.")
        exit(0) # Завершуємо навчання, якщо режим невідомий

    return x, y

# Завантажуємо та токенізуємо дані відповідно до обраних параметрів
# Лише один з цих трьох списків/тензорів буде заповнений, інші будуть порожніми.
data_for_fragments, encoded_lines_for_lines, encoded_dialogues_for_lines = \
    load_and_tokenize_data(DATA_SOURCE, DATA_PATH, tokenizer, LEN, SAMPLE_MODE)
val_data_for_fragments, val_encoded_lines_for_lines, val_encoded_dialogues_for_lines = \
    load_and_tokenize_data(DATA_SOURCE, VAL_DATA_PATH, tokenizer, LEN, SAMPLE_MODE)

# Перевірка, чи є дані для навчання
if (SAMPLE_MODE == "fragments" and len(data_for_fragments) == 0) or \
   (SAMPLE_MODE == "lines" and len(encoded_lines_for_lines) == 0) or \
   (SAMPLE_MODE == "dialogues" and len(encoded_dialogues_for_lines) == 0):
    print("Немає даних для навчання в обраному режимі. Завершення програми.")
    exit()

# --- Тренування ---
print(f"Початок тренування в режимі семплювання: '{SAMPLE_MODE}'...")
for epoch in range(MAX_EPOCH * ACCUMULATE):

    #seq_len = LEN # Довжина послідовності для кожного батчу (контекстне вікно моделі)
    x = None # Вхідні дані для моделі
    y = None # Цільові дані для моделі

    u = None # Вхідні дані для моделі
    v = None # Цільові дані для моделі

    x, y = prepaire_forward_data(data_for_fragments, encoded_lines_for_lines, encoded_dialogues_for_lines)

    # Перевірка, чи були x та y успішно ініціалізовані
    if x is None or y is None:
        print("Помилка: x або y не були ініціалізовані. Пропущено епоху.")
        continue

    # Переміщуємо тензори на обраний пристрій (GPU/CPU)
    x, y = x.to(device), y.to(device)

    # Прямий прохід моделі
    logits = model(x)

    # Отримання карти уваги (якщо доступно). Це може бути корисно для аналізу.
    attn_map = None
    if hasattr(model.blocks[0], 'last_attn'): # Перевіряємо, чи існує 'last_attn'
        attn_map = model.blocks[0].last_attn  # Наприклад, attention першого блоку

    # Обчислення функції втрат (крос-ентропія)
    # Змінюємо форму logits та y для відповідності вимогам F.cross_entropy
    # logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
    # y: (batch_size, seq_len) -> (batch_size * seq_len)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    loss = loss / ACCUMULATE
    loss.backward() # Обчислюємо градієнти
    
    if epoch % ACCUMULATE == 0:
        # Зворотний прохід та оптимізація
        optimizer.step() # Оновлюємо ваги моделі
        scheduler.step(loss) # Оновлюємо швидкість навчання

        optimizer.zero_grad() # Обнуляємо градієнти

    if epoch % REPORT_RATE == 0:
        u, v = prepaire_forward_data(val_data_for_fragments, val_encoded_lines_for_lines, val_encoded_dialogues_for_lines)
        # Переміщуємо тензори на обраний пристрій (GPU/CPU)
        u, v = u.to(device), v.to(device)
        model.eval()
        logits = model(u)
        val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), v.view(-1))
        model.train()
        print(f"Епоха {(int)(epoch / ACCUMULATE)}, Втрати: {(loss.item() * ACCUMULATE):.4f}, Коефіцієнт навчання: {optimizer.param_groups[0]['lr']}, Втрати тестування: {val_loss.item():.4f}")
    if epoch % REPORT_RATE == 0 and not epoch == 0:
        torch.save(model.state_dict(), f"./models/as/ep-{(int)(epoch / ACCUMULATE)}.pth")

# Збереження навченої моделі після завершення всіх епох
torch.save(model.state_dict(), 'models/model.pth')
print("Навчання завершено. Модель збережено.")
