from simple_bpe import CharacterTokenizer
import os

def process_tokenizer_learning(
    dataset_path: str,
    output_path: str,
    alphabet: str = None,
    load_path: str = None,
    min_merge_frequency: int = 80,
    max_merges: int = None,
    verbose: bool = False
) -> None:
    """
    Навчає токенізатор Byte Pair Encoding (BPE).

    Args:
        dataset_path (str): Шлях до файлу датасету для навчання токенізатора.
        output_path (str): Шлях, куди буде збережено навчений токенізатор.
        alphabet (str, optional): Початковий алфавіт для токенізатора. За замовчуванням None.
        load_path (str, optional): Шлях до існуючого токенізатора для завантаження.
                                  Якщо вказано, токенізатор буде завантажений замість навчання.
                                  За замовчуванням None.
        min_merge_frequency (int, optional): Мінімальна частота об'єднання для BPE.
                                            За замовчуванням 80.
        max_merges (int, optional): Максимальна кількість об'єднань для BPE. За замовчуванням None.
        verbose (bool, optional): Якщо True, виводить детальну інформацію під час навчання.
                                  За замовчуванням False.

    Returns:
        None
    """

    # Перевірка наявності шляхів
    if not dataset_path:
        print("Помилка: Не вказано шлях до файлу датасету (dataset_path).")
        return
    if not output_path:
        print("Помилка: Не вказано шлях для збереження токенізатора (output_path).")
        return

    tokenizer = None
    if load_path:
        if not os.path.exists(load_path):
            print(f"Попередження: Файл токенізатора за шляхом '{load_path}' не знайдено. Буде ініціалізовано новий токенізатор.")
        else:
            try:
                tokenizer = CharacterTokenizer.load(load_path)
                print(f"Завантажено токенізатор на {tokenizer.vocab_size} слів з '{load_path}'.")
            except FileNotFoundError:
                print(f"Помилка: Файл токенізатора за шляхом '{load_path}' не знайдено.")
                return
            except Exception as e:
                print(f"Помилка під час завантаження токенізатора з '{load_path}': {e}")
                return

    if tokenizer is None:  # Ініціалізуємо новий токенізатор, якщо не завантажено або load_path не вказано
        if alphabet is None:
            print("Помилка: Не вказано шлях для алфавіт для створення токенізатора (alphabet).")
        tokenizer = CharacterTokenizer(
            text=None,
            max_merges=max_merges,
            min_merge_frequency=min_merge_frequency,
            initial_alphabet_str=alphabet
        )
        print("Ініціалізовано новий токенізатор.")

    try:
        with open(dataset_path, encoding="utf-8") as f:
            data = f.read()
        if not data:
            print(f"Попередження: Файл датасету '{dataset_path}' порожній.")
            return
    except FileNotFoundError:
        print(f"Помилка: Файл датасету за шляхом '{dataset_path}' не знайдено.")
        return
    except IOError as e:
        print(f"Помилка вводу/виводу під час завантаження датасету '{dataset_path}': {e}")
        return
    except Exception as e:
        print(f"Невідома помилка під час завантаження датасету '{dataset_path}': {e}")
        return

    tokenizer.verbose = verbose
    print(f"Починаємо навчання токенізатора на {len(data)} символах...")
    tokenizer.train(data)

    print(f"\nРозмір словника після навчання: {tokenizer.vocab_size}")
    print(f"\nКількість навчених об'єднань (merges): {len(tokenizer.merges)}")

    # Створюємо директорію, якщо вона не існує
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Створено директорію для збереження: '{output_dir}'")
        except OSError as e:
            print(f"Помилка: Не вдалося створити директорію '{output_dir}': {e}")
            return

    try:
        tokenizer.save(output_path)
        print(f"Токенізатор успішно збережено до '{output_path}'.")
    except Exception as e:
        print(f"Помилка під час збереження токенізатора до '{output_path}': {e}")

