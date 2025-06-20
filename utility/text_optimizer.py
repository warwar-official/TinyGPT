import os

def process_text_file(input_filepath, output_filepath, replacements, allowed_alphabet, sort_lines=False, remove_dublicates=False):
    """
    Обробляє текстовий файл: видаляє дублікати рядків, замінює символи
    та видаляє символи, що не входять до дозволеного алфавіту.
    Сортування рядків є опціональним.

    Args:
        input_filepath (str): Шлях до вхідного текстового файлу.
        output_filepath (str): Шлях до вихідного текстового файлу.
        replacements (dict): Словник замін, де ключ - символ для заміни,
                             а значення - символ, на який потрібно замінити.
        allowed_alphabet (str): Рядок, що містить усі дозволені символи.
        sort_lines (bool, optional): Якщо True, рядки будуть відсортовані
                                     перед подальшою обробкою. За замовчуванням False.
    """
    try:
        # 1. Завантаження текстового файлу та видалення дублікатів рядків
        # Зберігаємо порядок рядків, використовуючи список та множину для відстеження унікальності
        ordered_unique_lines = []
        seen_lines = set()

        with open(input_filepath, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                stripped_line = line.strip() # Видаляємо пробіли на початку/в кінці рядка
                if remove_dublicates:
                    if stripped_line not in seen_lines:
                        ordered_unique_lines.append(stripped_line)
                        seen_lines.add(stripped_line)
                else:
                    ordered_unique_lines.append(stripped_line)

        # Опціональне сортування
        if sort_lines:
            processed_lines = sorted(ordered_unique_lines)
        else:
            processed_lines = ordered_unique_lines # Зберігаємо оригінальний порядок

        #processed_lines = [line for line in processed_lines if not line.startswith("http")]
        #processed_lines = [line for line in processed_lines if not "Я не володію достатньою інформацією для відповіді." in line]
        processed_lines = [line for line in processed_lines if not "http" in line]
        # 2. Заміна символів згідно словника замін та видалення недозволених символів
        final_processed_content = []
        for line in processed_lines:
            modified_line = line
            # Заміна символів
            for old_char, new_char in replacements.items():
                modified_line = modified_line.replace(old_char, new_char)
            
            if any(modified_line.startswith(str(i) + ".") for i in range(11)):
                modified_line = modified_line.strip()
                cut_mark = 0
                for i in range(len(modified_line)):
                    if modified_line[i] == ' ':
                        cut_mark = i
                        break
                modified_line = modified_line[cut_mark:].strip()
            # Видалення символів, що не містяться в дозволеному алфавіті
            cleaned_line = "".join(char for char in modified_line if char in allowed_alphabet)
            final_processed_content.append(cleaned_line)

        # 3. Збереження отриманого результату в файл
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            for line in final_processed_content:
                f_out.write(line + '\n') # Додаємо новий рядок після кожного обробленого рядка

        print(f"Файл успішно оброблено. Результат збережено у: {output_filepath}")

    except FileNotFoundError:
        print(f"Помилка: Файл '{input_filepath}' не знайдено.")
    except Exception as e:
        print(f"Виникла помилка під час обробки файлу: {e}")

