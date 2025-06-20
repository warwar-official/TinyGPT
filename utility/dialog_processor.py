def _load_files(file_paths: list[str]) -> list[str]:
    result_lines = []
    for path in file_paths:
        lines = []
        try:
            with open(path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
            result_lines.extend(lines)
        except FileNotFoundError:
            print(f"Помилка: Файл '{path}' не знайдено.")
        except Exception as e:
            print(f"Виникла помилка під час читання файлу: {e}")
        print(f"Прочитано {len(lines)} з {path}.")
    print(f"Загалом прочитано {len(result_lines)}.")
    return result_lines

def _fix_raw_file(original_lines: list[str], marker: str = "---\n") -> list[str]:
    marked_lines = []
    cleared_lines = []
    final_lines = []
    lines_row = 0
    # Round 1. add markers
    for line in original_lines:
        if line.strip():
            lines_row += 1
            # additional markers
            if "Привіт" in line and "### Користувач:" in line:
                marked_lines.append(marker)    
            if "***" in line:
                marked_lines.append(marker)
            else:
                marked_lines.append(line)
            if "### ШІ: Будь ласка" in line:
                marked_lines.append(marker)
        else:
            if lines_row > 2:
                marked_lines.append(marker)
            else:
                marked_lines.append(line)
            lines_row = 0
    # Round 2. remove empty
    cleared_lines = [line for line in marked_lines if line.strip()]
    # Round 3. remove double markers
    for line in cleared_lines:
        if line == marker:
            lines_row += 1
        else:
            lines_row = 0
        if lines_row < 2:
            final_lines.append(line)
    return final_lines

def _lines2dialogues(lines: list[str], marker: str = "---\n", alert_lenght: int = 36, skip_too_long: bool = False) -> list[list[str]]:
    dialogues = []
    dialogue = []
    for line in lines:
        if line.strip():
            if line == marker:
                if len(dialogue) > alert_lenght:
                    if skip_too_long:
                        dialogue = []
                        continue
                dialogues.append(dialogue)
                dialogue = []
            else:
                dialogue.append(line)
    # add last dialogue
    if not dialogue == []:
        if len(dialogue) > alert_lenght:
            if not skip_too_long:
                dialogues.append(dialogue)
    return dialogues

def _dialogues2lines(dialogues: list[list[str]], marker: str = "---\n") -> list[str]:
    lines = []
    for dialogue in dialogues:
        for line in dialogue:
            lines.append(line)
        lines.append(marker)
    return lines

def _sort_dialogues(dialogues: list[list[str]], keywords: set) -> tuple[list[list[str]], list[list[str]]]:
    clear_dialogues = []
    marked_dialogues = []
    for dialogue in dialogues:
        is_marked = False
        for line in dialogue:
            is_marked = any(word in line for word in keywords)
            if is_marked:
                break
        if is_marked:
            marked_dialogues.append(dialogue)
        else:
            clear_dialogues.append(dialogue)
    return clear_dialogues, marked_dialogues
        

def process_text_corpus(input_files: list[str], output_file_main: str, output_file_val: str, sort_keywords: set, dialog_fix = False):
    lines = _load_files(input_files)

    if dialog_fix:
        lines = _fix_raw_file(lines)
        dialogues = _lines2dialogues(lines,skip_too_long=True)
        dialogues_main, dialogues_val = _sort_dialogues(dialogues, sort_keywords)
        lines_main = _dialogues2lines(dialogues_main)
        lines_val = _dialogues2lines(dialogues_val)
    else:
        dialogues = _lines2dialogues(lines,skip_too_long=False)
        dialogues_main, dialogues_val = _sort_dialogues(dialogues, sort_keywords)
        lines_main = _dialogues2lines(dialogues_main)
        lines_val = _dialogues2lines(dialogues_val)

    try:
        with open(output_file_main, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines_main)
    except Exception as e:
        print(f"Виникла помилка під час запису файлу: {e}")
    try:
        with open(output_file_val, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines_val)
    except Exception as e:
        print(f"Виникла помилка під час запису файлу: {e}")

"""
if __name__ == "__main__":
    input_files = ["./data/countries-data-2-0.txt","./data/countries-data-2-1.txt","./data/countries-data-2-2.txt","./data/countries-data-2-3.txt","./data/countries-data-2-4.txt","./data/countries-data-2-5.txt","./data/countries-data-2-6.txt"] 
    output_file_main = "./data/fixed_countries-data-chat.txt"
    output_file_val = "./data/fixed_countries-data-chat-val.txt"

    lines = load_files(input_files)
    
    lines = fix_raw_file(lines)
    dialogues = lines2dialogues(lines,skip_too_long=True)
    dialogues_main, dialogues_val = sort_dialogues(dialogues, {"Кені", "Тайван", "Бразилі", "Аргентин"})
    lines_main = dialogues2lines(dialogues_main)
    lines_val = dialogues2lines(dialogues_val)

    try:
        with open(output_file_main, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines_main)
    except Exception as e:
        print(f"Виникла помилка під час запису файлу: {e}")
    try:
        with open(output_file_val, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines_val)
    except Exception as e:
        print(f"Виникла помилка під час запису файлу: {e}")"""