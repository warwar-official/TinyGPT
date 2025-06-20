import collections
import pickle

class CharacterTokenizer:
    """
    Простий символьний токенайзер (Character-level BPE) для тестування та аналізу.
    Він підтримує створення, доповнення, збереження/завантаження,
    налаштування стратегій об'єднання токенів (за частотою), відсікання рідкісних об'єднань,
    початковий алфавіт та обробку невідомих символів.
    """

    # Спеціальні токени
    UNK_TOKEN = '[UNK]'  # Токен для невідомих символів
    EOS_TOKEN = '[EOS]'  # Токен для кінця послідовності
    PAD_TOKEN = '[PAD]'  # Токен для доповнення послідовності

    def __init__(self, text=None, max_merges=None,
                 min_merge_frequency=1, initial_alphabet_str=None, verbose=False):
        """
        Ініціалізує токенайзер.

        :param text: Початковий текст для навчання токенайзера.
        :param max_merges: Максимальна кількість об'єднань.
        :param min_merge_frequency: Мінімальна частота об'єднання для додавання до словника.
                                    Об'єднання з меншою частотою будуть ігноруватися.
        :param initial_alphabet_str: Рядок, що містить початковий алфавіт символів.
                                     Ці символи (та UNK, EOS, PAD) будуть базовими токенами.
        """
        self.merges = {}  # Словник для зберігання об'єднань: (char1_id, char2_id) -> new_token_id
        self.decodes = {} # Словник для декодування: token_id -> str (символ або послідовність символів)
        self.vocab = {}   # Словник для кодування: str -> token_id (символ або послідовність символів)
        self.min_merge_frequency = min_merge_frequency
        # Стратегія об'єднання тепер фіксована на 'frequency'
        self.merge_strategy = 'frequency' 
        self.max_merges = max_merges
        self.next_token_id = 0 # Починаємо з 0, оскільки немає фіксованих "базових" ID як у байтів

        self.verbose = verbose

        self.trie = {} # Ініціалізація префіксного дерева (Trie) для швидкого кодування

        # Ініціалізуємо словник символами з початкового алфавіту та спеціальними токенами
        self._initialize_vocab(initial_alphabet_str)

        if text:
            self.train(text)

    @property
    def vocab_size(self):
        """
        Повертає розмір словника токенайзера.
        """
        return len(self.vocab)

    def _add_to_trie(self, token_str, token_id):
        """
        Додає токен до префіксного дерева (Trie).
        """
        current_node = self.trie
        for char in token_str:
            if char not in current_node:
                current_node[char] = {}
            current_node = current_node[char]
        current_node['_id'] = token_id # Позначаємо, що цей вузол відповідає повному токену

    def _initialize_vocab(self, initial_alphabet_str):
        """
        Ініціалізує словник з початковим алфавітом та спеціальними токенами (UNK, EOS, PAD).
        """
        # Додаємо UNK_TOKEN першим, щоб його ID був фіксованим
        self.vocab[self.UNK_TOKEN] = self.next_token_id
        self.decodes[self.next_token_id] = self.UNK_TOKEN
        self._add_to_trie(self.UNK_TOKEN, self.next_token_id)
        self.next_token_id += 1

        # Додаємо EOS_TOKEN другим, щоб його ID був фіксованим
        self.vocab[self.EOS_TOKEN] = self.next_token_id
        self.decodes[self.next_token_id] = self.EOS_TOKEN
        self._add_to_trie(self.EOS_TOKEN, self.next_token_id)
        self.next_token_id += 1

        # Додаємо PAD_TOKEN третім, щоб його ID був фіксованим
        self.vocab[self.PAD_TOKEN] = self.next_token_id
        self.decodes[self.next_token_id] = self.PAD_TOKEN
        self._add_to_trie(self.PAD_TOKEN, self.next_token_id)
        self.next_token_id += 1

        if initial_alphabet_str:
            # Додаємо унікальні символи з початкового алфавіту
            for char in sorted(set(initial_alphabet_str)): # Сортуємо для консистентності ID
                if char not in self.vocab:
                    self.vocab[char] = self.next_token_id
                    self.decodes[self.next_token_id] = char
                    self._add_to_trie(char, self.next_token_id) # Додаємо до Trie
                    self.next_token_id += 1

    def _get_char_pairs(self, tokens_ids):
        """
        Знаходить усі унікальні пари послідовних ID токенів у списку.
        """
        pairs = collections.defaultdict(int)
        for i in range(len(tokens_ids) - 1):
            pairs[(tokens_ids[i], tokens_ids[i+1])] += 1
        return pairs

    def _perform_merge(self, tokens_ids, pair, new_token_id):
        """
        Виконує об'єднання пари ID токенів у списку.
        """
        new_tokens = []
        i = 0
        while i < len(tokens_ids):
            if i + 1 < len(tokens_ids) and (tokens_ids[i], tokens_ids[i+1]) == pair:
                new_tokens.append(new_token_id)
                i += 2
            else:
                new_tokens.append(tokens_ids[i])
                i += 1
        return new_tokens

    def train(self, text):
        """
        Навчає токенайзер на заданому тексті, виконуючи об'єднання символів.
        """
        # Початкове представлення тексту як списку ID базових символів
        # Невідомі символи замінюємо на UNK_TOKEN ID
        initial_char_tokens_ids = []
        unk_token_id = self.vocab[self.UNK_TOKEN]
        for char in text:
            if char in self.vocab:
                initial_char_tokens_ids.append(self.vocab[char])
            else:
                print(f"Симовл {char} не зміг бути декодованим!")
                initial_char_tokens_ids.append(unk_token_id)
        
        current_tokens_ids = initial_char_tokens_ids[:]

        merge_count = 0
        while True:
            if self.max_merges is not None and merge_count >= self.max_merges:
                break

            pairs = self._get_char_pairs(current_tokens_ids)
            if not pairs:
                break # Немає більше пар для об'єднання

            eligible_pairs = []
            for pair, freq in pairs.items():
                # Не об'єднуємо спеціальні токени, якщо вони є частиною пари
                # Це важливе правило для багатьох токенізаторів, щоб спеціальні токени
                # залишалися окремими одиницями.
                if (self.decodes[pair[0]] in [self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN] or
                    self.decodes[pair[1]] in [self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]):
                    continue

                if freq >= self.min_merge_frequency:
                    eligible_pairs.append((pair, freq))

            if not eligible_pairs:
                break # Немає пар, що відповідають мінімальній частоті або всі спеціальні

            best_pair_to_merge = None
            max_current_freq = -1

            # Шукаємо НАЙКРАЩУ ПАРУ, яка ЩЕ НЕ БУЛА ОБ'ЄДНАНА
            for pair, freq in eligible_pairs:
                if pair not in self.merges and freq > max_current_freq:
                    max_current_freq = freq
                    best_pair_to_merge = pair
            
            if best_pair_to_merge is None:
                # Якщо ми пройшли всі eligible_pairs і не знайшли жодної НОВОЇ
                # пари для об'єднання, тоді немає сенсу продовжувати.
                break

            new_token_id = self.next_token_id

            self.merges[best_pair_to_merge] = new_token_id

            # Оновлюємо словники: vocab та decodes
            chars1 = self.decodes[best_pair_to_merge[0]]
            chars2 = self.decodes[best_pair_to_merge[1]]
            new_chars = chars1 + chars2

            # --- КРИТИЧНЕ МІСЦЕ ДЛЯ ДІАГНОСТИКИ ---
            # Перевіряємо, чи новий токен вже існує, хоча цього не повинно бути
            if new_chars in self.vocab:
                 # Якщо це трапляється, це помилка в логіці (pair not in self.merges)
                 # або несподівана поведінка. next_token_id буде "пропущений".
                 #if self.verbose:
                 #   print(f"ПОМИЛКА: Спроба додати '{new_chars}' (ID {new_token_id}), але він вже є (ID {self.vocab[new_chars]})! Пропуск.")
                 # Тут ви можете вирішити, що робити: зупинитись, не додавати, тощо.
                 # Для цілей відладки, це ключовий момент.
                 # Якщо ви хочете "перестрибнути" цей ID і не створювати прогалин,
                 # вам потрібно буде відкотити next_token_id, або переглянути логіку вибору best_pair_to_merge.
                 # Але за правильної логіки `pair not in self.merges` цього не повинно відбуватись.
                 # Для відладки, давайте все одно додамо, щоб побачити, як себе поведе.
                 # Проте, якщо `new_chars` вже в `vocab`, але з іншим ID, це створить перезапис.
                 # Якщо `new_chars` вже в `vocab` з тим самим ID, це буде просто переприсвоєння.
                 continue
            # --- КІНЕЦЬ КРИТИЧНОГО МІСЦЯ ---

            self.next_token_id += 1

            self.vocab[new_chars] = new_token_id
            self.decodes[new_token_id] = new_chars
            
            # Додаємо новий об'єднаний токен до Trie
            self._add_to_trie(new_chars, new_token_id)

            # Застосовуємо об'єднання до поточних токенів
            current_tokens_ids = self._perform_merge(current_tokens_ids, best_pair_to_merge, new_token_id)
            merge_count += 1

            if self.verbose:
                if merge_count % 100 == 0 and not merge_count == 0:
                    print(f"Виконано поєднань: {merge_count}, доступних пар: {len(eligible_pairs)}")

    def encode(self, text, add_eos=False, pad_to_length=None):
        """
        Токенізує рядок тексту, віддаючи перевагу довшим токенам,
        використовуючи Trie для оптимізації пошуку.
        Невідомі символи замінюються на UNK_TOKEN.

        :param text: Вхідний рядок тексту.
        :param add_eos: Додати токен кінця послідовності ([EOS]) в кінець.
        :param pad_to_length: Доповнити послідовність до цієї довжини за допомогою [PAD] токенів.
                              Якщо послідовність довша, вона буде обрізана.
        :return: Список цілих чисел, що представляють токени.
        """
        tokens_ids = []
        i = 0
        n_chars = len(text)
        unk_token_id = self.vocab[self.UNK_TOKEN]

        while i < n_chars:
            best_match_token_id = None
            best_match_len = 0
            
            current_node = self.trie
            current_path_len = 0 # Довжина поточного префікса в Trie
            temp_best_id = None # ID найдовшого валідного токена, знайденого на поточному шляху

            # Траверсуємо Trie, щоб знайти найдовший збіг
            for j in range(i, n_chars):
                char = text[j]
                if char in current_node:
                    current_node = current_node[char]
                    current_path_len += 1
                    if '_id' in current_node: # Якщо цей вузол представляє повний токен
                        temp_best_id = current_node['_id']
                        best_match_len = current_path_len # Оновлюємо довжину найкращого збігу
                else:
                    # Якщо символ не знайдено в поточному вузлі Trie,
                    # це означає, що подальших збігів за цим шляхом немає.
                    break
            
            # Після траверсування, якщо ми знайшли будь-який збіг, використовуємо найдовший
            if temp_best_id is not None:
                tokens_ids.append(temp_best_id)
                i += best_match_len
            else:
                # Якщо жодного збігу не знайдено (навіть для одного символу),
                # це означає, що поточний символ не був у словнику.
                # Замінюємо його на UNK_TOKEN.
                tokens_ids.append(unk_token_id)
                i += 1 # Просуваємося на 1 символ (UNK_TOKEN завжди 1 символ)
            
            # Якщо pad_to_length встановлено і ми вже досягли цієї довжини, обрізаємо
            if pad_to_length is not None and len(tokens_ids) >= pad_to_length:
                tokens_ids = tokens_ids[:pad_to_length]
                break


        if add_eos:
            # Додаємо EOS, якщо його ще немає і довжина дозволяє
            if not (pad_to_length is not None and len(tokens_ids) >= pad_to_length):
                tokens_ids.append(self.vocab[self.EOS_TOKEN])
            # Якщо додавання EOS перевищить pad_to_length, воно буде обрізано на наступному кроці.

        # Доповнення або обрізка до pad_to_length
        if pad_to_length is not None:
            if len(tokens_ids) < pad_to_length:
                pad_token_id = self.vocab[self.PAD_TOKEN]
                tokens_ids.extend([pad_token_id] * (pad_to_length - len(tokens_ids)))
            elif len(tokens_ids) > pad_to_length:
                tokens_ids = tokens_ids[:pad_to_length]

        return tokens_ids

    def decode(self, tokens_ids):
        """
        Декодує список токенів назад у рядок.

        :param tokens_ids: Список цілих чисел, що представляють токени.
        :return: Декодований рядок.
        """
        decoded_chars = []
        for token_id in tokens_ids:
            # Якщо зустрічаємо EOS_TOKEN або PAD_TOKEN під час декодування,
            # зазвичай припиняємо або просто додаємо його текстове представлення.
            # Для простоти, додаємо його текстове представлення.
            if token_id in self.decodes:
                decoded_chars.append(self.decodes[token_id])
            else:
                # Цього не повинно траплятись, якщо модель генерує валідні токени.
                # Якщо трапилось, можливо, словник завантажений неправильно або
                # модель генерує неіснуючі ID. Повертаємо UNK_TOKEN як запасний варіант.
                decoded_chars.append(self.UNK_TOKEN)
        return "".join(decoded_chars)

    def save(self, filename):
        """
        Зберігає стан токенайзера у файл.

        :param filename: Ім'я файлу для збереження.
        """
        with open(filename, 'wb') as f:
            pickle.dump({
                'merges': self.merges,
                'decodes': self.decodes,
                'vocab': self.vocab,
                'next_token_id': self.next_token_id,
                'max_merges': self.max_merges,
                'min_merge_frequency': self.min_merge_frequency,
                'unk_token': self.UNK_TOKEN,
                'eos_token': self.EOS_TOKEN,
                'pad_token': self.PAD_TOKEN, # Зберігаємо PAD_TOKEN
                'trie': self.trie # Зберігаємо Trie
            }, f)
        print(f"Токенайзер успішно збережено у {filename}")

    @classmethod
    def load(cls, filename):
        """
        Завантажує стан токенайзера з файлу.

        :param filename: Ім'я файлу для завантаження.
        :return: Об'єкт CharacterTokenizer.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls() 
        tokenizer.merges = data['merges']
        tokenizer.decodes = data['decodes']
        tokenizer.vocab = data['vocab']
        tokenizer.next_token_id = data['next_token_id']
        tokenizer.max_merges = data.get('max_merges', None)
        tokenizer.min_merge_frequency = data.get('min_merge_frequency', 1)
        tokenizer.UNK_TOKEN = data.get('unk_token', '[UNK]') 
        tokenizer.EOS_TOKEN = data.get('eos_token', '[EOS]')
        tokenizer.PAD_TOKEN = data.get('pad_token', '[PAD]') # Завантажуємо PAD_TOKEN
        tokenizer.trie = data.get('trie', {}) # Завантажуємо Trie
        
        # Якщо Trie не було збережено (для старих версій), можна його перебудувати
        if not tokenizer.trie and tokenizer.vocab:
            print("Trie не знайдено у файлі, перебудовую Trie...")
            for token_str, token_id in tokenizer.vocab.items():
                tokenizer._add_to_trie(token_str, token_id)

        print(f"Токенайзер успішно завантажено з {filename}")
        return tokenizer
