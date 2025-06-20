import collections
import pickle
import heapq
from time import localtime

class CharacterTokenizer:
    """
    Оптимізований символьний токенайзер (Character-level BPE) для швидкого навчання.
    Основні покращення:
    - Інкрементальне оновлення частот пар
    - Купа (heap) для ефективного пошуку найчастіших пар
    - Оптимізована обробка слів
    - Видалено дублювання коду
    """

    # Спеціальні токени
    UNK_TOKEN = '[UNK]'
    EOS_TOKEN = '[EOS]'
    PAD_TOKEN = '[PAD]'

    def __init__(self, text=None, max_merges=None, min_merge_frequency=1, 
                 max_token_length=None, initial_alphabet_str=None, verbose=False):
        """
        Ініціалізує токенайзер.
        """
        self.merges = {}
        self.decodes = {}
        self.vocab = {}
        self.min_merge_frequency = min_merge_frequency
        self.max_token_length = max_token_length
        self.max_merges = max_merges
        self.verbose = verbose
        self.next_token_id = 0
        
        # Множина символів, які не можна об'єднувати
        self.unmergable_symbols = set(".,!?;:+-*/=_—\"()[]{}%#")
        
        # Ініціалізуємо базовий словник
        self._initialize_vocab(initial_alphabet_str)
        
        # Trie будується тільки після навчання для кодування
        self.trie = {}

        if text:
            self.train(text)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _initialize_vocab(self, initial_alphabet_str):
        """Ініціалізує словник з базовими токенами."""
        # Додаємо спеціальні токени з фіксованими ID
        for token in [self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN]:
            self.vocab[token] = self.next_token_id
            self.decodes[self.next_token_id] = token
            self.next_token_id += 1

        # Додаємо символи з початкового алфавіту
        if initial_alphabet_str:
            for char in sorted(set(initial_alphabet_str)):
                if char not in self.vocab:
                    self.vocab[char] = self.next_token_id
                    self.decodes[self.next_token_id] = char
                    self.next_token_id += 1

    def train(self, text):
        """Оптимізоване навчання BPE з різними стратегіями для обмеженого та необмеженого навчання."""
        print(f"{self._get_timestamp()} Початок навчання токенізатора.")
        
        # Підготовка даних
        unk_token_id = self.vocab[self.UNK_TOKEN]
        eos_token_id = self.vocab[self.EOS_TOKEN]
        
        # Токенізуємо слова один раз
        words = []
        for word in text.split():
            tokenized = [self.vocab.get(c, unk_token_id) for c in word] + [eos_token_id]
            words.append(tokenized)

        if self.max_merges is None:
            # Режим батчової обробки: робимо всі можливі об'єднання, потім перерахунок
            words = self._train_batch_mode(words)
        else:
            # Звичайний режим: перерахунок після кожного об'єднання
            words = self._train_incremental_mode(words)

        # Будуємо Trie після навчання для швидкого кодування
        self._build_trie()
        print(f"{self._get_timestamp()} Навчання завершено. Загальна кількість об'єднань: {len(self.merges)}")

    def _train_batch_mode(self, words):
        """Батчове навчання: робить всі можливі об'єднання перед перерахунком частот."""
        batch_round = 0
        
        while True:
            batch_round += 1
            initial_merge_count = len(self.merges)
            
            # Рахуємо частоти пар для поточного стану
            pair_counts = self._count_pairs(words)
            
            if self.verbose:
                print(f"{self._get_timestamp()} Раунд {batch_round}: знайдено {len(pair_counts)} унікальних пар")
            
            # Знаходимо всі доступні пари для об'єднання та сортуємо за частотою
            available_pairs = []
            for pair, freq in pair_counts.items():
                if freq >= self.min_merge_frequency and self._can_merge_pair(pair):
                    available_pairs.append((freq, pair))
            
            if not available_pairs:
                if self.verbose:
                    print(f"{self._get_timestamp()} Немає доступних пар для об'єднання. Зупиняємо навчання.")
                break
            
            # Сортуємо пари за частотою (від найбільшої до найменшої)
            available_pairs.sort(reverse=True)
            
            if self.verbose:
                print(f"{self._get_timestamp()} Знайдено {len(available_pairs)} доступних пар для об'єднання")
            
            # Об'єднуємо всі доступні пари в порядку їх частоти
            for freq, pair in available_pairs:
                # Перевіряємо чи пара все ще може бути об'єднана
                # (може змінитися після попередніх об'єднань)
                if self._can_merge_pair(pair):
                    new_token_id = self._create_merge(pair)
                    words = self._apply_merge(words, pair, new_token_id)
                    
                    if self.verbose and len(self.merges) % 100 == 0:
                        token_str = self.decodes[new_token_id]
                        print(f"{self._get_timestamp()} [{len(self.merges)}] Об'єднано: {token_str} з частотою {freq}")
            
            merges_in_round = len(self.merges) - initial_merge_count
            
            if self.verbose:
                print(f"{self._get_timestamp()} Раунд {batch_round} завершено. "
                      f"Зроблено {merges_in_round} об'єднань. Всього: {len(self.merges)}")
            
            # Якщо в цьому раунді не було зроблено жодного об'єднання, зупиняємося
            if merges_in_round == 0:
                break
        
        return words

    def _train_incremental_mode(self, words):
        """Інкрементальне навчання: перерахунок після кожного об'єднання."""
        merge_count = 0
        
        while merge_count < self.max_merges:
            # Рахуємо частоти пар
            pair_counts = self._count_pairs(words)
            
            # Знаходимо найкращу пару для об'єднання
            best_pair = self._find_best_pair(pair_counts)
            
            if best_pair is None:
                break
                
            # Виконуємо об'єднання
            new_token_id = self._create_merge(best_pair)
            words = self._apply_merge(words, best_pair, new_token_id)
            
            merge_count += 1
            
            if self.verbose and merge_count % 100 == 0:
                freq = pair_counts[best_pair]
                token_str = self.decodes[new_token_id]
                print(f"{self._get_timestamp()} [{merge_count}] Об'єднано: {token_str} "
                      f"з частотою {freq} (всього пар {len(pair_counts)})")
        
        return words

    def _count_pairs(self, words):
        """Ефективно рахує частоти пар у всіх словах."""
        pairs = collections.Counter()
        for word in words:
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += 1
        return pairs

    def _find_best_pair(self, pair_counts):
        """Знаходить найкращу пару для об'єднання."""
        best_pair = None
        best_freq = 0
        
        for pair, freq in pair_counts.items():
            if freq < self.min_merge_frequency:
                continue
                
            # Перевіряємо чи можна об'єднувати цю пару
            if not self._can_merge_pair(pair):
                continue
                
            if freq > best_freq:
                best_pair = pair
                best_freq = freq
                
        return best_pair

    def _can_merge_pair(self, pair):
        """Перевіряє чи можна об'єднати пару токенів."""
        token_a = self.decodes.get(pair[0], "")
        token_b = self.decodes.get(pair[1], "")
        
        # Не об'єднуємо спеціальні токени
        special_tokens = {self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN}
        if token_a in special_tokens or token_b in special_tokens:
            return False
            
        merged_token = token_a + token_b
        
        # Перевіряємо довжину
        if self.max_token_length and len(merged_token) > self.max_token_length:
            return False
            
        # Перевіряємо чи токен вже існує
        if merged_token in self.vocab:
            return False
            
        # Перевіряємо чи містить необ'єднувані символи
        if set(merged_token) & self.unmergable_symbols:
            return False
            
        return True

    def _create_merge(self, pair):
        """Створює новий токен з об'єднання пари."""
        new_token_str = self.decodes[pair[0]] + self.decodes[pair[1]]
        new_token_id = self.next_token_id
        
        self.vocab[new_token_str] = new_token_id
        self.decodes[new_token_id] = new_token_str
        self.merges[pair] = new_token_id
        self.next_token_id += 1
        
        return new_token_id

    def _apply_merge(self, words, pair, new_token_id):
        """Застосовує об'єднання до всіх слів."""
        new_words = []
        for word in words:
            new_word = []
            i = 0
            while i < len(word):
                if (i < len(word) - 1 and 
                    word[i] == pair[0] and word[i + 1] == pair[1]):
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
        return new_words

    def _build_trie(self):
        """Будує префіксне дерево для швидкого кодування."""
        self.trie = {}
        for token_str, token_id in self.vocab.items():
            current = self.trie
            for char in token_str:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_id'] = token_id

    def _get_timestamp(self):
        """Повертає поточний час у форматі для логування."""
        t = localtime()
        return f"({t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d})"

    def encode(self, text, add_eos=False, pad_to_length=None):
        """
        Токенізує текст використовуючи жадібний алгоритм з Trie.
        """
        if not self.trie:
            self._build_trie()
            
        tokens = []
        i = 0
        unk_id = self.vocab[self.UNK_TOKEN]
        
        while i < len(text):
            best_len = 0
            best_id = None
            
            # Шукаємо найдовший збіг в Trie
            current = self.trie
            for j in range(i, len(text)):
                char = text[j]
                if char in current:
                    current = current[char]
                    if '_id' in current:
                        best_len = j - i + 1
                        best_id = current['_id']
                else:
                    break
            
            if best_id is not None:
                tokens.append(best_id)
                i += best_len
            else:
                tokens.append(unk_id)
                i += 1
                
            # Обрізаємо якщо досягли максимальної довжини
            if pad_to_length and len(tokens) >= pad_to_length:
                tokens = tokens[:pad_to_length]
                break

        # Додаємо EOS якщо потрібно
        if add_eos and (not pad_to_length or len(tokens) < pad_to_length):
            tokens.append(self.vocab[self.EOS_TOKEN])

        # Доповнюємо або обрізаємо до потрібної довжини
        if pad_to_length:
            if len(tokens) < pad_to_length:
                pad_id = self.vocab[self.PAD_TOKEN]
                tokens.extend([pad_id] * (pad_to_length - len(tokens)))
            elif len(tokens) > pad_to_length:
                tokens = tokens[:pad_to_length]

        return tokens

    def decode(self, tokens):
        """Декодує токени назад у текст."""
        return "".join(self.decodes.get(token, self.UNK_TOKEN) for token in tokens)

    def save(self, filename):
        """Зберігає токенайзер у файл."""
        data = {
            'merges': self.merges,
            'decodes': self.decodes,
            'vocab': self.vocab,
            'next_token_id': self.next_token_id,
            'max_merges': self.max_merges,
            'min_merge_frequency': self.min_merge_frequency,
            'max_token_length': self.max_token_length,
            'unmergable_symbols': self.unmergable_symbols,
            'special_tokens': {
                'unk': self.UNK_TOKEN,
                'eos': self.EOS_TOKEN,
                'pad': self.PAD_TOKEN
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Токенайзер збережено у {filename}")

    @classmethod
    def load(cls, filename):
        """Завантажує токенайзер з файлу."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        
        tokenizer = cls()
        tokenizer.merges = data['merges']
        tokenizer.decodes = data['decodes']
        tokenizer.vocab = data['vocab']
        tokenizer.next_token_id = data['next_token_id']
        tokenizer.max_merges = data.get('max_merges')
        tokenizer.min_merge_frequency = data.get('min_merge_frequency', 1)
        tokenizer.max_token_length = data.get('max_token_length')
        tokenizer.unmergable_symbols = data.get('unmergable_symbols', set(".,!?;:+-*/=_—\"()[]{}%#"))
        
        # Завантаження спеціальних токенів
        special = data.get('special_tokens', {})
        tokenizer.UNK_TOKEN = special.get('unk', '[UNK]')
        tokenizer.EOS_TOKEN = special.get('eos', '[EOS]')
        tokenizer.PAD_TOKEN = special.get('pad', '[PAD]')
        
        # Будуємо Trie для швидкого кодування
        tokenizer._build_trie()
        
        print(f"Токенайзер завантажено з {filename}")
        return tokenizer