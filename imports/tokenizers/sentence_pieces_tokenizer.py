import pickle

from imports.tokenizers.base_tokenizer import BaseTokenizer

class SentencePiecesTokenizer(BaseTokenizer):

    def __init__(self, tokens: list[str] = None):
        super().__init__()
        self.trie = {}
        # Додаємо службові токени з батьківського класу
        special_tokens = [
            self.UNK_TOKEN,
            self.EOS_TOKEN,
            self.PAD_TOKEN,
            self.SYS_TOKEN,
            self.RAG_TOKEN,
            self.USR_TOKEN,
            self.AST_TOKEN,
            '[LWR]',
            '[CAP]',
            '[CPS]'
        ]
        for token in special_tokens:
            self.add_token(token)
        # Додаємо додаткові токени, якщо вони передані
        if tokens:
            for token in tokens:
                self.add_token(token)
    
    def vocab_size(self) -> int:
        """
        Повертає кількість токенів в словнику.

        Returns:
            int: Кілткість токенів в словнику.
        """
        return len(self.vocab)

    def add_token(self, token: str):
        """
        Додає токен до словника, якщо його ще немає, і оновлює trie.
        """
        if token not in self.vocab:
            self.vocab[token] = self.next_token_id
            self.decodes[self.next_token_id] = token
            self.next_token_id += 1
            self._add_token_to_trie(token)
    
    def _add_token_to_trie(self, token: str):
        node = self.trie
        for char in token:
            node = node.setdefault(char, {})
        node['__token__'] = token

    def encode(self, text: str, add_eos: bool = False, pad_to_length: int = None) -> list[int]:
        """
        Кодує рядок у список токенів, використовуючи trie для пошуку.
        Слова маркуються токенами [LWR], [CAP], [CPS] залежно від регістру.
        """
        import re
        tokens = []
        # Розбиваємо текст на слова та не-слова
        parts = re.findall(r'\w+|\W+', text)
        for part in parts:
            if part.isalpha():
                if part.islower():
                    pass #tokens.append(self.vocab['[LWR]'])
                elif part.istitle():
                    tokens.append(self.vocab['[CAP]'])
                elif part.isupper():
                    tokens.append(self.vocab['[CPS]'])
                else:
                    pass #tokens.append(self.vocab['[LWR]']) # fallback
                word = part.lower()
                i = 0
                while i < len(word):
                    node = self.trie
                    match = None
                    match_len = 0
                    for j in range(i, len(word)):
                        c = word[j]
                        if c in node:
                            node = node[c]
                            if '__token__' in node:
                                match = node['__token__']
                                match_len = j - i + 1
                        else:
                            break
                    if match:
                        tokens.append(self.vocab[match])
                        i += match_len
                    else:
                        tokens.append(self.vocab[self.UNK_TOKEN])
                        i += 1
            else:
                # Не-слова кодуємо як є
                i = 0
                while i < len(part):
                    node = self.trie
                    match = None
                    match_len = 0
                    for j in range(i, len(part)):
                        c = part[j]
                        if c in node:
                            node = node[c]
                            if '__token__' in node:
                                match = node['__token__']
                                match_len = j - i + 1
                        else:
                            break
                    if match:
                        tokens.append(self.vocab[match])
                        i += match_len
                    else:
                        tokens.append(self.vocab[self.UNK_TOKEN])
                        i += 1
        if add_eos:
            tokens.append(self.vocab[self.EOS_TOKEN])
        if pad_to_length is not None:
            while len(tokens) < pad_to_length:
                tokens.append(self.vocab[self.PAD_TOKEN])
        return tokens
    
    def decode(self, tokens: list[int], ignore_pad: bool = True) -> str:
        """
        Декодує список токенів назад у рядок, враховуючи службові токени для регістру слів.
        """
        result = []
        mode = None
        word_buffer = []
        for token_id in tokens:
            token = self.decodes.get(token_id, self.UNK_TOKEN)
            if ignore_pad and token == self.PAD_TOKEN:
                continue
            if token in ['[LWR]', '[CAP]', '[CPS]']:
                # Якщо є слово в буфері, додаємо його
                if word_buffer:
                    word = ''.join(word_buffer)
                    if mode == '[LWR]':
                        result.append(word.lower())
                    elif mode == '[CAP]':
                        result.append(word.capitalize())
                    elif mode == '[CPS]':
                        result.append(word.upper())
                    else:
                        result.append(word)
                    word_buffer = []
                mode = token
            elif token in [self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN, self.SYS_TOKEN, self.RAG_TOKEN, self.USR_TOKEN, self.AST_TOKEN]:
                # Додаємо службові токени як є
                if word_buffer:
                    word = ''.join(word_buffer)
                    if mode == '[LWR]':
                        result.append(word.lower())
                    elif mode == '[CAP]':
                        result.append(word.capitalize())
                    elif mode == '[CPS]':
                        result.append(word.upper())
                    else:
                        result.append(word)
                    word_buffer = []
                result.append(token)
                mode = None
            elif token.isalpha():
                word_buffer.append(token)
            else:
                # Якщо є слово в буфері, додаємо його
                if word_buffer:
                    word = ''.join(word_buffer)
                    if mode == '[LWR]':
                        result.append(word.lower())
                    elif mode == '[CAP]':
                        result.append(word.capitalize())
                    elif mode == '[CPS]':
                        result.append(word.upper())
                    else:
                        result.append(word)
                    word_buffer = []
                result.append(token)
                mode = None
        # Додаємо останнє слово з буфера
        if word_buffer:
            word = ''.join(word_buffer)
            if mode == '[LWR]':
                result.append(word.lower())
            elif mode == '[CAP]':
                result.append(word.capitalize())
            elif mode == '[CPS]':
                result.append(word.upper())
            else:
                result.append(word)
        return ''.join(result)
    
    def save(self, filename: str):
        """
        Зберігає токенізатор у файл через pickle. Trie не зберігається.
        """
        
        data = {
            'vocab': self.vocab,
            'decodes': self.decodes,
            'next_token_id': self.next_token_id
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filename: str):
        """
        Завантажує токенізатор з файлу через pickle. Trie будується заново.
        """
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        obj = cls()
        obj.vocab = data['vocab']
        obj.decodes = data['decodes']
        obj.next_token_id = data['next_token_id']
        # Відновлюємо trie
        obj.trie = {}
        for token in obj.vocab:
            obj._add_token_to_trie(token)
        return obj