import pickle

from imports.tokenizers.base_tokenizer import BaseTokenizer


class CharLevelTokenizer(BaseTokenizer):
    """
    Простий символьний токенайзер без навчання.
    Підтримує спеціальні токени: [UNK], [EOS], [PAD] + [SYS], [CON], [USR], [AST]
    """

    SYS_TOKEN = "[SYS]"
    CON_TOKEN = "[CON]"
    USR_TOKEN = "[USR]"
    AST_TOKEN = "[AST]"

    def __init__(self, alphabet_str=None):
        super().__init__()
        self._initialize_vocab(alphabet_str)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _initialize_vocab(self, alphabet_str):
        # Додаємо спеціальні токени з фіксованими ID
        for token in [self.UNK_TOKEN, self.EOS_TOKEN, self.PAD_TOKEN, self.SYS_TOKEN, self.CON_TOKEN, self.USR_TOKEN, self.AST_TOKEN]:
            self.vocab[token] = self.next_token_id
            self.decodes[self.next_token_id] = token
            self.next_token_id += 1
        # Додаємо символи з алфавіту
        if alphabet_str:
            for char in sorted(set(alphabet_str)):
                if char not in self.vocab:
                    self.vocab[char] = self.next_token_id
                    self.decodes[self.next_token_id] = char
                    self.next_token_id += 1

    def encode(self, text, add_eos=False, pad_to_length=None):
        tokens = [self.vocab.get(c, self.vocab[self.UNK_TOKEN]) for c in text]
        if add_eos:
            tokens.append(self.vocab[self.EOS_TOKEN])
        if pad_to_length:
            if len(tokens) < pad_to_length:
                tokens.extend([self.vocab[self.PAD_TOKEN]] * (pad_to_length - len(tokens)))
            elif len(tokens) > pad_to_length:
                tokens = tokens[:pad_to_length]
        return tokens

    def decode(self, tokens, ignore_pad=False):
        pad_token_id = self.vocab[self.PAD_TOKEN]
        if ignore_pad:
            return "".join(self.decodes.get(token, self.UNK_TOKEN) for token in tokens if token != pad_token_id)
        else:
            return "".join(self.decodes.get(token, self.UNK_TOKEN) for token in tokens)

    def save(self, filename):
        data = {
            'vocab': self.vocab,
            'decodes': self.decodes,
            'next_token_id': self.next_token_id
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Токенайзер збережено у {filename}")

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        tokenizer = cls()
        tokenizer.vocab = data['vocab']
        tokenizer.decodes = data['decodes']
        tokenizer.next_token_id = data['next_token_id']
        print(f"Токенайзер завантажено з {filename}")
        return tokenizer

if __name__ == "__main__":
    # Приклад використання
    ukrainian_alphabet = "абвгґїдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    english_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    numbers_alphabet = "0123456789"
    symbol_alphabet = " .,!?;:+-*/^=_—'\"()[]{}%#<>\n"
    alphabet = ukrainian_alphabet + english_alphabet + numbers_alphabet + symbol_alphabet
    
    tokenizer = CharLevelTokenizer(alphabet_str=alphabet)
    
    text = "Hello, World!"
    encoded = tokenizer.encode(text, add_eos=True, pad_to_length=20)
    print("Encoded:", encoded)
    
    decoded = tokenizer.decode(encoded)
    print("Decoded:", decoded)
    
    # Збереження та завантаження токенайзера
    tokenizer.save("char_tokenizer.pkl")
    loaded_tokenizer = CharLevelTokenizer.load("char_tokenizer.pkl")
    
    assert loaded_tokenizer.decode(encoded) == decoded