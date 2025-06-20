#from word_pieses_tokenizer import CharacterTokenizer # погано працює для української мови
from sentence_pieses_tokenizer import CharacterTokenizer

LOAD = False

ukrainian_alphabet = "абвгґїдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ"
english_alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
numbers_alphabet = "01234567890"
symbol_alphabet = " .,!?;:+-*/=_—'\"()[]{}%#\n"


if LOAD:
    tokenizer = CharacterTokenizer.load("./models/tokenizer.pkl")
    print(f"Завантажено токенізатор на {tokenizer.vocab_size} слів.")
else:
    alphabet = ukrainian_alphabet + english_alphabet + numbers_alphabet + symbol_alphabet
    tokenizer = CharacterTokenizer(
        text=None,
        max_merges=None,
        min_merge_frequency=80,
        initial_alphabet_str=alphabet
    )

with open("data/countries1-processed.txt", encoding="utf-8") as f:
    data = f.read()
tokenizer.verbose = True
tokenizer.train(data)

print(f"\nРозмір словника після навчання: {tokenizer.vocab_size}")
print("\nКількість навчених об'єднань (merges):")
print(len(tokenizer.merges))

tokenizer.save("./models/tokenizer.pkl")
