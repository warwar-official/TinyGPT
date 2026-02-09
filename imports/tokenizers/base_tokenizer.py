class BaseTokenizer:
    """
    Базовий клас для всіх токенізаторів (метаклас).
    Всі токенізатори повинні наслідувати цей клас і реалізовувати encode, decode, save, load.
    """
    UNK_TOKEN = '[UNK]' # для невідомих токенів 
    EOS_TOKEN = '[EOS]' # кінець генерації
    PAD_TOKEN = '[PAD]' # для порожніх місць
    SYS_TOKEN = '[SYS]' # маркер початку системного промту
    RAG_TOKEN = '[RAG]' # маркер початку фрагменту RAG
    USR_TOKEN = '[USR]' # початок повідомлення користувача
    AST_TOKEN = '[AST]' # початок відповіді моделі

    def __init__(self):
        self.vocab = {}
        self.decodes = {}
        self.next_token_id = 0

    def encode(self, text: str, add_eos: bool = False, pad_to_length: int = None) -> list[int]:
        """
        Декодує список токенів назад у рядок.

        Args:
            text (str): Рядок для кодування.
            add_eos (bool): Якщо True, в кінці спику токенів додається [EOS].
            pad_to_length (bool): Якщо True, доповнює до заданої довжини за допомогою [PAD].
        Returns:
            list[int]: Список токенів (ID).
        """
        raise NotImplementedError

    def decode(self, tokens: list[int], ignore_pad: bool = True) -> str:
        """
        Декодує список токенів назад у рядок.

        Args:
            tokens (list[int]): Список токенів (ID).
            ignore_pad (bool): Якщо True, ігнорує [PAD] токени при декодуванні.
        
        Returns:
            str: Декодований рядок.
        """
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        """
        Повертає кількість токенів в словнику.

        Returns:
            int: Кілткість токенів в словнику.
        """
        return len(self.vocab)

    def save(self, filename: str):
        """
        Зберішає токенізатор в файл.

        Args:
            filename (str): Шлях до файлу збереження.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, filename: str):
        """
        Зберішає токенізатор з файлу.

        Args:
            filename (str): Шлях до файлу завантаження.
        Returns:
            BaseTokenizer: Об'єкт токенізатора прочитаний з файлу.
        """
        raise NotImplementedError
