import random
import json
from collections import deque
from imports.tokenizers.base_tokenizer import BaseTokenizer

class DatasetProcessor():

    def __init__(self,
                 file_path: str,
                 tokenizer: BaseTokenizer,
                 file_type: str = "txt",
                 sample_mode: str = "lines",
                 block_size = 10240,
                 shuffle = True):
        try:
            self.file = open(file_path, 'r', encoding="utf-8")
            self.tokenizer = tokenizer
            self.file_type = file_type
            self.sample_mode = sample_mode
            self.block_size = block_size
            self.shuffle = shuffle
            self.data = deque([])
        except Exception as e:
            print(f"Помилка завантаження датасету: {e}")

    def _load_block(self):
        data_block = []
        for i in range(self.block_size):
            line = self.file.readline()
            if not line:
                print("Читання досягло кінця файлу, повернення на початок.")
                self.file.seek(0)
                continue
            data_block.append(line)
        if not data_block == []:
            self._decode_block(data_block)
        else:
            raise Exception("Помилка завантаження. Список порожній.")

    def _get_token_id(self, token_attr, fallback="PAD_TOKEN"):
        if hasattr(self.tokenizer, token_attr):
            return self.tokenizer.vocab[getattr(self.tokenizer, token_attr)]
        print(f"[{token_attr}] токен не знайдено. Використовується [PAD].")
        return self.tokenizer.vocab[getattr(self.tokenizer, fallback)]

    def _decode_block(self, data_block: list[str]):
        sys_id = self._get_token_id("SYS_TOKEN")
        con_id = self._get_token_id("RAG_TOKEN")
        usr_id = self._get_token_id("USR_TOKEN")
        ast_id = self._get_token_id("AST_TOKEN")

        if self.file_type == "txt" and self.sample_mode == "lines":
            output_list = self._decode_txt_lines(data_block)
        elif self.file_type == "ndjson":
            if self.sample_mode == "lines":
                output_list = self._decode_ndjson_lines(data_block)
            elif self.sample_mode == "qa":
                output_list = self._decode_ndjson_qa(data_block, usr_id, ast_id)
            elif self.sample_mode == "cq_old":
                output_list = self._decode_ndjson_cq_old(data_block, con_id)
            elif self.sample_mode == "sqa":
                output_list = self._decode_ndjson_sqa(data_block, sys_id, usr_id, ast_id)
            elif self.sample_mode == "scqa":
                output_list = self._decode_ndjson_scqa(data_block, sys_id, con_id, usr_id, ast_id)
            else:
                raise ValueError(f"Невідомий тип семплювання: {self.sample_mode}")
        else:
            raise ValueError(f"Невідомий тип файлу: {self.file_type}")
        if self.shuffle:
            random.shuffle(output_list)
        self.data.extend(output_list)

    def _decode_txt_lines(self, data_block):
        return [self.tokenizer.encode(line, True) for line in data_block]

    def _decode_ndjson_lines(self, data_block):
        return [self.tokenizer.encode(json.loads(line)["line"], True) for line in data_block]

    def _decode_ndjson_qa(self, data_block, usr_id, ast_id):
        output = []
        for json_line in data_block:
            line = json.loads(json_line)
            tokenized_question = [usr_id] + self.tokenizer.encode(line["question"], True)
            tokenized_answer = [ast_id] + self.tokenizer.encode(line["answer"], True)
            output.append(tokenized_question + tokenized_answer)
        return output

    def _decode_ndjson_cq_old(self, data_block, con_id):
        output = []
        for json_line in data_block:
            line = json.loads(json_line)
            tokenized_context = [con_id] + self.tokenizer.encode(line["context"], True)
            for question in line["questions"]:
                tokenized_question = self.tokenizer.encode(question, True)
                output.append(tokenized_context + tokenized_question)
        return output

    def _decode_ndjson_sqa(self, data_block, sys_id, usr_id, ast_id):
        output = []
        for json_line in data_block:
            line = json.loads(json_line)
            tokenized_system = [sys_id] + self.tokenizer.encode(line["system"], True)
            for pair in line["lines"]:
                tokenized_question = [usr_id] + self.tokenizer.encode(pair["question"], True)
                tokenized_answer = [ast_id] + self.tokenizer.encode(pair["answer"], True)
                output.append(tokenized_system + tokenized_question + tokenized_answer)
        return output

    def _decode_ndjson_scqa(self, data_block, sys_id, con_id, usr_id, ast_id):
        output = []
        for json_line in data_block:
            line = json.loads(json_line)
            tokenized_system = [sys_id] + self.tokenizer.encode(line["system"], True)
            tokenized_context = [con_id] + self.tokenizer.encode(line["context"], True)
            for pair in line["lines"]:
                tokenized_question = [usr_id] + self.tokenizer.encode(pair["question"], True)
                tokenized_answer = [ast_id] + self.tokenizer.encode(pair["answer"], True)
                output.append(tokenized_system + tokenized_context + tokenized_question + tokenized_answer)
        return output

    def prepare_batch(self, size: int):
        pad_id = self.tokenizer.vocab[self.tokenizer.PAD_TOKEN]
        ast_id = self.tokenizer.vocab[self.tokenizer.AST_TOKEN]  # Get [AST] token ID
        
        max_len = 0
        batch_x = []
        batch_y = []
        batch_labels = []  # New: masked labels for loss calculation
        
        for _ in range(size):
            if len(self.data) < 1:
                self._load_block()
            line = self.data.popleft()
            max_len = max(len(line), max_len)
            batch_x.append(line)
            batch_y.append(line[1:] + [pad_id])
            
            # Create label mask
            labels = (line[1:] + [pad_id]).copy()
            
            # Find [AST] token position
            try:
                ast_pos = line.index(ast_id) if ast_id in line else -1
            except ValueError:
                ast_pos = -1
            
            # Mask everything before [AST] (inclusive)
            if ast_pos >= 0:
                for i in range(ast_pos):  # Mask up to [AST]
                    if i < len(labels):
                        labels[i] = -100
            
            batch_labels.append(labels)
        
        # Padding
        for x, y, labels in zip(batch_x, batch_y, batch_labels):
            if len(x) < max_len:
                x.extend([pad_id] * (max_len - len(x)))
            if len(y) < max_len:
                y.extend([pad_id] * (max_len - len(y)))
            if len(labels) < max_len:
                labels.extend([-100] * (max_len - len(labels)))  # Pad with -100
        
        return batch_x, batch_y, batch_labels
        