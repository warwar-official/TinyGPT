import os
import sys
import unittest
# Додаємо корінь проекту до sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from imports.dataset_loader import DatasetProcessor
from imports.tokenizers.sentence_pieces_tokenizer import SentencePiecesTokenizer

class TestDatasetProcessor(unittest.TestCase):
    def setUp(self):
        # Завантаження токенізатора з pickle
        tokenizer_path = os.path.join(os.path.dirname(__file__), '../objects/tokenizers/SPT_1_0.pkl')
        self.tokenizer = SentencePiecesTokenizer.load(tokenizer_path)
        self.txt_path = os.path.join(os.path.dirname(__file__), '../datasets/old-txt/capitals-explanation.txt')
        self.cq_old_path = os.path.join(os.path.dirname(__file__), '../datasets/old-f-s-gen-json/rag-random-val.json')
        self.scqa_path = os.path.join(os.path.dirname(__file__), '../datasets/old-f-s-gen-json/scqa-val.ndjson')

    def test_txt_lines(self):
        dp = DatasetProcessor(self.txt_path, self.tokenizer, file_type="txt", sample_mode="lines", block_size=5, shuffle=False)
        dp._load_block()
        self.assertTrue(len(dp.data) > 0)
        for item in dp.data:
            self.assertIsInstance(item, list)

    def test_prepare_batch(self):
        dp = DatasetProcessor(self.txt_path, self.tokenizer, file_type="txt", sample_mode="lines", block_size=5, shuffle=False)
        dp._load_block()
        batch_x, batch_y = dp.prepare_batch(2)
        self.assertEqual(len(batch_x), 2)
        self.assertEqual(len(batch_y), 2)
        self.assertEqual(len(batch_x[0]), len(batch_y[0]))

    def test_ndjson_cq_old(self):
        dp = DatasetProcessor(self.cq_old_path, self.tokenizer, file_type="ndjson", sample_mode="cq_old", block_size=5, shuffle=False)
        dp._load_block()
        self.assertTrue(len(dp.data) > 0)
        for item in dp.data:
            self.assertIsInstance(item, list)
            self.assertTrue(len(item) > 0)

    def test_ndjson_scqa(self):
        dp = DatasetProcessor(self.scqa_path, self.tokenizer, file_type="ndjson", sample_mode="scqa", block_size=5, shuffle=False)
        dp._load_block()
        self.assertTrue(len(dp.data) > 0)
        for item in dp.data:
            self.assertIsInstance(item, list)
            self.assertTrue(len(item) > 0)

    def test_ndjson_sqa(self):
        dp = DatasetProcessor(self.scqa_path, self.tokenizer, file_type="ndjson", sample_mode="sqa", block_size=5, shuffle=False)
        dp._load_block()
        self.assertTrue(len(dp.data) > 0)
        for item in dp.data:
            self.assertIsInstance(item, list)
            self.assertTrue(len(item) > 0)

if __name__ == "__main__":
    unittest.main()
