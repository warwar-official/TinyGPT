from imports.train_v2 import train_model, create_model
from imports.tokenizers.sentence_pieces_tokenizer import SentencePiecesTokenizer

if __name__ == "__main__":
    tokenizer = SentencePiecesTokenizer.load("objects/tokenizers/SPT_1_0.pkl")
    model, device = create_model(
        vocab_size=tokenizer.vocab_size(),
        emb_size=512,
        max_len=2048,
        num_layers=12,
        n_head=8,
        model_path="models/12l/v2/wiki-768/ep-10000.pth",
        device=None)

    train_model(
        model=model,
        tokenizer=tokenizer,
        train_data_path="datasets/new_ndjson/qa_pairs_full.ndjson",
        val_data_path="datasets/new_ndjson/qa_pairs.ndjson",
        file_type="ndjson",
        sample_mode="qa",
        batch_size=4,
        max_epochs=32000,
        learning_rate=5e-5,
        accumulate_steps=8,
        report_rate=1000,
        device=device,
        save_dir="models/12l/v2/qa",
        save_rate=1000,
    )