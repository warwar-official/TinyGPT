# TinyGPT

A collection of training scripts for creating and training miniature language models, including custom tokenizers, RAG (Retrieval-Augmented Generation) support, and model inference tools.

## Features

- **TinyGPT Model**: A compact Transformer-based language model implementation with support for key-value caching and windowed attention.
- **Custom Tokenizers**: SentencePiece-inspired tokenizer with support for special tokens for dialog and QA tasks.
- **Training Scripts**: Flexible training pipeline for dialog and QA data in NDJSON format.
- **RAG Integration**: Simple vector database using FAISS and sentence transformers for retrieval-augmented generation, with support for Ukrainian language.
- **Inference Tools**: Command-line interface for model testing and interactive chat with confidence estimation.

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Dependencies

Install the required packages:

```bash
pip install torch faiss-cpu sentence-transformers matplotlib seaborn numpy
```

For GPU support with FAISS:

```bash
pip install faiss-gpu
```

## Project Structure

```
TinyGPT/
├── main.py                 # Main training script
├── model_tester.py         # Inference and testing script
├── imports/
│   ├── dataset_loader.py   # Data loading and preprocessing utilities
│   ├── train_v2.py         # Training functions and model creation
│   ├── models/
│   │   ├── tinygpt.py      # Main TinyGPT model implementation
│   │   ├── tinygpt_kv.py   # Version with KV caching
│   │   └── tinygpt_no_kv.py # Version without KV caching
│   ├── tokenizers/
│   │   ├── base_tokenizer.py     # Base tokenizer class
│   │   ├── sentence_pieces_tokenizer.py  # SentencePiece-style tokenizer
│   │   └── CLE.py                # Another tokenizer implementation
│   └── RAG/
│       └── simple_RAG.py         # RAG vector database implementation
├── tests/
│   └── dataset_loader_tests.py   # Unit tests for data loading
└── TinyGPT/                      # Duplicate directory structure
    └── old/                      # Older versions and archived files
```

## Usage

### Training a Model

To train a TinyGPT model, run:

```bash
python main.py
```

This script loads a pre-trained tokenizer and model checkpoint, then continues training on QA pairs dataset. Modify the paths and hyperparameters in `main.py` as needed.

Key parameters in training:
- `vocab_size`: Size of the tokenizer vocabulary
- `emb_size`: Embedding dimension (default: 512)
- `num_layers`: Number of transformer layers (default: 12)
- `n_head`: Number of attention heads (default: 8)
- `max_len`: Maximum sequence length (default: 2048)

### Model Inference

For interactive inference:

```bash
python model_tester.py
```

This launches an interactive chat interface where you can ask questions and get responses from the trained model.

For single question inference:

```bash
python model_tester.py --question "What is the capital of France?"
```

#### Command Line Options

- `--model-path`: Path to the trained model checkpoint (.pth file)
- `--tokenizer-path`: Path to the saved tokenizer (.pkl file)
- `--rag-path`: Path to RAG vector database directory (leave empty to disable RAG)
- `--device`: Computation device ('cpu' or 'cuda')
- `--max-new-tokens`: Maximum number of tokens to generate
- `--top-k`: Top-K sampling parameter for generation diversity
- `--repetition-penalty`: Penalty for repeating tokens
- `--system-instruction`: System prompt for the model

### RAG Setup

To use Retrieval-Augmented Generation:

1. Prepare your documents and create a vector database using the `SimpleVectorDB` class
2. Save the database to a directory
3. Provide the path via `--rag-path` when running inference

The RAG system uses sentence transformers to encode documents and FAISS for efficient similarity search.

## Data Format

The training expects data in NDJSON format with QA pairs. Example:

```json
{"question": "What is Python?", "answer": "Python is a programming language."}
```

## Model Architecture

TinyGPT is a decoder-only Transformer model with the following components:

- Multi-head self-attention with optional windowed attention
- Feed-forward networks with GELU activation
- Layer normalization and dropout
- Key-value caching for efficient inference
- Support for special tokens ([SYS], [USR], [AST], [RAG], etc.)

## Testing

Run the tests:

```bash
python -m pytest tests/
```

## License

MIT License - see LICENSE file for details.

Copyright (c) 2025 warwar-official

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, and pull requests.

This project is primarily developed with Ukrainian language support in mind, but can be adapted for other languages.