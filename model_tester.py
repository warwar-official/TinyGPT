import argparse
import math
from typing import List, Tuple, Optional

import torch

from imports.models.tinygpt import TinyGPT  # noqa: F401 (інформація про тип)
from imports.RAG.simple_RAG import SimpleVectorDB
from imports.tokenizers.sentence_pieces_tokenizer import SentencePiecesTokenizer
from imports.train_v2 import create_model


DEFAULT_SYSTEM_INSTRUCTION = (
    ""#"Ти корисний асистент. Використовуй наданий контекст та відповідай стисло українською."
)
DEFAULT_RAG_PATH = "objects/RAG/test_vb"


def load_components(
    model_path: str,
    tokenizer_path: str,
    device_str: Optional[str] = None,
) -> Tuple[TinyGPT, SentencePiecesTokenizer, torch.device]:
    tokenizer = SentencePiecesTokenizer.load(tokenizer_path)

    if device_str is not None:
        device = torch.device(device_str)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, device = create_model(
        vocab_size=tokenizer.vocab_size(),
        model_path=model_path,
        emb_size=512,
        max_len=2048,
        num_layers=12,
        n_head=8,
        device=device,
    )
    model.eval()
    return model, tokenizer, device


def load_vector_db(
    rag_path: Optional[str],
    rag_device: str = "cpu",
    model_name: Optional[str] = None,
) -> Optional[SimpleVectorDB]:
    if not rag_path:
        return None

    if model_name:
        vector_db = SimpleVectorDB(model_name=model_name, device=rag_device)
    else:
        vector_db = SimpleVectorDB(device=rag_device)

    try:
        vector_db.load(rag_path)
    except Exception as exc:
        raise RuntimeError(f"Не вдалося завантажити векторну базу з '{rag_path}': {exc}") from exc
    return vector_db


def prepare_context(
    question: str,
    vector_db: Optional[SimpleVectorDB],
    top_k: int,
) -> str:
    if vector_db is None:
        return ""

    results = vector_db.search(question, top_k=top_k)
    if not results:
        return ""

    formatted_chunks: List[str] = []
    for chunk_text, _, metadata in results:
        source = metadata.get("source_file") if metadata else None
        if not source:
            source = "Невідоме джерело"
        formatted_chunks.append(f"Заголовок: {source}. Текст: {chunk_text}")

    return "\n\n".join(formatted_chunks)


def build_prompt(
    question: str,
    context: str,
    system_instruction: str,
    tokenizer: SentencePiecesTokenizer,
) -> List[int]:
    sys_id = tokenizer.vocab[tokenizer.SYS_TOKEN]
    con_id = tokenizer.vocab[tokenizer.RAG_TOKEN]
    usr_id = tokenizer.vocab[tokenizer.USR_TOKEN]
    ast_id = tokenizer.vocab[tokenizer.AST_TOKEN]
    encoded_prompt: List[int] = []
    encoded_prompt = [sys_id] + tokenizer.encode(system_instruction, add_eos=False)
    if context.strip():
        encoded_prompt += [con_id] + tokenizer.encode(context, add_eos=False)
    else:
        encoded_prompt += [con_id] + tokenizer.encode("Контекст відсутній.", add_eos=False)
    encoded_prompt += [usr_id] + tokenizer.encode(question, add_eos=False)
    encoded_prompt += [ast_id]
    return encoded_prompt


def decode_answer(
    tokenizer: SentencePiecesTokenizer,
    full_tokens: List[int],
) -> str:
    ast_id = tokenizer.vocab[tokenizer.AST_TOKEN]
    eos_id = tokenizer.vocab[tokenizer.EOS_TOKEN]

    answer_tokens: List[int] = []
    capture = False
    for token_id in full_tokens:
        if capture:
            if token_id == eos_id:
                break
            answer_tokens.append(token_id)
        elif token_id == ast_id:
            capture = True

    text = tokenizer.decode(answer_tokens, ignore_pad=True)
    for special in [
        tokenizer.USR_TOKEN,
        tokenizer.AST_TOKEN,
        tokenizer.SYS_TOKEN,
        "[CON]",
        tokenizer.RAG_TOKEN,
        tokenizer.EOS_TOKEN,
        tokenizer.PAD_TOKEN,
    ]:
        text = text.replace(special, "")
    return text.strip()


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: List[int],
    penalty: float,
) -> torch.Tensor:
    if penalty == 1.0 or not generated_ids:
        return logits

    penalized = logits.clone()
    unique_tokens = set(generated_ids)
    for token_id in unique_tokens:
        if 0 <= token_id < penalized.size(-1):
            if penalty > 1.0:
                penalized[token_id] /= penalty
            else:
                penalized[token_id] *= penalty
    return penalized


def sample_next_token(
    logits: torch.Tensor,
    top_k: int,
) -> Tuple[int, float]:
    if top_k is None or top_k <= 0:
        probs = torch.softmax(logits, dim=-1)
        next_token = int(torch.argmax(probs).item())
        return next_token, float(probs[next_token].item())

    k = min(top_k, logits.size(-1))
    top_values, top_indices = torch.topk(logits, k)
    top_probs = torch.softmax(top_values, dim=-1)
    sampled_idx = int(torch.multinomial(top_probs, num_samples=1).item())
    next_token = int(top_indices[sampled_idx].item())

    full_probs = torch.softmax(logits, dim=-1)
    return next_token, float(full_probs[next_token].item())


def confidence_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-12))).item()
    max_entropy = math.log(float(probs.numel()))
    if max_entropy == 0:
        return 0.0
    normalized = 1.0 - min(entropy / max_entropy, 1.0)
    return max(0.0, float(normalized))


def estimate_sequence_confidence(step_confidences: List[float]) -> float:
    if not step_confidences:
        return 0.0
    return float(sum(step_confidences) / len(step_confidences))


def generate_answer(
    model: TinyGPT,
    tokenizer: SentencePiecesTokenizer,
    question: str,
    context: str,
    system_instruction: str,
    device: torch.device,
    max_new_tokens: int = 128,
    top_k_sampling: int = 0,
    repetition_penalty: float = 1.0,
) -> Tuple[str, List[int], float]:
    model.eval()

    prompt_ids = build_prompt(
        question=question,
        context=context,
        system_instruction=system_instruction,
        tokenizer=tokenizer,
    )
    eos_id = tokenizer.vocab[tokenizer.EOS_TOKEN]
    max_pos = model.pos_embed.num_embeddings

    generated = prompt_ids.copy()
    input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

    kv_cache = None
    step_confidences: List[float] = []
    for _ in range(max_new_tokens):
        if input_ids.size(1) >= max_pos:
            break

        with torch.no_grad():
            if kv_cache is None:
                logits, kv_cache = model(input_ids, past_key_values=kv_cache, use_cache=True)
            else:
                logits, kv_cache = model(input_ids[:, -1:], past_key_values=kv_cache, use_cache=True)

        last_logits = logits[:, -1, :].squeeze(0)
        adjusted_logits = apply_repetition_penalty(last_logits, generated, repetition_penalty)
        step_confidences.append(confidence_from_logits(adjusted_logits))

        next_token_id, _ = sample_next_token(
            logits=adjusted_logits,
            top_k=top_k_sampling,
        )
        generated.append(next_token_id)
        input_ids = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)

        if next_token_id == eos_id:
            break

    answer = decode_answer(tokenizer, generated)
    confidence = estimate_sequence_confidence(step_confidences)
    return answer, generated, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple question-answer inference for TinyGPT.")
    parser.add_argument(
        "--model-path",
        default="models/12l/v2/qa/ep-32000.pth",
        help="Шлях до збереженої моделі (.pth).",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="objects/tokenizers/SPT_1_0.pkl",
        help="Шлях до збереженого токенізатора (.pkl).",
    )
    parser.add_argument(
        "--rag-path",
        default="",#DEFAULT_RAG_PATH,
        help="Шлях до директорії з векторною базою FAISS. Вкажіть '', щоб вимкнути RAG.",
    )
    parser.add_argument(
        "--rag-device",
        default="cpu",
        help="Пристрій для SentenceTransformer у RAG (cpu/cuda).",
    )
    parser.add_argument(
        "--rag-top-k",
        type=int,
        default=1,
        help="Кількість контекстних шматків для RAG.",
    )
    parser.add_argument(
        "--system-instruction",
        default=DEFAULT_SYSTEM_INSTRUCTION,
        help="Інструкція системи для блоку [SYS].",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Примусовий вибір пристрою (наприклад, 'cpu', 'cuda'). За замовчуванням визначається автоматично.",
    )
    parser.add_argument(
        "--question",
        "-q",
        help="Запитання для одноразового інференсу. Якщо не вказано, запускається інтерактивний режим.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Максимальна кількість нових токенів для генерації.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Top-K для семплінгу (0 означає жадібний вибір).",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Штраф за повтори токенів (>1.0 зменшує ймовірність повторів).",
    )
    return parser.parse_args()


def interactive_loop(
    model: TinyGPT,
    tokenizer: SentencePiecesTokenizer,
    vector_db: Optional[SimpleVectorDB],
    system_instruction: str,
    rag_top_k: int,
    device: torch.device,
    max_new_tokens: int,
    top_k_sampling: int,
    repetition_penalty: float,
) -> None:
    print("Інтерактивний режим. Введіть запитання або 'exit' для виходу.")
    while True:
        try:
            question = input("Q> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВихід.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Вихід.")
            break

        context = prepare_context(
            question=question,
            vector_db=vector_db,
            top_k=rag_top_k,
        )
        if context:
            print("=== Контекст RAG ===")
            print(context)
            print("====================")

        answer, _, confidence = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=question,
            context=context,
            system_instruction=system_instruction,
            device=device,
            max_new_tokens=max_new_tokens,
            top_k_sampling=top_k_sampling,
            repetition_penalty=repetition_penalty,
        )
        print(f"A> {answer}")
        print(f"(Впевненість моделі: {confidence:.2f})\n")


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_components(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device_str=args.device,
    )
    vector_db = load_vector_db(
        rag_path=args.rag_path,
        rag_device=args.rag_device,
    )

    if args.question:
        context = prepare_context(
            question=args.question,
            vector_db=vector_db,
            top_k=args.rag_top_k,
        )
        if context:
            print("=== Контекст RAG ===")
            print(context)
            print("====================")

        answer, _, confidence = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=args.question,
            context=context,
            system_instruction=args.system_instruction,
            device=device,
            max_new_tokens=args.max_new_tokens,
            top_k_sampling=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print(f"{answer}\nВпевненість: {confidence:.2f}")
    else:
        interactive_loop(
            model=model,
            tokenizer=tokenizer,
            vector_db=vector_db,
            system_instruction=args.system_instruction,
            rag_top_k=args.rag_top_k,
            device=device,
            max_new_tokens=args.max_new_tokens,
            top_k_sampling=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )


if __name__ == "__main__":
    main()
