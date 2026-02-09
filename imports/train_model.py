"""
Модуль для навчання малих мовних моделей на діалогових даних з підтримкою мульти-токенізаторів.

Цей модуль надає функціональність для створення та навчання TinyGPT моделей
на текстових даних у різних режимах семплювання з можливістю використання
декількох токенізаторів одночасно.
"""

import os
import random
import json
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

from imports.models.tinygpt import TinyGPT
from imports.tokenizers.base_tokenizer import BaseTokenizer

def load_tokenizers(tokenizer_paths: List[Union[str, Tuple[str, type]]]) -> List[BaseTokenizer]:
    """
    Завантажує список токенізаторів з файлів.
    
    Args:
        tokenizer_paths (List[str|Tuple[str, type]]):
            Список шляхів до файлів токенізаторів або кортежів (шлях, клас)
    
    Returns:
        List[BaseTokenizer]: Список завантажених токенізаторів
    
    Raises:
        FileNotFoundError: Якщо якийсь файл токенізатора не знайдено
    """
    tokenizers = []
    for i, item in enumerate(tokenizer_paths):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            path, cls = item
        else:
            path = item
            cls = None
        
        try:
            if cls is not None:
                tokenizer = cls.load(path)
            else:
                tokenizer = BaseTokenizer.load(path)
            tokenizers.append(tokenizer)
            print(f"Токенізатор {i+1} завантажено з: {path} ({type(tokenizer).__name__})")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл токенізатора не знайдено: {path}")
        except Exception as e:
            raise RuntimeError(f"Помилка завантаження токенізатора {path}: {e}")
    
    return tokenizers

def create_model(
    tokenizers: List[BaseTokenizer],
    input_lengths: List[int],
    inference_tokenizer_idx: int = 0,
    model_path: Optional[str] = None, 
    device: Optional[torch.device] = None,
    emb_size: int = 384,
    n_head: int = 6,
    num_layers: int = 7,
    fusion_heads: int = None
) -> Tuple[TinyGPT, torch.device]:
    """
    Створює та ініціалізує модель TinyGPT з підтримкою мульти-токенізаторів.
    
    Args:
        tokenizers (List[BaseTokenizer]): Список токенізаторів
        input_lengths (List[int]): Список довжин вхідних послідовностей для кожного токенізатора
        inference_tokenizer_idx (int): Індекс токенізатора для інференсу
        model_path (str, optional): Шлях до збереженої моделі для завантаження
        device (torch.device, optional): Пристрій для обчислень (GPU/CPU)
        emb_size (int): Розмір ембедингів
        n_head (int): Кількість голів уваги
        num_layers (int): Кількість шарів трансформера
    
    Returns:
        Tuple[TinyGPT, torch.device]: Ініціалізована модель та використовуваний пристрій
    
    Raises:
        ValueError: Якщо список токенізаторів порожній
    """
    if not tokenizers:
        raise ValueError("Список токенізаторів не може бути порожнім")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Використовується пристрій: {device}")
    print(f"Кількість токенізаторів: {len(tokenizers)}")
    
    # Отримуємо розміри словників для кожного токенізатора
    vocab_sizes = [tokenizer.vocab_size for tokenizer in tokenizers]
    print(f"Розміри словників: {vocab_sizes}")
    
    model = TinyGPT(
        vocab_sizes=vocab_sizes,
        input_lengths=input_lengths,
        emb_size=emb_size,
        n_head=n_head,
        num_layers=num_layers,
        max_length=input_lengths[inference_tokenizer_idx],
        out_vocab_size=vocab_sizes[inference_tokenizer_idx] if inference_tokenizer_idx < len(vocab_sizes) else None,
        fusion_heads=fusion_heads
    ).to(device)
    
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Модель успішно завантажено з: {model_path}")
        except Exception as e:
            print(f"Помилка при завантаженні моделі: {e}. Модель ініціалізована з нуля.")
    elif model_path:
        print(f"Файл моделі не знайдено: {model_path}. Модель ініціалізована з нуля.")
    else:
        print("Модель ініціалізована з нуля.")
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Кількість параметрів: {param_count}")
    
    return model, device

def train_model(
    model: TinyGPT,
    tokenizers: List[BaseTokenizer],
    input_lengths: List[int],
    train_data_path: str,
    val_data_path: str,
    device: torch.device,
    inference_tokenizer_idx: int = 0,
    sample_mode: str = "rag_like",
    batch_size: int = 1,
    seq_len: int = 768,
    max_epochs: int = 15000,
    accumulate_steps: int = 16,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    input_dropout: Optional[float] = None,
    report_rate: Optional[int] = None,
    save_rate: Optional[int] = None,
    save_dir: str = "./models/checkpoints"
) -> None:
    """
    Навчає модель на заданих даних з підтримкою мульти-токенізаторів.
    
    Args:
        model (TinyGPT): Модель для навчання
        tokenizers (List[BaseTokenizer]): Список токенізаторів
        train_data_path (str): Шлях до файлу з тренувальними даними
        val_data_path (str): Шлях до файлу з валідаційними даними
        device (torch.device): Пристрій для обчислень
        inference_tokenizer_idx (int): Індекс токенізатора для інференсу (target)
        sample_mode (str): Режим семплювання ("rag_like", "context_training")
        batch_size (int): Розмір батчу
        seq_len (int): Максимальна довжина послідовності
        max_epochs (int): Максимальна кількість епох
        accumulate_steps (int): Кількість кроків для накопичення градієнтів
        learning_rate (float): Швидкість навчання
        weight_decay (float): Коефіцієнт регуляризації
        report_rate (int, optional): Частота виведення статистики
        save_rate (int, optional): Частота збереження моделі
        save_dir (str): Директорія для збереження чекпоінтів
    
    Raises:
        ValueError: Якщо параметри навчання некоректні
    """
    if sample_mode not in ["rag_like", "context_training"]:
        raise ValueError("sample_mode має бути одним з:'rag_like'")
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Файл тренувальних даних не знайдено: {train_data_path}")
    
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Файл валідаційних даних не знайдено: {val_data_path}")
    
    if inference_tokenizer_idx >= len(tokenizers):
        raise ValueError(f"inference_tokenizer_idx ({inference_tokenizer_idx}) поза межами списку токенізаторів")
    
    # Встановлення значень за замовчуванням
    if report_rate is None:
        report_rate = 100 * accumulate_steps
    else:
        report_rate = report_rate * accumulate_steps
    if save_rate is None:
        save_rate = 100 * accumulate_steps
    else:
        save_rate = save_rate * accumulate_steps
    
    # Створення директорії для збереження
    os.makedirs(save_dir, exist_ok=True)
    
    # Ініціалізація оптимізатора та планувальника
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=500)
    
    # Завантаження та токенізація даних
    print("Завантаження тренувальних даних...")
    train_data = _load_and_tokenize_data(train_data_path, tokenizers, input_lengths, sample_mode)
    
    print("Завантаження валідаційних даних...")
    val_data = _load_and_tokenize_data(val_data_path, tokenizers, input_lengths, sample_mode)
    
    if not train_data:
        raise ValueError("Немає тренувальних даних для навчання")
    
    print(f"Початок навчання в режимі '{sample_mode}'...")
    print(f"Використовується токенізатор {inference_tokenizer_idx} для інференсу")
    
    model.train()
    
    total_steps = max_epochs * accumulate_steps
    
    fusion_mask = None

    
    for step in range(total_steps):
        batch_result = _prepare_batch_data(
            tokenized_data=train_data,
            sample_mode=sample_mode,
            batch_size=batch_size,
            seq_lens=input_lengths,
            device=device,
            tokenizers=tokenizers,
            mask_special_tokens=True,
            dropout=input_dropout,
        )
        
        if batch_result is None:
            continue
        
        xs, y = batch_result
        
        fusion_mask = _make_fusion_mask(xs, tokenizers)
        if not fusion_mask.any():
            raise ValueError("fusion_mask all False!")

        logits = model(xs, fusion_mask=fusion_mask)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100
        )
        loss = loss / accumulate_steps
        
        # Зворотний прохід
        loss.backward()
        
        # Оновлення параметрів
        if (step + 1) % accumulate_steps == 0:
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()
        
        # Виведення статистики
        if (step + 1) % (report_rate)  == 0:
            epoch = (step + 1) // accumulate_steps
            val_loss = _evaluate_model(
                model=model,
                val_data=val_data,
                sample_mode=sample_mode,
                batch_size=batch_size,
                input_lengths=input_lengths,
                device=device,
                inference_tokenizer_idx=inference_tokenizer_idx,
                tokenizers=tokenizers
            )
            
            print(f"Епоха {epoch}, "
                  f"Втрати: {loss.item() * accumulate_steps:.4f}, "
                  f"Валідаційні втрати: {val_loss:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Збереження моделі
        if (step + 1) % (save_rate) == 0 and step > 0:
            epoch = (step + 1) // accumulate_steps
            save_path = os.path.join(save_dir, f"ep-{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Модель збережено: {save_path}")
        
    print("Навчання завершено.")

def _evaluate_model(
    model: TinyGPT, 
    val_data: List[Union[torch.Tensor, List[List[int]]]],
    sample_mode: str, 
    batch_size: int,
    input_lengths: List[int],
    device: torch.device,
    inference_tokenizer_idx: int,
    tokenizers: List[BaseTokenizer]
) -> float:
    """
    Оцінює модель на валідаційних даних з мульти-токенізаторами.
    
    Returns:
        float: Значення функції втрат
    """
    model.eval()
    
    with torch.no_grad():
        batch_result = _prepare_batch_data(
            tokenized_data=val_data,
            sample_mode=sample_mode,
            batch_size=batch_size,
            seq_lens=input_lengths,
            device=device,
            tokenizers=tokenizers,
            mask_special_tokens=True,
            dropout=None,
        )
        
        if batch_result is None:
            return float('inf')
        
        xs, y = batch_result

        logits = model(xs)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=-100
        )

        inference_tokenizer = tokenizers[inference_tokenizer_idx]
        true_ids = y
        true_ids = true_ids.masked_fill(true_ids == -100, inference_tokenizer.vocab[inference_tokenizer.PAD_TOKEN])
        predicted_ids = logits.argmax(dim=-1)
        accuracy = (predicted_ids == true_ids).float().mean().item()
        
        for i in range(min(1, predicted_ids.size(0))):
            pred_tokens = inference_tokenizer.decode(predicted_ids[i].tolist(), ignore_pad=False)
            true_tokens = inference_tokenizer.decode(true_ids[i].tolist(), ignore_pad=True)
            print(f"Приклад {i+1}:")
            print(f"  Передбачено: {pred_tokens[:50]}...")
            print(f"  Правильно: {true_tokens[:50]}...")
        print(f"Валідаційна точність: {accuracy:.4f}")

    
    model.train()
    return loss.item()