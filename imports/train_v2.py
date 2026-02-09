"""
Модуль для навчання малих мовних моделей на діалогових даних.

Цей модуль надає функціональність для створення та навчання TinyGPT моделей
на текстових даних у різних режимах семплювання.
"""

import os
from typing import List, Tuple, Optional, Union

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from imports.models.tinygpt import TinyGPT

from imports.tokenizers.base_tokenizer import BaseTokenizer

from datetime import datetime

from imports.dataset_loader import DatasetProcessor

def create_model(
    vocab_size: int,
    emb_size: int = 384,
    max_len: int = 768,
    model_path: Optional[str] = None, 
    device: Optional[torch.device] = None,
    n_head: int = 6,
    num_layers: int = 7
) -> Tuple[TinyGPT, torch.device]:
    """
    Створює та ініціалізує модель TinyGPT з підтримкою мульти-токенізаторів.
    
    Args:
        vocab_size (int): розмір словника моделі
        model_path (str, optional): Шлях до збереженої моделі для завантаження
        device (torch.device, optional): Пристрій для обчислень (GPU/CPU)
        emb_size (int): Розмір ембедингів
        num_layers (int): Кількість шарів моделі
        d_state (int): Розмір вектора внутрішнього стану
    
    Returns:
        Tuple[TinyGPT, torch.device]: Ініціалізована модель та використовуваний пристрій
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Використовується пристрій: {device}")
    print(f"Розміри словника: {vocab_size}")
    
    model = TinyGPT(
        vocab_size=vocab_size,
        emb_size=emb_size,
        n_head=n_head,
        num_layers=num_layers,
        max_len=max_len,
        use_windowed=True
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
    
    param_count = model.get_num_params()
    print(f"Кількість параметрів: {param_count}")
    
    return model, device

def train_model(
    model: TinyGPT,
    tokenizer: BaseTokenizer,
    train_data_path: str,
    val_data_path: str,
    device: torch.device,
    file_type: str = "txt",
    sample_mode: str = "fragments",
    batch_size: int = 16,
    max_epochs: int = 15000,
    accumulate_steps: int = 1,
    learning_rate: float = 5e-5,
    weight_decay: float = 1e-4,
    report_rate: Optional[int] = None,
    save_rate: Optional[int] = None,
    save_dir: str = "./models/checkpoints"
) -> None:
    """
    Навчає модель на заданих даних.
    
    Args:
        model (TinyGPT): Модель для навчання
        tokenizer (BaseTokenizer): Токенізатор для обробки тексту
        train_data_path (str): Шлях до файлу з тренувальними даними
        val_data_path (str): Шлях до файлу з валідаційними даними
        device (torch.device): Пристрій для обчислень
        file_type (str): Тип файлу датасету ("txt", "ndjson")
        sample_mode (str): Режим семплювання (для txt: "lines"; для ndjson: "lines", "qa", "cq_old", "sqa", "scqa")
        batch_size (int): Розмір батчу
        max_epochs (int): Максимальна кількість епох
        accumulate_steps (int): Кількість кроків для накопичення градієнтів
        learning_rate (float): Швидкість навчання
        weight_decay (float): Коефіцієнт регуляризації
        report_rate (int, optional): Частота виведення статистики
        save_rate (int, optional): Частота збереження моделі
        save_dir (str): Директорія для збереження чекпоінтів
    
    Raises:
        FileNotFoundError: Якщо файли даних не знайдено
        ValueError: Якщо параметри навчання некоректні
    """
    
    if file_type == "txt":
        if sample_mode not in ["line"]:
            raise ValueError("sample_mode для txt має бути одним з:'lines'")
    elif file_type == "ndjson":
        if sample_mode not in ["lines", "qa", "cq_old", "sqa", "scqa"]:
            raise ValueError("sample_mode для ndjson має бути одним з: 'lines', 'qa', 'cq_old', 'sqa', 'scqa'")
    else:
        raise ValueError("file_type має бути одним з:'txt', 'ndjson'")

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Файл тренувальних даних не знайдено: {train_data_path}")
    
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Файл валідаційних даних не знайдено: {val_data_path}")
    
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10000, min_lr=5e-6)
    
    # Завантаження та токенізація даних
    print("Завантаження тренувальних даних...")
    train_data = DatasetProcessor(
        file_path=train_data_path,
        tokenizer=tokenizer,
        file_type=file_type,
        sample_mode=sample_mode,
        block_size=1024)
    
    print("Завантаження валідаційних даних...")
    val_data = DatasetProcessor(
        file_path=val_data_path,
        tokenizer=tokenizer,
        file_type=file_type,
        sample_mode=sample_mode,
        block_size=1024)
    
    timestamp1 = datetime.now()
    timebegin = timestamp1

    print(f"Початок навчання в режимі '{sample_mode}'...")
    model.train()
    
    total_steps = max_epochs * accumulate_steps
    
    for step in range(total_steps):
        # Підготовка даних для поточного кроку
        batch_x, batch_y, batch_labels = train_data.prepare_batch(batch_size)
        x = torch.stack([torch.tensor(x_list, dtype=torch.long) for x_list in batch_x]).to(device)
        y = torch.stack([torch.tensor(y_list, dtype=torch.long) for y_list in batch_y]).to(device)
        labels = torch.stack([torch.tensor(label_list, dtype=torch.long) for label_list in batch_labels]).to(device)
        
        if x is None or y is None:
            print("Помилка підготовки даних. Пропущено крок.")
            step -= 1
            continue
        if len(batch_x[0]) > model.pos_embed.num_embeddings:
            print(f"Перевищено максимальну довжину послідовності моделі ({model.pos_embed.num_embeddings}). Пропущено крок.")
            step -= 1
            continue
        
        # Прямий прохід
        logits = model(x)
        
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        loss = loss / accumulate_steps
        
        # Оновлення параметрів
        if (step + 1) % accumulate_steps == 0:
            # Зворотний прохід
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            optimizer.zero_grad()
        
        # Виведення статистики
        if (step + 1) % report_rate == 0:
            epoch = (step + 1) // accumulate_steps
            val_loss, accuracy = _evaluate_model(model, val_data, batch_size, device, calculate_accuracy=True)
            
            print(f"Епоха {epoch}, "
                  f"Втрати: {loss.item() * accumulate_steps:.4f}, "
                  f"Валідаційні втрати: {val_loss:.4f}, "
                  f"Точність: {accuracy:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

            timestamp2 = datetime.now()
            timedelta = timestamp2 - timestamp1
            timestamp1 = timestamp2
            print(f"Продуктивність начання: {(timedelta / report_rate * accumulate_steps)} / крок.")

            if True:
                _visualize_attentions(model, full_attention=False, save_path=f"{save_dir}/attention_ep-{epoch}.png")
        
        # Збереження моделі
        if (step + 1) % save_rate == 0 and step > 0:
            epoch = (step + 1) // accumulate_steps
            save_path = os.path.join(save_dir, f"ep-{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Модель збережено: {save_path}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("Навчання завершено.")
    timeend = datetime.now()
    print(f"Загальний час навчання: {(timeend - timebegin)}")

def _evaluate_model(
    model: TinyGPT,
    val_data: DatasetProcessor,
    batch_size: int, 
    device: torch.device,
    calculate_accuracy: bool = False
) -> float:
    """
    Оцінює модель на валідаційних даних.
    
    Args:
        model (TinyGPT): Модель для оцінки
        val_data: Валідаційні дані
        batch_size (int): Розмір батчу
        device (torch.device): Пристрій для обчислень
    
    Returns:
        float: Значення функції втрат
    """
    model.eval()
    
    with torch.no_grad():
        batch_x, batch_y, batch_labels = val_data.prepare_batch(batch_size)
        x = torch.stack([torch.tensor(x_list, dtype=torch.long) for x_list in batch_x]).to(device)
        y = torch.stack([torch.tensor(y_list, dtype=torch.long) for y_list in batch_y]).to(device)
        labels = torch.stack([torch.tensor(label_list, dtype=torch.long) for label_list in batch_labels]).to(device)
        
        if x is None or y is None:
            return float('inf')
        
        if len(batch_x[0]) > model.pos_embed.num_embeddings:
            print(f"Перевищено максимальну довжину послідовності моделі ({model.pos_embed.num_embeddings}). Пропущено валідацію.")
            return float('inf'), 0.0

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        accuracy = -1.0
        if calculate_accuracy:
            accuracy = _calculate_accuracy(logits, labels)

    
    model.train()
    return loss.item(), accuracy

def _visualize_attentions(model: TinyGPT, full_attention: bool, save_path: str) -> None:
    """
    Візуалізує механізми уваги моделі.
    
    Args:
        model (TinyGPT): Модель для візуалізації
        full_attention (bool): Якщо True, візуалізує всі голови уваги
        save_path (str): Шлях для збереження візуалізації
    """
    if hasattr(model.blocks[0], 'last_attention'):
        # Візуалізація передбачає форму уваги B, H, T, T
        if model.blocks[0].last_attention == None:
            print("Відсутні дані уваги для візуалізації.")
            return
        if model.blocks[0].last_attention.dim() != 4:
            print("Форма уваги не відповідає очікуваній (B, H, T, T). Візуалізація пропущена.")
            return
        num_cols = 0
        num_rows = 0
        if full_attention:
            num_cols = model.blocks[0].n_head
            num_rows = len(model.blocks)
        else:
            num_cols = 4
            num_rows = (int)(np.ceil(len(model.blocks) / num_cols))
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
        for i, block in enumerate(model.blocks):
            attn = block.last_attention[0].cpu().numpy()  # Візуалізуємо перший приклад у батчі
            if not full_attention:
                head_idx = i % model.blocks[0].n_head
            else:
                head_idx = slice(None)  # Всі голови
            attn_to_plot = attn[head_idx]
            if not full_attention:
                ax = axes[i // num_cols, i % num_cols]
                sns.heatmap(attn_to_plot, ax=ax, cmap="viridis")
                ax.set_title(f"Блок {i} - Голова {head_idx if full_attention else head_idx}")
                ax.set_xlabel("Позиція ключа")
                ax.set_ylabel("Позиція запиту")
            else:
                for h in range(attn_to_plot.shape[0]):
                    ax = axes[i, h]
                    sns.heatmap(attn_to_plot[h], ax=ax, cmap="viridis")
                    ax.set_title(f"Блок {i+1} - Голова {h}")
                    ax.set_xlabel("Позиція ключа")
                    ax.set_ylabel("Позиція запиту")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Візуалізація уваги збережена: {save_path}")
    else:
        print("Модель не має необхідних атрибутів (last_attention) в блоках.")

def _calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Обчислює точність передбачень моделі.
    
    Args:
        logits (torch.Tensor): Вихідні логіти моделі
        labels (torch.Tensor): Істинні мітки
    
    Returns:
        float: Точність передбачень
    """
    preds = torch.argmax(logits, dim=-1)
    mask = labels != -100  # Ігноруємо позиції з -100
    correct = (preds == labels) & mask
    accuracy = correct.sum().item() / mask.sum().item()
    return accuracy