import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_head, dropout=0.15):
        super().__init__()
        self.attn = nn.MultiheadAttention(emb_size, n_head, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size * 4),
            nn.GELU(),
            nn.Linear(emb_size * 4, emb_size),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(emb_size)
        self.alpha_attn = nn.Parameter(torch.tensor(0.0))
        self.alpha_ff = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, mask):
        x_ = self.ln1(x)  # Pre-LN для attention
        attn_output, _ = self.attn(x_, x_, x_, attn_mask=mask, average_attn_weights=False)
        x = x + self.alpha_attn * attn_output  # ReZero: α * F(x)

        x_ = self.ln2(x)  # Pre-LN для FFN
        ff_output = self.ff(x_)
        x = x + self.alpha_ff * ff_output      # ReZero: α * F(x)

        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, emb_size=384, n_head=6, num_layers=7, max_len=768):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.pos_embed = nn.Embedding(max_len, emb_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_head)
            for _ in range(num_layers)
        ])

        self.out = nn.Linear(emb_size, vocab_size)

        self.apply(self.init_weights)  # ⬅️ виклик ініціалізації

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        x = self.embed(x) + self.pos_embed(pos)

        # causal mask
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.out(x)
        return logits
