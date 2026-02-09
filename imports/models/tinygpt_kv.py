import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_head, dropout=0.15):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_size // n_head
        if self.head_dim * n_head != emb_size:
            raise ValueError("emb_size must be divisible by n_head")

        self.q_proj = nn.Linear(emb_size, emb_size)
        self.k_proj = nn.Linear(emb_size, emb_size)
        self.v_proj = nn.Linear(emb_size, emb_size)
        self.out_proj = nn.Linear(emb_size, emb_size)

        self.attn_dropout = nn.Dropout(dropout)
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

        self.last_attention = None

    def forward(self, x, attn_mask=None, past_key_value=None, use_cache=False):
        residual = x
        x_ = self.ln1(x)  # Pre-LN для attention

        q = self.q_proj(x_)
        k = self.k_proj(x_)
        v = self.v_proj(x_)

        B, T, C = q.size()
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        else:
            total_len = k.size(-2)
            if use_cache:
                if T > 0:
                    future_mask = torch.triu(torch.ones((T, total_len), device=x.device, dtype=torch.bool), diagonal=1 + past_len)
                    attn_scores = attn_scores.masked_fill(future_mask, float('-inf'))
            else:
                if T > 0:
                    future_mask = torch.triu(torch.ones((T, T), device=x.device, dtype=torch.bool), diagonal=1)
                    attn_scores = attn_scores.masked_fill(future_mask, float('-inf'))

        attn_weight = F.softmax(attn_scores, dim=-1)
        attn_weight = self.attn_dropout(attn_weight)
        attn_output = torch.matmul(attn_weight, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)

        x = residual + self.alpha_attn * attn_output  # ReZero: α * F(x)

        x_ = self.ln2(x)  # Pre-LN для FFN
        ff_output = self.ff(x_)
        x = x + self.alpha_ff * ff_output      # ReZero: α * F(x)

        self.last_attention = attn_weight

        if use_cache:
            present = (k, v)
            return x, present

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

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, past_key_values=None, use_cache=False):
        B, T = x.shape

        past_length = 0
        if use_cache and past_key_values:
            for pkv in past_key_values:
                if pkv is not None:
                    past_length = pkv[0].size(2)
                    break

        positions = torch.arange(past_length, past_length + T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embed(x) + self.pos_embed(positions)

        attn_mask = None
        if not use_cache:
            attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        presents = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i]
            if use_cache:
                x, present = block(x, attn_mask=attn_mask, past_key_value=past, use_cache=True)
                presents.append(present)
            else:
                x = block(x, attn_mask=attn_mask)

        logits = self.out(x)

        if use_cache:
            return logits, presents

        return logits
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
