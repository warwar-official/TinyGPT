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
        self.alpha_attn = nn.Parameter(torch.tensor(1e-3))
        self.alpha_ff = nn.Parameter(torch.tensor(1e-3))

    def forward(self, x, mask):
        x_ = self.ln1(x)  # Pre-LN для attention
        attn_output, _ = self.attn(x_, x_, x_, attn_mask=mask, average_attn_weights=False)
        x = x + self.alpha_attn * attn_output  # ReZero: α * F(x)

        x_ = self.ln2(x)  # Pre-LN для FFN
        ff_output = self.ff(x_)
        x = x + self.alpha_ff * ff_output      # ReZero: α * F(x)

        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_sizes, emb_size=384, n_head=6, num_layers=7, max_len=768, out_len=None):
        """
        vocab_sizes: list of vocab sizes, one per embedding type
        out_len: output layer size (vocab size for output). If None, uses vocab_sizes[0].
        """
        super().__init__()
        self.n_embeddings = len(vocab_sizes)
        self.emb_size = emb_size
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, emb_size) for vocab_size in vocab_sizes
        ])
        self.pos_embed = nn.Embedding(max_len, emb_size)


        # Attention-based fusion parameters (MLP + bias + temp + gating)
        self.fusion_attn = nn.Sequential(
            nn.Linear(emb_size, emb_size // 2),
            nn.GELU(),
            nn.Linear(emb_size // 2, 1, bias=False)
        )
        # 1. Learnable bias per embedding type
        self.fusion_bias = nn.Parameter(torch.zeros(self.n_embeddings))
        # 2. Learnable temperature parameter
        self.fusion_temp = nn.Parameter(torch.tensor(1.0))
        # 3. Gated fusion (sigmoid gate for each embedding type)
        self.fusion_gate = nn.Sequential(
            nn.Linear(emb_size, self.n_embeddings),
            nn.Sigmoid()
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_head)
            for _ in range(num_layers)
        ])

        if out_len is None:
            out_len = vocab_sizes[0]
        self.out = nn.Linear(emb_size, out_len)

        self.apply(self.init_weights)

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

    def forward(self, xs):
        """
        Forward pass with proper attention masking.
        
        Args:
            xs: List of input tensors [B, T]
            attention_mask: Unified attention mask [B, T] or None
        """
        B, T = xs[0].shape
        pos = torch.arange(T, device=xs[0].device).unsqueeze(0).expand(B, T)

        # Compute all embeddings: [n_embed, B, T, emb_size]
        embeds = [emb(x) for emb, x in zip(self.embeddings, xs)]
        embeds = torch.stack(embeds, dim=0)  # [n_embed, B, T, emb_size]

        # Add position embedding to each embedding for fusion attention
        pos_emb = self.pos_embed(pos)  # [B, T, emb_size]
        pos_emb_exp = pos_emb.unsqueeze(0)  # [1, B, T, emb_size]
        fusion_input = embeds + pos_emb_exp  # [n_embed, B, T, emb_size]

        # Compute attention weights for fusion (MLP + bias + temp)
        fusion_scores = self.fusion_attn(fusion_input)  # [n_embed, B, T, 1]
        fusion_scores = fusion_scores.squeeze(-1).permute(1,2,0)  # [B, T, n_embed]
        fusion_scores = fusion_scores + self.fusion_bias  # [B, T, n_embed] + [n_embed]
        fusion_weights = torch.softmax(fusion_scores / self.fusion_temp, dim=-1)  # [B, T, n_embed]

        # 3. Gated fusion: compute gates for each embedding type at each position
        # Use the mean embedding across all types as context for gating
        embeds_for_gate = embeds.mean(dim=0)  # [B, T, emb_size]
        gate_values = self.fusion_gate(embeds_for_gate)  # [B, T, n_embed]
        # 5. Dynamic fusion: multiply fusion_weights by gate values, then renormalize
        fusion_weights = fusion_weights * gate_values
        fusion_weights = fusion_weights / (fusion_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Weighted sum of embeddings (without pos_emb, only for fusion)
        embeds = embeds.permute(1,2,0,3)  # [B, T, n_embed, emb_size]
        x = torch.sum(fusion_weights.unsqueeze(-1) * embeds, dim=2)  # [B, T, emb_size]

        # Add position embedding after fusion
        x = x + pos_emb

        # Masking logic
        causal_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        logits = self.out(x)
        return logits
