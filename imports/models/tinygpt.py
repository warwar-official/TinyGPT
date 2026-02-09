import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, emb_size, n_head, dropout=0.15, window_size=256, use_windowed=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = emb_size // n_head
        self.window_size = window_size
        self.use_windowed = use_windowed
        
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

    def create_windowed_mask(self, seq_len, device, global_mask=None):
        """
        Create attention mask with sliding window + global tokens (VECTORIZED - GPU friendly)
        
        Args:
            seq_len: sequence length
            device: torch device
            global_mask: [seq_len] boolean tensor, True = global token
        
        Returns:
            mask: [seq_len, seq_len] attention mask (0 = attend, -inf = mask)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)
        row_idx = positions.unsqueeze(1)  # [seq_len, 1]
        col_idx = positions.unsqueeze(0)  # [1, seq_len]
        
        # Causal mask: can only attend to past (col <= row)
        causal = col_idx <= row_idx
        
        # Local window: distance within window_size
        distance = row_idx - col_idx
        local_window = (distance >= 0) & (distance < self.window_size)
        
        # Combine: within window AND causal
        mask = local_window & causal
        
        if global_mask is not None and global_mask.any():
            # Global tokens can attend to everything in the past
            is_global_row = global_mask.unsqueeze(1)  # [seq_len, 1]
            global_attention = is_global_row & causal
            
            # All tokens can attend to global tokens in the past
            is_global_col = global_mask.unsqueeze(0)  # [1, seq_len]
            attend_to_global = is_global_col & causal
            
            # Combine all: local OR (global row) OR (global col)
            mask = mask | global_attention | attend_to_global
        
        # Convert boolean mask to attention mask format
        # True = can attend (0), False = cannot attend (-inf)
        attn_mask = torch.zeros((seq_len, seq_len), device=device)
        attn_mask = attn_mask.masked_fill(~mask, float('-inf'))
        
        return attn_mask

    def forward(self, x, attn_mask=None, global_mask=None, past_key_value=None, use_cache=False):
        """
        Args:
            x: input tensor [B, T, C]
            attn_mask: optional custom attention mask
            global_mask: [T] boolean tensor indicating global positions
            past_key_value: cached keys and values for generation
            use_cache: whether to return cache
        """
        residual = x
        x_ = self.ln1(x)

        q = self.q_proj(x_)
        k = self.k_proj(x_)
        v = self.v_proj(x_)

        B, T, C = q.size()
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            past_len = past_k.size(2)
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply attention mask
        if attn_mask is not None:
            # Custom mask provided
            attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
        elif self.use_windowed and not use_cache:
            # Use windowed attention with global tokens
            windowed_mask = self.create_windowed_mask(T, x.device, global_mask)
            attn_scores = attn_scores + windowed_mask.unsqueeze(0).unsqueeze(0)
        else:
            # Standard causal mask
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

        x = residual + self.alpha_attn * attn_output

        x_ = self.ln2(x)
        ff_output = self.ff(x_)
        x = x + self.alpha_ff * ff_output

        self.last_attention = attn_weight

        if use_cache:
            present = (k, v)
            return x, present

        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, emb_size=384, n_head=6, num_layers=7, max_len=768,
                 window_size=256, use_windowed=False, global_period=64):
        """
        Args:
            vocab_size: vocabulary size
            emb_size: embedding dimension
            n_head: number of attention heads
            num_layers: number of transformer blocks
            max_len: maximum sequence length
            window_size: local attention window size (e.g., 256)
            use_windowed: whether to use windowed attention
            global_period: period for global tokens (every Nth token), 0 = no periodic global
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_size)
        self.pos_embed = nn.Embedding(max_len, emb_size)
        self.use_windowed = use_windowed
        self.global_period = global_period
        self.max_len = max_len

        self.blocks = nn.ModuleList([
            TransformerBlock(emb_size, n_head, window_size=window_size, use_windowed=use_windowed)
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

    def create_global_mask(self, seq_len, device, custom_global_positions=None):
        """
        Create global attention mask
        
        Args:
            seq_len: sequence length
            device: torch device
            custom_global_positions: optional list/tensor of custom global positions
        
        Returns:
            global_mask: [seq_len] boolean tensor
        """
        global_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        
        if custom_global_positions is not None:
            # Use custom positions
            if isinstance(custom_global_positions, list):
                custom_global_positions = torch.tensor(custom_global_positions, device=device)
            global_mask[custom_global_positions] = True
        elif self.global_period > 0:
            # Periodic global tokens
            global_positions = torch.arange(0, seq_len, self.global_period, device=device)
            global_mask[global_positions] = True
        
        return global_mask

    def forward(self, x, global_positions=None, past_key_values=None, use_cache=False):
        """
        Args:
            x: input tensor [B, T]
            global_positions: optional custom global token positions [list or tensor]
            past_key_values: cached keys/values for generation
            use_cache: whether to cache for generation
        """
        B, T = x.shape

        past_length = 0
        if use_cache and past_key_values:
            for pkv in past_key_values:
                if pkv is not None:
                    past_length = pkv[0].size(2)
                    break

        positions = torch.arange(past_length, past_length + T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embed(x) + self.pos_embed(positions)

        # Create global mask if using windowed attention
        global_mask = None
        if self.use_windowed and not use_cache:
            global_mask = self.create_global_mask(T, x.device, global_positions)

        attn_mask = None
        if not use_cache and not self.use_windowed:
            # Standard causal mask (only if not using windowed)
            attn_mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)

        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        presents = []
        for i, block in enumerate(self.blocks):
            past = past_key_values[i]
            if use_cache:
                x, present = block(x, attn_mask=attn_mask, global_mask=global_mask, 
                                  past_key_value=past, use_cache=True)
                presents.append(present)
            else:
                x = block(x, attn_mask=attn_mask, global_mask=global_mask)

        logits = self.out(x)

        if use_cache:
            return logits, presents

        return logits
    
    def get_num_params(self):
        """Return the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Example usage
if __name__ == "__main__":
    # Standard model (no windowed attention)
    model_standard = TinyGPT(
        vocab_size=50257,
        emb_size=384,
        n_head=6,
        num_layers=7,
        max_len=768,
        use_windowed=False
    )
    
    # Windowed attention model with periodic global tokens
    model_windowed = TinyGPT(
        vocab_size=50257,
        emb_size=384,
        n_head=6,
        num_layers=7,
        max_len=2048,  # Can handle longer sequences now
        window_size=256,
        use_windowed=True,
        global_period=64  # Every 64th token is global
    )
    
    # Test forward pass
    batch_size = 2
    seq_len = 512
    x = torch.randint(0, 50257, (batch_size, seq_len))
    
    # Standard attention
    print("Standard attention:")
    logits = model_standard(x)
    print(f"Output shape: {logits.shape}")
    
    # Windowed attention with automatic periodic global tokens
    print("\nWindowed attention (periodic global):")
    logits = model_windowed(x)
    print(f"Output shape: {logits.shape}")
    
    # Windowed attention with custom global positions (e.g., first token only)
    print("\nWindowed attention (custom global positions):")
    logits = model_windowed(x, global_positions=[0])
    print(f"Output shape: {logits.shape}")
    
    print(f"\nModel parameters: {model_windowed.get_num_params():,}")