import torch
import torch.nn as nn
import math


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + self.drop(attn_out))
        # Feed-forward with residual
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class DecisionTransformerModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        context_len: int = 20,
        max_ep_len: int = 1000,
        action_tanh: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.d_model = d_model
        self.context_len = context_len
        self.action_tanh = action_tanh

        # Embeddings
        self.rtg_embed = nn.Linear(1, d_model)
        self.obs_embed = nn.Linear(obs_dim, d_model)
        self.act_embed = nn.Linear(act_dim, d_model)
        self.timestep_embed = nn.Embedding(max_ep_len, d_model)

        self.embed_ln = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # Transformer
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, 4 * d_model, dropout) for _ in range(n_layers)]
        )

        # Output head for action prediction
        self.action_head = nn.Linear(d_model, act_dim)

    def forward(
        self,
        states: torch.Tensor,  # (B, T, obs_dim)
        actions: torch.Tensor,  # (B, T, act_dim)
        returns_to_go: torch.Tensor,  # (B, T, 1)
        timesteps: torch.Tensor,  # (B, T) long
    ) -> torch.Tensor:
        B, T, _ = states.shape

        # Embed each modality
        t_embed = self.timestep_embed(timesteps)  # (B, T, d_model)

        rtg_emb = self.embed_drop(self.embed_ln(self.rtg_embed(returns_to_go) + t_embed))
        obs_emb = self.embed_drop(self.embed_ln(self.obs_embed(states) + t_embed))
        act_emb = self.embed_drop(self.embed_ln(self.act_embed(actions) + t_embed))

        # Interleave: [rtg_0, obs_0, act_0, rtg_1, obs_1, act_1, ...]
        # Shape: (B, 3*T, d_model)
        seq = torch.stack([rtg_emb, obs_emb, act_emb], dim=2).reshape(B, 3 * T, self.d_model)

        # Causal mask
        seq_len = 3 * T
        mask = torch.triu(torch.ones(seq_len, seq_len, device=states.device), diagonal=1).bool()
        # Convert to float mask for nn.MultiheadAttention (True = ignore)

        x = seq
        for block in self.blocks:
            x = block(x, mask=mask.float() * -1e9)

        # Extract action predictions from obs positions (index 1, 4, 7, ...)
        obs_positions = torch.arange(1, 3 * T, 3, device=states.device)
        action_preds = self.action_head(x[:, obs_positions, :])  # (B, T, act_dim)

        if self.action_tanh:
            action_preds = torch.tanh(action_preds)

        return action_preds
