"""Stage 0 training heads (discarded after pretraining; only the encoder is kept)."""
import torch
import torch.nn as nn


class ReconstructionHead(nn.Module):
    """Each subsampled step predicts the `subsample` input frames it covers."""

    def __init__(self, d_model: int, out_dim: int, subsample: int):
        super().__init__()
        self.out_dim, self.subsample = out_dim, subsample
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                 nn.Linear(d_model, subsample * out_dim))

    def forward(self, h: torch.Tensor, t_out: int) -> torch.Tensor:
        b, t_sub, _ = h.shape
        y = self.net(h).reshape(b, t_sub * self.subsample, self.out_dim)
        return y[:, :t_out]                                   # (B, T, out_dim)


class WordPresenceHead(nn.Module):
    """Pools the clip and predicts which vocabulary words occur in its sentence."""

    def __init__(self, d_model: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(d_model, 1)                    # attention pooling weights
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(d_model, vocab_size))

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(dim=-1)
        pooled = (w.unsqueeze(-1) * h).sum(dim=1)
        return self.net(pooled)                               # (B, vocab) logits
