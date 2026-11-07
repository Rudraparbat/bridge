"""
PoseEncoder: (B, T, D) standardized pose features -> (B, T/4, d_model) hidden states.

    per-part MLPs (body | L-hand | R-hand | face | presence)  -> concat -> d_model
    [optional: replace masked frames with a learned [MASK] vector]   (Stage 0)
    Conv subsampling x4 (two stride-2 convs)
    + sinusoidal positions
    N x Conformer block (FFN/2 -> self-attention -> depthwise conv -> FFN/2)
"""
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    group_names: list[str] = field(default_factory=list)
    group_dims: list[int] = field(default_factory=list)   # columns per group, in input order
    part_dim: int = 64          # width of each per-part MLP
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    ff_mult: int = 4
    conv_kernel: int = 15       # depthwise conv over ~15 subsampled steps (~2.4 s at 25 fps / 4)
    dropout: float = 0.1
    subsample: int = 4          # power of 2
    max_len: int = 2048         # max subsampled length for positional encoding

    @property
    def in_dim(self) -> int:
        return sum(self.group_dims)


# --------------------------------------------------------------------------- #
class PartEmbedding(nn.Module):
    """A small MLP per body part so the hands are not drowned out by larger groups."""

    def __init__(self, group_dims, part_dim, d_model):
        super().__init__()
        self.group_dims = list(group_dims)
        self.parts = nn.ModuleList(
            nn.Sequential(nn.Linear(d, part_dim), nn.GELU(), nn.Linear(part_dim, part_dim))
            for d in self.group_dims
        )
        self.merge = nn.Sequential(nn.Linear(part_dim * len(self.group_dims), d_model), nn.LayerNorm(d_model))

    def forward(self, x):
        chunks = torch.split(x, self.group_dims, dim=-1)
        return self.merge(torch.cat([mlp(c) for mlp, c in zip(self.parts, chunks)], dim=-1))


class ConvSubsampler(nn.Module):
    """Stride-2 convs: T -> ceil(T/2) per stage; shorter sequences = faster attention."""

    def __init__(self, d_model, factor):
        super().__init__()
        n_stages = int(math.log2(factor))
        if 2 ** n_stages != factor:
            raise ValueError("subsample must be a power of 2")
        self.stages = nn.ModuleList(
            nn.Sequential(nn.Conv1d(d_model, d_model, 3, stride=2, padding=1), nn.GELU())
            for _ in range(n_stages)
        )

    def forward(self, h, lengths):
        h = h.transpose(1, 2)                      # (B, d, T)
        for stage in self.stages:
            h = stage(h)
            lengths = (lengths + 1) // 2
        return h.transpose(1, 2), lengths


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, h):
        if h.shape[1] > self.pe.shape[0]:
            raise ValueError(f"sequence of {h.shape[1]} steps exceeds max_len {self.pe.shape[0]}")
        return h + self.pe[: h.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, d, mult, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d * mult), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(d * mult, d), nn.Dropout(dropout))

    def forward(self, h):
        return self.net(h)


class ConvModule(nn.Module):
    """Captures local motion (handshape transitions) that attention can blur."""

    def __init__(self, d, kernel, dropout):
        super().__init__()
        self.norm_in = nn.LayerNorm(d)
        self.pointwise_in = nn.Linear(d, 2 * d)
        self.depthwise = nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
        self.norm_mid = nn.LayerNorm(d)
        self.pointwise_out = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, mask):
        y = nn.functional.glu(self.pointwise_in(self.norm_in(h)), dim=-1)
        y = y.masked_fill(~mask[..., None], 0.0)             # padding must not leak into the conv
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        y = nn.functional.silu(self.norm_mid(y))
        return self.dropout(self.pointwise_out(y))


class ConformerBlock(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d = cfg.d_model
        self.ff1 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.attn_norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.conv = ConvModule(d, cfg.conv_kernel, cfg.dropout)
        self.ff2 = FeedForward(d, cfg.ff_mult, cfg.dropout)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, h, mask):
        h = h + 0.5 * self.ff1(h)
        a = self.attn_norm(h)
        a, _ = self.attn(a, a, a, key_padding_mask=~mask, need_weights=False)
        h = h + self.attn_drop(a)
        h = h + self.conv(h, mask)
        h = h + 0.5 * self.ff2(h)
        return self.out_norm(h)


# --------------------------------------------------------------------------- #
class PoseEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.part_embed = PartEmbedding(cfg.group_dims, cfg.part_dim, cfg.d_model)
        self.mask_token = nn.Parameter(torch.zeros(cfg.d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.subsample = ConvSubsampler(cfg.d_model, cfg.subsample)
        self.positions = SinusoidalPositions(cfg.d_model, cfg.max_len)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(ConformerBlock(cfg) for _ in range(cfg.n_layers))

    @property
    def out_dim(self) -> int:
        return self.cfg.d_model

    def forward(self, x, mask, frame_mask=None):
        """
        x:          (B, T, in_dim) standardized features
        mask:       (B, T) bool, True = real frame
        frame_mask: (B, T) bool, True = hide this frame (Stage 0 masked modelling only)
        returns     h (B, T', d_model), out_mask (B, T') with T' = ceil(T / subsample)
        """
        h = self.part_embed(x)
        if frame_mask is not None:
            h = torch.where(frame_mask[..., None], self.mask_token.to(h.dtype), h)
        h = h.masked_fill(~mask[..., None], 0.0)

        h, lengths = self.subsample(h, mask.sum(dim=1))
        out_mask = torch.arange(h.shape[1], device=h.device)[None, :] < lengths[:, None]

        h = self.dropout(self.positions(h))
        for block in self.blocks:
            h = block(h, out_mask)
        return h, out_mask

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    @classmethod
    def from_checkpoint(cls, path, device="cpu", eval_mode: bool = True) -> "PoseEncoder":
        ckpt = torch.load(Path(path), map_location=device, weights_only=False)
        enc = cls(EncoderConfig(**ckpt["encoder_config"]))
        enc.load_state_dict(ckpt["encoder_state_dict"])
        enc.to(device)
        if eval_mode:
            enc.eval()
        return enc
