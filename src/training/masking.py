"""Span masking for masked pose modelling."""
import math

import torch


def block_span_mask(lengths: torch.Tensor, t_max: int, block: int = 4, ratio: float = 0.4,
                    span: int = 2, generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Hide ~`ratio` of each clip's real frames, in contiguous spans of `span` blocks of `block`
    frames (block = encoder subsample factor, so hidden regions line up with encoder steps).
    Hiding spans instead of single frames stops the model from just interpolating neighbours.

    returns (B, t_max) bool, True = hidden. Never hides padding.
    """
    lengths = lengths.cpu()
    out = torch.zeros(len(lengths), t_max, dtype=torch.bool)
    for i, length in enumerate(lengths.tolist()):
        n_blocks = math.ceil(length / block)
        target = max(1, round(ratio * n_blocks))
        hidden = torch.zeros(n_blocks, dtype=torch.bool)
        for start in torch.randperm(n_blocks, generator=generator).tolist():
            hidden[start : start + span] = True
            if int(hidden.sum()) >= target:
                break
        frames = hidden.repeat_interleave(block)[:length]
        out[i, :length] = frames
    return out
