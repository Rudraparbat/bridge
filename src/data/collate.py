"""Pad a list of clips into one batch."""
from dataclasses import dataclass

import torch


@dataclass
class PoseBatch:
    x: torch.Tensor          # (B, T, D) standardized features, zero-padded
    mask: torch.Tensor       # (B, T) bool, True = real frame
    lengths: torch.Tensor    # (B,) number of real frames
    texts: list[str]
    uids: list[str]

    def to(self, device) -> "PoseBatch":
        return PoseBatch(self.x.to(device), self.mask.to(device), self.lengths.to(device),
                         self.texts, self.uids)


def collate_pose(items: list[dict]) -> PoseBatch:
    lengths = torch.tensor([it["x"].shape[0] for it in items], dtype=torch.long)
    t_max, dim = int(lengths.max()), items[0]["x"].shape[1]
    x = torch.zeros(len(items), t_max, dim)
    for i, it in enumerate(items):
        x[i, : it["x"].shape[0]] = it["x"]
    mask = torch.arange(t_max)[None, :] < lengths[:, None]
    return PoseBatch(x, mask, lengths, [it["text"] for it in items], [it["uid"] for it in items])
