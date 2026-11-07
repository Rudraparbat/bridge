"""
Stage 1 input when the encoder is frozen: precomputed embeddings from `python -m src.export`.

    ds = EmbeddingDataset("handoff", "train")
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_embeddings)
    for batch in loader:
        batch.h     # (B, T', 256) float32 encoder outputs, zero-padded
        batch.mask  # (B, T') bool, True = real step   -> attention_mask for the LLM
        batch.texts # list[str] target sentences
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(self, handoff_dir, split: str, max_clips: int | None = None):
        self.dir = Path(handoff_dir)
        self.split = split
        self.index = pd.read_csv(self.dir / f"index_{split}.csv")
        if max_clips:
            self.index = self.index.head(max_clips)
        self._emb = None  # opened lazily so DataLoader workers don't copy the whole array

    @property
    def emb(self) -> np.ndarray:
        if self._emb is None:
            self._emb = np.load(self.dir / f"embeddings_{self.split}.npy", mmap_mode="r")
        return self._emb

    @property
    def lengths(self) -> list[int]:
        return self.index["length"].tolist()

    @property
    def texts(self) -> list[str]:
        return self.index["text"].tolist()

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        r = self.index.iloc[i]
        h = np.asarray(self.emb[r["offset"] : r["offset"] + r["length"]], dtype=np.float32)
        return {"h": torch.from_numpy(h), "text": r["text"], "uid": r["uid"]}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_emb"] = None
        return state


@dataclass
class EmbeddingBatch:
    h: torch.Tensor
    mask: torch.Tensor
    texts: list[str]
    uids: list[str]

    def to(self, device) -> "EmbeddingBatch":
        return EmbeddingBatch(self.h.to(device), self.mask.to(device), self.texts, self.uids)


def collate_embeddings(items: list[dict]) -> EmbeddingBatch:
    lengths = torch.tensor([it["h"].shape[0] for it in items])
    t_max, dim = int(lengths.max()), items[0]["h"].shape[1]
    h = torch.zeros(len(items), t_max, dim)
    for i, it in enumerate(items):
        h[i, : it["h"].shape[0]] = it["h"]
    mask = torch.arange(t_max)[None, :] < lengths[:, None]
    return EmbeddingBatch(h, mask, [it["text"] for it in items], [it["uid"] for it in items])
