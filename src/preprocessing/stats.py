"""Per-feature mean/std, accumulated over the TRAIN split only."""
from pathlib import Path

import numpy as np


class RunningStats:
    """Streaming mean/std over many (frames, dim) arrays without holding them in memory."""

    def __init__(self, dim: int):
        self.count = 0
        self.sum = np.zeros(dim, dtype=np.float64)
        self.sumsq = np.zeros(dim, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        self.count += x.shape[0]
        self.sum += x.sum(axis=0)
        self.sumsq += np.square(x).sum(axis=0)

    def finalize(self, skip_columns=()) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std). `skip_columns` (e.g. 0/1 flags) get mean 0, std 1."""
        if self.count == 0:
            raise ValueError("no data accumulated")
        mean = self.sum / self.count
        var = np.maximum(self.sumsq / self.count - mean ** 2, 0.0)
        std = np.sqrt(var)
        std[std < 1e-6] = 1.0                     # constant columns: leave unscaled
        skip = list(skip_columns)
        mean[skip], std[skip] = 0.0, 1.0
        return mean.astype(np.float32), std.astype(np.float32)

    def save(self, path: Path, skip_columns=()) -> None:
        mean, std = self.finalize(skip_columns)
        np.savez(path, mean=mean, std=std, frames=self.count)
