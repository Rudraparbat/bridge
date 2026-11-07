"""
ISLPoseDataset: one item = one clip.

    metadata.csv row  ->  processed/<uid>.npy (frames, 492)
                      ->  [augment]  ->  [crop]  ->  pick feature groups  ->  standardize
                      ->  {"x": (frames, D) float tensor, "text": str, "uid": str}
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import DEFAULT_DATA_DIR
from ..preprocessing.layout import FeatureLayout

# body + both hands + face expressions (nose-relative) + detection flags = 360 columns
DEFAULT_GROUPS = ("body", "left_hand", "right_hand", "face_local", "presence")


class ISLPoseDataset(Dataset):
    def __init__(self, split: str, data_dir: Path = DEFAULT_DATA_DIR,
                 feature_groups=DEFAULT_GROUPS, max_frames: int | None = 512,
                 random_crop: bool = False, augment=None, max_clips: int | None = None):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.split = split
        self.feature_groups = tuple(feature_groups)
        self.max_frames = max_frames
        self.random_crop = random_crop
        self.augment = augment

        meta = pd.read_csv(self.data_dir / "metadata.csv")
        meta = meta[meta["split"] == split].reset_index(drop=True)
        if max_clips:
            meta = meta.head(max_clips)
        if meta.empty:
            raise ValueError(f"no clips for split '{split}' in {self.data_dir / 'metadata.csv'}")
        self.meta = meta

        layout = FeatureLayout()
        self.columns = np.array(layout.columns(*self.feature_groups))
        self.group_dims = [layout.slices[g][1] - layout.slices[g][0] for g in self.feature_groups]

        stats = np.load(self.data_dir / "stats.npz")
        self.mean = stats["mean"][self.columns].astype(np.float32)
        self.std = stats["std"][self.columns].astype(np.float32)

    # ---- info used by the sampler / model ---------------------------------- #
    @property
    def feature_dim(self) -> int:
        return len(self.columns)

    @property
    def lengths(self) -> list[int]:
        n = self.meta["num_frames"].to_numpy()
        return list(np.minimum(n, self.max_frames) if self.max_frames else n)

    @property
    def texts(self) -> list[str]:
        return self.meta["text"].tolist()

    def __len__(self) -> int:
        return len(self.meta)

    # ---- one item ---------------------------------------------------------- #
    def _crop(self, arr: np.ndarray) -> np.ndarray:
        if not self.max_frames or arr.shape[0] <= self.max_frames:
            return arr
        if self.random_crop:
            start = int(torch.randint(0, arr.shape[0] - self.max_frames + 1, (1,)))
        else:
            start = (arr.shape[0] - self.max_frames) // 2
        return arr[start : start + self.max_frames]

    def __getitem__(self, idx: int) -> dict:
        row = self.meta.iloc[idx]
        arr = np.load(self.processed_dir / f"{row['uid']}.npy")  # (T, 492), unstandardized
        if self.augment is not None:
            arr = self.augment(arr)
        arr = self._crop(arr)
        x = (arr[:, self.columns] - self.mean) / self.std
        return {"x": torch.from_numpy(x.astype(np.float32)), "text": row["text"], "uid": row["uid"]}
