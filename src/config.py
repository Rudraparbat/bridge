"""Project-wide constants and configuration objects."""
import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

# --------------------------------------------------------------------------- #
# iSign on Hugging Face
# --------------------------------------------------------------------------- #
REPO_ID = "Exploration-Lab/iSign"
LABELS_CSV = "iSign_v1.1.csv"
POSE_PARTS = (
    "iSign-poses_v1.1_part_aa",
    "iSign-poses_v1.1_part_ab",
    "iSign-poses_v1.1_part_ac",
    "iSign-poses_v1.1_part_ad",
)
ARCHIVE_PREFIX = "iSign-poses_v1.1"

# --------------------------------------------------------------------------- #
# Landmark selection (MediaPipe Holistic layout used by iSign .pose files)
# --------------------------------------------------------------------------- #
BODY = "POSE_LANDMARKS"            # 33 points
LEFT_HAND = "LEFT_HAND_LANDMARKS"  # 21 points
RIGHT_HAND = "RIGHT_HAND_LANDMARKS"  # 21 points
FACE = "FACE_LANDMARKS"            # 468 points (we keep a subset)

# MediaPipe Face Mesh indices that carry sign-language information.
FACE_SUBSET = {
    "lips_outer": (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                   291, 409, 270, 269, 267, 0, 37, 39, 40, 185),
    "lips_inner": (78, 308, 13, 14),
    "brow_right": (70, 63, 105, 66, 107),
    "brow_left": (336, 296, 334, 293, 300),
    "eye_right": (33, 133, 159, 145),
    "eye_left": (362, 263, 386, 374),
    "head": (1, 152),  # nose tip, chin
}
FACE_POINTS = tuple(i for group in FACE_SUBSET.values() for i in group)  # 44 points
FACE_NOSE, FACE_EYE_RIGHT_OUTER, FACE_EYE_LEFT_OUTER = 1, 33, 263



@dataclass
class PreprocessConfig:
    data_dir: Path = DEFAULT_DATA_DIR      # where poses/ and the labels CSV live
    out_dir: Path | None = None            # where outputs go (defaults to data_dir)

    target_fps: float = 25.0               # every clip is resampled to this rate
    confidence_threshold: float = 1e-6     # below this a landmark counts as "not detected"
    min_frames: int = 16                   # shorter clips (after resampling) are dropped
    min_hand_ratio: float = 0.2            # drop clips where no hand is seen in >= 20% of frames

    train_pct: int = 80                    # hash bucket < 80  -> train
    val_pct: int = 90                      # hash bucket < 90  -> val, else test

    workers: int = field(default_factory=lambda: max(1, (os.cpu_count() or 2) - 1))
    overwrite: bool = False                # re-process clips that already have a .npy
    max_clips: int | None = None           # debug: only process the first N clips

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.out_dir = Path(self.out_dir) if self.out_dir else self.data_dir

    # inputs
    @property
    def pose_dir(self) -> Path:
        return self.data_dir / "poses"

    @property
    def labels_csv(self) -> Path:
        return self.data_dir / LABELS_CSV

    # outputs
    @property
    def processed_dir(self) -> Path:
        return self.out_dir / "processed"

    @property
    def metadata_csv(self) -> Path:
        return self.out_dir / "metadata.csv"

    @property
    def dropped_csv(self) -> Path:
        return self.out_dir / "dropped.csv"

    @property
    def stats_path(self) -> Path:
        return self.out_dir / "stats.npz"

    @property
    def layout_path(self) -> Path:
        return self.out_dir / "feature_layout.json"

    @property
    def splits_dir(self) -> Path:
        return self.out_dir / "splits"
