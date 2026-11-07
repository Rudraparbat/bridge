"""
Training-time augmentation on the raw processed array (frames, 492), BEFORE standardization.

- Spatial: one random rotation + scale + shear of the x,y plane, applied to every
  shoulder-normalized point (body, hands, face) around the shoulder midpoint (= origin).
- Temporal: random speed change (0.8x - 1.2x) by linear resampling.

Uses torch's RNG so each DataLoader worker gets different random values.
"""
import math

import numpy as np
import torch

from ..preprocessing.layout import FeatureLayout


class PoseAugment:
    SPATIAL_GROUPS = ("body", "left_hand", "right_hand", "face")

    def __init__(self, rotate_deg: float = 10.0, scale=(0.9, 1.1), shear: float = 0.1,
                 speed=(0.8, 1.2), p_spatial: float = 0.8, p_speed: float = 0.5):
        self.rotate = math.radians(rotate_deg)
        self.scale, self.shear, self.speed = scale, shear, speed
        self.p_spatial, self.p_speed = p_spatial, p_speed
        layout = FeatureLayout()
        self.spatial_slices = [layout.slices[g] for g in self.SPATIAL_GROUPS]
        self.presence = layout.slices["presence"]

    @staticmethod
    def _uniform(lo: float, hi: float) -> float:
        return lo + (hi - lo) * float(torch.rand(1))

    def _affine(self, arr: np.ndarray) -> np.ndarray:
        a = self._uniform(-self.rotate, self.rotate)
        s = self._uniform(*self.scale)
        sh = self._uniform(-self.shear, self.shear)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]], dtype=np.float32)
        m = (s * rot @ np.array([[1.0, sh], [0.0, 1.0]], dtype=np.float32)).astype(np.float32)
        out = arr.copy()
        frames = arr.shape[0]
        for start, end in self.spatial_slices:
            pts = out[:, start:end].reshape(frames, -1, 3)
            pts[..., :2] = pts[..., :2] @ m.T
            pts[..., 2] *= s
            out[:, start:end] = pts.reshape(frames, -1)
        return out

    def _speed(self, arr: np.ndarray) -> np.ndarray:
        factor = self._uniform(*self.speed)           # >1 = faster = fewer frames
        frames = arr.shape[0]
        n_out = max(2, int(round(frames / factor)))
        pos = np.linspace(0, frames - 1, n_out)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, frames - 1)
        w = (pos - lo)[:, None].astype(np.float32)
        out = (1 - w) * arr[lo] + w * arr[hi]
        ps, pe = self.presence
        out[:, ps:pe] = np.round(out[:, ps:pe])       # keep flags binary
        return out.astype(np.float32)

    def __call__(self, arr: np.ndarray) -> np.ndarray:
        if float(torch.rand(1)) < self.p_spatial:
            arr = self._affine(arr)
        if float(torch.rand(1)) < self.p_speed and arr.shape[0] > 4:
            arr = self._speed(arr)
        return arr
