"""
Small, single-purpose transforms applied to one clip.

Arrays are (frames, points, 3) float32 unless stated otherwise.
"""
import numpy as np


class GapFiller:
    """Fill frames where a landmark was not detected by linear interpolation over time.

    Points never detected in the whole clip cannot be interpolated; they are reported in
    `never_seen` so the caller can zero them after normalization.
    """

    def __call__(self, xyz: np.ndarray, valid: np.ndarray):
        frames = xyz.shape[0]
        out = xyz.copy()
        t = np.arange(frames)
        counts = valid.sum(axis=0)
        never_seen = counts == 0
        for p in np.where((counts > 0) & (counts < frames))[0]:
            v = valid[:, p]
            for d in range(xyz.shape[2]):
                out[:, p, d] = np.interp(t, t[v], xyz[v, p, d])  # edges hold the nearest value
        out[:, never_seen] = 0.0
        return out, never_seen


class FrameRateResampler:
    """Resample any (frames, ...) array to a fixed frame rate with linear interpolation."""

    def __init__(self, target_fps: float = 25.0):
        self.target_fps = target_fps

    def __call__(self, arr: np.ndarray, src_fps: float) -> np.ndarray:
        frames = arr.shape[0]
        if frames < 2 or abs(src_fps - self.target_fps) < 0.5:
            return arr
        duration = (frames - 1) / src_fps
        n_out = int(np.floor(duration * self.target_fps)) + 1
        pos = np.arange(n_out) * (src_fps / self.target_fps)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, frames - 1)
        w = (pos - lo).reshape(-1, *([1] * (arr.ndim - 1)))
        return ((1.0 - w) * arr[lo] + w * arr[hi]).astype(np.float32)


class ShoulderNormalizer:
    """Center every point on the shoulder midpoint (per frame) and divide by shoulder width.

    Width uses the clip median, so per-frame detection jitter does not rescale the signer.
    Result: camera position, zoom and video resolution no longer matter.
    """

    def __init__(self, left_idx: int, right_idx: int):
        self.left_idx, self.right_idx = left_idx, right_idx

    def __call__(self, xyz: np.ndarray) -> np.ndarray:
        left, right = xyz[:, self.left_idx, :2], xyz[:, self.right_idx, :2]
        origin = (left + right) / 2.0
        width = np.linalg.norm(left - right, axis=-1)
        good = width[width > 1e-6]
        scale = float(np.median(good)) if good.size else 1.0
        out = xyz.copy()
        out[..., :2] -= origin[:, None, :]
        return out / scale


class FaceLocalizer:
    """Face points relative to the nose tip, scaled by outer-eye distance.

    Makes small expressions (eyebrow raise, mouth opening) visible independent of head motion.
    """

    def __init__(self, nose_idx: int, eye_right_idx: int, eye_left_idx: int):
        self.nose_idx, self.eye_right_idx, self.eye_left_idx = nose_idx, eye_right_idx, eye_left_idx

    def __call__(self, face_xyz: np.ndarray) -> np.ndarray:
        nose = face_xyz[:, self.nose_idx]
        width = np.linalg.norm(face_xyz[:, self.eye_right_idx, :2] - face_xyz[:, self.eye_left_idx, :2], axis=-1)
        good = width[width > 1e-6]
        scale = float(np.median(good)) if good.size else 1.0
        return (face_xyz - nose[:, None, :]) / scale
