"""Turn one raw .pose file into a (frames x 492) float32 feature matrix."""
from pathlib import Path

import numpy as np
from pose_format import Pose

from .layout import FeatureLayout, PointSelector
from .transforms import FaceLocalizer, FrameRateResampler, GapFiller, ShoulderNormalizer


class PoseProcessor:
    """
    Pipeline for a single clip:

        read .pose -> select 119 points (body, hands, 44 face) -> presence flags
        -> fill gaps -> resample to target fps -> shoulder-normalize
        -> face-local copy -> flatten to (frames, 492)
    """

    def __init__(self, target_fps: float = 25.0, confidence_threshold: float = 1e-6):
        self.selector = PointSelector()
        self.layout = FeatureLayout(n_face=len(self.selector.face_points))
        self.gap_filler = GapFiller()
        self.resampler = FrameRateResampler(target_fps)
        self.conf_thr = confidence_threshold

    def process_file(self, path: Path) -> np.ndarray:
        return self.process_bytes(Path(path).read_bytes())

    def process_bytes(self, raw: bytes) -> np.ndarray:
        pose = Pose.read(raw)
        sel = self.selector(pose.header)

        data = np.ma.filled(pose.body.data, 0.0).astype(np.float32)[:, 0]          # (F, 576, 3), person 0
        conf = np.asarray(np.ma.filled(pose.body.confidence, 0.0), np.float32)[:, 0]  # (F, 576)
        idx = list(sel.indices)
        xyz, valid = data[:, idx], conf[:, idx] > self.conf_thr                       # (F, 119, 3), (F, 119)

        presence = np.stack([valid[:, sel.left_hand].any(1),
                             valid[:, sel.right_hand].any(1),
                             valid[:, sel.face].any(1)], axis=1).astype(np.float32)

        xyz, never_seen = self.gap_filler(xyz, valid)

        fps = float(pose.body.fps)
        xyz = self.resampler(xyz, fps)
        presence = (self.resampler(presence, fps) >= 0.5).astype(np.float32)

        normalized = ShoulderNormalizer(sel.left_shoulder, sel.right_shoulder)(xyz)
        normalized[:, never_seen] = 0.0

        face_local = FaceLocalizer(sel.face_nose, sel.face_eye_right, sel.face_eye_left)(xyz[:, sel.face])
        face_local[:, never_seen[sel.face]] = 0.0

        frames = normalized.shape[0]
        features = np.concatenate([
            normalized.reshape(frames, -1),   # body | left_hand | right_hand | face
            face_local.reshape(frames, -1),
            presence,
        ], axis=1).astype(np.float32)

        if features.shape[1] != self.layout.dim:
            raise ValueError(f"expected {self.layout.dim} columns, got {features.shape[1]}")
        if not np.isfinite(features).all():
            raise ValueError("non-finite values after preprocessing")
        return features
