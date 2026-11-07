"""Deterministic train/val/test split by video_id (no clip of a video leaks across splits)."""
import hashlib


class VideoSplitter:
    """
    split = bucket(md5(video_id) mod 100):  < train_pct -> train, < val_pct -> val, else test.

    A hash (not a shuffle) means a video's split never changes, even when more clips
    are downloaded later, and anyone can reproduce it without a shared file.
    """

    def __init__(self, train_pct: int = 80, val_pct: int = 90):
        if not 0 < train_pct < val_pct < 100:
            raise ValueError("need 0 < train_pct < val_pct < 100")
        self.train_pct, self.val_pct = train_pct, val_pct

    @staticmethod
    def video_id(uid: str) -> str:
        """'Mj0pgX8-2Xs--12' -> 'Mj0pgX8-2Xs-'; uids without a '-N' suffix are their own video."""
        return uid.rsplit("-", 1)[0]

    @staticmethod
    def bucket(video_id: str) -> int:
        return int(hashlib.md5(video_id.encode("utf-8")).hexdigest(), 16) % 100

    def split_of(self, video_id: str) -> str:
        b = self.bucket(video_id)
        if b < self.train_pct:
            return "train"
        if b < self.val_pct:
            return "val"
        return "test"
