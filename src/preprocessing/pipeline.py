"""
Dataset-level preprocessing: raw .pose files -> processed .npy + metadata + split + stats.

Outputs (in out_dir, default data/):
    processed/<uid>.npy        (frames, 492) float32, see layout.py
    metadata.csv               kept clips: uid, video_id, split, text, num_frames, ...
    dropped.csv                removed clips and the reason
    stats.npz                  per-column mean/std from TRAIN clips only
    feature_layout.json        column map
    splits/{train,val,test}_video_ids.txt
"""
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..config import PreprocessConfig
from .layout import FeatureLayout
from .processor import PoseProcessor
from .splitter import VideoSplitter
from .stats import RunningStats

# --------------------------------------------------------------------------- #
# Worker process helpers (top-level so they can be pickled on macOS "spawn")
# --------------------------------------------------------------------------- #
_PROCESSOR: PoseProcessor | None = None


def _init_worker(target_fps: float, conf_thr: float) -> None:
    global _PROCESSOR
    _PROCESSOR = PoseProcessor(target_fps, conf_thr)


def _process_clip(job: tuple[str, str, str]) -> tuple[str, str | None]:
    uid, pose_path, out_path = job
    try:
        features = _PROCESSOR.process_file(Path(pose_path))
        out = Path(out_path)
        tmp = out.with_name(f".{out.stem}.tmp.npy")
        np.save(tmp, features)
        tmp.replace(out)  # atomic: no half-written files
        return uid, None
    except Exception as e:
        return uid, repr(e)


# --------------------------------------------------------------------------- #
class PreprocessPipeline:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.layout = FeatureLayout()
        self.splitter = VideoSplitter(cfg.train_pct, cfg.val_pct)
        self.dropped: list[dict] = []

    # ---- step 1: labels + which clips we have ------------------------------ #
    @staticmethod
    def clean_text(text) -> str:
        if not isinstance(text, str):
            return ""
        return re.sub(r"\s+", " ", text).strip()

    def build_catalog(self) -> pd.DataFrame:
        labels = pd.read_csv(self.cfg.labels_csv)
        on_disk = {p.stem for p in self.cfg.pose_dir.glob("*.pose")}
        df = labels[labels["uid"].isin(on_disk)].copy()
        df["text"] = df["text"].map(self.clean_text)
        df["video_id"] = df["uid"].map(self.splitter.video_id)
        df["split"] = df["video_id"].map(self.splitter.split_of)

        no_text = df["text"] == ""
        self._drop(df.loc[no_text, "uid"], "no_text")
        df = df[~no_text].sort_values("uid").reset_index(drop=True)
        print(f"Catalog: {len(on_disk)} .pose files on disk | {len(df)} with text | "
              f"{int(no_text.sum())} without text")
        if self.cfg.max_clips:
            df = df.head(self.cfg.max_clips)
            print(f"(debug) limited to the first {len(df)} clips")
        return df

    # ---- step 2: .pose -> .npy (parallel, resumable) ----------------------- #
    def npy_path(self, uid: str) -> Path:
        return self.cfg.processed_dir / f"{uid}.npy"

    def process_clips(self, df: pd.DataFrame) -> None:
        self.cfg.processed_dir.mkdir(parents=True, exist_ok=True)
        jobs = [(u, str(self.cfg.pose_dir / f"{u}.pose"), str(self.npy_path(u)))
                for u in df["uid"] if self.cfg.overwrite or not self.npy_path(u).exists()]
        print(f"To process: {len(jobs)} (already done: {len(df) - len(jobs)}) using {self.cfg.workers} workers")
        if not jobs:
            return
        failures = []
        with ProcessPoolExecutor(self.cfg.workers, initializer=_init_worker,
                                 initargs=(self.cfg.target_fps, self.cfg.confidence_threshold)) as pool:
            for uid, err in tqdm(pool.map(_process_clip, jobs, chunksize=16), total=len(jobs), unit="clip"):
                if err:
                    failures.append(uid)
                    tqdm.write(f"Failed {uid}: {err}")
        self._drop(failures, "processing_error")

    # ---- step 3: quality filter + train stats (one pass over the .npy) ----- #
    def index_and_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        failed = {d["uid"] for d in self.dropped if d["reason"] == "processing_error"}
        pres_l, pres_r, _ = range(*self.layout.slices["presence"])
        stats = RunningStats(self.layout.dim)
        rows = []
        for r in tqdm(df.itertuples(index=False), total=len(df), desc="indexing", unit="clip"):
            if r.uid in failed:
                continue
            arr = np.load(self.npy_path(r.uid), mmap_mode="r")
            n = arr.shape[0]
            left, right = float(arr[:, pres_l].mean()), float(arr[:, pres_r].mean())
            if n < self.cfg.min_frames:
                self._drop([r.uid], "too_short")
                continue
            if max(left, right) < self.cfg.min_hand_ratio:
                self._drop([r.uid], "no_hands")
                continue
            if r.split == "train":
                stats.update(arr)
            rows.append({"uid": r.uid, "video_id": r.video_id, "split": r.split, "text": r.text,
                         "num_frames": n, "duration_s": round(n / self.cfg.target_fps, 2),
                         "left_hand_ratio": round(left, 3), "right_hand_ratio": round(right, 3)})
        presence_cols = self.layout.columns("presence")
        stats.save(self.cfg.stats_path, skip_columns=presence_cols)
        return pd.DataFrame(rows)

    # ---- step 4: write everything ------------------------------------------ #
    def write_outputs(self, meta: pd.DataFrame) -> None:
        meta.to_csv(self.cfg.metadata_csv, index=False)
        pd.DataFrame(self.dropped, columns=["uid", "reason"]).to_csv(self.cfg.dropped_csv, index=False)
        layout = self.layout.to_dict() | {"target_fps": self.cfg.target_fps}
        self.cfg.layout_path.write_text(json.dumps(layout, indent=2))
        self.cfg.splits_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            ids = sorted(meta.loc[meta["split"] == split, "video_id"].unique())
            (self.cfg.splits_dir / f"{split}_video_ids.txt").write_text("\n".join(ids) + "\n")

    def _drop(self, uids, reason: str) -> None:
        self.dropped.extend({"uid": u, "reason": reason} for u in uids)

    # ---- run all ------------------------------------------------------------ #
    def run(self) -> pd.DataFrame:
        start = time.time()
        df = self.build_catalog()
        self.process_clips(df)
        meta = self.index_and_filter(df)
        self.write_outputs(meta)
        self._report(meta, time.time() - start)
        return meta

    def _report(self, meta: pd.DataFrame, seconds: float) -> None:
        print(f"\nDone in {seconds / 60:.1f} min. Kept {len(meta)} clips, dropped {len(self.dropped)}.")
        if self.dropped:
            print("Dropped by reason:", pd.Series([d["reason"] for d in self.dropped]).value_counts().to_dict())
        summary = meta.groupby("split").agg(videos=("video_id", "nunique"), clips=("uid", "size"),
                                            median_frames=("num_frames", "median"))
        summary["% clips"] = (100 * summary["clips"] / summary["clips"].sum()).round(1)
        print(summary.reindex(["train", "val", "test"]).to_string())
        size_gb = sum(self.npy_path(u).stat().st_size for u in meta["uid"]) / 1e9
        print(f"Features: {self.layout.dim} columns/frame | {size_gb:.2f} GB in {self.cfg.processed_dir}")
        print(f"Outputs: {self.cfg.metadata_csv.name}, {self.cfg.dropped_csv.name}, "
              f"{self.cfg.stats_path.name}, {self.cfg.layout_path.name}, splits/")
