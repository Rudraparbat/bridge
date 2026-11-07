"""Download raw .pose files (one per clip) into data/poses/. Resumable."""
import os
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ..config import ARCHIVE_PREFIX, DEFAULT_DATA_DIR, LABELS_CSV, POSE_PARTS, REPO_ID
from .remote_zip import RemoteMultipartZip, read_member


class PoseDownloader:
    def __init__(self, token: str, data_dir: Path = DEFAULT_DATA_DIR, workers: int = 8):
        self.token = token
        self.data_dir = Path(data_dir)
        self.pose_dir = self.data_dir / "poses"
        self.workers = workers
        self._local = threading.local()
        self._infos: dict[str, zipfile.ZipInfo] = {}
        self._part_sizes = None

    # ---- step 1: labels ---------------------------------------------------- #
    def load_labels(self) -> pd.DataFrame:
        csv_path = self.data_dir / LABELS_CSV
        if not csv_path.exists():
            print(f"Downloading {LABELS_CSV} ...")
            hf_hub_download(REPO_ID, LABELS_CSV, repo_type="dataset",
                            local_dir=str(self.data_dir), token=self.token)
        df = pd.read_csv(csv_path)
        df["video_id"] = df["uid"].str.rsplit("-", n=1).str[0]
        return df

    # ---- step 2: which clips ----------------------------------------------- #
    @staticmethod
    def select_clips(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
        """Pick whole videos (shuffled with `seed`) until ~`limit` clips are reached."""
        if not limit:
            return df
        video_ids = df["video_id"].unique()
        np.random.RandomState(seed).shuffle(video_ids)
        sizes = df.groupby("video_id").size()
        chosen, count = [], 0
        for v in video_ids:
            chosen.append(v)
            count += sizes[v]
            if count >= limit:
                break
        return df[df["video_id"].isin(chosen)]

    # ---- step 3: archive index --------------------------------------------- #
    def load_index(self) -> None:
        print("Reading remote archive index ...")
        root = RemoteMultipartZip(REPO_ID, POSE_PARTS, self.token)
        index = zipfile.ZipFile(root)
        self._infos = {Path(i.filename).stem: i for i in index.infolist()
                       if i.filename.startswith(ARCHIVE_PREFIX) and i.filename.endswith(".pose")}
        self._part_sizes = root.sizes()
        print(f"Archive: {root.total_size / 1e9:.1f} GB, {len(self._infos)} pose files")

    def _is_done(self, uid: str) -> bool:
        out = self.pose_dir / f"{uid}.pose"
        return out.exists() and out.stat().st_size == self._infos[uid].file_size

    # ---- step 4: download -------------------------------------------------- #
    def _reader(self) -> RemoteMultipartZip:
        if not hasattr(self._local, "reader"):  # one HTTP session per thread
            self._local.reader = RemoteMultipartZip(REPO_ID, POSE_PARTS, self.token,
                                                    part_sizes=self._part_sizes)
        return self._local.reader

    def _download_one(self, uid: str):
        try:
            raw = read_member(self._reader(), self._infos[uid])
            tmp = self.pose_dir / f".{uid}.pose.tmp"
            tmp.write_bytes(raw)
            tmp.replace(self.pose_dir / f"{uid}.pose")  # atomic
            return uid, None
        except Exception as e:
            return uid, repr(e)

    def run(self, limit: int | None = None, seed: int = 42) -> None:
        self.pose_dir.mkdir(parents=True, exist_ok=True)
        df = self.select_clips(self.load_labels(), limit, seed)
        self.load_index()

        uids = df["uid"].tolist()
        missing = [u for u in uids if u not in self._infos]
        present = [u for u in uids if u in self._infos]
        todo = [u for u in present if not self._is_done(u)]
        todo_bytes = sum(self._infos[u].file_size for u in todo)
        print(f"Selected {len(uids)} clips from {df['video_id'].nunique()} videos | "
              f"already downloaded: {len(present) - len(todo)} | "
              f"to download: {len(todo)} ({todo_bytes / 1e9:.2f} GB) | not in archive: {len(missing)}")
        if not todo:
            print("Nothing to do.")
            return

        failed, start = [], time.time()
        try:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futures = [pool.submit(self._download_one, u) for u in todo]
                with tqdm(total=todo_bytes, unit="B", unit_scale=True, unit_divisor=1024) as bar:
                    for fut in as_completed(futures):
                        uid, err = fut.result()
                        if err:
                            failed.append(uid)
                            tqdm.write(f"Failed {uid}: {err}")
                        bar.update(self._infos[uid].file_size)
        except KeyboardInterrupt:
            print("\nStopped. Finished files are saved - rerun the same command to continue.")
            os._exit(1)

        for tmp in self.pose_dir.glob(".*.tmp"):
            tmp.unlink(missing_ok=True)
        print(f"\nDone in {(time.time() - start) / 60:.1f} min. "
              f"Downloaded {len(todo) - len(failed)}, failed {len(failed)} (rerun to retry).")
