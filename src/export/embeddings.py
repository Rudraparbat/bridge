"""
Run the trained (frozen) encoder over every clip once and save its outputs.

Hand-off folder layout (default: handoff/):
    embeddings_{train,val,test}.npy   (total_steps, 256) float16 - all clips of a split, stacked
    index_{train,val,test}.csv        uid, video_id, text, offset, length  (rows offset:offset+length)
    encoder_stage0.pt                 the encoder itself (for end-to-end fine-tuning / inference)
    stats.npz, feature_layout.json    needed to preprocess NEW clips for the encoder
    info.json                         shapes, settings, metrics
"""
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import DEFAULT_DATA_DIR, PROJECT_ROOT
from ..data import ISLPoseDataset, LengthBucketSampler, collate_pose
from ..models import PoseEncoder
from ..training.utils import pick_device

SPLITS = ("train", "val", "test")


def subsampled_length(frames: int, factor: int) -> int:
    """Encoder output length: each stride-2 stage maps T -> ceil(T/2)."""
    n = frames
    for _ in range(int(np.log2(factor))):
        n = (n + 1) // 2
    return n


class EmbeddingExporter:
    def __init__(self, checkpoint: Path, data_dir: Path = DEFAULT_DATA_DIR,
                 out_dir: Path = PROJECT_ROOT / "handoff", batch_size: int = 64, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.data_dir = Path(data_dir)
        self.out_dir = Path(out_dir)
        self.batch_size = batch_size
        self.device = pick_device(device)
        self.ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        self.encoder = PoseEncoder.from_checkpoint(self.checkpoint, device=self.device)  # eval mode
        self.groups = self.ckpt["feature_groups"]
        self.factor = self.encoder.cfg.subsample
        self.dim = self.encoder.out_dim

    @torch.no_grad()
    def export_split(self, split: str) -> dict:
        ds = ISLPoseDataset(split, self.data_dir, self.groups, max_frames=None, augment=None)
        index = ds.meta[["uid", "video_id", "text", "num_frames"]].copy()
        index["length"] = index["num_frames"].map(lambda n: subsampled_length(int(n), self.factor))
        index["offset"] = np.concatenate([[0], np.cumsum(index["length"].to_numpy())[:-1]])
        offsets = dict(zip(index["uid"], zip(index["offset"], index["length"])))

        total = int(index["length"].sum())
        path = self.out_dir / f"embeddings_{split}.npy"
        tmp = path.with_name(f".{path.name}.tmp")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(total, self.dim))

        loader = DataLoader(ds, batch_sampler=LengthBucketSampler(ds.lengths, self.batch_size, shuffle=False),
                            collate_fn=collate_pose, num_workers=4, persistent_workers=False)
        for batch in tqdm(loader, desc=f"export {split}", unit="batch"):
            b = batch.to(self.device)
            h, h_mask = self.encoder(b.x, b.mask)
            h = h.float().cpu().numpy()
            lengths = h_mask.sum(1).cpu().tolist()
            for i, uid in enumerate(b.uids):
                off, length = offsets[uid]
                if lengths[i] != length:
                    raise RuntimeError(f"{uid}: expected {length} steps, got {lengths[i]}")
                out[off : off + length] = h[i, :length].astype(np.float16)
        out.flush()
        del out
        tmp.replace(path)

        index[["uid", "video_id", "text", "offset", "length"]].to_csv(self.out_dir / f"index_{split}.csv", index=False)
        return {"clips": len(index), "steps": total, "size_mb": round(path.stat().st_size / 1e6, 1)}

    def run(self, make_zip: bool = False) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        print(f"Encoder: {self.checkpoint} (epoch {self.ckpt.get('epoch')}) | device {self.device} | "
              f"features {', '.join(self.groups)}")

        splits = {s: self.export_split(s) for s in SPLITS}

        shutil.copy(self.checkpoint, self.out_dir / "encoder_stage0.pt")
        shutil.copy(self.data_dir / "stats.npz", self.out_dir / "stats.npz")
        shutil.copy(self.data_dir / "feature_layout.json", self.out_dir / "feature_layout.json")
        info = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "embedding_dim": self.dim,
            "dtype": "float16",
            "subsample": self.factor,
            "frame_rate_after_subsample": 25 / self.factor,
            "feature_groups": self.groups,
            "encoder_config": self.ckpt["encoder_config"],
            "encoder_epoch": self.ckpt.get("epoch"),
            "encoder_val_metrics": self.ckpt.get("val_metrics"),
            "splits": splits,
            "how_to_load": "from src.data import EmbeddingDataset; ds = EmbeddingDataset('handoff', 'train')",
        }
        (self.out_dir / "info.json").write_text(json.dumps(info, indent=2))

        total_mb = sum(p.stat().st_size for p in self.out_dir.iterdir() if p.is_file()) / 1e6
        print(f"\nDone in {(time.time() - start) / 60:.1f} min -> {self.out_dir} ({total_mb:.0f} MB)")
        for s, v in splits.items():
            print(f"  {s:5s}: {v['clips']} clips, {v['steps']} steps, {v['size_mb']} MB")

        if make_zip:
            archive = shutil.make_archive(str(self.out_dir), "zip", root_dir=self.out_dir)
            print(f"Zipped: {archive} ({Path(archive).stat().st_size / 1e6:.0f} MB)")
