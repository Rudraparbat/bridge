"""
Stage 0: pretrain the PoseEncoder (no LLM).

Objectives
  1. Masked pose modelling: hide ~40% of frames (spans), reconstruct them (L1).   [no text needed]
  2. Word presence (optional, --word-weight): predict which of the top-K train words occur
     in the clip's sentence (BCE). Teaches the encoder sign -> meaning cues.

Run
  uv run python -m src.training.stage0 --overfit                 # sanity check, ~1 min
  uv run python -m src.training.stage0 --epochs 1 --max-steps 50 # speed test
  uv run python -m src.training.stage0                           # real run (resumable: --resume)

Outputs (checkpoints/stage0/)
  best.pt      encoder + config + vocab + metrics  (hand-off: PoseEncoder.from_checkpoint)
  last.pt      everything incl. optimizer, for --resume
  log.csv      one row per epoch
  config.json  the exact settings of this run
"""
import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import DEFAULT_DATA_DIR, PROJECT_ROOT
from ..data import (DEFAULT_GROUPS, ISLPoseDataset, LengthBucketSampler, PoseAugment,
                    WordVocab, collate_pose)
from ..models import EncoderConfig, PoseEncoder, ReconstructionHead, WordPresenceHead
from .masking import block_span_mask
from .utils import count_params, pick_device, set_seed, warmup_cosine


@dataclass
class Stage0Config:
    data_dir: str = str(DEFAULT_DATA_DIR)
    out_dir: str = str(PROJECT_ROOT / "checkpoints" / "stage0")
    feature_groups: list[str] = field(default_factory=lambda: list(DEFAULT_GROUPS))

    # model
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 4
    part_dim: int = 64
    dropout: float = 0.1

    # objectives
    mask_ratio: float = 0.4
    mask_span: int = 2
    word_weight: float = 0.1      # 0 disables the word-presence head (word loss starts ~7.6 = ln 2000)
    vocab_size: int = 2000

    # optimisation
    epochs: int = 30
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int = 6             # early stop after N epochs without val improvement
    max_frames: int = 512         # random crop length for training (~20 s)

    # runtime
    device: str = "auto"
    num_workers: int = 4
    seed: int = 42
    augment: bool = True
    max_steps: int | None = None  # debug: limit steps per epoch
    max_clips: int | None = None  # debug: limit clips per split
    overfit: bool = False         # sanity check: one batch, no augmentation/dropout
    resume: str | None = None


class Stage0Trainer:
    def __init__(self, cfg: Stage0Config):
        self.cfg = cfg
        set_seed(cfg.seed)
        self.device = pick_device(cfg.device)
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.overfit:
            cfg.augment, cfg.dropout, cfg.max_clips = False, 0.0, cfg.batch_size

        self._build_data()
        self._build_model()
        self.optimizer = torch.optim.AdamW(self.params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        steps = len(self.train_loader) if not cfg.max_steps else min(cfg.max_steps, len(self.train_loader))
        self.scheduler = warmup_cosine(self.optimizer, steps * cfg.epochs, cfg.warmup_frac)
        self.start_epoch, self.best_val, self.stale = 1, float("inf"), 0
        if cfg.resume:
            self._load_resume(cfg.resume)

    # ---- setup ------------------------------------------------------------- #
    def _build_data(self):
        c = self.cfg
        aug = PoseAugment() if c.augment else None
        self.train_ds = ISLPoseDataset("train", c.data_dir, c.feature_groups, c.max_frames,
                                       random_crop=True, augment=aug, max_clips=c.max_clips)
        val_split = "train" if c.overfit else "val"
        self.val_ds = ISLPoseDataset(val_split, c.data_dir, c.feature_groups, max_frames=1024,
                                     max_clips=c.max_clips)

        loader_kw = dict(collate_fn=collate_pose, num_workers=c.num_workers,
                         persistent_workers=c.num_workers > 0)
        self.train_sampler = LengthBucketSampler(self.train_ds.lengths, c.batch_size,
                                                 shuffle=not c.overfit, seed=c.seed)
        self.train_loader = DataLoader(self.train_ds, batch_sampler=self.train_sampler, **loader_kw)
        self.val_loader = DataLoader(
            self.val_ds, batch_sampler=LengthBucketSampler(self.val_ds.lengths, c.batch_size, shuffle=False),
            **loader_kw)

        self.vocab = WordVocab.build(self.train_ds.texts, c.vocab_size) if c.word_weight > 0 else None

    def _build_model(self):
        c = self.cfg
        self.enc_cfg = EncoderConfig(group_names=list(c.feature_groups), group_dims=self.train_ds.group_dims,
                                     part_dim=c.part_dim, d_model=c.d_model, n_layers=c.n_layers,
                                     n_heads=c.n_heads, dropout=c.dropout)
        self.encoder = PoseEncoder(self.enc_cfg).to(self.device)
        self.recon_head = ReconstructionHead(c.d_model, self.enc_cfg.in_dim, self.enc_cfg.subsample).to(self.device)
        self.word_head = (WordPresenceHead(c.d_model, len(self.vocab), c.dropout).to(self.device)
                          if self.vocab else None)
        modules = [self.encoder, self.recon_head] + ([self.word_head] if self.word_head else [])
        self.params = [p for m in modules for p in m.parameters()]

    # ---- one step ---------------------------------------------------------- #
    def _forward(self, batch, generator=None):
        b = batch.to(self.device)
        hidden = block_span_mask(b.lengths, b.x.shape[1], self.enc_cfg.subsample,
                                 self.cfg.mask_ratio, self.cfg.mask_span, generator).to(self.device)
        h, h_mask = self.encoder(b.x, b.mask, frame_mask=hidden)

        pred = self.recon_head(h, b.x.shape[1])
        sel = hidden & b.mask
        recon = (pred - b.x).abs()[sel].mean()

        out = {"recon": recon, "loss": recon}
        if self.word_head is not None:
            logits = self.word_head(h, h_mask)
            target = self.vocab.multi_hot(b.texts).to(self.device)
            word = self._word_loss(logits, target)
            out.update(word=word, loss=recon + self.cfg.word_weight * word, logits=logits, target=target)
        return out

    @staticmethod
    def _word_loss(logits, target):
        """Soft-target cross-entropy: spread probability over the sentence's words.

        Plain per-word BCE is dominated by the ~99.5% absent words and collapses to predicting
        word frequencies; this loss instead rewards RANKING the clip's own words first.
        """
        n_pos = target.sum(dim=-1)
        has = n_pos > 0
        if not has.any():
            return logits.sum() * 0.0
        dist = target[has] / n_pos[has, None]
        return -(dist * logits[has].log_softmax(dim=-1)).sum(dim=-1).mean()

    @staticmethod
    def _recall_at_k(logits, target, k=10) -> float:
        top = logits.topk(k, dim=-1).indices
        hits = target.gather(1, top).sum(1)
        denom = target.sum(1).clamp(max=k)
        valid = denom > 0
        return float((hits[valid] / denom[valid]).mean()) if valid.any() else 0.0

    # ---- epochs ------------------------------------------------------------ #
    def _set_train(self, mode: bool):
        for m in (self.encoder, self.recon_head, self.word_head):
            if m is not None:
                m.train(mode)

    def train_epoch(self, epoch: int) -> dict:
        self._set_train(True)
        self.train_sampler.set_epoch(epoch)
        totals, n = {"loss": 0.0, "recon": 0.0, "word": 0.0}, 0
        limit = self.cfg.max_steps or len(self.train_loader)
        bar = tqdm(self.train_loader, total=limit, desc=f"epoch {epoch} train", leave=False)
        for step, batch in enumerate(bar):
            if step >= limit:
                break
            out = self._forward(batch)
            self.optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            nn.utils.clip_grad_norm_(self.params, self.cfg.grad_clip)
            self.optimizer.step()
            self.scheduler.step()
            for key in totals:
                if key in out:
                    totals[key] += out[key].detach().item()
            n += 1
            bar.set_postfix(loss=f"{totals['loss'] / n:.4f}", lr=f"{self.scheduler.get_last_lr()[0]:.1e}")
        return {k: v / max(n, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate(self) -> dict:
        self._set_train(False)
        g = torch.Generator().manual_seed(1234)  # same masks every epoch -> comparable val loss
        totals, recall, n = {"loss": 0.0, "recon": 0.0, "word": 0.0}, 0.0, 0
        limit = self.cfg.max_steps or len(self.val_loader)
        for step, batch in enumerate(tqdm(self.val_loader, total=limit, desc="val", leave=False)):
            if step >= limit:
                break
            out = self._forward(batch, generator=g)
            for key in totals:
                if key in out:
                    totals[key] += float(out[key])
            if "logits" in out:
                recall += self._recall_at_k(out["logits"], out["target"])
            n += 1
        res = {k: v / max(n, 1) for k, v in totals.items()}
        res["word_recall@10"] = recall / max(n, 1)
        return res

    # ---- checkpoints ------------------------------------------------------- #
    def _checkpoint(self, epoch, val, full: bool) -> dict:
        ckpt = {
            "stage": 0,
            "epoch": epoch,
            "encoder_config": asdict(self.enc_cfg),
            "encoder_state_dict": self.encoder.state_dict(),
            "feature_groups": list(self.cfg.feature_groups),
            "stats_path": str(Path(self.cfg.data_dir) / "stats.npz"),
            "vocab": self.vocab.words if self.vocab else None,
            "val_metrics": val,
            "train_clips": len(self.train_ds),
            "stage0_config": asdict(self.cfg),
        }
        if full:
            ckpt.update(
                recon_head=self.recon_head.state_dict(),
                word_head=self.word_head.state_dict() if self.word_head else None,
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                best_val=self.best_val, stale=self.stale,
            )
        return ckpt

    def _load_resume(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(ckpt["encoder_state_dict"])
        self.recon_head.load_state_dict(ckpt["recon_head"])
        if self.word_head is not None and ckpt.get("word_head"):
            self.word_head.load_state_dict(ckpt["word_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.start_epoch = ckpt["epoch"] + 1
        self.best_val, self.stale = ckpt["best_val"], ckpt["stale"]
        print(f"Resumed from {path} at epoch {self.start_epoch}")

    # ---- main loop --------------------------------------------------------- #
    def fit(self):
        c = self.cfg
        (self.out_dir / "config.json").write_text(json.dumps(asdict(c), indent=2))
        print(f"Device: {self.device} | train clips {len(self.train_ds)} | val clips {len(self.val_ds)} | "
              f"features {self.train_ds.feature_dim} ({', '.join(c.feature_groups)})")
        print(f"Encoder params: {count_params(self.encoder) / 1e6:.2f}M | "
              f"word vocab: {len(self.vocab) if self.vocab else 'off'} | "
              f"steps/epoch: {c.max_steps or len(self.train_loader)}")
        if self.vocab:
            freq = self.vocab.multi_hot(self.train_ds.texts).mean(0)
            val_t = self.vocab.multi_hot(self.val_ds.texts)
            base = self._recall_at_k(freq.expand(len(val_t), -1), val_t)
            print(f"Word R@10 baseline (always guess the most frequent words): {base:.3f} "
                  f"- the encoder must beat this to have learned meaning")

        log_path = self.out_dir / "log.csv"
        if not log_path.exists() or not c.resume:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(["epoch", "train_loss", "train_recon", "train_word",
                                        "val_loss", "val_recon", "val_word", "val_word_recall@10",
                                        "lr", "seconds"])

        for epoch in range(self.start_epoch, c.epochs + 1):
            t0 = time.time()
            tr = self.train_epoch(epoch)
            va = self.evaluate()
            secs = time.time() - t0
            lr = self.scheduler.get_last_lr()[0]
            print(f"epoch {epoch:3d} | train {tr['loss']:.4f} (recon {tr['recon']:.4f}, word {tr['word']:.4f}) | "
                  f"val {va['loss']:.4f} (recon {va['recon']:.4f}, word R@10 {va['word_recall@10']:.3f}) | "
                  f"lr {lr:.1e} | {secs / 60:.1f} min")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([epoch, tr["loss"], tr["recon"], tr["word"], va["loss"], va["recon"],
                                        va["word"], va["word_recall@10"], lr, round(secs, 1)])

            if va["loss"] < self.best_val:
                self.best_val, self.stale = va["loss"], 0
                torch.save(self._checkpoint(epoch, va, full=False), self.out_dir / "best.pt")
                print(f"  -> new best, saved {self.out_dir / 'best.pt'}")
            else:
                self.stale += 1
            torch.save(self._checkpoint(epoch, va, full=True), self.out_dir / "last.pt")
            if self.stale >= c.patience and not c.overfit:
                print(f"No improvement for {c.patience} epochs - stopping early.")
                break
        print(f"Done. Best val loss {self.best_val:.4f}. Encoder: {self.out_dir / 'best.pt'}")


# --------------------------------------------------------------------------- #
def parse_args() -> Stage0Config:
    defaults = Stage0Config()
    ap = argparse.ArgumentParser(description="Stage 0: pretrain the pose encoder")
    for f in fields(Stage0Config):
        value = getattr(defaults, f.name)
        flag = "--" + f.name.replace("_", "-")
        if isinstance(value, bool):
            ap.add_argument(flag, action=argparse.BooleanOptionalAction, default=value)
        elif isinstance(value, list):
            ap.add_argument(flag, nargs="+", default=value)
        elif value is None:
            ap.add_argument(flag, type=str if f.name == "resume" else int, default=None)
        else:
            ap.add_argument(flag, type=type(value), default=value)
    return Stage0Config(**vars(ap.parse_args()))


def main():
    Stage0Trainer(parse_args()).fit()


if __name__ == "__main__":
    main()
