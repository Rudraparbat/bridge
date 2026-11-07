"""
Preprocess downloaded .pose files.

    uv run python -m src.preprocessing                  # everything in data/poses
    uv run python -m src.preprocessing --max-clips 200  # quick test
"""
import argparse

from ..config import DEFAULT_DATA_DIR, PreprocessConfig
from .pipeline import PreprocessPipeline


def main():
    d = PreprocessConfig()
    ap = argparse.ArgumentParser(description="Raw iSign .pose -> processed .npy + metadata + stats")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="folder with poses/ and the labels CSV")
    ap.add_argument("--out-dir", default=None, help="where outputs go (default: same as --data-dir)")
    ap.add_argument("--target-fps", type=float, default=d.target_fps)
    ap.add_argument("--min-frames", type=int, default=d.min_frames)
    ap.add_argument("--min-hand-ratio", type=float, default=d.min_hand_ratio)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--max-clips", type=int, default=None, help="debug: only the first N clips")
    ap.add_argument("--overwrite", action="store_true", help="re-process clips that already have a .npy")
    args = ap.parse_args()

    cfg = PreprocessConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, target_fps=args.target_fps,
        min_frames=args.min_frames, min_hand_ratio=args.min_hand_ratio,
        workers=args.workers, max_clips=args.max_clips, overwrite=args.overwrite,
    )
    PreprocessPipeline(cfg).run()


if __name__ == "__main__":
    main()
