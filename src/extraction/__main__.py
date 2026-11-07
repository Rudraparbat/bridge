"""
Download raw iSign .pose files.

    uv run python -m src.extraction --limit 30000
    uv run python -m src.extraction              # all ~127k clips
"""
import argparse

from ..config import DEFAULT_DATA_DIR
from ..env import get_hf_token
from .downloader import PoseDownloader


def main():
    ap = argparse.ArgumentParser(description="Download raw iSign .pose files into data/poses/")
    ap.add_argument("--limit", type=int, default=None, help="approx. number of clips (whole videos); omit for all")
    ap.add_argument("--seed", type=int, default=42, help="which videos are picked when --limit is set")
    ap.add_argument("--workers", type=int, default=8, help="parallel downloads")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = ap.parse_args()

    PoseDownloader(get_hf_token(), args.data_dir, args.workers).run(args.limit, args.seed)


if __name__ == "__main__":
    main()
