"""
Export encoder embeddings for the Stage 1 hand-off.

    uv run python -m src.export                       # -> handoff/
    uv run python -m src.export --zip                 # also creates handoff.zip
"""
import argparse

from ..config import DEFAULT_DATA_DIR, PROJECT_ROOT
from .embeddings import EmbeddingExporter


def main():
    ap = argparse.ArgumentParser(description="Run the frozen encoder over all clips and save embeddings")
    ap.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "stage0" / "best.pt"))
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "handoff"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--zip", action="store_true", help="also create <out-dir>.zip for sharing")
    args = ap.parse_args()

    EmbeddingExporter(args.checkpoint, args.data_dir, args.out_dir, args.batch_size, args.device).run(args.zip)


if __name__ == "__main__":
    main()
