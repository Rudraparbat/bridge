import os
from pathlib import Path

from .config import PROJECT_ROOT


def load_env(env_path: Path = PROJECT_ROOT / ".env") -> None:
    """Put KEY=VALUE pairs from .env into os.environ (existing variables win)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def get_hf_token() -> str:
    load_env()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not found. Add it to .env in the project root:\n"
            "    HF_TOKEN=hf_xxxxxxxxxxxxxxxxx\n"
            "and accept the license at https://huggingface.co/datasets/Exploration-Lab/iSign"
        )
    return token
