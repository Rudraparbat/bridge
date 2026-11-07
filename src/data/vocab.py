"""Word vocabulary for the auxiliary word-presence objective (built from TRAIN text only)."""
import json
import re
from collections import Counter
from pathlib import Path

import torch

STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being am
it its this that these those there here i you he she we they me him her us them my your his our their
do does did done have has had having will would shall should can could may might must not no
so than then too very just also about into over after before up down out off again s t
""".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


class WordVocab:
    def __init__(self, words: list[str]):
        self.words = list(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, texts, size: int = 2000, min_count: int = 3) -> "WordVocab":
        counts = Counter(w for t in texts for w in set(tokenize(t)) if w not in STOPWORDS)
        return cls([w for w, c in counts.most_common(size) if c >= min_count])

    def __len__(self) -> int:
        return len(self.words)

    def multi_hot(self, texts: list[str]) -> torch.Tensor:
        out = torch.zeros(len(texts), len(self.words))
        for i, t in enumerate(texts):
            ids = [self.index[w] for w in set(tokenize(t)) if w in self.index]
            out[i, ids] = 1.0
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.words))

    @classmethod
    def load(cls, path: Path) -> "WordVocab":
        return cls(json.loads(Path(path).read_text()))
