"""Batch clips of similar length together -> much less padding, faster epochs."""
import torch
from torch.utils.data import Sampler


class LengthBucketSampler(Sampler[list[int]]):
    """
    Sort indices by length, cut into batches, shuffle the ORDER of batches each epoch
    (plus a little noise in the sort so batch composition also changes).
    """

    def __init__(self, lengths, batch_size: int, shuffle: bool = True, noise: float = 0.1, seed: int = 0):
        self.lengths = torch.as_tensor(list(lengths), dtype=torch.float)
        self.batch_size, self.shuffle, self.noise = batch_size, shuffle, noise
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        keys = self.lengths
        if self.shuffle and self.noise > 0:
            keys = keys * (1 + self.noise * (torch.rand(len(keys), generator=g) - 0.5))
        order = torch.argsort(keys).tolist()
        batches = [order[i : i + self.batch_size] for i in range(0, len(order), self.batch_size)]
        if self.shuffle:
            batches = [batches[i] for i in torch.randperm(len(batches), generator=g).tolist()]
        yield from batches
