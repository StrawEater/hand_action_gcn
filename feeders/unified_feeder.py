import numpy as np
import torch
import sys

sys.path.extend(['../'])
from feeders.feeder import Feeder


class UnifiedFeeder(Feeder):
    """
    Feeder for the unified OakInk2+H2O dataset (23 classes, T=32, V=42).

    Drop-in replacement for Feeder — same __getitem__ contract (data, label, index).
    Extras:
      - self.sources: str array, 'oakink' or 'h2o' per sample (parsed from sample_name prefix)
      - self.wrist: optional (N, T, 2, 3) absolute wrist positions when wrist_path is given
      - get_source_weights(): per-sample inverse-frequency weights for WeightedRandomSampler
    """

    def __init__(self, data_path, label_path,
                 wrist_path=None,
                 random_choose=False, random_shift=False, random_move=False,
                 window_size=-1, normalization=False, debug=False, use_mmap=True,
                 data_fraction=1.0):
        self._wrist_path = wrist_path
        super().__init__(
            data_path=data_path,
            label_path=label_path,
            random_choose=random_choose,
            random_shift=random_shift,
            random_move=random_move,
            window_size=window_size,
            normalization=normalization,
            debug=debug,
            use_mmap=use_mmap,
            data_fraction=data_fraction,
        )

    def load_data(self):
        super().load_data()

        self.sources = np.array([
            'oakink' if n.startswith('oak_') else 'h2o'
            for n in self.sample_name
        ])

        if self._wrist_path is not None:
            self.wrist = np.load(self._wrist_path, mmap_mode='r' if self.use_mmap else None)
        else:
            self.wrist = None

    def get_source_weights(self):
        """
        Per-sample weights inversely proportional to source size.
        Pass to torch.utils.data.WeightedRandomSampler to balance OakInk2 / H2O.
        """
        unique, counts = np.unique(self.sources, return_counts=True)
        freq = {src: cnt for src, cnt in zip(unique, counts)}
        weights = np.array([1.0 / freq[s] for s in self.sources])
        return torch.DoubleTensor(weights)
