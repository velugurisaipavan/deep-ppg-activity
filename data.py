"""Histories never cross subjects or time gaps."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def load_cache(root):
    fields = {key: [] for key in ("spectra", "features", "hr", "activity", "start_seconds", "subject")}
    files = sorted(Path(root).glob("S*.npz"))
    if not files:
        raise ValueError("No cached subjects; run preprocessing first")
    for file in files:
        with np.load(file, allow_pickle=False) as archive:
            for key in fields:
                fields[key].append(np.repeat(file.stem, len(archive["hr"])) if key == "subject" else archive[key])
    return {key: np.concatenate(value) for key, value in fields.items()}


def eligible_rows(data, history):
    if history < 1:
        raise ValueError("history must be positive")
    rows = np.arange(history - 1, len(data["subject"]))
    valid = np.ones(len(rows), dtype=bool)
    for lag in range(1, history):
        valid &= data["subject"][rows] == data["subject"][rows - lag]
        valid &= data["start_seconds"][rows] - data["start_seconds"][rows - lag] == 2 * lag
    return rows[valid]


class WindowDataset(Dataset):
    def __init__(self, data, rows, probabilities, history, mean, std):
        self.data, self.rows, self.probabilities = data, rows, probabilities
        self.history, self.mean, self.std = history, mean, std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        i = self.rows[index]
        x = self.data["spectra"][i - self.history + 1:i + 1].transpose(1, 0, 2).copy()
        y = (self.data["hr"][i] - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(self.probabilities[i]), torch.tensor(y, dtype=torch.float32)
