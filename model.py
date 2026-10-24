"""Compact reconstruction with channel/history fusion and late activity fusion."""
import torch
from torch import nn


class ActivityAwareCNN(nn.Module):
    def __init__(self, history=3, activity_dim=9, dropout=.1):
        super().__init__()
        if history < 1 or activity_dim not in (0, 9):
            raise ValueError("history >=1, activity_dim 0 or 9 required")
        self.history, self.activity_dim = history, activity_dim
        self.channel_fusion = nn.Sequential(nn.Conv2d(4, 32, 1), nn.ELU())
        self.history_fusion = nn.Sequential(nn.Conv2d(32, 32, (history, 3), padding=(0, 1)), nn.ELU())
        blocks = []
        for _ in range(7):
            blocks += [nn.Conv1d(32, 32, 3, padding=1), nn.ELU(), nn.MaxPool1d(2)]
        self.blocks = nn.Sequential(*blocks)
        self.compress = nn.Sequential(nn.Conv1d(32, 6, 1), nn.ELU())
        self.fc1 = nn.Sequential(nn.Linear(48, 64), nn.ELU(), nn.Dropout(dropout))
        self.fc2 = nn.Linear(64 + activity_dim, 1)

    def forward(self, spectra, activity=None):
        if spectra.ndim != 4 or tuple(spectra.shape[1:]) != (4, self.history, 1025):
            raise ValueError("Expected spectra [batch,4,history,1025]")
        x = self.history_fusion(self.channel_fusion(spectra)).squeeze(2)
        x = self.fc1(self.compress(self.blocks(x)).flatten(1))
        if self.activity_dim:
            if activity is None or tuple(activity.shape) != (len(x), 9):
                raise ValueError("Expected predicted activity probabilities [batch,9]")
            x = torch.cat([x, activity], dim=1)
        return self.fc2(x).squeeze(1)
