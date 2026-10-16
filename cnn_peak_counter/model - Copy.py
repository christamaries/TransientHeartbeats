"""1D CNN that predicts the number of parent pulses in a GRB light curve."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PulseCountCNN(nn.Module):
    """
    Input:  (batch, L) raw (peak-normalized) light curve time series.
    Output: (batch,) predicted pulse count, as a continuous value >= 1
            (round it to get a discrete count).
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, L)
        z = self.features(x)
        raw = self.head(z).squeeze(-1)
        # softplus + 1 keeps the output positive and >= 1, matching the fact
        # that n_parent_pulses is always >= 1 by construction (avalanche.py
        # redraws mu_s until it's nonzero)
        return F.softplus(raw) + 1.0
