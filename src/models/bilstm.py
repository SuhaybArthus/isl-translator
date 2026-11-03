import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, num_classes))

    def forward(self, x):                      # x: (B, T, F)
        out, _ = self.lstm(self.norm(x))       # (B, T, 2*hidden)
        feat = torch.cat([out.mean(1), out.max(1).values], dim=1)
        return self.head(feat)