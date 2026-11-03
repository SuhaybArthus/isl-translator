import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, heads=4, layers=3,
                 ff=256, dropout=0.2, max_len=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_classes))

    def forward(self, x):                                   # x: (B, T, F)
        B, T, _ = x.shape
        h = self.proj(self.norm(x)) + self.pos[:, :T]
        h = torch.cat([self.cls.expand(B, -1, -1), h], dim=1)  # prepend class token
        h = self.encoder(h)
        return self.head(self.out_norm(h[:, 0]))