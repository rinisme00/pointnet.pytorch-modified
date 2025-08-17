# llama_backbone/heads.py
import torch.nn as nn

class SegHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, h):
        # h: [B, N, d]
        return self.net(h)  # [B, N, C]