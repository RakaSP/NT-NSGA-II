import torch
import torch.nn as nn
from typing import Dict, Tuple

class SecondNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.logits_head = nn.Linear(hidden, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)
        nn.init.normal_(self.logits_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.logits_head.bias, 0.0)

    def sample_actions(self, x: torch.Tensor):
        x = self.norm(x)                              # <-- normalize features
        logits = self.logits_head(self.net(x))
        logits = torch.clamp(logits, -10.0, 10.0)     # <-- clamp to prevent saturation
        rates = torch.sigmoid(logits)
        cx, mut = rates[0], rates[1]
        logp = (torch.log(rates + 1e-8)).sum()
        info = {"logits": logits.detach(), "rates": rates.detach()}
        return cx.item(), mut.item(), logp, info
