import torch
import torch.nn as nn


class SecondNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )

        # Separate output heads for means
        self.cx_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()  # Outputs in [0, 1]
        )
        self.mut_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
            nn.Sigmoid()  # Outputs in [0, 1]
        )

        # Learnable log standard deviations (initialized to give std ~0.1)
        self.log_std_cx = nn.Parameter(torch.tensor([-2.3]))
        self.log_std_mut = nn.Parameter(torch.tensor([-2.3]))

    def forward(self, x: torch.Tensor):
        features = self.shared_net(x)

        # Get means from heads
        cx_mean = self.cx_head(features).squeeze(-1)
        mut_mean = self.mut_head(features).squeeze(-1)

        # Get standard deviations (clamped for stability)
        cx_std = torch.exp(self.log_std_cx).clamp(0.01, 0.3)
        mut_std = torch.exp(self.log_std_mut).clamp(0.01, 0.3)

        # Create distributions
        dist_cx = torch.distributions.Normal(cx_mean, cx_std)
        dist_mut = torch.distributions.Normal(mut_mean, mut_std)

        # Sample actions
        cx_rate_raw = dist_cx.rsample()
        mut_rate_raw = dist_mut.rsample()

        # Clamp to valid range
        cx_rate = torch.clamp(cx_rate_raw, 0.0, 1.0)
        mut_rate = torch.clamp(mut_rate_raw, 0.0, 1.0)

        # CRITICAL FIX: Use actual log probabilities from distributions
        # Must use raw (unclamped) values for correct gradients
        logp_cx = dist_cx.log_prob(cx_rate_raw)
        logp_mut = dist_mut.log_prob(mut_rate_raw)
        logp = logp_cx + logp_mut

        info = {
            "logits": torch.stack([cx_mean, mut_mean]).detach(),
            "sampled": torch.stack([cx_rate, mut_rate]),
        }

        return cx_rate.item(), mut_rate.item(), logp, info
