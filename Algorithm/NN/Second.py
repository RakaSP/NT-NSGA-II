import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class SecondNN(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        )

        # Separate output heads for means (keep your Sigmoids)
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
        eps = 1e-6

        features = self.shared_net(x)

        # Heads output (0,1)
        cx_mean = self.cx_head(features).squeeze(-1)   # (B,)
        mut_mean = self.mut_head(features).squeeze(-1)  # (B,)

        # Use your learnable log-stds (no detach, keep grads)
        cx_std_u = self.log_std_cx.exp().clamp(1e-5, 1.0)  # scalar; broadcasts
        mut_std_u = self.log_std_mut.exp().clamp(1e-5, 1.0)

        # Move means to unconstrained space via inverse-sigmoid (logit)
        cx_mu_u = torch.logit(cx_mean.clamp(eps, 1 - eps))   # (B,)
        mut_mu_u = torch.logit(mut_mean.clamp(eps, 1 - eps))  # (B,)

        # Base Gaussians in R (reparameterized)
        dist_cx_u = Normal(cx_mu_u,  cx_std_u)
        dist_mut_u = Normal(mut_mu_u, mut_std_u)

        u_cx = dist_cx_u.rsample()   # (B,)
        u_mut = dist_mut_u.rsample()  # (B,)

        # Squash to (0,1) smoothly (NO hard clamp)
        cx_rate = torch.sigmoid(u_cx)   # (B,)
        mut_rate = torch.sigmoid(u_mut)  # (B,)

        # Log-prob of squashed actions with Jacobian correction:
        # y = sigmoid(u) => dy/du = y*(1-y)
        # log p_Y(y) = log p_U(u) - [log y + log(1-y)]
        logp_cx = dist_cx_u.log_prob(
            u_cx) - (torch.log(cx_rate + eps) + torch.log1p(-cx_rate))
        logp_mut = dist_mut_u.log_prob(
            u_mut) - (torch.log(mut_rate + eps) + torch.log1p(-mut_rate))
        logp = logp_cx + logp_mut  # shape (B,)

        info = {
            # ok to .detach() if only for logging
            "logits": torch.stack([cx_mean, mut_mean]),
            # actions in (0,1)
            "sampled": torch.stack([cx_rate, mut_rate])
        }

        # IMPORTANT: return tensors, NOT .item(), so the graph stays intact
        return cx_rate, mut_rate, logp, info
