import math
import torch
import torch.nn as nn
from torch.distributions import Normal


class SecondNN(nn.Module):
    """
    Policy network that outputs a distribution over (cx, mut) in (0, 1)^2.

    - Shared trunk processes the EAControlEnv observation.
    - Two heads produce unconstrained means in R for crossover and mutation.
    - We sample in R and squash via sigmoid to keep actions in (0, 1).
    """

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()

        # Shared feature extraction
        self.shared_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        # Separate output heads for means in R (unconstrained)
        self.cx_mean_head = nn.Linear(hidden, 1)
        self.mut_mean_head = nn.Linear(hidden, 1)

        # Learnable log standard deviations
        self.log_std_cx = nn.Parameter(torch.tensor([-1.5]))
        self.log_std_mut = nn.Parameter(torch.tensor([-1.2]))

        # ======== IMPORTANT: initialize to your baseline (cx=0.9, mut=0.35) ========
        with torch.no_grad():
            # target means in (0,1)
            cx_base = 0.9
            mut_base = 0.35

            def logit(p: float) -> float:
                return math.log(p / (1.0 - p))

            # 1) make heads initially ignore features (pure bias)
            self.cx_mean_head.weight.zero_()
            self.mut_mean_head.weight.zero_()

            # 2) set biases so that sigmoid(bias) ≈ desired rate
            self.cx_mean_head.bias.fill_(logit(cx_base))
            self.mut_mean_head.bias.fill_(logit(mut_base))

            # Optional: start with smaller std (less randomness)
            self.log_std_cx.data.fill_(math.log(0.1))   # std ≈ 0.1
            self.log_std_mut.data.fill_(math.log(0.1))
        # ============================================================================

    def forward(self, x: torch.Tensor):
        """
        x: (B, in_dim)
        Returns:
            cx_rate: (B,) sampled crossover rates in (0, 1)
            mut_rate: (B,) sampled mutation rates in (0, 1)
            logp: (B,) joint log-prob under the policy
            info: dict with debug info (means, entropy, samples, ...)
        """
        eps = 1e-6

        if x.dim() == 1:
            x = x.unsqueeze(0)  # make it (1, in_dim) if single obs

        features = self.shared_net(x)  # (B, hidden)

        # Means in unconstrained space R
        cx_mu_u = self.cx_mean_head(features).squeeze(-1)   # (B,)
        mut_mu_u = self.mut_mean_head(features).squeeze(-1)  # (B,)

        # Standard deviations (scalar) - broadcast across batch
        cx_std_u = self.log_std_cx.exp().clamp(1e-5, 1.0)
        mut_std_u = self.log_std_mut.exp().clamp(1e-5, 1.0)

        # Base Gaussians in R
        dist_cx_u = Normal(cx_mu_u, cx_std_u)
        dist_mut_u = Normal(mut_mu_u, mut_std_u)

        # Reparameterized sampling
        u_cx = dist_cx_u.rsample()   # (B,)
        u_mut = dist_mut_u.rsample()  # (B,)

        # Squash to (0,1) smoothly
        cx_rate = torch.sigmoid(u_cx)    # (B,)
        mut_rate = torch.sigmoid(u_mut)  # (B,)

        # Log-prob of squashed actions with Jacobian correction:
        # y = sigmoid(u) => dy/du = y*(1-y)
        # log p_Y(y) = log p_U(u) - [log y + log(1-y)]
        logp_cx = dist_cx_u.log_prob(u_cx) - (
            torch.log(cx_rate + eps) + torch.log1p(-cx_rate)
        )
        logp_mut = dist_mut_u.log_prob(u_mut) - (
            torch.log(mut_rate + eps) + torch.log1p(-mut_rate)
        )
        logp = logp_cx + logp_mut  # (B,)

        # Entropy in the *unconstrained* space
        entropy_u = dist_cx_u.entropy() + dist_mut_u.entropy()

        info = {
            "mu_u": torch.stack([cx_mu_u, mut_mu_u]),      # (2, B)
            "sampled": torch.stack([cx_rate, mut_rate]),   # (2, B)
            "entropy_u": entropy_u,                        # (B,)
        }

        return cx_rate, mut_rate, logp, info
