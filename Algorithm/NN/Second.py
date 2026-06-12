import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class SecondNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        cx_base: float = 0.9,
        mut_base: float = 0.35,
        std_init: float = 0.4,
    ):
        super().__init__()

        self.shared_net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )

        self.cx_mean_head = nn.Linear(hidden, 1)
        self.mut_mean_head = nn.Linear(hidden, 1)

        self.cx_rho_head = nn.Linear(hidden, 1)
        self.mut_rho_head = nn.Linear(hidden, 1)

        with torch.no_grad():
            self.cx_mean_head.weight.zero_()
            self.mut_mean_head.weight.zero_()
            self.cx_mean_head.bias.fill_(_logit(cx_base))
            self.mut_mean_head.bias.fill_(_logit(mut_base))

            inv_sp = math.log(math.exp(std_init) - 1.0)
            self.cx_rho_head.weight.zero_()
            self.mut_rho_head.weight.zero_()
            self.cx_rho_head.bias.fill_(inv_sp)
            self.mut_rho_head.bias.fill_(inv_sp)

        for m in self.shared_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)

        with torch.no_grad():
            self.cx_mean_head.weight.zero_()
            self.mut_mean_head.weight.zero_()
            self.cx_mean_head.bias.fill_(_logit(cx_base))
            self.mut_mean_head.bias.fill_(_logit(mut_base))

            inv_sp = math.log(math.exp(std_init) - 1.0)
            self.cx_rho_head.weight.zero_()
            self.mut_rho_head.weight.zero_()
            self.cx_rho_head.bias.fill_(inv_sp)
            self.mut_rho_head.bias.fill_(inv_sp)


    def forward(self, x: torch.Tensor):
        eps = 1e-6

        if x.dim() == 1:
            x = x.unsqueeze(0)


        h = self.shared_net(x)

        cx_mean= self.cx_mean_head(h).squeeze(-1)
        mut_mean = self.mut_mean_head(h).squeeze(-1)

        cx_std = self.cx_rho_head(h).squeeze(-1)
        mut_std = self.mut_rho_head(h).squeeze(-1)
        
        cx_std = F.softplus(cx_std) + eps
        mut_std = F.softplus(mut_std) + eps

        dist_cx = Normal(cx_mean, cx_std)
        dist_mut = Normal(mut_mean, mut_std)

        u_cx = dist_cx.rsample()
        u_mut = dist_mut.rsample()

        cx_rate = torch.sigmoid(u_cx).clamp(eps, 1.0 - eps)
        mut_rate = torch.sigmoid(u_mut).clamp(eps, 1.0 - eps)

        logp_cx = dist_cx.log_prob(u_cx) - (torch.log(cx_rate) + torch.log1p(-cx_rate))
        logp_mut = dist_mut.log_prob(u_mut) - (torch.log(mut_rate) + torch.log1p(-mut_rate))
        logp = logp_cx + logp_mut


        info = {
            "mu_u": torch.stack([cx_mean, mut_mean], dim=0),
            "std_u": torch.stack([cx_std, mut_std], dim=0),
            "sampled": torch.stack([cx_rate, mut_rate], dim=0),
            "entropy_u": dist_cx.entropy() + dist_mut.entropy(),
        }
        return cx_rate, mut_rate, logp, info
