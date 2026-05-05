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
        hidden: int = 256,
        cx_base: float = 0.9,
        mut_base: float = 0.35,
        std_init: float = 0.4,
        std_min: float = 0.05,
        std_max: float = 1.5,
        safe_check: bool = False,
    ):
        super().__init__()
        self.safe_check = safe_check
        self.std_min = std_min
        self.std_max = std_max

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

    def _check(self, t: torch.Tensor, name: str):
        if self.safe_check and not torch.isfinite(t).all():
            bad = t[~torch.isfinite(t)]
            raise RuntimeError(f"[NaN/Inf] {name}: examples={bad[:5].detach().cpu().tolist()}")

    def forward(self, x: torch.Tensor):
        eps = 1e-6

        if x.dim() == 1:
            x = x.unsqueeze(0)

        self._check(x, "input")

        h = self.shared_net(x)
        self._check(h, "features")

        cx_mu_u = self.cx_mean_head(h).squeeze(-1)
        mut_mu_u = self.mut_mean_head(h).squeeze(-1)

        cx_rho = self.cx_rho_head(h).squeeze(-1)
        mut_rho = self.mut_rho_head(h).squeeze(-1)

        cx_std_u = F.softplus(cx_rho).clamp(self.std_min, self.std_max)
        mut_std_u = F.softplus(mut_rho).clamp(self.std_min, self.std_max)

        self._check(cx_mu_u, "cx_mu_u")
        self._check(mut_mu_u, "mut_mu_u")
        self._check(cx_std_u, "cx_std_u")
        self._check(mut_std_u, "mut_std_u")

        dist_cx = Normal(cx_mu_u, cx_std_u)
        dist_mut = Normal(mut_mu_u, mut_std_u)

        u_cx = dist_cx.rsample()
        u_mut = dist_mut.rsample()

        cx_rate = torch.sigmoid(u_cx).clamp(eps, 1.0 - eps)
        mut_rate = torch.sigmoid(u_mut).clamp(eps, 1.0 - eps)

        logp_cx = dist_cx.log_prob(u_cx) - (torch.log(cx_rate) + torch.log1p(-cx_rate))
        logp_mut = dist_mut.log_prob(u_mut) - (torch.log(mut_rate) + torch.log1p(-mut_rate))
        logp = logp_cx + logp_mut

        self._check(logp, "logp")

        info = {
            "mu_u": torch.stack([cx_mu_u, mut_mu_u], dim=0),
            "std_u": torch.stack([cx_std_u, mut_std_u], dim=0),
            "sampled": torch.stack([cx_rate, mut_rate], dim=0),
            "entropy_u": dist_cx.entropy() + dist_mut.entropy(),
        }
        return cx_rate, mut_rate, logp, info
