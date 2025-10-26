from torch.distributions import Poisson
import torch.nn as nn
import torch
from typing import Dict, Tuple

class FirstNN(nn.Module):
    """
    Predicts integer (iters, population) with constant Poisson lambdas:
    - Network outputs probabilities p_pop, p_iter in (0,1) via sigmoid
    - Constant lambdas: lambda_pop=260, lambda_iter=1600 (so p=0.5 gives ~130 pop and ~800 iters)
    - Sample from Poisson and add offsets (+2 for pop, +1 for iters)
    """
    # CONSTANT lambdas - calculated to give desired K values at p=0.5
    LAMBDA_POP = 100.0    # Poisson(260) mean=260, p=0.5 -> sample~130
    LAMBDA_ITER = 500.0  # Poisson(1600) mean=1600, p=0.5 -> sample~800

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.prob_pop_head = nn.Linear(hidden, 1)
        self.prob_iter_head = nn.Linear(hidden, 1)

        # Initialize to output ~0.5 probabilities
        nn.init.constant_(self.prob_pop_head.bias, 0.0)
        nn.init.constant_(self.prob_iter_head.bias, 0.0)

    def _forward_raw(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.net(x)
        logit_pop = self.prob_pop_head(h).squeeze(-1)
        logit_iter = self.prob_iter_head(h).squeeze(-1)
        return h, logit_pop, logit_iter

    def forward_probs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, logit_pop, logit_iter = self._forward_raw(x)
        p_pop = torch.sigmoid(logit_pop)   # in (0,1)
        p_iter = torch.sigmoid(logit_iter)  # in (0,1)
        return {"p_pop": p_pop, "p_iter": p_iter, "logit_pop": logit_pop, "logit_iter": logit_iter}

    def sample_actions(self, x: torch.Tensor, min_pop: int = 2, min_iters: int = 1):
        probs = self.forward_probs(x)

        # Use probabilities as sampling weights with CONSTANT lambdas
        p_pop = probs["p_pop"]
        p_iter = probs["p_iter"]

        # Scale the Poisson samples by probabilities
        # p=1.0 -> full Poisson sample, p=0.0 -> sample near 0
        pop_sample = Poisson(self.LAMBDA_POP * p_pop).sample()
        iter_sample = Poisson(self.LAMBDA_ITER * p_iter).sample()

        # Convert to integers with offsets
        population_size = (pop_sample.to(torch.long) + min_pop).clamp(min=2)
        iters = (iter_sample.to(torch.long) + min_iters).clamp(min=1)

        # Log probabilities - use the actual distributions we sampled from
        log_prob_pop = Poisson(self.LAMBDA_POP * p_pop).log_prob(pop_sample)
        log_prob_iter = Poisson(
            self.LAMBDA_ITER * p_iter).log_prob(iter_sample)
        log_prob = log_prob_pop + log_prob_iter

        return (
            population_size.item(),
            iters.item(),
            log_prob,
            {
                "p_pop": probs["p_pop"].detach(),
                "p_iter": probs["p_iter"].detach(),
                "logit_pop": probs["logit_pop"].detach(),
                "logit_iter": probs["logit_iter"].detach(),
                "pop_sample": pop_sample.detach(),
                "iter_sample": iter_sample.detach(),
                "scaled_lambda_pop": (self.LAMBDA_POP * p_pop).detach(),
                "scaled_lambda_iter": (self.LAMBDA_ITER * p_iter).detach(),
            },
        )
