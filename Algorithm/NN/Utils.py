from typing import Any, Dict, List, Tuple
import torch
import math

# -----------------------------
# Utility helpers
# -----------------------------
def _flatten(lst_of_lsts: List[List[int]]) -> List[int]:
    return [x for lst in lst_of_lsts for x in lst]


# -----------------------------
# Feature Extraction
# -----------------------------
def extract_vrp_features(vrp: Dict[str, Any]) -> torch.Tensor:
    """
    Handcrafted global features from a VRP dict.
    (Capacity removed)
    """
    nodes = vrp["nodes"]
    vehicles = vrp["vehicles"]
    D = vrp["D"]
    T = vrp["T"]

    N = len(nodes)
    V = len(vehicles)

    # demand stats (kept)
    demands = [float(getattr(n, "demand", 0.0) or 0.0) for n in nodes]

    mean_dem = float(sum(demands) / max(1, len(demands)))
    sq_dem = sum((d - mean_dem) ** 2 for d in demands)
    std_dem = float(math.sqrt(sq_dem / max(1, len(demands))))

    # distance/time matrix stats (assume dense 2D arrays with numeric types)
    dij = []
    tij = []
    nD0 = len(D)
    for i in range(nD0):
        for j in range(nD0):
            if i == j:
                continue
            dij.append(float(D[i, j]))
            tij.append(float(T[i, j]))

    if dij:
        d_mean = float(sum(dij) / len(dij))
        d_min = float(min(dij))
        d_max = float(max(dij))
        d_sq = sum((x - d_mean) ** 2 for x in dij)
        d_std = float(math.sqrt(d_sq / len(dij)))
    else:
        d_mean = d_min = d_max = d_std = 0.0

    if tij:
        t_mean = float(sum(tij) / len(tij))
        t_min = float(min(tij))
        t_max = float(max(tij))
        t_sq = sum((x - t_mean) ** 2 for x in tij)
        t_std = float(math.sqrt(t_sq / len(tij)))
    else:
        t_mean = t_min = t_max = t_std = 0.0

    # NOTE: mean_cap/std_cap removed
    feats = torch.tensor(
        [
            float(N),
            float(V),
            mean_dem,
            std_dem,
            d_mean,
            d_std,
            d_min,
            d_max,
            t_mean,
            t_std,
            t_min,
            t_max,
        ],
        dtype=torch.float32,
    )
    return feats


def extract_population_features(
    objectives: List[Tuple[float, float, float]],
) -> torch.Tensor:
    primary_scores = [obj[0] for obj in objectives]
    best = min(primary_scores)
    worst = max(primary_scores)
    mean = sum(primary_scores) / len(primary_scores)

    best_mean = best - mean
    mean_worst = mean - worst

    feats = torch.tensor(
        [
            float(best),
            float(best_mean),
            float(mean),
            float(mean_worst),
            float(worst),
        ],
        dtype=torch.float32,
    )
    return feats
