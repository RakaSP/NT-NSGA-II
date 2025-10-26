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
    Handcrafted global features from a VRP dict to feed FirstNN.
    Shape: [F1]
    """
    nodes = vrp["nodes"]
    vehicles = vrp["vehicles"]
    D = vrp["D"]
    T = vrp["T"]

    N = len(nodes)
    V = len(vehicles)

    demands = [float(getattr(n, "demand", 0.0)) for n in nodes]
    cap = [float(getattr(v, "max_capacity", 0.0)) for v in vehicles]

    mean_dem = float(sum(demands) / max(1, len(demands)))
    sq_dem = sum((d - mean_dem) ** 2 for d in demands)
    std_dem = float(math.sqrt(sq_dem / max(1, len(demands))))

    mean_cap = float(sum(cap) / max(1, len(cap)))
    sq_cap = sum((c - mean_cap) ** 2 for c in cap)
    std_cap = float(math.sqrt(sq_cap / max(1, len(cap))))

    # distance/time matrix stats (assume dense 2D arrays with numeric types)
    # skip diagonals
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

    feats = torch.tensor(
        [
            float(N),
            float(V),
            mean_dem,
            std_dem,
            mean_cap,
            std_cap,
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
    population: List[List[int]],
    objectives: List[Tuple[float, float, float]],
    iter_idx: int,
    total_iters: int,
) -> torch.Tensor:
    """
    Per-iteration features for SecondNN.
    Includes basic population stats (size, iteration progress) and
    objective distribution stats (mean/std/min/max for each objective).
    Also includes a cheap diversity proxy (#unique genes at first position / pop).
    Shape: [F2]
    """
    P = len(population)
    progress = 0.0 if total_iters <= 1 else float(
        iter_idx) / float(total_iters - 1)

    if objectives:
        # per objective (cost, dist, time)
        cols = list(zip(*objectives))
        stats = []
        for col in cols:
            arr = list(col)
            mean = float(sum(arr) / len(arr))
            var = sum((x - mean) ** 2 for x in arr) / len(arr)
            std = float(math.sqrt(var))
            stats.extend([mean, std, float(min(arr)), float(max(arr))])
    else:
        stats = [0.0] * 12  # 3 objectives * 4 stats

    # cheap diversity: distinct head-genes proportion
    if P > 0 and len(population[0]) > 0:
        head = [ind[0] for ind in population]
        div_head_ratio = float(len(set(head))) / float(P)
    else:
        div_head_ratio = 0.0

    feats = torch.tensor(
        [float(P), progress, div_head_ratio] + stats,
        dtype=torch.float32,
    )
    return feats