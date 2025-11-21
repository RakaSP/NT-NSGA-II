# vrp_core/scorer/distance.py
from __future__ import annotations
from typing import Any, Dict, List


def score_solution(solution: List[Dict[str, Any]]) -> float:
    # literally: sum the distances already present in the solution entries
    return float(sum(float(v["total_distance"]) for v in solution))
