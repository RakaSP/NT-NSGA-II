from __future__ import annotations

from typing import Any, Dict, List

from .models.node import node


def decode_route(perm: List[int], vrp: Dict[str, Any]) -> List[List[int]]:
    """
    Deterministic decoder.

    - Takes the permutation as-is
    - Wraps it with depot at start and end
    - Returns a single route
    """

    nodes: List[node] = vrp["nodes"]
    if not nodes:
        raise ValueError("VRP has no nodes")

    depot_id = int(nodes[0].id)

    if not perm:
        return [[depot_id, depot_id]]

    route = [depot_id] + [int(x) for x in perm] + [depot_id]
    return [route]
