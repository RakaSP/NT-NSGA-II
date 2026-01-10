from __future__ import annotations

from typing import Any, Dict, List, Mapping
from Algorithm.BaseAlgorithm import BaseAlgorithm
from vrp_core.decoding import decode_minimize, decode_split_equal
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


class BaseBugReplicated(BaseAlgorithm):
    """
    Variant of BaseAlgorithm that replicates the bug from the cited paper:

    - Still uses decode_minimize / decode_split_equal to build routes
      (which add depot at start and end).
    - BUT:
        * depot is removed from the stored 'route'
        * distance is computed only between customer nodes
          (no depot->first, no last->depot).
    """

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        # Decode same as base class (routes WITH depots)
        if self.decode_mode == "minimize":
            full_routes = decode_minimize(perm, self.vrp)
        else:
            full_routes = decode_split_equal(perm, self.vrp)

        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        sol: List[Dict[str, Any]] = []

        for r, veh in zip(full_routes, vehicles):
            # r typically looks like [depot, c1, c2, ..., ck, depot]
            # BUG REPLICATION: drop depot completely
            customer_route: List[int] = [int(x) for x in r if int(x) != self.depot_id]

            # Distance: only customer-to-customer edges
            if len(customer_route) < 2:
                total_distance = 0.0
            else:
                d = 0.0
                for a, b in zip(customer_route[:-1], customer_route[1:]):
                    d += float(D[a][b])
                total_distance = d

            sol.append({
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "vehicle_distance_cost": veh.distance_cost,
                "vehicle_initial_cost": veh.initial_cost,
                "total_distance": float(total_distance),  # BUGGY: excludes depot legs
                "route": customer_route,                  # BUGGY: no depot in route
            })

        return sol
