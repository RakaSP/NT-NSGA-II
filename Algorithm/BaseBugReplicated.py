from __future__ import annotations

from typing import Any, Dict, List, Mapping
from Algorithm.BaseAlgorithm import BaseAlgorithm
from vrp_core.decoding import decode_route
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


class BaseBugReplicated(BaseAlgorithm):
    """
    Variant of BaseAlgorithm that replicates the bug from Azam's thesis:

    - Uses decode_route to build routes (which add depot at start and end).
    - BUT implements the bug:
        * depot is removed from the stored 'route'
        * distance is computed only between customer nodes
          (no depot->first, no last->depot).
        * time is computed only between customer nodes
          (no depot->first, no last->depot).
    
    This means travel from depot to first customer and from last customer 
    to depot is completely ignored in both distance and time calculations.
    """

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        """
        Build solution from permutation, replicating the depot-exclusion bug.
        
        Args:
            perm: Permutation of customer IDs (not indices)
            
        Returns:
            List of route dictionaries with buggy distance/time calculations
        """
        # Decode routes (will include depot at start and end)
        routes = decode_route(perm, self.vrp)

        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]

        sol: List[Dict[str, Any]] = []

        for r, veh in zip(routes, vehicles):
            # r typically looks like [depot, c1, c2, ..., ck, depot]
            # BUG REPLICATION: drop depot completely from stored route
            customer_route: List[int] = [int(x) for x in r if int(x) != self.depot_id]

            # BUGGY Distance: only customer-to-customer edges
            # Excludes depot->first and last->depot
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

    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        """
        Compute detailed solution information with buggy calculations.
        
        This override ensures that even the detailed metrics replicate the bug,
        calculating both distance and time without depot travel legs.
        
        Args:
            perm: Permutation of customer IDs
            
        Returns:
            Dictionary with solution details using buggy calculations
        """
        solution = self._solution_from_perm(perm)

        # Total distance already computed incorrectly in _solution_from_perm
        total_distance = float(sum(s["total_distance"] for s in solution))

        # BUGGY Time: only customer-to-customer travel time
        # Excludes depot->first and last->depot
        Tm: Mapping[int, Mapping[int, float]] = self.vrp["T"]
        total_time = 0.0
        for s in solution:
            customer_route = s["route"]  # Already has depot removed
            if len(customer_route) < 2:
                continue
            # Calculate time only between customers
            for a, b in zip(customer_route[:-1], customer_route[1:]):
                total_time += float(Tm[a][b])

        # Score calculation uses the buggy solution
        try:
            if self.scorer == "distance":
                from vrp_core.scorer.distance import score_solution as sdist
                score_val = float(sdist(solution))
            else:
                from vrp_core.scorer.cost import score_solution as scost
                score_val = float(scost(solution, self.vrp["nodes"], self.vrp["vehicles"], self.vrp["D"]))
        except Exception:
            score_val = float("nan")

        return {
            "summary": {
                "vehicles_used": len(solution),
                "total_distance": total_distance,      # BUGGY: excludes depot travel
                "total_time": float(total_time),       # BUGGY: excludes depot travel
                "scorer": self.scorer,
                "score": score_val,
            },
            "per_vehicle": solution,
            "permutation": list(perm),  # best permutation (ID space)
            "depot_id": self.depot_id,
        }