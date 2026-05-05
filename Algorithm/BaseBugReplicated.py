from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping
from Algorithm.BaseAlgorithm import BaseAlgorithm
from vrp_core.decoding import decode_route
from vrp_core.models.node import node
from vrp_core.models.vehicle import vehicle


class BaseBugReplicated(BaseAlgorithm):
    """
    Variant of BaseAlgorithm that replicates TWO bugs from Azam's thesis:

    BUG 1 - Depot exclusion:
    - Uses decode_route to build routes (which add depot at start and end).
    - BUT depot is removed from stored 'route'
    - Distance is computed only between customer nodes
      (no depot->first, no last->depot).
    - Time is computed only between customer nodes
      (no depot->first, no last->depot).
    
    BUG 2 - Haversine distance:
    - Distance and time are calculated using straight-line haversine distance
      instead of actual road distance from OSRM.
    - This assumes the driver "goes crazy and hits all structures in between 2 nodes"
      (straight line through everything).
    """

    def __init__(self, seed, vrp: Dict[str, Any], scorer: str = "cost") -> None:
        """
        Initialize with haversine-based distance/time matrices.
        
        Args:
            seed: Random seed
            vrp: VRP problem dict (will be modified to use haversine D/T)
            scorer: Scoring method ("cost" or "distance")
        """
        # Create haversine-based D and T matrices before calling super().__init__
        vrp_copy = dict(vrp)
        vrp_copy["D"], vrp_copy["T"] = self._build_haversine_matrices(vrp_copy["nodes"])
        
        # Now initialize base class with modified VRP
        super().__init__(seed, vrp_copy, scorer)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate haversine distance between two points in meters.
        
        This is the straight-line distance "as the crow flies", ignoring roads,
        buildings, terrain, etc.
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c

    def _build_haversine_matrices(
        self, nodes: List[node]
    ) -> tuple[Dict[int, Dict[int, float]], Dict[int, Dict[int, float]]]:
        """
        Build distance and time matrices using haversine (straight-line) distance.
        
        BUG REPLICATION: Instead of using actual road distances from OSRM,
        this calculates straight-line distances, assuming vehicles can travel
        through any obstacle.
        
        Time is set equal to distance (in meters).
        
        Args:
            nodes: List of node objects with id, lat, lon
            
        Returns:
            Tuple of (D, T) where:
                D[i][j] = haversine distance from node i to j in meters
                T[i][j] = haversine distance from node i to j in meters (same as D)
        """
        D: Dict[int, Dict[int, float]] = {}
        T: Dict[int, Dict[int, float]] = {}
        
        for n1 in nodes:
            id1 = int(n1.id)
            D[id1] = {}
            T[id1] = {}
            
            for n2 in nodes:
                id2 = int(n2.id)
                
                if id1 == id2:
                    D[id1][id2] = 0.0
                    T[id1][id2] = 0.0
                else:
                    # Calculate straight-line distance (THE BUG)
                    dist = self._haversine_distance(
                        float(n1.lat), float(n1.lon),
                        float(n2.lat), float(n2.lon)
                    )
                    
                    D[id1][id2] = float(dist)
                    
                    # TIME = DISTANCE (in meters)
                    T[id1][id2] = float(dist)
        
        return D, T

    def _solution_from_perm(self, perm: List[int]) -> List[Dict[str, Any]]:
        """
        Build solution from permutation, replicating the depot-exclusion bug.
        
        BUG 1 REPLICATION: Depot legs are excluded from distance/time calculations.
        BUG 2 REPLICATION: Uses haversine D/T matrices (set in __init__).
        
        Args:
            perm: Permutation of customer IDs (not indices)
            
        Returns:
            List of route dictionaries with buggy distance/time calculations
        """
        # Decode routes (will include depot at start and end)
        routes = decode_route(perm, self.vrp)

        vehicles: List[vehicle] = self.vrp["vehicles"]
        D: Mapping[int, Mapping[int, float]] = self.vrp["D"]  # Haversine-based
        T: Mapping[int, Mapping[int, float]] = self.vrp["T"]  # Haversine-based (same as D)

        sol: List[Dict[str, Any]] = []

        for r, veh in zip(routes, vehicles):
            # r typically looks like [depot, c1, c2, ..., ck, depot]
            # BUG 1 REPLICATION: drop depot completely from stored route
            customer_route: List[int] = [int(x) for x in r if int(x) != self.depot_id]

            # BUGGY Distance: only customer-to-customer edges using HAVERSINE
            # Excludes depot->first and last->depot
            if len(customer_route) < 2:
                total_distance = 0.0
                total_time = 0.0
            else:
                d = 0.0
                t = 0.0
                for a, b in zip(customer_route[:-1], customer_route[1:]):
                    d += float(D[a][b])  # Haversine distance
                    t += float(T[a][b])  # Haversine distance (same as D)
                total_distance = d
                total_time = t

            sol.append({
                "vehicle_id": veh.id,
                "vehicle_name": veh.vehicle_name,
                "vehicle_distance_cost": veh.distance_cost,
                "vehicle_initial_cost": veh.initial_cost,
                "total_distance": float(total_distance),  # BUGGY: haversine + excludes depot
                "total_time": float(total_time),          # BUGGY: haversine + excludes depot
                "route": customer_route,                  # BUGGY: no depot in route
            })

        return sol

    def _compute_details(self, perm: List[int]) -> Dict[str, Any]:
        """
        Compute detailed solution information with buggy calculations.
        
        This override ensures that even the detailed metrics replicate both bugs:
        - BUG 1: Excludes depot travel legs
        - BUG 2: Uses haversine distance/time
        
        Args:
            perm: Permutation of customer IDs
            
        Returns:
            Dictionary with solution details using buggy calculations
        """
        solution = self._solution_from_perm(perm)

        # Total distance and time already computed incorrectly in _solution_from_perm
        total_distance = float(sum(s["total_distance"] for s in solution))
        total_time = float(sum(s["total_time"] for s in solution))

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
                "total_distance": total_distance,      # BUGGY: haversine + excludes depot
                "total_time": total_time,              # BUGGY: haversine + excludes depot (now same as distance)
                "scorer": self.scorer,
                "score": score_val,
            },
            "per_vehicle": solution,
            "permutation": list(perm),  # best permutation (ID space)
            "depot_id": self.depot_id,
        }