# vrp_core/models/vehicle.py
from dataclasses import dataclass, asdict, field
from typing import Optional, Iterable, List, Any


@dataclass(slots=True)
class vehicle:
    id: int
    vehicle_name: str
    initial_cost: float
    distance_cost: float

    # runtime fields
    distance: Optional[float] = None
    route: List[int] = field(default_factory=list)

    # ---------- Constructors ----------
    @classmethod
    def from_dict(cls, d: dict) -> "vehicle":
        return cls(
            id=int(d["id"]),
            vehicle_name=str(d["vehicle_name"]),
            initial_cost=float(d["initial_cost"]),
            distance_cost=float(d["distance_cost"]),
            distance=None if d.get("distance", None) in ("", None) else float(d["distance"]),
            route=cls._coerce_route(d.get("route", [])),
        )

    @classmethod
    def from_csv_row(cls, row: Iterable[str]) -> "vehicle":
        # expected CSV: id,vehicle_name,initial_cost,distance_cost
        id_, name, init, dist = row
        return cls(
            id=int(id_),
            vehicle_name=str(name),
            initial_cost=float(init),
            distance_cost=float(dist),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("id must be int")
        if not isinstance(self.vehicle_name, str):
            raise TypeError("vehicle_name must be str")

        for fname in ("initial_cost", "distance_cost"):
            v = getattr(self, fname)
            if not isinstance(v, (float, int)):
                raise TypeError(f"{fname} must be float-like")
            setattr(self, fname, float(v))

        if self.initial_cost < 0 or self.distance_cost < 0:
            raise ValueError("costs must be >= 0")

        if self.distance is not None:
            self._set_distance(self.distance)

        self.route = self._coerce_route(self.route)

    # ---------- Safe mutators ----------
    def set_distance(self, value: Optional[float]) -> None:
        self._set_distance(value)

    def add_distance(self, delta: float) -> None:
        if delta is None:
            raise TypeError("delta distance cannot be None")
        self._set_distance((self.distance or 0.0) + float(delta))

    def add_stop(self, node_id: Any) -> None:
        try:
            nid = int(node_id)
        except Exception as e:
            raise TypeError("route node_id must be int-castable") from e
        self.route.append(nid)

    def reset_trip(self) -> None:
        self.distance = None
        self.route.clear()

    # ---------- Internal helpers ----------
    def _set_distance(self, value: Optional[float]) -> None:
        if value is None:
            self.distance = None
            return
        val = float(value)
        if val < 0:
            raise ValueError("distance must be >= 0")
        self.distance = val

    @staticmethod
    def _coerce_route(route_like: Iterable[Any]) -> List[int]:
        if route_like is None:
            return []
        try:
            return [int(x) for x in route_like]
        except Exception as e:
            raise TypeError("route must be an iterable of int-castable values") from e
