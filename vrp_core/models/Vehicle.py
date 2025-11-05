from dataclasses import dataclass, asdict, field
from typing import Optional, Iterable, List, Any


@dataclass(slots=True)
class Vehicle:
    id: int
    vehicle_name: str
    max_capacity: float
    initial_cost: float
    distance_cost: float

    # new runtime fields
    used_capacity: Optional[float] = None
    distance: Optional[float] = None
    route: List[int] = field(default_factory=list)

    # ---------- Constructors ----------
    @classmethod
    def from_dict(cls, d: dict) -> "Vehicle":
        return cls(
            id=int(d["id"]),
            vehicle_name=str(d["vehicle_name"]),
            max_capacity=float(
                d["max_capacity"] if "max_capacity" in d else d["capacity"]),
            initial_cost=float(d["initial_cost"]),
            distance_cost=float(d["distance_cost"]),
            used_capacity=None if d.get("used_capacity", None) in (
                "", None) else float(d["used_capacity"]),
            distance=None if d.get("distance", None) in (
                "", None) else float(d["distance"]),
            route=cls._coerce_route(d.get("route", [])),
        )

    @classmethod
    def from_csv_row(cls, row: Iterable[str]) -> "Vehicle":
        # Supports your original CSV: id,vehicle_name,capacity,initial_cost,distance_cost
        id_, name, cap, init, dist = row
        return cls(
            id=int(id_),
            vehicle_name=str(name),
            max_capacity=float(cap),
            initial_cost=float(init),
            distance_cost=float(dist),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # ---------- Validation & safeguards ----------
    def __post_init__(self):
        # basic type checks
        if not isinstance(self.id, int):
            raise TypeError("id must be int")
        if not isinstance(self.vehicle_name, str):
            raise TypeError("vehicle_name must be str")
        for fname in ("max_capacity", "initial_cost", "distance_cost"):
            v = getattr(self, fname)
            if not isinstance(v, (float, int)):  # allow ints but store as float
                raise TypeError(f"{fname} must be float-like")
            setattr(self, fname, float(v))

        # non-negativity
        if self.max_capacity < 0:
            raise ValueError("max_capacity must be >= 0")
        if self.initial_cost < 0 or self.distance_cost < 0:
            raise ValueError("costs must be >= 0")

        # optional fields
        if self.used_capacity is not None:
            self._set_used_capacity(self.used_capacity)  # validates & casts
        if self.distance is not None:
            self._set_distance(self.distance)            # validates & casts

        # route sanity
        self.route = self._coerce_route(self.route)

    # ---------- Safe mutators ----------
    def set_used_capacity(self, value: Optional[float]) -> None:
        self._set_used_capacity(value)

    def add_used_capacity(self, delta: float) -> None:
        new_val = (self.used_capacity or 0.0) + float(delta)
        self._set_used_capacity(new_val)

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
        self.used_capacity = None
        self.distance = None
        self.route.clear()

    # ---------- Internal helpers ----------
    def _set_used_capacity(self, value: Optional[float]) -> None:
        if value is None:
            self.used_capacity = None
            return
        val = float(value)
        if val < 0:
            raise ValueError("used_capacity must be >= 0")
        if val > self.max_capacity:
            raise ValueError(
                f"used_capacity ({val}) cannot exceed max_capacity ({self.max_capacity})")
        self.used_capacity = val

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
            lst = [int(x) for x in route_like]
        except Exception as e:
            raise TypeError(
                "route must be an iterable of int-castable values") from e
        return lst
