# vrp_core/models/node.py
from dataclasses import dataclass, asdict
from typing import Optional, Iterable


@dataclass(slots=True)
class node:
    id: int
    lat: float
    lon: float
    demand: Optional[float] = None  # empty -> None

    @classmethod
    def from_dict(cls, d: dict) -> "node":
        # accept strings or numbers; treat "" as None for demand
        demand = d.get("demand", None)
        if demand == "" or demand is None:
            demand_val = None
        else:
            demand_val = float(demand)
        return cls(
            id=int(d["id"]),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
            demand=demand_val,
        )

    @classmethod
    def from_csv_row(cls, row: Iterable[str]) -> "node":
        id_, lat, lon, demand = row
        return cls(
            id=int(id_),
            lat=float(lat),
            lon=float(lon),
            demand=None if demand == "" else float(demand),
        )

    def to_dict(self) -> dict:
        return asdict(self)
