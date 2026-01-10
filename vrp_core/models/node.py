# vrp_core/models/node.py
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(slots=True)
class node:
    id: int
    lat: float
    lon: float

    @classmethod
    def from_dict(cls, d: dict) -> "node":
        return cls(
            id=int(d["id"]),
            lat=float(d["lat"]),
            lon=float(d["lon"]),
        )

    @classmethod
    def from_csv_row(cls, row: Iterable[str]) -> "node":
        # expects: id, lat, lon
        id_, lat, lon = row
        return cls(
            id=int(id_),
            lat=float(lat),
            lon=float(lon),
        )

    def to_dict(self) -> dict:
        return asdict(self)
