from pathlib import Path
import csv
import json
import math
import re
import time
import urllib.request


RESULTS_DIR = Path("results/testing/ntnsga2_run1")
NODES_CSV = Path("Problems/GIB.csv")
OUT_DIR = RESULTS_DIR / "distance_check"

OSRM_BASE_URL = "https://router.project-osrm.org"
# OSRM_BASE_URL = "http://localhost:5000"

EARTH_RADIUS_M = 6_371_000.0
REQUEST_SLEEP_S = 0.15
TIMEOUT_S = 30


def haversine_m(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_nodes(path):
    nodes = {}

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            node_id = int(row["id"])
            nodes[node_id] = {
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "address": row.get("address", ""),
            }

    return nodes


def cluster_sort_key(path):
    match = re.search(r"cluster_(\d+)$", path.name)

    if match:
        return int(match.group(1))

    return 10**9


def osrm_route_for_stops(stops, nodes):
    coords = []

    for node_id in stops:
        node = nodes[int(node_id)]
        coords.append(f"{node['lon']},{node['lat']}")

    coord_string = ";".join(coords)

    url = (
        f"{OSRM_BASE_URL}/route/v1/driving/{coord_string}"
        f"?overview=false&steps=false&annotations=false"
    )

    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
        data = json.loads(response.read().decode("utf-8"))

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    return data["routes"][0]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct_diff(a, b):
    if b in (None, 0):
        return None

    return round((a - b) / b * 100, 6)


def main():
    nodes = load_nodes(NODES_CSV)

    leg_rows = []
    route_rows = []
    solution_rows = []
    error_rows = []

    global_reported_distance = 0.0
    global_haversine_distance = 0.0
    global_osrm_distance = 0.0
    global_osrm_available = True

    cluster_dirs = sorted(
        [p for p in RESULTS_DIR.iterdir() if p.is_dir() and p.name.startswith("cluster_")],
        key=cluster_sort_key,
    )

    for cluster_dir in cluster_dirs:
        json_path = cluster_dir / "solution_routes.json"

        if not json_path.exists():
            print(f"Missing: {json_path}")
            continue

        print(f"Checking {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        solution_reported_distance = 0.0
        solution_haversine_distance = 0.0
        solution_osrm_distance = 0.0
        solution_osrm_available = True

        solution_reported_time = 0.0
        solution_osrm_duration = 0.0

        for route_index, route in enumerate(data["routes"]):
            stops = [int(x) for x in route["stops_by_id"]]

            reported_route_distance = float(route.get("distance_m", 0.0))
            reported_route_time = float(route.get("time_s", 0.0))

            haversine_route_distance = 0.0
            osrm_route_distance = None
            osrm_route_duration = None
            osrm_legs = []

            try:
                osrm_data = osrm_route_for_stops(stops, nodes)
                osrm_route_distance = float(osrm_data["distance"])
                osrm_route_duration = float(osrm_data["duration"])
                osrm_legs = osrm_data.get("legs", [])

                time.sleep(REQUEST_SLEEP_S)

            except Exception as e:
                solution_osrm_available = False
                global_osrm_available = False

                error_rows.append(
                    {
                        "cluster": cluster_dir.name,
                        "route_index": route_index,
                        "vehicle_id": route.get("vehicle_id"),
                        "error": repr(e),
                        "stops_by_id": ",".join(str(x) for x in stops),
                    }
                )

            for leg_index, (from_id, to_id) in enumerate(zip(stops[:-1], stops[1:])):
                a = nodes[from_id]
                b = nodes[to_id]

                hav_dist = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
                haversine_route_distance += hav_dist

                reported_leg_distance = None
                reported_leg_time = None

                if "legs" in route and leg_index < len(route["legs"]):
                    reported_leg_distance = float(route["legs"][leg_index]["distance_m"])
                    reported_leg_time = float(route["legs"][leg_index]["time_s"])

                osrm_leg_distance = None
                osrm_leg_duration = None

                if leg_index < len(osrm_legs):
                    osrm_leg_distance = float(osrm_legs[leg_index]["distance"])
                    osrm_leg_duration = float(osrm_legs[leg_index]["duration"])

                leg_rows.append(
                    {
                        "cluster": cluster_dir.name,
                        "route_index": route_index,
                        "vehicle_id": route.get("vehicle_id"),
                        "vehicle_name": route.get("vehicle_name"),
                        "leg_index": leg_index,
                        "from_id": from_id,
                        "to_id": to_id,

                        "reported_leg_distance_m": reported_leg_distance,
                        "haversine_leg_distance_m": round(hav_dist, 3),
                        "osrm_leg_distance_m": (
                            round(osrm_leg_distance, 3)
                            if osrm_leg_distance is not None
                            else None
                        ),

                        "reported_minus_haversine_m": (
                            round(reported_leg_distance - hav_dist, 3)
                            if reported_leg_distance is not None
                            else None
                        ),
                        "reported_minus_osrm_m": (
                            round(reported_leg_distance - osrm_leg_distance, 3)
                            if reported_leg_distance is not None and osrm_leg_distance is not None
                            else None
                        ),
                        "osrm_minus_haversine_m": (
                            round(osrm_leg_distance - hav_dist, 3)
                            if osrm_leg_distance is not None
                            else None
                        ),

                        "reported_leg_time_s": reported_leg_time,
                        "osrm_leg_duration_s": (
                            round(osrm_leg_duration, 3)
                            if osrm_leg_duration is not None
                            else None
                        ),
                    }
                )

            route_rows.append(
                {
                    "cluster": cluster_dir.name,
                    "route_index": route_index,
                    "vehicle_id": route.get("vehicle_id"),
                    "vehicle_name": route.get("vehicle_name"),

                    "reported_route_distance_m": round(reported_route_distance, 3),
                    "haversine_route_distance_m": round(haversine_route_distance, 3),
                    "osrm_route_distance_m": (
                        round(osrm_route_distance, 3)
                        if osrm_route_distance is not None
                        else None
                    ),

                    "reported_minus_haversine_m": round(
                        reported_route_distance - haversine_route_distance, 3
                    ),
                    "reported_minus_osrm_m": (
                        round(reported_route_distance - osrm_route_distance, 3)
                        if osrm_route_distance is not None
                        else None
                    ),
                    "osrm_minus_haversine_m": (
                        round(osrm_route_distance - haversine_route_distance, 3)
                        if osrm_route_distance is not None
                        else None
                    ),

                    "reported_minus_haversine_pct": pct_diff(
                        reported_route_distance,
                        haversine_route_distance,
                    ),
                    "reported_minus_osrm_pct": (
                        pct_diff(reported_route_distance, osrm_route_distance)
                        if osrm_route_distance is not None
                        else None
                    ),

                    "reported_route_time_s": reported_route_time,
                    "osrm_route_duration_s": (
                        round(osrm_route_duration, 3)
                        if osrm_route_duration is not None
                        else None
                    ),

                    "stops_by_id": ",".join(str(x) for x in stops),
                }
            )

            solution_reported_distance += reported_route_distance
            solution_haversine_distance += haversine_route_distance
            solution_reported_time += reported_route_time

            if osrm_route_distance is not None:
                solution_osrm_distance += osrm_route_distance
                solution_osrm_duration += osrm_route_duration or 0.0

        solution_rows.append(
            {
                "cluster": cluster_dir.name,
                "solution_file": str(json_path),

                "reported_final_distance_m": round(solution_reported_distance, 3),
                "haversine_final_distance_m": round(solution_haversine_distance, 3),
                "osrm_final_distance_m": (
                    round(solution_osrm_distance, 3)
                    if solution_osrm_available
                    else None
                ),

                "reported_minus_haversine_m": round(
                    solution_reported_distance - solution_haversine_distance, 3
                ),
                "reported_minus_osrm_m": (
                    round(solution_reported_distance - solution_osrm_distance, 3)
                    if solution_osrm_available
                    else None
                ),
                "osrm_minus_haversine_m": (
                    round(solution_osrm_distance - solution_haversine_distance, 3)
                    if solution_osrm_available
                    else None
                ),

                "reported_minus_haversine_pct": pct_diff(
                    solution_reported_distance,
                    solution_haversine_distance,
                ),
                "reported_minus_osrm_pct": (
                    pct_diff(solution_reported_distance, solution_osrm_distance)
                    if solution_osrm_available
                    else None
                ),

                "reported_final_time_s": round(solution_reported_time, 3),
                "osrm_final_duration_s": (
                    round(solution_osrm_duration, 3)
                    if solution_osrm_available
                    else None
                ),

                "routes_count": len(data["routes"]),
            }
        )

        global_reported_distance += solution_reported_distance
        global_haversine_distance += solution_haversine_distance

        if solution_osrm_available:
            global_osrm_distance += solution_osrm_distance

    solution_rows.append(
        {
            "cluster": "__ALL_CLUSTERS__",
            "solution_file": "",

            "reported_final_distance_m": round(global_reported_distance, 3),
            "haversine_final_distance_m": round(global_haversine_distance, 3),
            "osrm_final_distance_m": (
                round(global_osrm_distance, 3)
                if global_osrm_available
                else None
            ),

            "reported_minus_haversine_m": round(
                global_reported_distance - global_haversine_distance, 3
            ),
            "reported_minus_osrm_m": (
                round(global_reported_distance - global_osrm_distance, 3)
                if global_osrm_available
                else None
            ),
            "osrm_minus_haversine_m": (
                round(global_osrm_distance - global_haversine_distance, 3)
                if global_osrm_available
                else None
            ),

            "reported_minus_haversine_pct": pct_diff(
                global_reported_distance,
                global_haversine_distance,
            ),
            "reported_minus_osrm_pct": (
                pct_diff(global_reported_distance, global_osrm_distance)
                if global_osrm_available
                else None
            ),

            "reported_final_time_s": "",
            "osrm_final_duration_s": "",
            "routes_count": "",
        }
    )

    write_csv(OUT_DIR / "distance_check_legs.csv", leg_rows)
    write_csv(OUT_DIR / "distance_check_routes.csv", route_rows)
    write_csv(OUT_DIR / "distance_check_final.csv", solution_rows)
    write_csv(OUT_DIR / "distance_check_errors.csv", error_rows)

    print()
    print(f"Wrote {OUT_DIR / 'distance_check_legs.csv'}")
    print(f"Wrote {OUT_DIR / 'distance_check_routes.csv'}")
    print(f"Wrote {OUT_DIR / 'distance_check_final.csv'}")
    print(f"Wrote {OUT_DIR / 'distance_check_errors.csv'}")

    print()
    print("FINAL DISTANCE CHECK")
    print(f"reported : {round(global_reported_distance, 3)} m")
    print(f"haversine: {round(global_haversine_distance, 3)} m")
    print(f"osrm     : {round(global_osrm_distance, 3) if global_osrm_available else None} m")


if __name__ == "__main__":
    main()