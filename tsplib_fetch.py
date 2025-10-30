#!/usr/bin/env python3
from __future__ import annotations

import re
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import requests
import tsplib95


DEFAULT_BASE_URL = "https://comopt.ifi.uni-heidelberg.de/software/TSPLIB95"
TSP_SUBDIR = "tsp"   # both .tsp and .opt.tour typically live here


def fetch_text(url: str, timeout: int = 30) -> str:
    headers = {
        "User-Agent": "tsplib-grabber/1.0 (+https://example.local)"
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    # Most TSPLIB files are ASCII/plain text.
    r.encoding = r.encoding or "utf-8"
    return r.text


def list_instances_with_opt(base_url: str) -> List[str]:
    """
    Parse the directory listing to get all instance names that have *.opt.tour.
    Returns the bare stems (e.g., 'a280', 'gr666').
    """
    index_url = f"{base_url}/{TSP_SUBDIR}/"
    html = fetch_text(index_url)
    # Find links like 'a280.opt.tour' (case-insensitive just in case)
    names = sorted(set(m.group(1) for m in re.finditer(
        r'href=["\']([A-Za-z0-9_\-]+)\.opt\.tour["\']', html, flags=re.IGNORECASE)))
    return names


def download_file(url: str, dest: Path, timeout: int = 30) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "tsplib-grabber/1.0 (+https://example.local)"}
    with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)


def compute_opt_length(tsp_path: Path, opt_tour_path: Path) -> Dict[str, Optional[int]]:
    prob = tsplib95.load(str(tsp_path))
    tour = tsplib95.load(str(opt_tour_path))
    # trace_tours returns list(s) of lengths; we take the first
    weights = prob.trace_tours(tour.tours)  # type: ignore[attr-defined]
    return {
        "name": prob.name,
        "dimension": int(prob.dimension) if getattr(prob, "dimension", None) else None,
        "optimal_length": int(weights[0]) if weights else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download TSPLIB TSP cases that have global optimal tours and write a manifest.")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help=f"TSPLIB base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--dest", type=str, default="TSPLIB95_local",
                        help="Destination directory to store files (default: ./TSPLIB95_local)")
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Optional list of instance names to restrict to (e.g., a280 gr666 lin318)")
    parser.add_argument("--sleep", type=float, default=0.2,
                        help="Seconds to sleep between downloads (default: 0.2)")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    dest_root = Path(args.dest)
    tsp_dir = dest_root / TSP_SUBDIR

    print(f"[INFO] Scanning {base_url}/{TSP_SUBDIR}/ for *.opt.tour …")
    all_with_opt = list_instances_with_opt(base_url)
    if not all_with_opt:
        print("[WARN] No *.opt.tour files found. Check the base URL or connectivity.")
        return

    if args.only:
        requested = {n.lower() for n in args.only}
        names = [n for n in all_with_opt if n.lower() in requested]
        missing = requested - {n.lower() for n in names}
        if missing:
            print(f"[WARN] The following requested names were not found with .opt.tour: {sorted(missing)}")
    else:
        names = all_with_opt

    print(f"[INFO] Found {len(names)} instance(s) with *.opt.tour.")

    rows: List[Dict[str, object]] = []
    for i, name in enumerate(names, 1):
        tsp_url = f"{base_url}/{TSP_SUBDIR}/{name}.tsp"
        tour_url = f"{base_url}/{TSP_SUBDIR}/{name}.opt.tour"

        tsp_path = tsp_dir / f"{name}.tsp"
        tour_path = tsp_dir / f"{name}.opt.tour"

        try:
            if not tsp_path.exists():
                print(f"[{i}/{len(names)}] Downloading {name}.tsp …")
                download_file(tsp_url, tsp_path)
                time.sleep(args.sleep)
            else:
                print(f"[{i}/{len(names)}] {name}.tsp already exists, skipping.")

            if not tour_path.exists():
                print(f"[{i}/{len(names)}] Downloading {name}.opt.tour …")
                download_file(tour_url, tour_path)
                time.sleep(args.sleep)
            else:
                print(f"[{i}/{len(names)}] {name}.opt.tour already exists, skipping.")

            info = compute_opt_length(tsp_path, tour_path)
            row = {
                "name": info["name"],
                "dimension": info["dimension"],
                "optimal_length": info["optimal_length"],
                "tsp_path": str(tsp_path.resolve()),
                "opt_tour_path": str(tour_path.resolve()),
            }
            rows.append(row)
            print(f"      ✓ OPT={row['optimal_length']}  n={row['dimension']}  ({name})")
        except requests.HTTPError as e:
            print(f"      ✗ HTTP error for {name}: {e}")
        except Exception as e:
            print(f"      ✗ Failed to process {name}: {e}")

    # Write CSV manifest
    manifest_path = dest_root / "tsplib_opt_cases.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["name", "dimension", "optimal_length", "tsp_path", "opt_tour_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Done. Wrote manifest with {len(rows)} rows to: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
