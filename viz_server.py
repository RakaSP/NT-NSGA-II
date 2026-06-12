#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import urllib.request
import urllib.parse

from vrp_core import load_config, load_problem_from_config

def _safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False

def _safe_listdir(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except Exception:
        return []

def _int_or_none(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

class _SimpleNode:
    __slots__ = ("id", "lat", "lon")
    def __init__(self, nid: int, lat: float, lon: float):
        self.id = nid
        self.lat = lat
        self.lon = lon

def _resolve_path(p: str, base_dir: str) -> str:
    p = str(p)
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(base_dir, p))

def _load_nodes_only(nodes_path: str, base_dir: str) -> List[_SimpleNode]:
    path = _resolve_path(nodes_path, base_dir)
    if not _safe_exists(path):
        raise FileNotFoundError(f"nodes file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    def get_lat_lon(rec: Any) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(rec, dict):
            return None, None
        lat = rec.get("lat", rec.get("latitude"))
        lon = rec.get("lon", rec.get("lng", rec.get("longitude")))
        # sometimes coord: [lat, lon] or [lon, lat]
        if (lat is None or lon is None) and "coord" in rec and isinstance(rec["coord"], (list, tuple)) and len(rec["coord"]) >= 2:
            a, b = rec["coord"][0], rec["coord"][1]
            try:
                a = float(a); b = float(b)
                if abs(a) <= 90 and abs(b) <= 180:
                    lat, lon = a, b
                else:
                    lon, lat = a, b
            except Exception:
                pass
        try:
            latf = float(lat) if lat is not None else None
            lonf = float(lon) if lon is not None else None
            return latf, lonf
        except Exception:
            return None, None

    def get_id(rec: Any, fallback: int) -> int:
        if isinstance(rec, dict):
            for k in ("id", "node_id", "index", "idx"):
                if k in rec:
                    iv = _int_or_none(rec.get(k))
                    if iv is not None:
                        return iv
        return fallback

    out: List[_SimpleNode] = []

    if ext == ".json":
        data = _safe_read_json(path)
        if isinstance(data, dict):
            cand = None
            for k in ("nodes", "locations", "stops", "customers"):
                if k in data and isinstance(data[k], list):
                    cand = data[k]
                    break
            if cand is None and isinstance(data.get("data"), list):
                cand = data["data"]
            nodes_list = cand if isinstance(cand, list) else []
        elif isinstance(data, list):
            nodes_list = data
        else:
            nodes_list = []

        for i, rec in enumerate(nodes_list):
            if not isinstance(rec, dict):
                continue
            nid = get_id(rec, i)
            lat, lon = get_lat_lon(rec)
            if lat is None or lon is None:
                continue
            out.append(_SimpleNode(int(nid), float(lat), float(lon)))

        if not out:
            raise ValueError(f"Could not parse any nodes with lat/lon from JSON: {path}")
        return out

    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        df = pd.read_csv(path, sep=sep)
        cols = {c.lower(): c for c in df.columns}

        def col(*names: str) -> Optional[str]:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_id = col("id", "node_id", "index", "idx")
        c_lat = col("lat", "latitude")
        c_lon = col("lon", "lng", "longitude")

        if c_lat is None or c_lon is None:
            raise ValueError(f"CSV missing lat/lon columns: {path}")

        for i, row in df.iterrows():
            nid = int(row[c_id]) if (c_id and pd.notna(row[c_id])) else int(i)
            try:
                lat = float(row[c_lat])
                lon = float(row[c_lon])
            except Exception:
                continue
            out.append(_SimpleNode(nid, lat, lon))

        if not out:
            raise ValueError(f"Could not parse any nodes with lat/lon from CSV: {path}")
        return out

    raise ValueError(f"Unsupported nodes file type for fallback load: {path}")


def _extract_routes(payload: Any) -> List[List[int]]:
    def is_route_list(x: Any) -> bool:
        return isinstance(x, list) and all(isinstance(i, (int, str)) for i in x) and len(x) >= 2

    def to_int_route(x: List[Any]) -> List[int]:
        out = []
        for v in x:
            iv = _int_or_none(v)
            if iv is not None:
                out.append(iv)
        return out

    if isinstance(payload, list):
        if all(isinstance(r, list) for r in payload) and all(is_route_list(r) for r in payload):
            return [to_int_route(r) for r in payload]

    if isinstance(payload, dict):
        if "routes" in payload and isinstance(payload["routes"], list):
            routes: List[List[int]] = []
            for item in payload["routes"]:
                if isinstance(item, dict) and "stops_by_id" in item and isinstance(item["stops_by_id"], list):
                    if is_route_list(item["stops_by_id"]):
                        routes.append(to_int_route(item["stops_by_id"]))
            if routes:
                return routes

        for key in ("route_list", "routes_id", "routes_ids"):
            if key in payload and isinstance(payload[key], list):
                cand = payload[key]
                if all(isinstance(r, list) for r in cand) and all(is_route_list(r) for r in cand):
                    return [to_int_route(r) for r in cand]

        for key in ("per_vehicle", "vehicles", "vehicle_routes"):
            if key in payload and isinstance(payload[key], list):
                routes = []
                for item in payload[key]:
                    if isinstance(item, dict) and "route" in item and isinstance(item["route"], list):
                        if is_route_list(item["route"]):
                            routes.append(to_int_route(item["route"]))
                if routes:
                    return routes

        for key in ("solution", "best", "data"):
            if key in payload:
                routes = _extract_routes(payload[key])
                if routes:
                    return routes

        for v in payload.values():
            routes = _extract_routes(v)
            if routes:
                return routes

    return []

def _extract_route_records_from_solution_routes(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return out
    routes = payload.get("routes")
    if not isinstance(routes, list):
        return out

    for item in routes:
        if not isinstance(item, dict):
            continue
        stops = item.get("stops_by_id")
        if not isinstance(stops, list):
            continue
        stops_i = []
        for v in stops:
            iv = _int_or_none(v)
            if iv is not None:
                stops_i.append(iv)
        if len(stops_i) >= 2:
            rec = dict(item)
            rec["stops_by_id"] = stops_i
            out.append(rec)
    return out

def _find_metrics_csv(cluster_dir: str) -> Optional[str]:
    for fn in _safe_listdir(cluster_dir):
        if fn.startswith("metrics") and fn.endswith(".csv"):
            return os.path.join(cluster_dir, fn)
    return None

def _read_metrics(metrics_path: str) -> Dict[str, Any]:
    try:
        df = pd.read_csv(metrics_path)
        out: Dict[str, Any] = {}
        for col in ("iter", "best", "mean"):
            if col in df.columns:
                out[col] = df[col].tolist()
        return out
    except Exception:
        return {}

def _scan_clusters(run_dir: str) -> List[int]:
    clusters = []
    for x in _safe_listdir(run_dir):
        m = re.match(r"^cluster_(\d+)$", x)
        if m:
            clusters.append(int(m.group(1)))
    return sorted(clusters)

def _looks_like_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for name in _safe_listdir(path):
        if re.match(r"^cluster_\d+$", name):
            return True
    return False

def _discover_runs_under_results(results_root: str) -> List[str]:
    results_root = os.path.abspath(results_root)
    out: List[str] = []
    for name in _safe_listdir(results_root):
        p = os.path.join(results_root, name)
        if _looks_like_run_dir(p):
            out.append(os.path.abspath(p))
    out.sort()
    return out

def _osrm_route_full(
    coords_lonlat: List[List[float]],
    base_url: str,
    profile: str = "driving",
) -> Tuple[Optional[List[List[float]]], Optional[float], Optional[float]]:
    if len(coords_lonlat) < 2:
        return None, None, None

    coord_str = ";".join([f"{lon:.6f},{lat:.6f}" for lon, lat in coords_lonlat])
    url = f"{base_url.rstrip('/')}/route/v1/{profile}/{coord_str}"
    qs = urllib.parse.urlencode({
        "overview": "full",
        "geometries": "geojson",
        "steps": "false",
    })
    full = url + "?" + qs

    try:
        with urllib.request.urlopen(full, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("code") != "Ok":
            return None, None, None

        routes = data.get("routes") or []
        if not routes:
            return None, None, None

        r0 = routes[0]
        dist = r0.get("distance")
        dur = r0.get("duration")

        geom = r0.get("geometry")
        if not geom or geom.get("type") != "LineString":
            return None, float(dist) if dist is not None else None, float(dur) if dur is not None else None

        line_lonlat = geom.get("coordinates") or []
        coords_latlon = [[float(lat), float(lon)] for lon, lat in line_lonlat]
        return coords_latlon, (float(dist) if dist is not None else None), (float(dur) if dur is not None else None)
    except Exception:
        return None, None, None


# ----------------------------
# HTML page (single-file)
# ----------------------------
INDEX_HTML = r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>VRP Visualizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />

  <!-- Leaflet -->
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>

  <!-- Chart.js -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>

  <style>
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
    header { padding: 10px 14px; border-bottom: 1px solid #eee; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
    header .title { font-weight: 700; margin-right: 10px; }
    select, button { padding: 6px 8px; }
    #wrap { display: grid; grid-template-columns: 1.4fr 1fr; gap: 0; height: calc(100vh - 56px); }
    #map { width: 100%; height: 100%; }
    #side { padding: 12px; overflow: auto; border-left: 1px solid #eee; }
    .card { border: 1px solid #eee; border-radius: 10px; padding: 10px; margin-bottom: 12px; }
    .muted { color: #666; font-size: 13px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 12px; white-space: pre-wrap; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid #eee; padding: 6px 6px; vertical-align: top; }
    th { text-align: left; background: #fafafa; position: sticky; top: 0; z-index: 1; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #f2f2f2; font-size: 12px; }
    .click { cursor: pointer; }
    .rowhi { background: #fff7e6; }
    .small { font-size: 12px; color: #444; }
    .two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .right { text-align: right; }
    .nowrap { white-space: nowrap; }
  </style>
</head>

<body>
<header>
  <div class="title">VRP Visualizer</div>

  <label>Run:</label>
  <select id="runSel"></select>

  <label>Cluster:</label>
  <select id="clusterSel"></select>

  <button id="reloadBtn">Reload</button>
  <span class="muted" id="status"></span>
</header>

<div id="wrap">
  <div id="map"></div>
  <div id="side">
    <div class="card">
      <div style="font-weight:700; display:flex; justify-content:space-between; align-items:center;">
        <span>Summary</span>
        <span class="pill" id="sumPill">—</span>
      </div>
      <div id="summaryTable" class="small" style="margin-top:8px;">—</div>
    </div>

    <div class="card">
      <div style="font-weight:700;">Distance check</div>
      <div class="muted">Compare: file vs OSRM-route. Click a route row to highlight it.</div>

      <div class="two" style="margin-top:10px;">
        <div>
          <div class="small" style="font-weight:600; margin-bottom:6px;">Totals</div>
          <div id="totalsTable">—</div>
        </div>
        <div>
          <div class="small" style="font-weight:600; margin-bottom:6px;">Selected route</div>
          <div id="selectedRouteBox" class="small">—</div>
        </div>
      </div>

      <div style="margin-top:10px;">
        <div class="small" style="font-weight:600; margin-bottom:6px;">Routes</div>
        <div id="routesCheckTable">—</div>
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700;">Metrics</div>
      <div class="muted">Best/Mean over iterations (if metrics_*.csv exists)</div>
      <canvas id="chart" height="140"></canvas>
    </div>

    <div class="card">
      <div style="font-weight:700;">Routes</div>
      <div class="muted">Readable list. Click a route to highlight it.</div>
      <div id="routesList" class="small" style="margin-top:8px;">—</div>
    </div>
  </div>
</div>

<script>
let map, nodesLayer, routesLayer;
let chart;
let polylineRefs = [];
let selectedRouteIdx = null;      // row index
let selectedPolylineIdx = null;   // map polyline index

function setStatus(msg) {
  document.getElementById("status").textContent = msg || "";
}

function randColor(i) {
  const hue = (i * 57) % 360;
  return `hsl(${hue}, 75%, 45%)`;
}

function fmtNum(x, digits=2) {
  if (x == null || Number.isNaN(x)) return "—";
  const n = Number(x);
  if (!Number.isFinite(n)) return String(x);
  return n.toFixed(digits);
}

function fmtMeters(m) {
  if (m == null || Number.isNaN(m)) return "—";
  const n = Number(m);
  if (!Number.isFinite(n)) return String(m);
  if (n >= 1000) return `${(n/1000).toFixed(3)} km`;
  return `${Math.round(n)} m`;
}

function fmtSeconds(s) {
  if (s == null || Number.isNaN(s)) return "—";
  const n = Number(s);
  if (!Number.isFinite(n)) return String(s);
  if (n >= 3600) return `${(n/3600).toFixed(2)} h`;
  if (n >= 60) return `${(n/60).toFixed(2)} min`;
  return `${Math.round(n)} s`;
}

function fmtPct(p) {
  if (p == null || Number.isNaN(p)) return "—";
  const n = Number(p);
  if (!Number.isFinite(n)) return String(p);
  const sign = n > 0 ? "+" : "";
  return `${sign}${n.toFixed(2)}%`;
}

async function apiGet(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

function el(tag, attrs={}, children=[]) {
  const e = document.createElement(tag);
  for (const [k,v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function renderKVTable(obj) {
  if (!obj || typeof obj !== "object") return el("div", {class:"small"}, ["—"]);
  const keys = Object.keys(obj);
  if (keys.length === 0) return el("div", {class:"small"}, ["—"]);

  const table = el("table");
  const tbody = el("tbody");
  for (const k of keys) {
    const v = obj[k];
    let display = v;
    if (typeof v === "number") display = fmtNum(v, 6);
    else if (typeof v === "object") display = JSON.stringify(v);
    tbody.appendChild(el("tr", {}, [
      el("td", {class:"nowrap", style:"width:40%; font-weight:600;"}, [k]),
      el("td", {}, [String(display)]),
    ]));
  }
  table.appendChild(tbody);
  return table;
}

async function loadRuns() {
  const data = await apiGet("/api/runs");
  const runSel = document.getElementById("runSel");
  runSel.innerHTML = "";
  for (const r of data.runs) {
    const opt = document.createElement("option");
    opt.value = r;
    opt.textContent = r;
    runSel.appendChild(opt);
  }
  if (data.runs.length === 0) {
    setStatus("No runs found under --root. Expected: --root results");
  }
}

async function loadClusters() {
  const runKey = document.getElementById("runSel").value;
  const sel = document.getElementById("clusterSel");
  const prev = sel.value;

  const data = await apiGet(`/api/clusters?run=${encodeURIComponent(runKey)}`);
  sel.innerHTML = "";

  for (const c of data.clusters) {
    const opt = document.createElement("option");
    opt.value = String(c);
    opt.textContent = `cluster_${c}`;
    sel.appendChild(opt);
  }

  // NEW: all clusters
  const optAll = document.createElement("option");
  optAll.value = "all";
  optAll.textContent = "All clusters";
  sel.appendChild(optAll);

  // restore previous if possible
  const hasPrev = [...sel.options].some(o => o.value === prev);
  if (hasPrev) sel.value = prev;
  else if (data.clusters.length > 0) sel.value = String(data.clusters[0]);
  else sel.value = "all";

  if (data.clusters.length === 0) {
    setStatus(`No cluster_* folders inside ${runKey}`);
  }
}

function ensureMap() {
  if (map) return;
  map = L.map("map");
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);

  nodesLayer = L.layerGroup().addTo(map);
  routesLayer = L.layerGroup().addTo(map);
}

function drawNodes(nodes) {
  nodesLayer.clearLayers();
  if (!nodes || nodes.length === 0) return;

  const latlngs = [];
  for (const n of nodes) {
    const ll = [n.lat, n.lon];
    latlngs.push(ll);
    const isDepot = (n.is_depot === true);
    const radius = isDepot ? 7 : 4;
    const color = isDepot ? "#000" : "#333";
    L.circleMarker(ll, { radius, color, weight: 2, fillOpacity: 0.7 })
      .bindPopup(`id=${n.id}${isDepot ? " (depot)" : ""}`)
      .addTo(nodesLayer);
  }
  const bounds = L.latLngBounds(latlngs);
  map.fitBounds(bounds.pad(0.15));
}

function clearRoutes() {
  routesLayer.clearLayers();
  polylineRefs = [];
}

function highlightRoute(idx) {
  polylineRefs.forEach((p, i) => {
    try {
      if (idx == null) {
        p.setStyle({ weight: 4, opacity: 0.9 });
      } else if (i === idx) {
        p.setStyle({ weight: 7, opacity: 1.0 });
      } else {
        p.setStyle({ weight: 4, opacity: 0.55 });
      }
    } catch {}
  });
}

function drawRoutes(routePolylines) {
  clearRoutes();
  if (!routePolylines || routePolylines.length === 0) return;

  routePolylines.forEach((r, idx) => {
    const color = randColor(idx);
    const line = L.polyline(r.coords, { color, weight: 4, opacity: 0.9 });

    const d = (r.osrm_distance_m != null) ? ` • ${fmtMeters(r.osrm_distance_m)}` : "";
    const label = (r.cluster_id != null)
      ? `cluster_${r.cluster_id} • route #${(r.route_index != null ? r.route_index : "?")}`
      : `route #${idx}`;

    line.bindPopup(`${label} (${r.mode})${d}`);
    line.addTo(routesLayer);
    polylineRefs.push(line);
  });

  highlightRoute(selectedPolylineIdx);
}

function drawChart(metrics) {
  const ctx = document.getElementById("chart");
  const it = metrics?.iter || [];
  const best = metrics?.best || [];
  const mean = metrics?.mean || [];

  const data = {
    labels: it,
    datasets: [
      { label: "best", data: best, borderWidth: 2, pointRadius: 0 },
      { label: "mean", data: mean, borderWidth: 2, pointRadius: 0 },
    ]
  };

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: "line",
    data,
    options: {
      responsive: true,
      animation: false,
      scales: { x: { display: true }, y: { display: true } }
    }
  });
}

function renderTotals(totals) {
  const wrap = document.getElementById("totalsTable");
  wrap.innerHTML = "";
  if (!totals) { wrap.textContent = "—"; return; }

  const rows = [
    ["Sum file distance", fmtMeters(totals.sum_file_distance_m)],
    ["Sum OSRM route distance", fmtMeters(totals.sum_osrm_route_distance_m)],
    ["Diff OSRM - file", `${fmtMeters(totals.diff_sum_osrm_vs_file_m)} (${fmtPct(totals.diff_sum_osrm_vs_file_pct)})`],
  ];

  const table = el("table");
  const tbody = el("tbody");
  for (const [k,v] of rows) {
    tbody.appendChild(el("tr", {}, [
      el("td", {style:"font-weight:600;"}, [k]),
      el("td", {class:"right nowrap"}, [v]),
    ]));
  }
  table.appendChild(tbody);
  wrap.appendChild(table);
}

function renderRoutesCheck(routes) {
  const wrap = document.getElementById("routesCheckTable");
  wrap.innerHTML = "";
  if (!routes || routes.length === 0) { wrap.textContent = "—"; return; }

  const showCluster = routes.some(r => r.cluster_id != null);

  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, [
    el("th", {}, ["#"]),
    ...(showCluster ? [el("th", {}, ["Cluster"])] : []),
    el("th", {}, ["Stops"]),
    el("th", {}, ["File dist"]),
    el("th", {}, ["OSRM dist"]),
    el("th", {}, ["OSRM-file"]),
  ]));
  table.appendChild(thead);

  const tbody = el("tbody");
  routes.forEach((r, idx) => {
    const tr = el("tr", {
      class: `click ${selectedRouteIdx===idx ? "rowhi" : ""}`,
      onclick: () => selectRoute(idx, routes)
    }, [
      el("td", {}, [String(idx)]),
      ...(showCluster ? [el("td", {class:"nowrap"}, [`cluster_${r.cluster_id}`])] : []),
      el("td", {class:"nowrap"}, [String((r.stops_by_id || []).length)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.file_distance_m)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.osrm_route_distance_m)]),
      el("td", {class:"nowrap"}, [`${fmtMeters(r.diff_osrm_vs_file_m)} (${fmtPct(r.diff_osrm_vs_file_pct)})`]),
    ]);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
}

function renderSelectedRouteBox(r) {
  const box = document.getElementById("selectedRouteBox");
  box.innerHTML = "";
  if (!r) { box.textContent = "—"; return; }

  const stops = (r.stops_by_id || []).join(" → ");
  const rows = [
    ...(r.cluster_id != null ? [["Cluster", `cluster_${r.cluster_id}`]] : []),
    ["Stops", String((r.stops_by_id || []).length)],
    ["File dist / time", `${fmtMeters(r.file_distance_m)} • ${fmtSeconds(r.file_time_s)}`],
    ["OSRM dist / time", `${fmtMeters(r.osrm_route_distance_m)} • ${fmtSeconds(r.osrm_route_time_s)}`],
    ["OSRM-file", `${fmtMeters(r.diff_osrm_vs_file_m)} (${fmtPct(r.diff_osrm_vs_file_pct)})`],
  ];

  const table = el("table");
  const tbody = el("tbody");
  for (const [k,v] of rows) {
    tbody.appendChild(el("tr", {}, [
      el("td", {style:"font-weight:600;"}, [k]),
      el("td", {class:"right nowrap"}, [v]),
    ]));
  }
  table.appendChild(tbody);

  box.appendChild(table);
  box.appendChild(el("div", {class:"small", style:"margin-top:8px;"}, [
    el("div", {style:"font-weight:600; margin-bottom:4px;"}, ["Stops sequence"]),
    el("div", {class:"mono"}, [stops || "—"]),
  ]));
}

function selectRoute(idx, routesCheck) {
  selectedRouteIdx = idx;
  const r = routesCheck && routesCheck[idx] ? routesCheck[idx] : null;

  // map highlight: use polyline_index when provided (all-clusters mode)
  const pi = (r && r.polyline_index != null) ? Number(r.polyline_index) : idx;
  selectedPolylineIdx = Number.isFinite(pi) ? pi : idx;

  renderSelectedRouteBox(r);
  renderRoutesCheck(routesCheck);
  highlightRoute(selectedPolylineIdx);
}

function renderRoutesList(routeChecks) {
  const wrap = document.getElementById("routesList");
  wrap.innerHTML = "";
  if (!routeChecks || routeChecks.length === 0) { wrap.textContent = "—"; return; }

  const showCluster = routeChecks.some(r => r.cluster_id != null);

  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, [
    el("th", {}, ["#"]),
    ...(showCluster ? [el("th", {}, ["Cluster"])] : []),
    el("th", {}, ["Stops (preview)"]),
    el("th", {}, ["File dist"]),
    el("th", {}, ["OSRM dist"]),
  ]));
  table.appendChild(thead);

  const tbody = el("tbody");
  routeChecks.forEach((r, idx) => {
    const stops = r.stops_by_id || [];
    const preview = stops.length <= 10
      ? stops.join(" → ")
      : (stops.slice(0,4).join(" → ") + " → … → " + stops.slice(-3).join(" → "));

    tbody.appendChild(el("tr", {
      class: `click ${selectedRouteIdx===idx ? "rowhi" : ""}`,
      onclick: () => selectRoute(idx, routeChecks),
    }, [
      el("td", {}, [String(idx)]),
      ...(showCluster ? [el("td", {class:"nowrap"}, [`cluster_${r.cluster_id}`])] : []),
      el("td", {class:"mono"}, [preview]),
      el("td", {class:"nowrap"}, [fmtMeters(r.file_distance_m)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.osrm_route_distance_m)]),
    ]));
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
}

function renderSummary(summary) {
  const pill = document.getElementById("sumPill");
  const box = document.getElementById("summaryTable");
  box.innerHTML = "";

  if (!summary || typeof summary !== "object" || Object.keys(summary).length === 0) {
    pill.textContent = "no summary";
    box.textContent = "—";
    return;
  }

  const algo = summary.algorithm || "—";
  const scorer = summary.scorer || "—";
  pill.textContent = `${algo} / ${scorer}`;

  const preferred = [
    "algorithm","scorer","final_distance","final_cost","final_time","runtime_s","iterations"
  ];
  const ordered = {};
  for (const k of preferred) if (k in summary) ordered[k] = summary[k];
  for (const k of Object.keys(summary)) if (!(k in ordered)) ordered[k] = summary[k];

  box.appendChild(renderKVTable(ordered));
}

async function reloadAll() {
  ensureMap();

  const runKey = document.getElementById("runSel").value;
  const clusterId = document.getElementById("clusterSel").value;

  setStatus(`Loading ${runKey} / ${clusterId === "all" ? "all" : "cluster_" + clusterId} ...`);
  selectedRouteIdx = null;
  selectedPolylineIdx = null;

  const data = await apiGet(`/api/view?run=${encodeURIComponent(runKey)}&cluster=${encodeURIComponent(clusterId)}`);

  drawNodes(data.nodes || []);
  drawRoutes(data.route_polylines || []);
  drawChart(data.metrics || {});
  renderSummary(data.summary || {});

  const dc = data.distance_check || {};
  renderTotals(dc.totals || {});
  renderRoutesCheck(dc.routes || []);
  renderRoutesList(dc.routes || []);

  if ((dc.routes || []).length > 0) {
    selectRoute(0, dc.routes);
  } else {
    document.getElementById("selectedRouteBox").textContent = "—";
  }

  setStatus(`Loaded ${runKey} / ${clusterId === "all" ? "all clusters" : "cluster_" + clusterId}`);
}

async function init() {
  await loadRuns();
  await loadClusters();
  await reloadAll();

  document.getElementById("runSel").addEventListener("change", async () => {
    await loadClusters();
    await reloadAll();
  });
  document.getElementById("clusterSel").addEventListener("change", reloadAll);
  document.getElementById("reloadBtn").addEventListener("click", reloadAll);
}

init().catch(e => {
  console.error(e);
  setStatus("Error: " + e.message);
});
</script>
</body>
</html>
"""


# ----------------------------
# Server state
# ----------------------------
class AppState:
    def __init__(self, cfg_path: str, root_dir: str, osrm_base_url: str):
        self.cfg_path = cfg_path
        self.cfg = load_config(cfg_path)
        self.osrm_base_url = osrm_base_url

        self.root_dir = os.path.abspath(root_dir)

        self.routes_out = "solution_routes.json"
        self.summary_out = "solution_summary.json"

        self.run_dirs = _discover_runs_under_results(self.root_dir)

        base_dir = os.path.dirname(os.path.abspath(cfg_path))
        try:
            vrp = load_problem_from_config(self.cfg)
        except FileNotFoundError as e:
            print(f"[viz] WARNING: {e}")
            print("[viz] Continuing with nodes-only fallback.")
            nodes_only = _load_nodes_only(str(self.cfg.get("nodes", "")), base_dir=base_dir)
            vrp = {"nodes": nodes_only}

        nodes = vrp.get("nodes", [])
        self.depot_id = int(nodes[0].id) if nodes else 0
        self.nodes = [
            {"id": int(n.id), "lat": float(n.lat), "lon": float(n.lon), "is_depot": int(n.id) == self.depot_id}
            for n in nodes
        ]
        self.nodes_map = {int(n.id): (float(n.lat), float(n.lon)) for n in nodes}

    def runs(self) -> List[str]:
        return [os.path.basename(p) for p in self.run_dirs]

    def _get_run_dir(self, run_name: str) -> str:
        for p in self.run_dirs:
            if os.path.basename(p) == run_name:
                return p
        return self.run_dirs[0] if self.run_dirs else ""

    def clusters(self, run_name: str) -> List[int]:
        run_dir = self._get_run_dir(run_name)
        return _scan_clusters(run_dir)

    def cluster_dir(self, run_name: str, cluster_id: int) -> str:
        run_dir = self._get_run_dir(run_name)
        return os.path.join(run_dir, f"cluster_{cluster_id}")

    @staticmethod
    def _pct(diff: Optional[float], base: Optional[float]) -> Optional[float]:
        if diff is None or base is None:
            return None
        if abs(base) < 1e-9:
            return None
        return 100.0 * diff / base

    def load_view(self, run_name: str, cluster_id: int) -> Dict[str, Any]:
        cdir = self.cluster_dir(run_name, cluster_id)

        summary = {}
        p_sum = os.path.join(cdir, self.summary_out)
        if _safe_exists(p_sum):
            try:
                summary = _safe_read_json(p_sum)
            except Exception:
                summary = {}

        routes_path = os.path.join(cdir, self.routes_out)
        routes: List[List[int]] = []
        route_records: List[Dict[str, Any]] = []
        if _safe_exists(routes_path):
            try:
                routes_payload = _safe_read_json(routes_path)
                routes = _extract_routes(routes_payload)
                route_records = _extract_route_records_from_solution_routes(routes_payload)
            except Exception:
                routes = []
                route_records = []

        polylines: List[Dict[str, Any]] = []
        distance_check: Dict[str, Any] = {"routes": [], "totals": {}}

        sum_file = 0.0
        sum_osrm = 0.0
        have_file = False
        have_osrm = False

        for idx, r in enumerate(routes):
            lonlat: List[List[float]] = []
            for nid in r:
                if nid in self.nodes_map:
                    lat, lon = self.nodes_map[nid]
                    lonlat.append([lon, lat])

            if len(lonlat) < 2:
                continue

            road_coords, osrm_dist_m, osrm_dur_s = _osrm_route_full(
                lonlat, base_url=self.osrm_base_url, profile="driving"
            )

            if road_coords and len(road_coords) >= 2:
                polylines.append({
                    "coords": road_coords,
                    "mode": "osrm",
                    "osrm_distance_m": osrm_dist_m,
                    "osrm_duration_s": osrm_dur_s,
                    "route_index": idx,
                })
            else:
                straight = [[lat, lon] for lon, lat in lonlat]
                polylines.append({
                    "coords": straight,
                    "mode": "straight",
                    "osrm_distance_m": None,
                    "osrm_duration_s": None,
                    "route_index": idx,
                })

            file_dist = None
            file_time = None
            if idx < len(route_records):
                rec = route_records[idx]
                file_dist = rec.get("distance_m")
                file_time = rec.get("time_s")
                try:
                    file_dist = float(file_dist) if file_dist is not None else None
                except Exception:
                    file_dist = None
                try:
                    file_time = float(file_time) if file_time is not None else None
                except Exception:
                    file_time = None

            if file_dist is not None:
                have_file = True
                sum_file += file_dist
            if osrm_dist_m is not None:
                have_osrm = True
                sum_osrm += osrm_dist_m

            diff_osrm_file = (osrm_dist_m - file_dist) if (osrm_dist_m is not None and file_dist is not None) else None

            distance_check["routes"].append({
                "route_index": idx,
                "stops_by_id": r,
                "file_distance_m": file_dist,
                "file_time_s": file_time,
                "osrm_route_distance_m": osrm_dist_m,
                "osrm_route_time_s": osrm_dur_s,
                "diff_osrm_vs_file_m": diff_osrm_file,
                "diff_osrm_vs_file_pct": self._pct(diff_osrm_file, file_dist),
            })

        distance_check["totals"] = {
            "sum_file_distance_m": (sum_file if have_file else None),
            "sum_osrm_route_distance_m": (sum_osrm if have_osrm else None),
            "diff_sum_osrm_vs_file_m": (sum_osrm - sum_file) if (have_osrm and have_file) else None,
            "diff_sum_osrm_vs_file_pct": self._pct((sum_osrm - sum_file) if (have_osrm and have_file) else None, (sum_file if have_file else None)),
        }

        metrics = {}
        mp = _find_metrics_csv(cdir)
        if mp:
            metrics = _read_metrics(mp)

        return {
            "run": run_name,
            "cluster_id": int(cluster_id),
            "cluster_dir": cdir,
            "nodes": self.nodes,
            "routes": routes,
            "route_polylines": polylines,
            "metrics": metrics,
            "summary": summary,
            "distance_check": distance_check,
        }

    def load_view_all(self, run_name: str) -> Dict[str, Any]:
        """
        Overlay all cluster_* inside a run.
        We DO NOT assume any extra files. We just reuse existing per-cluster outputs.
        """
        clusters = self.clusters(run_name)

        all_polylines: List[Dict[str, Any]] = []
        all_routes_check: List[Dict[str, Any]] = []
        metrics_any: Dict[str, Any] = {}
        summary_any: Dict[str, Any] = {"note": "all clusters"}

        for cid in clusters:
            v = self.load_view(run_name, cid)
            pol = v.get("route_polylines", []) or []
            dc = v.get("distance_check", {}) or {}
            rc = dc.get("routes", []) or []

            offset = len(all_polylines)

            for p in pol:
                p2 = dict(p)
                p2["cluster_id"] = cid
                all_polylines.append(p2)

            for local_i, r in enumerate(rc):
                r2 = dict(r)
                r2["cluster_id"] = cid
                r2["polyline_index"] = offset + local_i
                all_routes_check.append(r2)

        return {
            "run": run_name,
            "cluster_id": "all",
            "all_clusters": True,
            "nodes": self.nodes,
            "routes": [],
            "route_polylines": all_polylines,
            "metrics": metrics_any,
            "summary": summary_any,
            "distance_check": {
                "routes": all_routes_check,
                "totals": {},
            },
        }


class Handler(BaseHTTPRequestHandler):
    state: AppState = None

    def _send(self, status: int, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: Any, status: int = 200):
        body = json.dumps(payload).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/api/runs":
            self._send_json({"runs": self.state.runs()})
            return

        if path == "/api/clusters":
            runs = self.state.runs()
            default_run = runs[0] if runs else ""
            run_name = (qs.get("run", [default_run])[0] or default_run)
            self._send_json({"run": run_name, "clusters": self.state.clusters(run_name)})
            return

        if path == "/api/view":
            runs = self.state.runs()
            default_run = runs[0] if runs else ""
            run_name = (qs.get("run", [default_run])[0] or default_run)

            cluster = qs.get("cluster", ["0"])[0]

            if str(cluster).lower() in ("all", "*"):
                try:
                    payload = self.state.load_view_all(run_name)
                    self._send_json(payload)
                except Exception as e:
                    self._send_json({"error": str(e)}, status=500)
                return

            cid = _int_or_none(cluster)
            if cid is None:
                self._send_json({"error": "cluster must be int or 'all'"}, status=400)
                return

            try:
                payload = self.state.load_view(run_name, cid)
                self._send_json(payload)
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)
            return

        self._send(404, b"Not Found", "text/plain; charset=utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="solver_config.yaml", help="Path to solver_config.yaml")
    ap.add_argument(
        "--root",
        default="results/GIB",
        help="Results root folder containing runs: results/<run_name>/cluster_*",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--osrm", default="https://router.project-osrm.org", help="OSRM base URL (e.g., http://localhost:5000)")
    args = ap.parse_args()

    state = AppState(args.config, root_dir=args.root, osrm_base_url=args.osrm)
    Handler.state = state

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)

    print(f"[viz] results_root={state.root_dir}")
    print(f"[viz] discovered_runs={len(state.run_dirs)}")
    if state.run_dirs:
        print("[viz] discovered runs:")
        for p in state.run_dirs:
            rel = os.path.relpath(p, start=state.root_dir)
            print(f"  - {rel}")
    else:
        print("[viz] (none) expected layout: results/<run_name>/cluster_0 ...")

    print(f"[viz] osrm={state.osrm_base_url}")
    print(f"[viz] open: http://{args.host}:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()