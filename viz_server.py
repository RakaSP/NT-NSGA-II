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


# ----------------------------
# Helpers
# ----------------------------
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

def _extract_routes(payload: Any) -> List[List[int]]:
    """
    Extract routes as list-of-list of node IDs from various JSON shapes.

    MUST support your format:
      solution_routes.json:
      {
        "routes": [
          {"stops_by_id": [0, 56, ..., 0], ...},
          ...
        ]
      }
    """
    def is_route_list(x: Any) -> bool:
        return isinstance(x, list) and all(isinstance(i, (int, str)) for i in x) and len(x) >= 2

    def to_int_route(x: List[Any]) -> List[int]:
        out = []
        for v in x:
            iv = _int_or_none(v)
            if iv is not None:
                out.append(iv)
        return out

    # 1) direct list-of-lists
    if isinstance(payload, list):
        if all(isinstance(r, list) for r in payload) and all(is_route_list(r) for r in payload):
            return [to_int_route(r) for r in payload]

    # 2) dict common keys
    if isinstance(payload, dict):
        # ---- your exact format ----
        if "routes" in payload and isinstance(payload["routes"], list):
            routes: List[List[int]] = []
            for item in payload["routes"]:
                if isinstance(item, dict) and "stops_by_id" in item and isinstance(item["stops_by_id"], list):
                    if is_route_list(item["stops_by_id"]):
                        routes.append(to_int_route(item["stops_by_id"]))
            if routes:
                return routes

        # legacy: direct list-of-lists under common keys
        for key in ("route_list", "routes_id", "routes_ids"):
            if key in payload and isinstance(payload[key], list):
                cand = payload[key]
                if all(isinstance(r, list) for r in cand) and all(is_route_list(r) for r in cand):
                    return [to_int_route(r) for r in cand]

        # per_vehicle style
        for key in ("per_vehicle", "vehicles", "vehicle_routes"):
            if key in payload and isinstance(payload[key], list):
                routes = []
                for item in payload[key]:
                    if isinstance(item, dict) and "route" in item and isinstance(item["route"], list):
                        if is_route_list(item["route"]):
                            routes.append(to_int_route(item["route"]))
                if routes:
                    return routes

        # nested solution-ish
        for key in ("solution", "best", "data"):
            if key in payload:
                routes = _extract_routes(payload[key])
                if routes:
                    return routes

        # last resort: scan values
        for v in payload.values():
            routes = _extract_routes(v)
            if routes:
                return routes

    return []

def _extract_route_records_from_solution_routes(payload: Any) -> List[Dict[str, Any]]:
    """
    For your exact solution_routes.json format:
    {
      "routes": [
        {
          "distance_m": ...,
          "time_s": ...,
          "stops_by_id": [...],
          ...
        }, ...
      ]
    }
    Returns list of dicts, each guaranteed to have 'stops_by_id' as List[int].
    """
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
    """
    results_root/
      <run_name_1>/cluster_0 ...
      <run_name_2>/cluster_0 ...
    Return absolute paths to each <run_name>.
    """
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
    """
    coords_lonlat: [[lon,lat], [lon,lat], ...]
    Returns:
      - geometry as [[lat,lon], ...] (Leaflet-friendly)
      - distance_m (float)
      - duration_s (float)
    """
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
      <div class="muted">Compare: file vs OSRM-route vs OSRM-table (D[a][b]). Click a route row to see legs.</div>

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

      <div style="margin-top:10px;">
        <div class="small" style="font-weight:600; margin-bottom:6px;">Legs (D[a][b])</div>
        <div id="legsTable">—</div>
      </div>
    </div>

    <div class="card">
      <div style="font-weight:700;">Metrics</div>
      <div class="muted">Best/Mean over iterations (if metrics_*.csv exists)</div>
      <canvas id="chart" height="140"></canvas>
    </div>

    <div class="card">
      <div style="font-weight:700;">Routes</div>
      <div class="muted">Readable list. Click a route to highlight it + show legs above.</div>
      <div id="routesList" class="small" style="margin-top:8px;">—</div>
    </div>
  </div>
</div>

<script>
let map, nodesLayer, routesLayer;
let chart;
let lastPolylines = [];
let polylineRefs = [];
let selectedRouteIdx = null;

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
  const data = await apiGet(`/api/clusters?run=${encodeURIComponent(runKey)}`);
  const sel = document.getElementById("clusterSel");
  sel.innerHTML = "";
  for (const c of data.clusters) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = `cluster_${c}`;
    sel.appendChild(opt);
  }
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
  // crude highlight: set weight/opacity
  polylineRefs.forEach((p, i) => {
    try {
      if (i === idx) p.setStyle({ weight: 7, opacity: 1.0 });
      else p.setStyle({ weight: 4, opacity: 0.55 });
    } catch {}
  });
}

function drawRoutes(routePolylines) {
  clearRoutes();
  lastPolylines = routePolylines || [];
  if (!routePolylines || routePolylines.length === 0) return;

  routePolylines.forEach((r, idx) => {
    const color = randColor(idx);
    const line = L.polyline(r.coords, { color, weight: 4, opacity: 0.9 });
    const d = (r.osrm_distance_m != null) ? ` • ${fmtMeters(r.osrm_distance_m)}` : "";
    line.bindPopup(`route #${idx} (${r.mode})${d}`);
    line.addTo(routesLayer);
    polylineRefs.push(line);
  });
  highlightRoute(selectedRouteIdx);
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
    ["Sum table distance (D[a][b])", fmtMeters(totals.sum_table_distance_m)],
    ["Diff OSRM - file", `${fmtMeters(totals.diff_sum_osrm_vs_file_m)} (${fmtPct(totals.diff_sum_osrm_vs_file_pct)})`],
    ["Diff OSRM - table", `${fmtMeters(totals.diff_sum_osrm_vs_table_m)} (${fmtPct(totals.diff_sum_osrm_vs_table_pct)})`],
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

  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, [
    el("th", {}, ["#"]),
    el("th", {}, ["Stops"]),
    el("th", {}, ["File dist"]),
    el("th", {}, ["OSRM dist"]),
    el("th", {}, ["Table dist"]),
    el("th", {}, ["OSRM-file"]),
    el("th", {}, ["OSRM-table"]),
  ]));
  table.appendChild(thead);

  const tbody = el("tbody");
  routes.forEach((r, idx) => {
    const tr = el("tr", {
      class: `click ${selectedRouteIdx===idx ? "rowhi" : ""}`,
      onclick: () => selectRoute(idx, routes)
    }, [
      el("td", {}, [String(idx)]),
      el("td", {class:"nowrap"}, [String((r.stops_by_id || []).length)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.file_distance_m)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.osrm_route_distance_m)]),
      el("td", {class:"nowrap"}, [fmtMeters(r.table_distance_m)]),
      el("td", {class:"nowrap"}, [`${fmtMeters(r.diff_osrm_vs_file_m)} (${fmtPct(r.diff_osrm_vs_file_pct)})`]),
      el("td", {class:"nowrap"}, [`${fmtMeters(r.diff_osrm_vs_table_m)} (${fmtPct(r.diff_osrm_vs_table_pct)})`]),
    ]);
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
}

function renderLegs(legs) {
  const wrap = document.getElementById("legsTable");
  wrap.innerHTML = "";
  if (!legs || legs.length === 0) { wrap.textContent = "—"; return; }

  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, [
    el("th", {}, ["From"]),
    el("th", {}, ["To"]),
    el("th", {}, ["Table dist"]),
  ]));
  table.appendChild(thead);

  const tbody = el("tbody");
  legs.forEach((l) => {
    tbody.appendChild(el("tr", {}, [
      el("td", {class:"nowrap"}, [String(l.from)]),
      el("td", {class:"nowrap"}, [String(l.to)]),
      el("td", {class:"nowrap"}, [fmtMeters(l.table_distance_m)]),
    ]));
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
    ["Stops", String((r.stops_by_id || []).length)],
    ["File dist / time", `${fmtMeters(r.file_distance_m)} • ${fmtSeconds(r.file_time_s)}`],
    ["OSRM dist / time", `${fmtMeters(r.osrm_route_distance_m)} • ${fmtSeconds(r.osrm_route_time_s)}`],
    ["Table dist", fmtMeters(r.table_distance_m)],
    ["OSRM-file", `${fmtMeters(r.diff_osrm_vs_file_m)} (${fmtPct(r.diff_osrm_vs_file_pct)})`],
    ["OSRM-table", `${fmtMeters(r.diff_osrm_vs_table_m)} (${fmtPct(r.diff_osrm_vs_table_pct)})`],
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
  renderSelectedRouteBox(r);
  renderLegs(r ? r.legs : null);

  // re-render route table highlight
  renderRoutesCheck(routesCheck);

  // map highlight
  highlightRoute(idx);
}

function renderRoutesList(routeChecks) {
  const wrap = document.getElementById("routesList");
  wrap.innerHTML = "";
  if (!routeChecks || routeChecks.length === 0) { wrap.textContent = "—"; return; }

  const table = el("table");
  const thead = el("thead");
  thead.appendChild(el("tr", {}, [
    el("th", {}, ["#"]),
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

  // nicer ordering if these exist
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

  setStatus(`Loading ${runKey} / cluster_${clusterId} ...`);
  selectedRouteIdx = null;

  const data = await apiGet(`/api/view?run=${encodeURIComponent(runKey)}&cluster=${encodeURIComponent(clusterId)}`);

  drawNodes(data.nodes || []);
  drawRoutes(data.route_polylines || []);
  drawChart(data.metrics || {});
  renderSummary(data.summary || {});

  const dc = data.distance_check || {};
  renderTotals(dc.totals || {});
  renderRoutesCheck(dc.routes || []);
  renderRoutesList(dc.routes || []);

  // auto-select first route if exists
  if ((dc.routes || []).length > 0) {
    selectRoute(0, dc.routes);
  } else {
    document.getElementById("selectedRouteBox").textContent = "—";
    document.getElementById("legsTable").textContent = "—";
  }

  setStatus(`Loaded ${runKey} / cluster_${clusterId}`);
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

        # root_dir should be "results"
        self.root_dir = os.path.abspath(root_dir)

        # fixed filenames
        self.routes_out = "solution_routes.json"
        self.summary_out = "solution_summary.json"

        # discover run dirs inside results/
        self.run_dirs = _discover_runs_under_results(self.root_dir)

        # Load one full problem for node coordinates + (optional) D/T matrices
        vrp = load_problem_from_config(self.cfg)

        nodes = vrp.get("nodes", [])
        self.depot_id = int(nodes[0].id) if nodes else 0
        self.nodes = [
            {"id": int(n.id), "lat": float(n.lat), "lon": float(n.lon), "is_depot": int(n.id) == self.depot_id}
            for n in nodes
        ]
        self.nodes_map = {int(n.id): (float(n.lat), float(n.lon)) for n in nodes}

        # OSRM-table distances used by solver (if present)
        self.D = vrp.get("D", None)  # D[a][b] meters
        self.T = vrp.get("T", None)  # T[a][b] seconds

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

    def _sum_table_distance(self, stops: List[int]) -> Tuple[Optional[float], List[Dict[str, Any]]]:
        """
        Sum D[a][b] over consecutive legs.
        Returns (total_m or None, legs list).
        """
        if not self.D:
            return None, []

        legs: List[Dict[str, Any]] = []
        total = 0.0
        ok_any = False
        for a, b in zip(stops[:-1], stops[1:]):
            d = None
            try:
                d = float(self.D[a][b])
                ok_any = True
                total += d
            except Exception:
                d = None
            legs.append({"from": int(a), "to": int(b), "table_distance_m": d})
        return (total if ok_any else None), legs

    @staticmethod
    def _pct(diff: Optional[float], base: Optional[float]) -> Optional[float]:
        if diff is None or base is None:
            return None
        if abs(base) < 1e-9:
            return None
        return 100.0 * diff / base

    def load_view(self, run_name: str, cluster_id: int) -> Dict[str, Any]:
        cdir = self.cluster_dir(run_name, cluster_id)

        # Summary (solution_summary.json)
        summary = {}
        p_sum = os.path.join(cdir, self.summary_out)
        if _safe_exists(p_sum):
            try:
                summary = _safe_read_json(p_sum)
            except Exception:
                summary = {}

        # Routes JSON (solution_routes.json)
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

        # Map routes -> OSRM geometry + distance check
        polylines: List[Dict[str, Any]] = []
        distance_check: Dict[str, Any] = {"routes": [], "totals": {}}

        sum_file = 0.0
        sum_osrm = 0.0
        sum_table = 0.0
        have_file = False
        have_osrm = False
        have_table = False

        for idx, r in enumerate(routes):
            # Build lonlat waypoints for OSRM
            lonlat: List[List[float]] = []
            for nid in r:
                if nid in self.nodes_map:
                    lat, lon = self.nodes_map[nid]
                    lonlat.append([lon, lat])  # OSRM expects [lon,lat]

            if len(lonlat) < 2:
                continue

            # OSRM full route geometry + its route distance
            road_coords, osrm_dist_m, osrm_dur_s = _osrm_route_full(
                lonlat, base_url=self.osrm_base_url, profile="driving"
            )

            if road_coords and len(road_coords) >= 2:
                polylines.append({
                    "coords": road_coords,
                    "mode": "osrm",
                    "osrm_distance_m": osrm_dist_m,
                    "osrm_duration_s": osrm_dur_s,
                })
            else:
                straight = [[lat, lon] for lon, lat in lonlat]
                polylines.append({
                    "coords": straight,
                    "mode": "straight",
                    "osrm_distance_m": None,
                    "osrm_duration_s": None,
                })

            # file distance/time (if present in solution_routes.json)
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

            # table distance using D[a][b]
            table_total, legs = self._sum_table_distance(r)

            # totals
            if file_dist is not None:
                have_file = True
                sum_file += file_dist
            if osrm_dist_m is not None:
                have_osrm = True
                sum_osrm += osrm_dist_m
            if table_total is not None:
                have_table = True
                sum_table += table_total

            # diffs
            diff_osrm_file = (osrm_dist_m - file_dist) if (osrm_dist_m is not None and file_dist is not None) else None
            diff_osrm_table = (osrm_dist_m - table_total) if (osrm_dist_m is not None and table_total is not None) else None
            diff_table_file = (table_total - file_dist) if (table_total is not None and file_dist is not None) else None

            distance_check["routes"].append({
                "route_index": idx,
                "stops_by_id": r,
                "file_distance_m": file_dist,
                "file_time_s": file_time,
                "osrm_route_distance_m": osrm_dist_m,
                "osrm_route_time_s": osrm_dur_s,
                "table_distance_m": table_total,
                "diff_osrm_vs_file_m": diff_osrm_file,
                "diff_osrm_vs_file_pct": self._pct(diff_osrm_file, file_dist),
                "diff_osrm_vs_table_m": diff_osrm_table,
                "diff_osrm_vs_table_pct": self._pct(diff_osrm_table, table_total),
                "diff_table_vs_file_m": diff_table_file,
                "diff_table_vs_file_pct": self._pct(diff_table_file, file_dist),
                "legs": legs,  # per-edge D[a][b]
            })

        distance_check["totals"] = {
            "sum_file_distance_m": (sum_file if have_file else None),
            "sum_osrm_route_distance_m": (sum_osrm if have_osrm else None),
            "sum_table_distance_m": (sum_table if have_table else None),
            "diff_sum_osrm_vs_file_m": (sum_osrm - sum_file) if (have_osrm and have_file) else None,
            "diff_sum_osrm_vs_file_pct": self._pct((sum_osrm - sum_file) if (have_osrm and have_file) else None, (sum_file if have_file else None)),
            "diff_sum_osrm_vs_table_m": (sum_osrm - sum_table) if (have_osrm and have_table) else None,
            "diff_sum_osrm_vs_table_pct": self._pct((sum_osrm - sum_table) if (have_osrm and have_table) else None, (sum_table if have_table else None)),
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


# ----------------------------
# HTTP Handler
# ----------------------------
class Handler(BaseHTTPRequestHandler):
    state: AppState = None  # type: ignore

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
            cid = _int_or_none(cluster)
            if cid is None:
                self._send_json({"error": "cluster must be int"}, status=400)
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
        default="results",
        help="Results root folder containing runs: results/<run_name>/cluster_*",
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    ap.add_argument("--osrm", default="https://router.project-osrm.org", help="OSRM base URL (e.g., http://localhost:5000)")
    args = ap.parse_args()

    state = AppState(args.config, root_dir=args.root, osrm_base_url=args.osrm)
    Handler.state = state  # type: ignore

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
