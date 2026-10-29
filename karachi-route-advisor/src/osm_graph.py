from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import networkx as nx
import requests

from .config import (
    BBOX,
    DEFAULT_MAX_SPEED_KPH,
    GRAPH_PATH,
    HIGHWAY_SPEEDS_KPH,
    BBox,
)
from .eta import haversine_m, parse_maxspeed

OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"

DRIVE_HIGHWAY_TYPES: frozenset[str] = frozenset(
    {
        "motorway",
        "motorway_link",
        "trunk",
        "trunk_link",
        "primary",
        "primary_link",
        "secondary",
        "secondary_link",
        "tertiary",
        "tertiary_link",
        "unclassified",
        "residential",
        "living_street",
        "service",
        "road",
    }
)

_USEFUL_WAY_TAGS = (
    "highway",
    "name",
    "maxspeed",
    "oneway",
    "surface",
    "lit",
    "toll",
    "access",
)


class GraphLoadError(RuntimeError):
    pass


def _osmnx() -> Any:
    try:
        import osmnx as ox
    except ImportError as exc:
        raise GraphLoadError(
            "OSMnx is not installed. Install requirements.txt or provide a cached GraphML file."
        ) from exc
    return ox


def _highway_values(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        raw_values: Any = value
    else:
        text = str(value).strip()
        if not text:
            return ()
        raw_values = (text,)
        if text.startswith(("[", "(")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
            if isinstance(parsed, (list, tuple, set)):
                raw_values = parsed
    values = (str(item).strip().lower() for item in raw_values)
    return tuple(item for item in values if item)


def is_drive_highway(value: Any) -> bool:
    values = _highway_values(value)
    return bool(values) and all(item in DRIVE_HIGHWAY_TYPES for item in values)


def filter_drive_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    filtered = nx.MultiDiGraph()
    filtered.graph.update(graph.graph)
    kept_nodes: set[Any] = set()
    for start, end, key, data in graph.edges(keys=True, data=True):
        if not is_drive_highway(data.get("highway")):
            continue
        filtered.add_edge(start, end, key=key, **data)
        kept_nodes.add(start)
        kept_nodes.add(end)
    for node, data in graph.nodes(data=True):
        if node in kept_nodes:
            filtered.add_node(node, **data)
    return filtered


def _edge_road_type(data: dict[str, Any]) -> str:
    road = data.get("road_type")
    if road is not None and is_drive_highway(road):
        values = _highway_values(road)
        if values:
            return values[0]
    highway = _highway_values(data.get("highway"))
    if highway and all(item in DRIVE_HIGHWAY_TYPES for item in highway):
        return highway[0]
    return "unclassified"


def _configure_useful_tags(ox: Any) -> None:
    ox.settings.useful_tags_way = list(
        set(ox.settings.useful_tags_way + list(_USEFUL_WAY_TAGS))
    )


def load_graph(
    path: Path = GRAPH_PATH,
    bbox: BBox = BBOX,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force_refresh:
        graph = _osmnx().load_graphml(path)
        return prepare_graph(graph)

    ox = _osmnx()
    try:
        graph = _graph_from_osm_api(bbox, path)
    except Exception as api_error:
        try:
            graph = ox.graph_from_bbox(
                bbox=bbox.osmnx,
                network_type="drive",
                simplify=True,
                retain_all=False,
            )
            graph = prepare_graph(graph)
        except Exception as overpass_error:
            raise GraphLoadError(
                "Could not fetch the OSM graph from the OSM API or Overpass. "
                f"OSM API: {type(api_error).__name__}; "
                f"Overpass: {type(overpass_error).__name__}."
            ) from overpass_error
    ox.save_graphml(graph, path)
    return graph


def _graph_from_osm_api(bbox: BBox, graph_path: Path) -> nx.MultiDiGraph:
    xml_path = graph_path.with_suffix(".osm")
    response = requests.get(
        OSM_API_URL,
        params={"bbox": f"{bbox.west},{bbox.south},{bbox.east},{bbox.north}"},
        headers={"User-Agent": "karachi-route-advisor/1.0"},
        timeout=120,
    )
    if response.status_code != 200:
        raise GraphLoadError(f"OSM API returned HTTP {response.status_code}.")
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_bytes(response.content)
    return graph_from_osm_xml(xml_path)


def graph_from_osm_xml(path: Path) -> nx.MultiDiGraph:
    ox = _osmnx()
    _configure_useful_tags(ox)
    graph = ox.graph_from_xml(path, simplify=False, retain_all=True)
    graph = filter_drive_graph(graph)
    if graph.number_of_edges() > 0:
        graph = ox.simplify_graph(graph)
    return prepare_graph(graph)


def prepare_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    graph = filter_drive_graph(graph)
    graph = largest_strongly_connected_component(graph)
    _ensure_edge_attributes(graph)
    return graph


def largest_strongly_connected_component(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if graph.number_of_nodes() == 0:
        return graph
    components = nx.strongly_connected_components(graph)
    largest = max(components, key=len)
    return graph.subgraph(largest).copy()


def _ensure_edge_attributes(graph: nx.MultiDiGraph) -> None:
    for _, _, _, data in graph.edges(keys=True, data=True):
        if "length_m" not in data:
            data["length_m"] = float(data.get("length", 0.0))
        road = _edge_road_type(data)
        data["road_type"] = road
        if "road_name" not in data:
            name = data.get("name", "")
            if isinstance(name, (list, tuple)):
                name = name[0] if name else ""
            data["road_name"] = str(name)
        if "speed_kph" not in data:
            parsed = parse_maxspeed(data.get("maxspeed"))
            data["speed_kph"] = (
                parsed
                if parsed is not None and parsed > 0
                else HIGHWAY_SPEEDS_KPH.get(str(road), DEFAULT_MAX_SPEED_KPH)
            )
        data["base_time_s"] = float(data["length_m"]) / (
            float(data["speed_kph"]) * 1000.0 / 3600.0
        )
        surface = data.get("surface", "unknown")
        if isinstance(surface, (list, tuple)):
            surface = surface[0] if surface else "unknown"
        data["surface"] = str(surface)
        lit = data.get("lit", "unknown")
        if isinstance(lit, (list, tuple)):
            lit = lit[0] if lit else "unknown"
        data["lit"] = str(lit).lower()
        toll = data.get("toll", "no")
        if isinstance(toll, (list, tuple)):
            toll = toll[0] if toll else "no"
        data["toll"] = str(toll).lower() in {"yes", "true", "1"}
        access = data.get("access", "")
        if isinstance(access, (list, tuple)):
            access = access[0] if access else ""
        data["access"] = str(access).lower()


def to_routing_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    drive_graph = filter_drive_graph(graph)
    routing = nx.DiGraph()
    for node, data in drive_graph.nodes(data=True):
        routing.add_node(node, x=data.get("x"), y=data.get("y"))
    for start, end, key, data in drive_graph.edges(keys=True, data=True):
        existing = routing.get_edge_data(start, end)
        candidate = {
            "key": key,
            "length_m": float(data.get("length_m", data.get("length", 0.0))),
            "base_time_s": float(data.get("base_time_s", 0.0)),
            "speed_kph": float(data.get("speed_kph", DEFAULT_MAX_SPEED_KPH)),
            "road_type": _edge_road_type(data),
            "road_name": data.get("road_name", ""),
            "surface": data.get("surface", "unknown"),
            "lit": data.get("lit", "unknown"),
            "toll": bool(data.get("toll", False)),
            "access": data.get("access", ""),
            "geometry": data.get("geometry"),
        }
        if existing is None or candidate["base_time_s"] < existing["base_time_s"]:
            routing.add_edge(start, end, **candidate)
    return routing


def nearest_node_with_distance(
    graph: nx.MultiDiGraph, lat: float, lon: float
) -> tuple[int, float]:
    best_node: int | None = None
    best_distance = float("inf")
    for node, data in graph.nodes(data=True):
        distance = haversine_m(lat, lon, float(data["y"]), float(data["x"]))
        if distance < best_distance:
            best_node = int(node)
            best_distance = distance
    if best_node is None:
        raise GraphLoadError("The graph has no nodes.")
    return best_node, best_distance


def nearest_node(graph: nx.MultiDiGraph, lat: float, lon: float) -> int:
    node, _ = nearest_node_with_distance(graph, lat, lon)
    return node


def node_latlon(graph: nx.Graph, node: int) -> tuple[float, float]:
    data = graph.nodes[node]
    return float(data["y"]), float(data["x"])


def edge_geometry(graph: nx.Graph, start: int, end: int) -> list[tuple[float, float]]:
    data = graph.edges[start, end]
    geometry = data.get("geometry")
    if geometry is not None and hasattr(geometry, "coords"):
        return [(float(lat), float(lon)) for lon, lat in geometry.coords]
    return [node_latlon(graph, start), node_latlon(graph, end)]


def path_geometry(
    graph: nx.Graph,
    nodes: tuple[int, ...] | list[int],
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for start, end in zip(nodes, nodes[1:]):
        segment = edge_geometry(graph, start, end)
        if points and segment and points[-1] == segment[0]:
            points.extend(segment[1:])
        else:
            points.extend(segment)
    if not points and nodes:
        points.append(node_latlon(graph, nodes[0]))
    return tuple(points)


def graph_stats(graph: nx.Graph) -> dict[str, float | int]:
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "largest_scc": 0, "largest_scc_ratio": 0.0}
    largest = max((len(component) for component in nx.strongly_connected_components(graph)), default=0)
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "largest_scc": largest,
        "largest_scc_ratio": largest / graph.number_of_nodes(),
    }
