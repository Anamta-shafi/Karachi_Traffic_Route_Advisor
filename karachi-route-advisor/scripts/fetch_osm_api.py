from __future__ import annotations

import sys
from pathlib import Path

import osmnx as ox
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BBOX, CACHE_DIR, GRAPH_PATH
from src.osm_graph import graph_from_osm_xml, is_drive_highway

OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"
XML_PATH = GRAPH_PATH.with_suffix(".osm")


def main() -> int:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = {
        "bbox": f"{BBOX.west},{BBOX.south},{BBOX.east},{BBOX.north}",
    }
    response = requests.get(
        OSM_API_URL,
        params=params,
        headers={"User-Agent": "karachi-route-advisor/1.0"},
        timeout=120,
    )
    print(f"http_status={response.status_code}")
    print(f"bytes={len(response.content)}")
    if response.status_code != 200:
        print(response.text[:500])
        return 1

    XML_PATH.write_bytes(response.content)
    graph = graph_from_osm_xml(XML_PATH)
    non_drive_edges = sum(
        1
        for _, _, _, data in graph.edges(keys=True, data=True)
        if not is_drive_highway(data.get("highway"))
    )
    print(f"non_drive_edges={non_drive_edges}")
    if non_drive_edges:
        print("error=non-drive edges remain in the fetched graph")
        return 1
    ox.save_graphml(graph, GRAPH_PATH)
    print(f"graphml={GRAPH_PATH}")
    print(f"nodes={graph.number_of_nodes()}")
    print(f"edges={graph.number_of_edges()}")
    sample = next(iter(graph.edges(keys=True, data=True)))
    print(f"sample_edge={sample[0]}->{sample[1]} key={sample[2]}")
    print(
        "sample_attrs="
        + str(
            {
                key: sample[3].get(key)
                for key in (
                    "highway",
                    "name",
                    "maxspeed",
                    "speed_kph",
                    "length_m",
                    "surface",
                    "lit",
                    "toll",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
