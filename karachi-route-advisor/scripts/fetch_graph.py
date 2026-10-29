from __future__ import annotations

import sys
from pathlib import Path

import osmnx as ox

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import GRAPH_PATH
from src.osm_graph import is_drive_highway, load_graph


def main() -> int:
    ox.settings.requests_timeout = 60
    ox.settings.overpass_rate_limit = True
    try:
        graph = load_graph(GRAPH_PATH, force_refresh=True)
    except Exception as exc:
        print(f"failed={type(exc).__name__}: {exc}")
        return 1
    non_drive_edges = sum(
        1
        for _, _, _, data in graph.edges(keys=True, data=True)
        if not is_drive_highway(data.get("highway"))
    )
    print(f"saved={GRAPH_PATH}")
    print(f"nodes={graph.number_of_nodes()}")
    print(f"edges={graph.number_of_edges()}")
    print(f"non_drive_edges={non_drive_edges}")
    return 1 if non_drive_edges else 0


if __name__ == "__main__":
    raise SystemExit(main())
