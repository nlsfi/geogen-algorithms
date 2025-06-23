#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict

from shapely.geometry import LineString, MultiLineString, Point


def find_all_endpoints(
    lines: list[LineString | MultiLineString],
) -> list[tuple[Point, int, int]]:
    """Find all endpoints (start and end points) of line features.

    Args:
    ----
        lines: A list of shapely LineString or MultiLineString geometries

    Returns:
    -------
        A list of tuples in the format (endpoint, line_index, occurrence_count), where
            occurrence_count indicates how many lines share the same endpoint location
            up to 5 decimals.

    """
    endpoint_map = defaultdict(list)
    min_coordinates_for_line = 2

    # Iterate over each geometry and extract its start and end points
    for idx, geom in enumerate(lines):
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = [
                geometry for geometry in geom.geoms if isinstance(geometry, LineString)
            ]

        for part in parts:
            coords = list(part.coords)
            if len(coords) >= min_coordinates_for_line:
                for endpoint in [Point(coords[0]), Point(coords[-1])]:
                    # Round coordinates to five decimal places
                    key = (round(endpoint.x, 5), round(endpoint.y, 5))
                    endpoint_map[key].append((endpoint, idx))

    endpoints = []

    # Flatten the dictionary into a list of (point, line index, count of lines)
    for entries in endpoint_map.values():
        for point, idx in entries:
            endpoints.append((point, idx, len(entries)))

    return endpoints
