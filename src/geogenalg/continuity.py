#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict

import geopandas as gpd
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


def connect_nearby_endpoints(
    input_gdf: gpd.GeoDataFrame, gap_threshold: float
) -> gpd.GeoDataFrame:
    """Generate helper lines to close small gaps between line endpoints.

    Args:
    ----
        input_gdf: A GeoDataFrame containing the original line network as LineStrings
            or MultiLineStrings
        gap_threshold: Maximum distance between endpoints to be connected

    Returns:
    -------
        A new GeoDataFrame containing only the generated helper lines

    """
    new_lines = []
    used_ends: set[int] = set()

    lines = list(input_gdf.geometry)

    endpoints = find_all_endpoints(lines)

    for i, (point_1, _, count_1) in enumerate(endpoints):
        if count_1 != 1 or (i in used_ends):
            continue

        closest_distance = float("inf")
        best_point = None

        # Search for the nearest other endpoint that is within the gap_threshold
        for j, (point_2, _, _) in enumerate(endpoints):
            if i == j:
                continue

            distance = point_1.distance(point_2)
            if distance < gap_threshold and distance < closest_distance:
                closest_distance = distance
                best_point = point_2

        if best_point:
            new_line = LineString([point_1, best_point])
            new_lines.append(new_line)
            used_ends.add(i)

    # Return a new GeoDataFrame containing only the helper lines
    return gpd.GeoDataFrame(geometry=new_lines, crs=input_gdf.crs)
