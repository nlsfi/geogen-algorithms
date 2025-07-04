#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING

import geopandas as gpd

if TYPE_CHECKING:
    from shapely.geometry import Point

from geogenalg.continuity import find_all_endpoints


def remove_disconnected_short_lines(
    input_gdf: gpd.GeoDataFrame, length_threshold: float
) -> gpd.GeoDataFrame:
    """Remove short lines from a GeoDataFrame unless they are connected to other lines.

    A line is retained if:
    - Its length is greater than or equal to the length_threshold, OR
    - It is connected to at least two other lines via shared endpoints.

    Args:
    ----
        input_gdf: A GeoDataFrame containing LineString geometries
        length_threshold: The minimum length a line must have to be kept, unless it's
            well-connected.

    Returns:
    -------
        A filtered GeoDataFrame containing only the lines that meet the length or
            connectivity criteria.

    """
    lines = list(input_gdf.geometry)
    endpoints = find_all_endpoints(lines)
    min_connections = 2

    # Build mapping: Point -> set of line indices touching this point
    point_to_lines: defaultdict[Point, set[int]] = defaultdict(set)
    for point, idx, _ in endpoints:
        point_to_lines[point].add(idx)

    # Count how many endpoints of each line are shared with other lines
    connected_ends: defaultdict[int, int] = defaultdict(int)
    for indices in point_to_lines.values():
        if len(indices) > 1:
            for idx in indices:
                connected_ends[idx] += 1

    # Determine which lines to keep
    lines_to_keep = []
    for idx, geom in enumerate(lines):
        line_length = geom.length
        connection_count = connected_ends[idx]
        keep = (line_length >= length_threshold) or (
            connection_count >= min_connections
        )
        lines_to_keep.append(keep)

    return input_gdf[lines_to_keep]


def split_polygons_by_point_intersection(
    polygon_gdf: gpd.GeoDataFrame,
    point_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Split polygons into two GeoDataFrames based on whether they contain any point.

    Args:
    ----
        polygon_gdf: A GeoDataFrame containing polygon geometries
        point_gdf: A GeoDataFrame containing point geometries

    Returns:
    -------
        A tuple of two GeoDataFrames:
        - The first contains polygons that contain at least one point.
        - The second contains polygons that do not contain any points.

    """
    polygons_intersecting_point = polygon_gdf.geometry.apply(
        lambda poly: point_gdf.geometry.intersects(poly).any()
    )

    polygons_with_point: gpd.GeoDataFrame = polygon_gdf[polygons_intersecting_point]
    polygons_without_point: gpd.GeoDataFrame = polygon_gdf[~polygons_intersecting_point]

    return polygons_with_point, polygons_without_point


def remove_parts_of_lines_on_polygon_edges(
    lines_gdf: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Remove line parts that lie exactly on the edges of the given polygons.

    Args:
    ----
        lines_gdf: A GeoDataFrame containing LineString or MultiLineString geometries
        polygons_gdf: A GeoDataFrame containing Polygon or MultiPolygon geometries

    Returns:
    -------
        A Filtered GeoDataFrame with line parts not lying on polygon boundaries.

    """
    boundary = polygons_gdf.union_all().boundary

    if boundary is None:
        return lines_gdf.copy()

    result_gdf = lines_gdf.copy()
    result_gdf.geometry = lines_gdf.geometry.difference(boundary)

    # Remove rows with missing geometry (None) or empty geometry (is_empty)
    return result_gdf.dropna(subset=[result_gdf.geometry.name]).loc[
        ~result_gdf.geometry.is_empty
    ]
