#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import TYPE_CHECKING

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    from shapely.geometry import Point

from geogenalg.continuity import find_all_endpoints
from geogenalg.core.exceptions import GeometryTypeError


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


def remove_large_polygons(
    input_gdf: gpd.GeoDataFrame, area_threshold: float
) -> gpd.GeoDataFrame:
    """Remove polygons from a GeoDataFrame whose area exceeds the given threshold.

    Args:
    ----
        input_gdf: Input GeoDataFrame with polygons
        area_threshold: Maximum allowed area

    Returns:
    -------
        A new GeoDataFrame containing only polygons with area below or equal to the
              threshold.

    """
    return input_gdf[input_gdf.geometry.area <= area_threshold]


def remove_small_polygons(
    input_gdf: gpd.GeoDataFrame, area_threshold: float
) -> gpd.GeoDataFrame:
    """Remove polygons that are smaller than the area_threshold.

    Args:
    ----
        input_gdf: Input GeoDataFrame with polygons
        area_threshold: Minimum allowed area

    Returns:
    -------
        A new GeoDataFrame containing only polygons with area greater or equal to the
              threshold.

    """
    return input_gdf[
        ~(
            input_gdf.geometry.apply(
                lambda geom: isinstance(geom, (Polygon, MultiPolygon))
            )
        )
        | (input_gdf.geometry.area >= area_threshold)
    ]


def remove_small_holes(
    input_gdf: gpd.GeoDataFrame, hole_threshold: float
) -> gpd.GeoDataFrame:
    """Remove too small polygon holes.

    Args:
    ----
        input_gdf: Input GeoDataFrame with polygons
        hole_threshold: Minimum area for a hole

    Returns:
    -------
        A GeoDataFrame containing original polygons without too small holes.

    """

    def filter_holes(geometry: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
        """Filter out holes in a polygon that are smaller than the hole_threshold.

        Args:
        ----
            geometry: A shapely Polygon or MultiPolygon geometry potentially containing
                  interior rings

        Returns:
        -------
            A new Polygon or MultiPolygon with only the retained holes.

        """
        if isinstance(geometry, Polygon):
            if geometry.interiors:
                new_interiors = [
                    hole
                    for hole in geometry.interiors
                    if Polygon(hole).area >= hole_threshold
                ]
                return Polygon(geometry.exterior, new_interiors)
            return geometry

        if isinstance(geometry, MultiPolygon):
            filtered_polygons = [filter_holes(polygon) for polygon in geometry.geoms]
            return MultiPolygon(filtered_polygons)

        return geometry

    input_gdf.geometry = input_gdf.geometry.apply(filter_holes)

    return input_gdf


def reduce_nearby_points_by_selecting(
    input_points_gdf: gpd.GeoDataFrame,
    reference_points_gdf: gpd.GeoDataFrame | None,
    distance_threshold: float,
    priority_column: str,
) -> gpd.GeoDataFrame:
    """Reduce points by removing those too close to others based on given rules.

    Args:
    ----
        input_points_gdf: A GeoDataFrame containing the points to be reduced
        reference_points_gdf: A GeoDataFrame containing the points used for distance and
              priority comparison. If None, the same GeoDataFrame as input gdf is used.
        distance_threshold: Maximum distance for two points to be considered "too close"
        priority_column: Column name whose higher value determines which point is kept

    Returns:
    -------
        A GeoDataFrame containing the reduced set of points.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrames contain other than
              point geometries.

    """
    # Use self-comparison if no reference set is provided
    if reference_points_gdf is None:
        reference_points_gdf = input_points_gdf
    if input_points_gdf.empty:
        return input_points_gdf

    if input_points_gdf.geometry.type.unique().tolist() != [
        "Point"
    ] or reference_points_gdf.geometry.type.unique().tolist() != ["Point"]:
        msg = "reduce_nearby_points_by_selecting only supports Point geometries."
        raise GeometryTypeError(msg)

    reference_points_sindex = reference_points_gdf.sindex

    to_remove = set()

    for idx, point in input_points_gdf.iterrows():
        if idx in to_remove:
            continue

        possible_matches_index = list(
            reference_points_sindex.intersection(
                point.geometry.buffer(distance_threshold).bounds
            )
        )
        possible_matches: gpd.GeoDataFrame = reference_points_gdf.iloc[
            possible_matches_index
        ]

        for other_idx, other_point in possible_matches.iterrows():
            if other_idx == idx or other_idx in to_remove:
                continue

            if point.geometry.distance(other_point.geometry) <= distance_threshold:
                # Compare priority and decide which to remove
                if point[priority_column] >= other_point[priority_column]:
                    to_remove.add(other_idx)
                else:
                    to_remove.add(idx)
                    break

    return input_points_gdf.drop(index=to_remove)
