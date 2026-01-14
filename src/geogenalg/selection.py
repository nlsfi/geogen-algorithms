#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import TYPE_CHECKING

from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Polygon

from geogenalg.utility.validation import check_gdf_geometry_type

if TYPE_CHECKING:
    from shapely.geometry import Point

from geogenalg.continuity import find_all_endpoints
from geogenalg.core.exceptions import GeometryTypeError


def remove_disconnected_short_lines(
    input_gdf: GeoDataFrame, length_threshold: float
) -> GeoDataFrame:
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
    polygon_gdf: GeoDataFrame,
    point_gdf: GeoDataFrame,
) -> tuple[GeoDataFrame, GeoDataFrame]:
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

    polygons_with_point: GeoDataFrame = polygon_gdf[polygons_intersecting_point]
    polygons_without_point: GeoDataFrame = polygon_gdf[~polygons_intersecting_point]

    return polygons_with_point, polygons_without_point


def remove_parts_of_lines_on_polygon_edges(
    lines_gdf: GeoDataFrame, polygons_gdf: GeoDataFrame
) -> GeoDataFrame:
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
    input_gdf: GeoDataFrame, area_threshold: float
) -> GeoDataFrame:
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
    polygons_gdf: GeoDataFrame, area_threshold: float
) -> GeoDataFrame:
    """Remove polygons smaller than a minimum area threshold.

    Args:
    ----
        polygons_gdf: GeoDataFrame containing Polygon or MultiPolygon geometries.
        area_threshold: Minimum area in the units of the projected CRS
            (e.g. square metres). Polygons with smaller area are removed.

    Returns:
    -------
        GeoDataFrame
            A version of the input GeoDataFrame where only polygons with
            area greater than or equal to the threshold remain.

    Raises:
    ------
    ValueError: If area_threshold <= 0 or CRS is not projected.

    """
    if area_threshold <= 0:
        msg = "area_threshold must be > 0."
        raise ValueError(msg)

    if polygons_gdf.crs is None or not polygons_gdf.crs.is_projected:
        msg = "Input layer must have a projected CRS to compute area."
        raise ValueError(msg)

    # Drop missing/empty geometries before area calculation
    gdf = polygons_gdf.dropna(subset=[polygons_gdf.geometry.name]).loc[
        ~polygons_gdf.geometry.is_empty
    ]

    mask = gdf.geometry.area >= area_threshold
    return gdf.loc[mask].copy()


def remove_small_holes(input_gdf: GeoDataFrame, hole_threshold: float) -> GeoDataFrame:
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
                  interior rings.

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

    output_gdf = input_gdf.copy()
    output_gdf.geometry = output_gdf.geometry.apply(filter_holes)
    return output_gdf


def reduce_nearby_points_by_selecting(
    input_points_gdf: GeoDataFrame,
    reference_points_gdf: GeoDataFrame | None,
    distance_threshold: float,
    priority_column: str,
) -> GeoDataFrame:
    """Reduce nearby points by keeping the higher-priority point within a threshold.

    Args:
    ----
        input_points_gdf: A GeoDataFrame containing the points to be reduced
        reference_points_gdf: A GeoDataFrame containing the points used for distance and
              priority comparison. If None, the same GeoDataFrame as input gdf is used.
        distance_threshold: Maximum distance for two points to be considered "too close"
        priority_column: Column name whose higher value determines which point is kept

    Returns:
    -------
        A GeoDataFrame containing the reduced set of points after applying the
        distance and priority rules.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrames contain other than point
              geometries.

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

    reference_sindex = reference_points_gdf.sindex
    to_remove = set()

    gdf = input_points_gdf.copy()

    for input_point in gdf.itertuples():
        input_point_index = input_point.Index
        if input_point_index in to_remove:
            continue

        possible_matches_idx = list(
            reference_sindex.intersection(
                input_point.geometry.buffer(distance_threshold).bounds
            )
        )
        possible_matches: GeoDataFrame = reference_points_gdf.iloc[possible_matches_idx]

        for other_point in possible_matches.itertuples():
            other_point_index = other_point.Index
            if other_point_index == input_point_index or other_point_index in to_remove:
                continue

            distance = input_point.geometry.distance(other_point.geometry)

            if distance < distance_threshold and (
                getattr(input_point, priority_column)
            ) <= getattr(other_point, priority_column):
                to_remove.add(input_point_index)
                break

    return gdf.loc[~gdf.index.isin(to_remove)]


def remove_close_line_segments(
    lines: GeoDataFrame,
    reference_lines: GeoDataFrame,
    buffer_size: float,
) -> GeoDataFrame:
    """Remove line segments close to reference lines.

    Buffers reference lines and removes intersecting line segments.
    Only the intersecting areas are removed, so a whole input line feature
    may be fully removed or split into segments (and change type to
    MultiLineString).

    Args:
    ----
        lines: GeoDataFrame with lines to process.
        reference_lines: GeoDataFrame with lines to use for interescting.
        buffer_size: Size used to buffer reference lines and determine
            the intersection area.

    Returns:
    -------
        Remaining lines.

    Raises:
    ------
        GeometryTypeError: If input lines or reference lines contain other
            than line geometries.

    """
    if not check_gdf_geometry_type(
        lines, ["LineString", "MultiLineString"]
    ) or not check_gdf_geometry_type(
        reference_lines, ["LineString", "MultiLineString"]
    ):
        msg = "remove_close_line_segments expects only line geometries."
        raise GeometryTypeError(msg)

    result = lines.copy()
    result.geometry = lines.difference(
        reference_lines.buffer(buffer_size).union_all(),
    )
    return result.loc[~result.geometry.is_empty & result.geometry.notna()]


def remove_short_lines(lines: GeoDataFrame, length_threshold: float) -> GeoDataFrame:
    """Remove short line features.

    Removes all lines that are shorter than the given length
    threshold value.

    Args:
    ----
        lines: GeoDataFrame with lines to process.
        length_threshold: Threshold that determines too short lines.

    Returns:
    -------
        Remaining lines.

    Raises:
    ------
        GeometryTypeError: If input lines or reference lines contain other
            than line geometries.

    """
    if not check_gdf_geometry_type(lines, ["LineString", "MultiLineString"]):
        msg = "remove_short_lines expects only line geometries."
        raise GeometryTypeError(msg)

    result = lines.copy()
    return result.loc[result.geometry.length >= length_threshold]
