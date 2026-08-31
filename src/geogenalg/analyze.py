#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from collections.abc import Hashable
from typing import Literal

import numpy as np
from geopandas import GeoDataFrame, overlay
from pandas import Series
from shapely import (
    BufferJoinStyle,
    MultiLineString,
    concave_hull,
    convex_hull,
    union_all,
)
from shapely.geometry import GeometryCollection, LineString, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import (
    angle_difference,
    ensure_geoms,
    explode_line,
    line_mean_direction,
    remove_holes,
    segment_bearing,
    segment_direction,
)
from geogenalg.utility.dataframe_processing import copy_gdf_as_empty
from geogenalg.utility.validation import check_gdf_geometry_type


def _group_parallel_lines(
    gdf: GeoDataFrame,
    id_column: str,
    parallel_with_column: str,
    parallel_group_column: str,
) -> None:
    """Recursively determine which lines are parallel.

    In practice this means that if we consider the lines:

    A - parallel with just B
    B - parallel with A and C
    C - parallel just with B

    All of these will be considered to be parallel with each other.

    Input GeoDataFrame is edited in place.

    """
    groups = []
    processed = set()

    def _process_row(index: int, row: Series, group: set[int]) -> None:
        if index in processed:
            return

        parallels = row[parallel_with_column]

        processed.add(index)

        group.add(index)
        group.update(parallels)

        for parallel in parallels:
            if parallel in gdf.index:
                _process_row(
                    parallel, gdf.loc[gdf[id_column] == parallel].iloc[0], group
                )

    for index, row in gdf.iterrows():
        if index in processed:
            continue

        group: set[int] = set()
        _process_row(index, row, group)
        groups.append(group)

    gdf[parallel_group_column] = -1

    for i, group in enumerate(groups, start=1):
        for idx in group:
            if idx in gdf.index:
                gdf.at[idx, parallel_group_column] = i  # noqa: PD008


def flag_parallel_lines(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    allowed_direction_difference: float,
    *,
    segmentize_distance: float = 0,
) -> GeoDataFrame:
    """Detect which lines are parallel with each other within given parameters.

    Args:
    ----
        input_gdf: GeoDataFrame with LineStrings.
        parallel_distance: Maximum distance at which lines are considered to be
            parallel.
        allowed_direction_difference: If the absolute value of the difference of
            direction (relative to north) between lines is under this value they
            will still be considered to parallel.
        segmentize_distance: If above zero, input lines will be segmentized
            resulting in more precise detection.

    Returns:
    -------
        GeoDataFrame containing detected parallel lines, with columns:
            `parallel_group` denoting which other lines a line is parallel with.
            `parallel_direction` direction of line relative to north.
            `parallel_with` set of number ids of other lines a line is parallel with.
            `parallel_id` number id of line.


    """
    gdf = input_gdf.copy()

    column_direction = "parallel_direction"
    column_group = "parallel_group"
    column_parallel_with = "parallel_with"
    column_id = "parallel_id"
    column_parallel_check = "_parallel_check"

    def _empty_gdf() -> GeoDataFrame:
        # This ensures in case we get an empty result, the output consistently
        # has these columns, with these dtypes.
        gdf[column_direction] = Series(dtype="float64")
        gdf[column_group] = Series(dtype="int64")
        gdf[column_parallel_with] = Series(dtype="object")
        gdf[column_id] = Series(dtype="int64")

        return gdf

    if input_gdf.empty:
        return _empty_gdf()

    if segmentize_distance > 0:
        gdf.geometry = gdf.geometry.segmentize(segmentize_distance)

    gdf.geometry = gdf.geometry.apply(explode_line)
    gdf = gdf.explode().reset_index(drop=True)

    # Normalize so that comparing the direction of segments later on is consistent
    gdf[column_direction] = Series(gdf.geometry.normalize().apply(segment_bearing))
    gdf[column_parallel_check] = gdf.geometry.buffer(
        parallel_distance, cap_style="flat"
    ).buffer(0.01, cap_style="square")
    gdf[column_parallel_with] = None
    gdf[column_id] = gdf.index

    for index, row in gdf.iterrows():
        parallel_geom = row[column_parallel_check]
        parallel_geom_direction = row[column_direction]
        row_geom = row[gdf.geometry.name]

        # This locates lines which a) are not the line we're iterating over
        # b) intersects the buffered area used to check for parallel lines and
        # c) are within the given direction bounds.
        crossing_lines = gdf.loc[
            (gdf.geometry.intersects(parallel_geom))
            & ~gdf.geometry.intersects(row_geom)
            & (
                ((gdf[column_direction] - parallel_geom_direction).abs())
                < allowed_direction_difference
            )
        ]

        if crossing_lines.empty:
            continue

        gdf[column_parallel_with].to_numpy()[index] = set(crossing_lines.index)

    gdf = gdf.loc[gdf[column_parallel_with].notna()]

    _group_parallel_lines(
        gdf,
        column_id,
        column_parallel_with,
        column_group,
    )

    gdf = gdf.drop(column_parallel_check, axis=1)

    if gdf.empty:
        return _empty_gdf()

    return gdf


def get_polygons_for_parallel_lines(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    *,
    allowed_direction_difference: float = 10,
    segmentize_distance: float = 50,
    polygonize_function: Literal["concave", "convex"] = "concave",
) -> GeoDataFrame:
    """Find areas with parallel lines.

    Args:
    ----
        input_gdf: GeoDataFrame with (potential) parallel lines.
        parallel_distance: Distance threshold to still consider lines parallel.
        allowed_direction_difference: Acceptable difference in direction
            (calculated in relation to North), in degrees.
        segmentize_distance: Interval at which parallel lines are checked for,
            the lower the number the more precise result, with the cost of
            performance.
        polygonize_function: Whether polygonization of parallel lines will be
            done with the stricter concave hull or convex hull.

    Returns:
    -------
        GeoDataFrame of polygons enclosing the parallel lines which were found.

    """
    parallels = flag_parallel_lines(
        input_gdf,
        parallel_distance,
        allowed_direction_difference,
        segmentize_distance=segmentize_distance,
    )

    if parallels.empty:
        return GeoDataFrame({"parallel_direction": []}, geometry=[], crs=input_gdf.crs)

    column_direction = "parallel_direction"
    column_group = "parallel_group"
    column_parallel_with = "parallel_with"
    column_id = "parallel_id"

    compare = parallels[[column_id, parallels.geometry.name]].copy()

    def _polygonize_parallel(
        parallel_with: set[int],
        geom: LineString,
    ) -> Polygon:
        others = compare.loc[compare[column_id].isin(parallel_with)]

        geoms = [geom, *list(others.geometry)]

        return (
            concave_hull(MultiLineString(geoms))
            if polygonize_function == "concave"
            else convex_hull(MultiLineString(geoms))
        )

    parallels.geometry = parallels[
        [column_parallel_with, parallels.geometry.name]
    ].apply(lambda columns: _polygonize_parallel(*columns), axis=1)

    # Dissolve by group and calculate mean direction for group. Other columns
    # not included in the aggfunc parameter will be dropped.
    parallels = parallels.dissolve(
        by=[column_group],
        aggfunc={column_direction: "mean"},  # noqa: SC200
        as_index=False,
    )

    # There may be some intersections in the generated polygons. This is okay,
    # if the orientation of the lines the polygon was created from is
    # different, otherwise this is not desired, so round mean direction and
    # dissolve again according to it.
    parallels[column_direction] = parallels[column_direction].round(-1)

    return (
        parallels.dissolve(
            by=[column_direction],
            as_index=False,
        )
        .drop(column_group, axis=1)
        .explode(index_parts=False)
        .reset_index(drop=True)
    )  # Explode to single rows.


def calculate_coverage(
    overlay_features: GeoDataFrame,
    base_features: GeoDataFrame,
    coverage_attribute: str = "coverage",
) -> GeoDataFrame:
    """Calculate the percentage of area covered by overlay features on base features.

    Args:
    ----
        overlay_features: GeoDataFrame whose geometries are overlaid on top of the base.
        base_features: GeoDataFrame to which the coverage percentage is written.
        coverage_attribute: Name of the output column that will hold the coverage (%).

    Returns:
    -------
        GeoDataFrame
            A version of 'base_features' with an additional 'coverage_attribute' column.
            Unnecessary columns are dropped.

    """
    overlay_features = overlay_features.copy()
    base_features = base_features.copy()
    base_features["base_area"] = base_features.geometry.area

    base_features["base_feature_id"] = base_features.index

    intersections = overlay(
        overlay_features,
        base_features,
        how="intersection",
        keep_geom_type=False,
    )
    intersections["intersect_area"] = intersections.geometry.area

    combined_overlay_area = (
        intersections.groupby("base_feature_id")["intersect_area"]
        .sum()
        .reset_index()
        .rename(columns={"intersect_area": "overlay_area"})
    )

    base_features = base_features.merge(
        combined_overlay_area, on="base_feature_id", how="left"
    )
    base_features["overlay_area"] = base_features["overlay_area"].fillna(0)
    base_features[coverage_attribute] = (
        100 * base_features["overlay_area"] / base_features["base_area"]
    )
    return base_features.drop(
        columns=["base_feature_id", "overlay_area", "base_area"],
        errors="ignore",
    )


def calculate_main_angle(polygon: Polygon) -> float:
    """Calculate the main angle of a polygon based on its minimum bounding rectangle.

    Args:
    ----
        polygon: A Shapely Polygon geometry

    Returns:
    -------
        Orientation angle in degrees (range: 0 to 180), measured from the positive
        Y-axis.

    Raises:
    ------
        TypeError: If the input is not a Shapely Polygon.

    """
    if polygon is None or polygon.is_empty:
        return np.nan

    if not isinstance(polygon, Polygon):
        msg = "Input geometry must be a Shapely Polygon."
        raise TypeError(msg)

    coordinates = list(polygon.minimum_rotated_rectangle.exterior.coords)

    edges = [
        np.array(coordinates[1]) - np.array(coordinates[0]),
        np.array(coordinates[3]) - np.array(coordinates[0]),
    ]

    edge_lengths = [np.linalg.norm(edge) for edge in edges]

    longest_edge = edges[np.argmax(edge_lengths)]

    # Calculate and return the angle of the longest edge
    return np.degrees(np.arctan2(longest_edge[0], longest_edge[1])) % 180


def classify_polygons_by_size_of_minimum_bounding_rectangle(
    input_gdf: GeoDataFrame,
    side_threshold: float,
) -> dict[str, GeoDataFrame]:
    """Classifiy polygons as small or large based on the minimum bounding rectangle.

    Args:
    ----
        input_gdf: Input GeoDataFrame containing Polygon or MultiPolygon geometries
        side_threshold: Length threshold for the longest side of the bounding rectangle

    Returns:
    -------
        Dictionary with two keys:
            - "small_polygons": GeoDataFrame of polygons classified as small
            - "large_polygons": GeoDataFrame of polygons classified as large.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              polygon geometries.

    """
    if not check_gdf_geometry_type(input_gdf, {"Polygon", "MultiPolygon"}):
        msg = "Classify polygons only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    if input_gdf.empty:
        return {
            "small_polygons": copy_gdf_as_empty(input_gdf),
            "large_polygons": copy_gdf_as_empty(input_gdf),
        }

    large_polygon_indices = []
    small_polygon_indices = []

    for idx, row in input_gdf.iterrows():
        geom = row[input_gdf.geometry.name]
        coordinates = list(geom.minimum_rotated_rectangle.exterior.coords)

        longest_side = max(
            LineString([coordinates[i], coordinates[i + 1]]).length
            for i in range(len(coordinates) - 1)
        )

        if longest_side > side_threshold:
            large_polygon_indices.append(idx)
        else:
            small_polygon_indices.append(idx)

    large_gdf = input_gdf.loc[large_polygon_indices].copy()
    small_gdf = input_gdf.loc[small_polygon_indices].copy()

    return {
        "small_polygons": small_gdf,
        "large_polygons": large_gdf,
    }


def calculate_edge_adjacency(
    input_gdf: GeoDataFrame,
    reference_gdf: GeoDataFrame,
    buffer_size: float,
    result_column: str = "adjacency_ratio",
) -> GeoDataFrame:
    """Calculate adjacency ratio using outer and inner halo buffers.

    Args:
    ----
        input_gdf: Input GeoDataFrame with Polygon or MultiPolygon geometries.
        reference_gdf: Reference GeoDataFrame with Polygon or MultiPolygon geometries.
        buffer_size: Distance defining the zone in which overlap with the reference
              geometries is analyzed.
        result_column: Name of the output column storing the computed adjacency ratio.

    Returns:
    -------
        GeoDataFrame with an added column containing the computed adjacency ratio for
            each input feature.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
            polygon or MultiPolygon geometries.

    """
    if not check_gdf_geometry_type(input_gdf, {"Polygon", "MultiPolygon"}):
        msg = (
            "Calculate edge adjacency only supports Polygon or MultiPolygon geometries."
        )
        raise GeometryTypeError(msg)

    def intersection_ratio(halo: BaseGeometry) -> float | None:
        if halo.is_empty:
            return None

        area = halo.area
        if area <= 0:
            return None

        possible = reference_gdf.iloc[
            list(reference_gdf.sindex.intersection(halo.bounds))
        ]

        if possible.empty:
            return 0.0

        intersection_area = possible.intersection(halo).area.sum()
        return intersection_area / area

    def feature_ratio(geom: BaseGeometry) -> float | None:
        # Outer halo
        outer_halo = geom.buffer(buffer_size).difference(geom)
        outer_ratio = intersection_ratio(outer_halo)

        # Inner halo
        inner_buffer = geom.buffer(-buffer_size)
        if inner_buffer.is_empty:
            inner_ratio = None
        else:
            inner_halo = geom.difference(inner_buffer)
            inner_ratio = intersection_ratio(inner_halo)

        # Select appropriate value for overlapping ratio
        if inner_ratio is None:
            return outer_ratio
        if outer_ratio is None:
            return inner_ratio
        return max(inner_ratio, outer_ratio)

    result_gdf = input_gdf.copy()
    result_gdf[result_column] = [feature_ratio(geom) for geom in result_gdf.geometry]

    return result_gdf


def group_geometries_by_intersections_recursively(  # noqa: C901
    input_gdf: GeoDataFrame,
    geometry_group_column: str = "_geometry_group",
) -> GeoDataFrame:
    """Recursively determine which geometries intersect.

    In practice this means that if we consider the polygons:

    A - intersects with just B
    B - intersects with A and C
    C - intersects just with B

    All of these will be in the same group.

    The function works by determining which rows each row intersects via a
    spatial join. The function then goes through each row and checks which
    other rows it intersects, then recursively descending to check what other
    rows those rows intersect (skipping already processed rows) and collecting
    the indices of each row found.

    Args:
    ----
        input_gdf: GeoDataFrame with geometries to group.
        geometry_group_column: Name of column to save the group index of each feature.

    Returns:
    -------
        GeoDataFrame with group column added.

    """
    gdf = input_gdf.copy()

    # Initialize group column as each row being its own group, this way if row
    # truly does not belong to a group it'll still have a valid value.
    gdf[geometry_group_column] = range(gdf.shape[0])

    geometry_group_sets = []
    processed = set()

    joined_group = input_gdf.sjoin(
        input_gdf,
        how="inner",
        predicate="intersects",
    )

    if joined_group.empty:
        return gdf

    def group_by_geometry(
        index: Hashable,
        rows: GeoDataFrame,
        geometry_group: set[Hashable],
    ) -> None:
        if index in processed:
            return

        intersects = rows["index_right"]

        processed.add(index)

        geometry_group.add(index)
        geometry_group.update(intersects)

        for intersecting_feature in intersects:
            if intersecting_feature == index:
                continue

            if intersecting_feature in gdf.index:
                group_by_geometry(
                    intersecting_feature,
                    joined_group.loc[joined_group.index == intersecting_feature],
                    geometry_group,
                )

    unique_indices = joined_group.index.unique()

    for index in unique_indices:
        if index in processed:
            continue

        geometry_group: set[Hashable] = set()
        group_by_geometry(
            index,
            joined_group.loc[joined_group.index == index],
            geometry_group,
        )
        geometry_group_sets.append(geometry_group)

    for i, geometry_group in enumerate(geometry_group_sets, start=1):
        for idx in geometry_group:
            if idx in gdf.index:
                gdf.loc[idx, geometry_group_column] = i

    return gdf


def polygonize_parallel_lines(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    *,
    maximum_angle_difference: float = 15,
    postprocessing_join_style: BufferJoinStyle
    | Literal["round", "mitre", "bevel"] = "round",
    postprocessing_hole_threshold: float = 1000,
) -> GeoDataFrame:
    """Turn areas with parallel lines to polygons.

    Args:
    ----
        input_gdf: GeoDataFrame containing LineStrings.
        parallel_distance: Minimum distance for two lines to be considered to
            be parallel.
        maximum_angle_difference: Maximum allowed difference in line angle for
            two lines to still be considered to be parallel.
        postprocessing_join_style: Buffer join style for post-processing
            the generated polygons.
        postprocessing_hole_threshold: Area threshold for removing holes from
            the generated polygons.

    Returns:
    -------
        GeoDataFrame with polygons encompassing parallel lines.

    """
    if input_gdf.empty:
        return copy_gdf_as_empty(input_gdf)

    gdf = input_gdf.copy()
    polys_for_lines = gdf.union_all()

    segments = gdf.geometry.apply(explode_line).explode()

    # This approach works by going through each line segment in the input
    # dataset, searching for other line segments by a buffered polygon, and if
    # they are close enough and within the allowed angle difference, a convex
    # hull of the lines is created.
    hulls = []
    for geom in segments:
        parallel_check = geom.buffer(
            parallel_distance,
            join_style="mitre",
            cap_style="flat",
        )

        direction = segment_direction(geom)
        intersection = polys_for_lines.intersection(parallel_check)

        if isinstance(intersection, GeometryCollection):
            intersection = union_all(
                [
                    geom
                    for geom in intersection.geoms
                    if geom.geom_type in {"LineString", "MultiLineString"}
                ],
            )

        if not isinstance(intersection, MultiLineString | LineString):
            raise NotImplementedError

        intersection = MultiLineString(
            [
                line
                for line in ensure_geoms(intersection)
                if angle_difference(
                    line_mean_direction(line),
                    direction,
                )
                < maximum_angle_difference
            ]
        )

        if intersection.is_empty:
            continue

        polygonized_lines = convex_hull(intersection)

        if not isinstance(polygonized_lines, Polygon):
            continue

        hulls.append(polygonized_lines)

    # Now that each segment is processed, combine all the results
    polygonized_lines = union_all(hulls)

    # Do some post-processing by removing some of the smaller holes
    polygonized_lines = remove_holes(
        polygonized_lines,
        area_threshold=postprocessing_hole_threshold,
    )

    # Do some further post-processing and remove thin spikes
    polygonized_lines = polygonized_lines.buffer(
        parallel_distance / 10,
        join_style=postprocessing_join_style,
    )
    polygonized_lines = polygonized_lines.buffer(
        -(parallel_distance / 10),
        join_style=postprocessing_join_style,
    )
    polygonized_lines = remove_holes(
        polygonized_lines,
        area_threshold=postprocessing_hole_threshold,
    )

    if polygonized_lines.is_empty:
        return copy_gdf_as_empty(input_gdf)

    return (
        GeoDataFrame(geometry=[polygonized_lines], crs=input_gdf.crs)
        .explode()
        .reset_index(drop=True)
    )


def flag_parallel_lines2(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    allowed_direction_difference: float,
) -> GeoDataFrame:
    gdf = input_gdf.copy().reset_index(drop=True)

    column_direction = "parallel_direction"
    column_group = "parallel_group"
    column_parallel_with = "parallel_with"
    column_id = "parallel_id"
    column_parallel_check = "_parallel_check"

    def _empty_gdf() -> GeoDataFrame:
        return copy_gdf_as_empty(
            gdf,
            add_columns={
                column_direction: "float64",
                column_group: "int64",
                column_parallel_with: "object",
                column_id: "int64",
            },
        )

    if input_gdf.empty:
        return _empty_gdf()

    gdf[column_direction] = gdf.geometry.apply(line_mean_direction)
    gdf[column_parallel_check] = gdf.geometry.buffer(
        parallel_distance, cap_style="flat"
    ).buffer(0.01, cap_style="square")
    gdf[column_parallel_with] = None
    gdf[column_id] = gdf.index

    for index, row in gdf.iterrows():
        parallel_geom = row[column_parallel_check]
        parallel_geom_direction = row[column_direction]
        row_geom = row[gdf.geometry.name]

        # This locates lines which a) are not the line we're iterating over
        # b) intersects the buffered area used to check for parallel lines and
        # c) are within the given direction bounds.
        crossing_lines = gdf.loc[
            (gdf.geometry.intersects(parallel_geom))
            & ~gdf.geometry.intersects(row_geom)
            & (
                (
                    gdf[column_direction].apply(
                        angle_difference,
                        b=parallel_geom_direction,
                        )
                ) < allowed_direction_difference
            )
        ]

        if crossing_lines.empty:
            continue

        print()
        print(crossing_lines)
        print(index)
        print(gdf[column_parallel_with].to_numpy()[index])

        gdf[column_parallel_with].to_numpy()[index] = set(crossing_lines.index)

    gdf = gdf.loc[gdf[column_parallel_with].notna()]

    _group_parallel_lines(
        gdf,
        column_id,
        column_parallel_with,
        column_group,
    )

    gdf = gdf.drop(column_parallel_check, axis=1)

    if gdf.empty:
        return _empty_gdf()

    return gdf
