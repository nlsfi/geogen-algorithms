#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, overlay
from pandas import Series
from shapely import MultiLineString, concave_hull
from shapely.geometry import LineString, Polygon

from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import explode_line, segment_direction
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
        GeoDataFrame containing detected parallel lines, with a column named
        "parallel_group" denoting which other lines a line is parallel with.


    """
    gdf = input_gdf.copy()

    def _empty_gdf() -> GeoDataFrame:
        # This ensures in case we get an empty result, the output consistently
        # has these columns, with these dtypes.
        gdf["__direction"] = Series(dtype="float64")
        gdf["parallel_group"] = Series(dtype="int64")
        gdf["parallel_with"] = Series(dtype="object")
        gdf["__id"] = Series(dtype="int64")

        return gdf

    if len(input_gdf.index) == 0:
        return _empty_gdf()

    if segmentize_distance > 0:
        gdf.geometry = gdf.geometry.segmentize(segmentize_distance)

    gdf.geometry = gdf.geometry.apply(explode_line)
    gdf = gdf.explode().reset_index(drop=True)

    # Normalize so that comparing the direction of segments later on is consistent
    gdf["__direction"] = gdf.geometry.normalize().apply(segment_direction)
    gdf["__parallel_check"] = gdf.geometry.buffer(
        parallel_distance, cap_style="flat"
    ).buffer(0.01, cap_style="square")
    gdf["parallel_with"] = [set() for _ in range(len(gdf.index))]
    gdf["__id"] = gdf.index

    for index, row in gdf.iterrows():
        parallel_geom = row["__parallel_check"]
        parallel_geom_direction = row["__direction"]

        # This locates lines which a) are not the line we're iterating over
        # b) intersects the buffered area used to check for parallel lines and
        # c) are within the given direction bounds.
        crossing_lines = gdf.loc[
            (gdf.geometry.intersects(parallel_geom))
            & ~gdf.geometry.intersects(row.geometry)
            & (
                ((gdf["__direction"] - parallel_geom_direction).abs())
                < allowed_direction_difference
            )
        ]

        if len(crossing_lines.index) == 0:
            continue

        for crossing_index in crossing_lines.index:
            gdf["parallel_with"].to_numpy()[index].add(crossing_index)

    gdf = gdf.loc[gdf["parallel_with"].apply(len) != 0]

    _group_parallel_lines(
        gdf,
        "__id",
        "parallel_with",
        "parallel_group",
    )

    gdf["__direction"] = Series(gdf["__direction"].tolist())
    gdf = gdf.drop("__parallel_check", axis=1)

    if len(gdf.index) == 0:
        return _empty_gdf()

    return gdf


def get_parallel_line_areas(
    input_gdf: GeoDataFrame,
    parallel_distance: float,
    *,
    allowed_direction_difference: float = 10,
    segmentize_distance: float = 50,
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

    if len(parallels) == 0:
        return GeoDataFrame()

    compare = parallels[["__id", parallels.geometry.name]].copy()

    def _polygonize_parallel(
        parallel_with: set[int],
        geom: LineString,
    ) -> Polygon:
        others = compare.loc[compare["__id"].isin(parallel_with)]

        geoms = [geom, *list(others.geometry)]

        return concave_hull(MultiLineString(geoms))

    parallels.geometry = parallels[["parallel_with", parallels.geometry.name]].apply(
        lambda columns: _polygonize_parallel(*columns), axis=1
    )

    parallels = parallels.dissolve(
        by=["parallel_group"],
        aggfunc={"__direction": "mean"},  # noqa: SC200
    )

    # There may be some intersections in the generated polygons. This is okay,
    # if the orientation of the lines the polygon was created from is
    # different, otherwise this is not desired, so round mean direction and
    # dissolve again according to it.
    parallels["__direction"] = parallels["__direction"].round(-1)

    return (
        parallels.dissolve(
            by=["__direction"],
            as_index=False,
        )
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

    Raises:
    ------
    ValueError: If CRS is not projected for both layers.

    """
    if (
        overlay_features.crs is None
        or base_features.crs is None
        or not overlay_features.crs.is_projected
        or not base_features.crs.is_projected
    ):
        error_msg = "Both layers must have a projected CRS (metres)."
        raise ValueError(error_msg)

    overlay_features = overlay_features.copy()
    base_features = base_features.copy()
    base_features["base_area"] = base_features.geometry.area

    base_features["base_feature_id"] = base_features.index

    intersections = overlay(overlay_features, base_features, how="intersection")
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
    input_gdf: gpd.GeoDataFrame,
    side_threshold: float,
) -> dict[str, gpd.GeoDataFrame]:
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
    if not check_gdf_geometry_type(input_gdf, ["Polygon", "MultiPolygon"]):
        msg = "Classify polygons only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)

    if input_gdf.empty:
        return {
            "small_polygons": gpd.GeoDataFrame(
                columns=input_gdf.columns, crs=input_gdf.crs
            ),
            "large_polygons": gpd.GeoDataFrame(
                columns=input_gdf.columns, crs=input_gdf.crs
            ),
        }

    large_polygon_indices = []
    small_polygon_indices = []

    for idx, row in input_gdf.iterrows():
        geom = row.geometry
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
