#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame, overlay
from shapely.geometry import LineString, Polygon

from geogenalg.core.exceptions import GeometryTypeError


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
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
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
