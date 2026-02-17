#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from typing import Literal

from geopandas import GeoDataFrame


def inherit_attributes(
    source_gdf: GeoDataFrame, target_gdf: GeoDataFrame
) -> GeoDataFrame:
    """Copy attributes from the source_gdf to the geometries in target_gdf.

    For each target geometry, the attributes of the first intersecting source geometry
    are inherited. If no intersection is found, the attributes of the first source
    geometry are used instead.

    Args:
    ----
        source_gdf: GeoDataFrame containing polygons with original attributes
        target_gdf: GeoDataFrame with new geometries that need inherited attributes

    Returns:
    -------
        A result GeoDataFrame geometries contains attributes from the source dataframe
              but no attributes from the target dataframe.

    """
    new_features = []

    # Iterate over each row in the target GeoDataFrame and inherit the attributes from
    # the first intersecting source GeoDataFrame geometry.
    for _, target_row in target_gdf.iterrows():
        target_geom = target_row.geometry
        new_attrs = None

        candidates = source_gdf[source_gdf.geometry.intersects(target_geom)]

        for _, source_row in candidates.iterrows():
            intersection = target_geom.intersection(source_row.geometry)
            if not intersection.is_empty:
                new_attrs = source_row.drop(source_gdf.geometry.name).to_dict()
                break

        # If no intersection is found, the attributes of the first source row are used
        if new_attrs is None:
            new_attrs = source_gdf.iloc[0].drop(source_gdf.geometry.name).to_dict()

        new_attrs[source_gdf.geometry.name] = target_geom
        new_features.append(new_attrs)

    return GeoDataFrame(
        new_features,
        geometry=source_gdf.geometry.name,
        index=target_gdf.index,
        crs=source_gdf.crs,
    )


def inherit_attributes_from_largest(
    source_gdf: GeoDataFrame,
    target_gdf: GeoDataFrame,
    old_ids_column: str | None = None,
    *,
    measure_by: Literal["area", "length"] = "area",
) -> GeoDataFrame:
    """Copy attributes from a intersecting feature from source GeoDataFrame.

    If target feature intersects multiple features on the source GeoDataFrame,
    the largest feature is used.

    Args:
    ----
        source_gdf: GeoDataFrame containing features with original attributes
        target_gdf: GeoDataFrame with new geometries that need inherited attributes
        old_ids_column: If not None, indexes of all intersecting source features
            will be saved as a tuple to a column by this name.
        measure_by: How to measure which feature is largest, area for polygons
            or length for lines (or perimeter length for polygons).

    Returns:
    -------
        GeoDataFrame with the inherited attributes.

    """
    new_features = []
    for _, feature in target_gdf.iterrows():
        intersecting_features = source_gdf.loc[
            source_gdf.geometry.intersects(feature.geometry)
        ].copy()

        if intersecting_features.empty:
            # TODO: should this give a warning/even error?
            continue

        if measure_by == "area":
            intersecting_features["__temp_size"] = intersecting_features.geometry.area
        else:
            intersecting_features["__temp_size"] = intersecting_features.geometry.length

        intersecting_features = intersecting_features.sort_values(
            "__temp_size",
            ascending=False,
        )
        intersecting_features = intersecting_features.drop("__temp_size", axis=1)

        largest_feature = intersecting_features.iloc[0]

        new_feature = largest_feature.copy()
        new_feature.geometry = feature.geometry

        if old_ids_column is not None:
            new_feature[old_ids_column] = tuple(
                intersecting_features.index.to_list(),
            )
        new_features.append(new_feature)

    output = GeoDataFrame(new_features, crs=target_gdf.crs)
    output.index.name = source_gdf.index.name
    output.geometry.name = source_gdf.geometry.name

    return output
