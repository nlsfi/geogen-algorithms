#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

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
) -> GeoDataFrame:
    """Copy attributes from a intersecting feature from source GeoDataFrame.

    If target feature intersects multiple features on the source GeoDataFrame,
    the largest feature is used.

    Args:
    ----
        source_gdf: GeoDataFrame containing polygons with original attributes
        target_gdf: GeoDataFrame with new geometries that need inherited attributes
        old_ids_column: If not None, indexes of all intersecting source features
            will be saved as a tuple to a column by this name.

    Returns:
    -------
        GeoDataFrame with the inherited attributes.

    """
    new_features = []
    for _, line in target_gdf.iterrows():
        intersecting_poly_features = source_gdf.loc[
            source_gdf.geometry.intersects(line.geometry)
        ].copy()

        if len(intersecting_poly_features.index) == 0:
            # TODO: should this give a warning/even error?
            continue

        intersecting_poly_features["__area"] = intersecting_poly_features.geometry.area
        intersecting_poly_features = intersecting_poly_features.sort_values(
            "__area",
            ascending=False,
        )
        intersecting_poly_features = intersecting_poly_features.drop("__area", axis=1)

        largest_poly_feature = intersecting_poly_features.iloc[0]

        new_feature = largest_poly_feature.copy()
        new_feature.geometry = line.geometry

        if old_ids_column is not None:
            new_feature[old_ids_column] = tuple(
                intersecting_poly_features.index.to_list(),
            )
        new_features.append(new_feature)

    output = GeoDataFrame(new_features, crs=target_gdf.crs)
    output.index.name = source_gdf.index.name

    return output
