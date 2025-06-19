#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd


def inherit_attributes(
    source_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
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

    return gpd.GeoDataFrame(
        new_features, geometry=source_gdf.geometry.name, crs=source_gdf.crs
    )
