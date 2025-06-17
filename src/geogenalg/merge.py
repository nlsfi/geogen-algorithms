#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString

from geogenalg import attributes


def merge_connecting_lines_by_attribute(
    input_gdf: gpd.GeoDataFrame, attribute: str
) -> gpd.GeoDataFrame:
    """Merge LineStrings in the GeoDataFrame by a given attribute.

    Args:
    ----
        input_gdf: A GeoDataFrame containing the lines to merge.
        attribute: The attribute name used to group lines for merging.

    Returns:
    -------
        A new GeoDataFrame with merged line geometries and corresponding attributes.

    """
    merged_records = []

    for _, group in input_gdf.groupby(attribute):
        lines = []

        for _, row in group.iterrows():
            geom = row.geometry
            if geom is None:
                continue
            if isinstance(geom, LineString):
                lines.append((geom, row.drop("geometry").to_dict()))
            elif isinstance(geom, MultiLineString):
                lines.extend(
                    [(part, row.drop("geometry").to_dict()) for part in geom.geoms]
                )

        if not lines:
            continue

        # Extract just geometries for merging
        geometries = [geom for geom, _ in lines]
        merged = line_merge(MultiLineString(geometries))

        if isinstance(merged, LineString):
            merged_lines = [merged]
        elif isinstance(merged, MultiLineString):
            merged_lines = list(merged.geoms)
        else:
            continue

        properties = lines[0][1]

        merged_records.extend(
            [{**properties, "geometry": line} for line in merged_lines]
        )

    result_gdf = gpd.GeoDataFrame(merged_records, crs=input_gdf.crs)

    # Assign attributes to merged lines by mapping back to one of the source lines
    return attributes.inherit_attributes(input_gdf, result_gdf)
