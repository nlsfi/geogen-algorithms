#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import geopandas as gpd
import pandas as pd
from shapely import line_merge
from shapely.geometry import LineString, MultiLineString

from geogenalg import attributes
from geogenalg.core.exceptions import GeometryTypeError


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
                lines.append((geom, row.drop(input_gdf.geometry.name).to_dict()))
            elif isinstance(geom, MultiLineString):
                lines.extend(
                    [
                        (part, row.drop(input_gdf.geometry.name).to_dict())
                        for part in geom.geoms
                    ]
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
            [{**properties, input_gdf.geometry.name: line} for line in merged_lines]
        )

    result_gdf = gpd.GeoDataFrame(merged_records, crs=input_gdf.crs)

    # Assign attributes to merged lines by mapping back to one of the source lines
    return attributes.inherit_attributes(input_gdf, result_gdf)


def dissolve_and_inherit_attributes(
    input_gdf: gpd.GeoDataFrame,
    by_column: str,
    unique_key_column: str,
    dissolve_members_column: str = "dissolve_members",
) -> gpd.GeoDataFrame:
    """Dissolve polygons and inherit attributes from a representative original polygon.

    Args:
    ----
        input_gdf: Input GeoDataFrame with Polygon geometries. The GeoDataFrame must
              include a column with a unique key.
        by_column: Column name used to group and dissolve polygons
        unique_key_column: Name of the column containing unique identifiers
        dissolve_members_column: Name of the column that lists the original polygons
              that have been dissolved

    Returns:
    -------
        A GeoDataFrame with dissolved polygons.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
              polygon geometries.

    """
    if not all(input_gdf.geometry.type.isin(["Polygon", "MultiPolygon"])):
        msg = "Dissolve only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)
    if input_gdf.empty:
        return input_gdf

    dissolved_gdf: gpd.GeoDataFrame = (
        input_gdf.dissolve(by=by_column).explode(index_parts=True).reset_index()
    )

    dissolved_polygons_gdfs: list[gpd.GeoDataFrame] = []

    for _, dissolved_row in dissolved_gdf.iterrows():
        dissolved_geom = dissolved_row.geometry
        group_value = dissolved_row[by_column]

        # Only intersect polygons from the same group
        intersecting_polygons_gdf = input_gdf[
            (input_gdf[by_column] == group_value)
            & (input_gdf.geometry.intersects(dissolved_geom))
        ]

        # TODO: Add feature to choose how the representative point is selected
        min_id = intersecting_polygons_gdf[unique_key_column].min()
        representative_polygon_gdf: gpd.GeoDataFrame = intersecting_polygons_gdf.loc[
            intersecting_polygons_gdf[unique_key_column] == min_id
        ].copy()

        # If several polygons are dissolved, their keys are saved as dissolve members
        if len(intersecting_polygons_gdf) > 1:
            new_dissolve_members = intersecting_polygons_gdf[unique_key_column].tolist()
            if (
                dissolve_members_column in representative_polygon_gdf.columns
                and representative_polygon_gdf[dissolve_members_column] is not None
            ):
                existing_members = representative_polygon_gdf[dissolve_members_column]
                representative_polygon_gdf[dissolve_members_column] = (
                    existing_members + new_dissolve_members
                )
            else:
                representative_polygon_gdf[dissolve_members_column] = [
                    new_dissolve_members
                ]

        # For single polygons, the dissolve members field is set to None if the field
        # does not contain any dissolve members
        elif dissolve_members_column not in representative_polygon_gdf.columns:
            representative_polygon_gdf[dissolve_members_column] = None

        representative_polygon_gdf.geometry = [dissolved_geom]
        dissolved_polygons_gdfs.append(representative_polygon_gdf)

    return pd.concat(dissolved_polygons_gdfs).reset_index(drop=True)
