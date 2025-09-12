#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from itertools import chain

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

    input_gdf.geometry = input_gdf.buffer(0)

    dissolved_gdf: gpd.GeoDataFrame = (
        input_gdf.dissolve(by=by_column).explode(index_parts=True).reset_index()
    )

    dissolved_polygons_gdfs: list[gpd.GeoDataFrame] = []

    for _, dissolved_row in dissolved_gdf.iterrows():
        dissolved_geom = dissolved_row.geometry
        group_value = dissolved_row[by_column]

        # Only intersect polygons from the same group
        intersecting_polygons_gdf: gpd.GeoDataFrame = input_gdf[
            (input_gdf[by_column] == group_value)
            & (input_gdf.geometry.intersects(dissolved_geom))
        ].copy()

        if dissolve_members_column not in intersecting_polygons_gdf.columns:
            intersecting_polygons_gdf[dissolve_members_column] = None

        # TODO: Add feature to choose how the representative point is selected
        min_id = intersecting_polygons_gdf[unique_key_column].min()
        representative_polygon_gdf: gpd.GeoDataFrame = (
            intersecting_polygons_gdf.loc[
                intersecting_polygons_gdf[unique_key_column] == min_id
            ]
            .iloc[
                [0]
            ]  # if multiple polygons with same key value, only one is representing
            .copy()
        )

        if len(intersecting_polygons_gdf) > 1:
            _merge_dissolve_members_column(
                intersecting_polygons_gdf,
                dissolve_members_column,
                dissolved_row,
                unique_key_column,
                representative_polygon_gdf,
            )

        # For single polygons, the dissolve members field is set to None if the field
        # does not contain any dissolve members
        elif dissolve_members_column not in representative_polygon_gdf.columns:
            representative_polygon_gdf[dissolve_members_column] = None

        representative_polygon_gdf = representative_polygon_gdf.iloc[[0]].copy()
        representative_polygon_gdf.geometry = [dissolved_geom]
        dissolved_polygons_gdfs.append(representative_polygon_gdf)

    return pd.concat(dissolved_polygons_gdfs).reset_index(drop=True)


def _merge_dissolve_members_column(
    intersecting_polygons_gdf: gpd.GeoDataFrame,
    dissolve_members_column: str,
    dissolved_row: pd.Series,
    unique_key_column: str,
    representative_polygon_gdf: gpd.GeoDataFrame,
) -> None:
    """List all dissolve member keys into the dissolve_members_column."""
    existing_members = (
        intersecting_polygons_gdf[dissolve_members_column].dropna().to_list()
    )

    existing_members = list(chain.from_iterable(existing_members))

    new_dissolve_members = []
    if dissolved_row.get(dissolve_members_column):
        for value in dissolved_row[dissolve_members_column]:
            if isinstance(value, list):
                new_dissolve_members += value
            else:
                new_dissolve_members.append(value)

    dissolve_members = intersecting_polygons_gdf[unique_key_column].dropna().to_list()

    all_dissolve_members = sorted(
        {str(x) for x in (existing_members + new_dissolve_members + dissolve_members)}
    )

    representative_polygon_gdf[dissolve_members_column] = [all_dissolve_members]
