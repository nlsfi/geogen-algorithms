#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from itertools import chain

from geopandas import GeoDataFrame
from pandas import Series, concat
from shapely import GeometryCollection, MultiPolygon, Polygon, line_merge
from shapely.geometry import LineString, MultiLineString

from geogenalg.attributes import inherit_attributes
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type


def merge_connecting_lines_by_attribute(
    input_gdf: GeoDataFrame,
    attribute: str,
    old_ids_column: str = "old_ids",
) -> GeoDataFrame:
    """Merge LineStrings in the GeoDataFrame by a given attribute.

    Lines with the same grouping attribute value and connecting endpoints will
    get merged into MultiLineStrings.

    Args:
    ----
        input_gdf: A GeoDataFrame containing the (Multi)LineStrings to merge.
        attribute: The attribute name used to group lines for merging.
        old_ids_column: Name of the column in the output GeoDataFrame
            containing a tuple of the cluster's old identifiers.

    Returns:
    -------
        A new GeoDataFrame with merged line geometries and corresponding attributes.
        Identifiers of the original lines are in `old_ids_column` column.

    """
    merged_records: list[dict] = []

    for _, group in input_gdf.groupby(attribute):
        lines: list[tuple[LineString, dict, str]] = []  # geom, attributes, id

        for idx, row in group.iterrows():
            geom = row.geometry
            if geom is None:
                continue

            row_properties = row.drop(input_gdf.geometry.name).to_dict()
            source_id = str(idx)

            if isinstance(geom, LineString):
                lines.append((geom, row_properties, source_id))
            elif isinstance(geom, MultiLineString):
                lines.extend((part, row_properties, source_id) for part in geom.geoms)

        if not lines:
            continue

        # Extract just geometries for merging
        geometries = [geom for geom, _, _ in lines]
        merged = line_merge(MultiLineString(geometries))

        if isinstance(merged, LineString):
            merged_lines = [merged]
        elif isinstance(merged, MultiLineString):
            merged_lines = list(merged.geoms)
        else:
            continue

        for merged_line in merged_lines:
            old_ids = sorted(
                {
                    source_id
                    for source_line, _, source_id in lines
                    if (
                        source_line.within(merged_line)
                        or source_line.equals(merged_line)
                    )
                }
            )
            properties = lines[0][1].copy()
            properties[input_gdf.geometry.name] = merged_line
            properties[old_ids_column] = old_ids
            merged_records.append(properties)

    result_gdf = GeoDataFrame(data=merged_records, crs=input_gdf.crs)

    # Assign attributes to merged lines by mapping back to one of the source lines
    result_gdf_with_attributes = inherit_attributes(input_gdf, result_gdf)

    # Assign old ids back
    result_gdf_with_attributes[old_ids_column] = result_gdf[old_ids_column]

    return result_gdf_with_attributes


def dissolve_and_inherit_attributes(
    input_gdf: GeoDataFrame,
    by_column: str,
    unique_key_column: str,
    dissolve_members_column: str = "dissolve_members",
) -> GeoDataFrame:
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
    if not check_gdf_geometry_type(input_gdf, ["Polygon", "MultiPolygon"]):
        msg = "Dissolve only supports Polygon or MultiPolygon geometries."
        raise GeometryTypeError(msg)
    if input_gdf.empty:
        return input_gdf

    # Apply buffer(0) to clean geometries. It fixes invalid polygons and
    # ensures resulting geometries are valid before further processing.
    input_gdf.geometry = input_gdf.buffer(0)

    dissolved_gdf: GeoDataFrame = (
        input_gdf.dissolve(by=by_column).explode(index_parts=True).reset_index()
    )

    dissolved_polygons_gdfs: list[GeoDataFrame] = []

    for _, dissolved_row in dissolved_gdf.iterrows():
        dissolved_geom = dissolved_row.geometry
        group_value = dissolved_row[by_column]

        # Only intersect polygons from the same group
        intersecting_polygons_gdf: GeoDataFrame = input_gdf[
            (input_gdf[by_column] == group_value)
            & (input_gdf.geometry.intersects(dissolved_geom))
        ].copy()

        if dissolve_members_column not in intersecting_polygons_gdf.columns:
            intersecting_polygons_gdf[dissolve_members_column] = None

        # TODO: Add feature to choose how the representative point is selected
        min_id = intersecting_polygons_gdf[unique_key_column].min()
        representative_polygon_gdf: GeoDataFrame = (
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

    return concat(dissolved_polygons_gdfs).reset_index(drop=True)


def _merge_dissolve_members_column(
    intersecting_polygons_gdf: GeoDataFrame,
    dissolve_members_column: str,
    dissolved_row: Series,
    unique_key_column: str,
    representative_polygon_gdf: GeoDataFrame,
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


def buffer_and_merge_polygons(
    input_gdf: GeoDataFrame, buffer_distance: float
) -> GeoDataFrame:
    """Merge polygons that are close to each other using a buffer.

    Attributes and IDs of the input data are not preserved and
    included in the output.

    Steps:
    - Buffers polygons outward by `buffer_distance`
    - Merge touching and overlapping geometries
    - Buffer back inward to restore approximate original sizes and shapes

    Args:
    ----
        input_gdf: GeoDataFrame containing Polygon geometries.
        buffer_distance: Distance used for buffering.

    Returns:
    -------
        A GeoDataFrame where nearby polygons have been merged and each merged
        polygon is represented as its own row.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
            polygon geometries.

    """
    if not check_gdf_geometry_type(input_gdf, ["Polygon", "MultiPolygon"]):
        msg = "Buffer and merge polygons works only with (Multi)Polygon geometries."
        raise GeometryTypeError(msg)

    if input_gdf.empty:
        return input_gdf

    # Return early if input contains only one polygon
    if len(input_gdf) == 1 and isinstance(input_gdf.geometry.iloc[0], Polygon):
        return input_gdf

    # 1. Buffer outward
    buffered = input_gdf.geometry.buffer(buffer_distance)

    # 2. Union all buffered geometries
    merged = buffered.union_all()

    # 3. Collect individual polygons
    polygons: list[Polygon] = []
    if isinstance(merged, Polygon):
        polygons = [merged]
    elif isinstance(merged, MultiPolygon):
        polygons = list(merged.geoms)
    elif isinstance(merged, GeometryCollection):
        polygons = [geom for geom in merged.geoms if isinstance(geom, Polygon)]
    else:
        polygons = []

    # 4. Buffer inward to restore approximate original size
    out_geoms = [
        geom.buffer(-buffer_distance)
        for geom in polygons
        if geom is not None and not geom.is_empty
    ]

    # Filter out empty
    out_geoms = [geom for geom in out_geoms if geom is not None and not geom.is_empty]

    return GeoDataFrame(geometry=out_geoms, crs=input_gdf.crs)


def dissolve_polygon_layers(input_gdfs: list[GeoDataFrame]) -> GeoDataFrame:
    """Append and dissolve multiple polygon layers into one.

    Args:
        input_gdfs: List of GeoDataFrames to merge directly.

    Returns:
        GeoDataFrame containing dissolved polygon geometries (geometry only).

    Raises:
        GeometryTypeError: If input data is not Polygons or MultiPolygons.

    """
    # Combine all layers
    combined = concat(input_gdfs, ignore_index=True)

    # Validate geometry type
    if not check_gdf_geometry_type(combined, ["Polygon", "MultiPolygon"]):
        error_msg_2 = "Only Polygon or MultiPolygon geometries are supported."
        raise GeometryTypeError(error_msg_2)

    # Clean geometries and dissolve (remove duplicates / internal borders)
    combined.geometry = combined.buffer(0)
    dissolved = combined.dissolve().explode(index_parts=True, ignore_index=True)

    # Return geometry-only GeoDataFrame
    return dissolved[[dissolved.geometry.name]].set_geometry(dissolved.geometry.name)
