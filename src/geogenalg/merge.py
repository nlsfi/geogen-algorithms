#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import operator
from typing import Literal

from geopandas import GeoDataFrame
from numpy import zeros
from pandas import Series
from shapely import GeometryCollection, MultiPolygon, Polygon, line_merge
from shapely.geometry import LineString, MultiLineString, Point

from geogenalg.analyze import group_geometries_by_intersections_recursively
from geogenalg.attributes import inherit_attributes
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.dataframe_processing import (
    combine_gdfs,
    copy_gdf_as_empty,
)
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

    for _, group in input_gdf.groupby(attribute, dropna=False):
        lines: list[tuple[LineString, dict, str]] = []  # geom, attributes, id

        for idx, row in group.iterrows():
            geom = row[group.geometry.name]
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

    result_gdf = GeoDataFrame(
        data=merged_records,
        geometry=input_gdf.geometry.name,
        crs=input_gdf.crs,
    )

    # Assign attributes to merged lines by mapping back to one of the source lines
    result_gdf_with_attributes = inherit_attributes(input_gdf, result_gdf)

    # Assign old ids back
    result_gdf_with_attributes[old_ids_column] = result_gdf[old_ids_column]

    return result_gdf_with_attributes


def dissolve_and_inherit_attributes(  # noqa: C901
    input_gdf: GeoDataFrame,
    by_column: str | list[str] | None = None,
    old_ids_column: str = "old_ids",
    inherit_from: Literal["min_id", "most_intersection"] = "most_intersection",
) -> GeoDataFrame:
    """Dissolve polygons and inherit attributes from a representative original polygon.

    Args:
    ----
        input_gdf: Input GeoDataFrame with Polygon geometries. The GeoDataFrame must
            include a column with a unique key.
        by_column: Column(s) whose values define the groups to be dissolved.
            If left None, the entire dataframe is considered a single group to dissolve.
        old_ids_column: Name of the column in the output GeoDataFrame
            containing a tuple of the cluster's old identifiers.
        inherit_from: Method for determining which intersecting feature is considered as
            the representative original polygon to inherit attributes from.

    Returns:
    -------
        A GeoDataFrame with dissolved polygons. Identifiers of the original
        polygons are in `old_ids_column` column.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
            polygon geometries.

    """
    if not check_gdf_geometry_type(input_gdf, {"Polygon"}):
        msg = "Only works for single polygons."
        raise GeometryTypeError(msg)

    if input_gdf.empty:
        return copy_gdf_as_empty(input_gdf, add_columns={old_ids_column: "object"})

    gdf = input_gdf.copy()

    old_index_name = gdf.index.name
    gdf.index.name = None

    # Create a dummy column to get every row to the same group if no column(s)
    # was given.
    by = zeros(len(gdf), dtype="int64") if by_column is None else by_column

    # This function works by first grouping rows by the given column(s) (or all
    # in the same group if nothing was given). Then we group rows by their
    # geometries such that rows which intersect each other are grouped
    # together. Those rows are then dissolved together and their geometries
    # combined.

    # The reason for having special handling for dissolving instead of using
    # GeoPandas dissolve and exploding features out of the multipolygon is that
    # if disjoint features are added as parts of a multipolygon their vertex
    # order may be changed, which may affect simplify etc. results, but more
    # importantly index handling when the geometry is used to hash a new index.

    def most_intersection_sort(geoms: Series, union: Polygon) -> Series:
        return geoms.apply(lambda geom: union.intersection(geom).area)

    def dissolve_geometry_group(geometry_group: GeoDataFrame) -> GeoDataFrame:
        union = geometry_group.geometry.union_all()

        used_geometry_group = geometry_group
        single_touching_point_features = copy_gdf_as_empty(
            geometry_group,
            add_columns={old_ids_column: "object"},
        )

        if isinstance(union, MultiPolygon):
            # This means we have now have polygons which intersect some other
            # polygon in the group, but only by one point which means the
            # geometries can't be dissolved into a single polygon. We have to
            # extract out the features which will not dissolve into the union
            # and keep them as separate features.
            union_without_feature = geometry_group.geometry.apply(
                union.difference,
            )
            intersection = geometry_group.geometry.intersection(union_without_feature)
            single_intersection_point_only = intersection.apply(
                lambda geom: isinstance(geom, Point),
            )

            used_geometry_group = geometry_group.loc[~single_intersection_point_only]

            single_touching_point_features = geometry_group.loc[
                single_intersection_point_only
            ].copy()

            single_touching_point_features[old_ids_column] = (
                single_touching_point_features.index.to_series().apply(
                    lambda idx: (idx,)
                )
            )

            if used_geometry_group.empty:
                return single_touching_point_features

            union = used_geometry_group.union_all()

        match inherit_from:
            case "min_id":
                used_geometry_group = used_geometry_group.sort_index(
                    ascending=True,
                )
            case "most_intersection":
                used_geometry_group = used_geometry_group.sort_values(
                    by=used_geometry_group.geometry.name,
                    key=lambda geom: most_intersection_sort(geom, union),
                    ascending=False,
                )

        old_ids = tuple(used_geometry_group.index)

        dissolved_group = used_geometry_group[:1].copy()
        dissolved_group[old_ids_column] = [old_ids]
        dissolved_group.geometry = [union]

        if not single_touching_point_features.empty:
            dissolved_group = combine_gdfs(
                [
                    dissolved_group,
                    single_touching_point_features,
                ]
            )

        return dissolved_group

    def dissolve_attribute_group(group: GeoDataFrame) -> GeoDataFrame:
        grouped_by_geom = group_geometries_by_intersections_recursively(group)

        return (
            grouped_by_geom.groupby(
                by="_geometry_group",
                as_index=False,
                level=None,
                sort=False,
            )[grouped_by_geom.columns]
            .apply(
                dissolve_geometry_group,
                include_groups=False,
            )
            .drop("_geometry_group", axis=1)
        )

    attribute_groups = gdf.groupby(
        by=by,
        as_index=False,
        level=None,
        sort=False,
    )[gdf.columns].apply(
        dissolve_attribute_group,
        include_groups=False,
    )

    attribute_groups.index = attribute_groups[old_ids_column].apply(
        operator.itemgetter(0),
    )
    attribute_groups.index.name = old_index_name

    return attribute_groups


def buffer_and_merge_polygons(
    input_gdf: GeoDataFrame,
    buffer_distance: float,
    *,
    join_style: Literal["round", "mitre", "bevel"] = "bevel",
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
        join_style: Join style used for buffering.

    Returns:
    -------
        A GeoDataFrame where nearby polygons have been merged and each merged
        polygon is represented as its own row.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than
            polygon geometries.

    """
    if not check_gdf_geometry_type(input_gdf, {"Polygon", "MultiPolygon"}):
        msg = "Buffer and merge polygons works only with (Multi)Polygon geometries."
        raise GeometryTypeError(msg)

    if input_gdf.empty:
        return input_gdf

    # Return early if input contains only one polygon
    if len(input_gdf) == 1 and isinstance(input_gdf.geometry.iloc[0], Polygon):
        return input_gdf

    # 1. Buffer outward
    buffered = input_gdf.geometry.buffer(buffer_distance, join_style=join_style)

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
        geom.buffer(-buffer_distance, join_style=join_style)
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
    combined = combine_gdfs(input_gdfs, ignore_index=True)

    # Validate geometry type
    if not check_gdf_geometry_type(combined, {"Polygon", "MultiPolygon"}):
        error_msg_2 = "Only Polygon or MultiPolygon geometries are supported."
        raise GeometryTypeError(error_msg_2)

    # Clean geometries and dissolve (remove duplicates / internal borders)
    combined.geometry = combined.buffer(0)
    dissolved = combined.dissolve().explode(index_parts=True, ignore_index=True)

    # Return geometry-only GeoDataFrame
    return dissolved[[dissolved.geometry.name]].set_geometry(dissolved.geometry.name)
