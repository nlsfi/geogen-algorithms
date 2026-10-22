#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from typing import Literal, cast

import numpy as np
from geopandas import GeoDataFrame, GeoSeries
from pandas import Index, Series, factorize
from shapely import (
    GeometryCollection,
    MultiPolygon,
    Polygon,
    STRtree,
    area,
    get_dimensions,
    intersection,
    line_merge,
    multilinestrings,
    union_all,
)
from shapely.geometry.base import BaseGeometry

from geogenalg.analyze import group_intersecting_geometries
from geogenalg.core.exceptions import DuplicateIndexError, GeometryTypeError
from geogenalg.core.geometry import get_contiguous_line_components
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
    get merged..

    Args:
    ----
        input_gdf: A GeoDataFrame containing the LineStrings to merge.
        attribute: The attribute name used to group lines for merging.
        old_ids_column: Name of the column in the output GeoDataFrame
            containing a tuple of the cluster's old identifiers.

    Returns:
    -------
        A new GeoDataFrame with merged line geometries and corresponding attributes.
        Identifiers of the original lines are in `old_ids_column` column.

    Raises:
    ------
        DuplicateIndexError: if input data has duplicate indices.

    """
    if not input_gdf.index.is_unique:
        msg = "Duplicate indexes found in input, cannot proceed."
        raise DuplicateIndexError(msg)

    if input_gdf.empty:
        return copy_gdf_as_empty(
            input_gdf,
            add_columns={old_ids_column: "object"},
        )

    geoms = input_gdf.geometry.to_numpy()
    lengths = input_gdf.geometry.length.to_numpy()
    source_ids = input_gdf.index.to_numpy()

    representative_positions = []
    merged_geometries = []
    old_ids = []

    groups = input_gdf.groupby(
        attribute,
        dropna=False,
        observed=True,
        sort=True,
    )

    for positions in groups.indices.values():
        positions = cast("np.ndarray", positions)
        if len(positions) <= 0:
            continue

        component_ids, _ = get_contiguous_line_components(
            geoms[positions],
            precision=None,
        )

        # Groups lines by their components
        order = np.argsort(component_ids, kind="stable")
        boundaries = np.flatnonzero(np.diff(component_ids[order])) + 1

        for component in np.split(order, boundaries):
            source_positions = positions[component]

            merged_geometries.append(
                line_merge(multilinestrings(geoms[source_positions]))
            )

            representative_positions.append(
                source_positions[np.argmax(lengths[source_positions])]
            )
            old_ids.append(sorted(set(source_ids[source_positions])))

    if not representative_positions:
        return copy_gdf_as_empty(
            input_gdf,
            add_columns={old_ids_column: "object"},
        )

    # Select representative row from original gdf and update with merged geometries
    result = input_gdf.iloc[representative_positions].copy()

    result.geometry = cast("list[BaseGeometry]", merged_geometries)
    result[old_ids_column] = old_ids

    return result


def _separate_point_touching_polygons(
    group_geoms: np.ndarray,
    positions: np.ndarray,
) -> tuple[np.ndarray, list[int], BaseGeometry | None]:
    """dissolve_and_inherit_attributes() helper.

    Identifies which polygons in a group touch others only by points.
    """
    geom_count = len(group_geoms)
    if geom_count == 2:  # noqa: PLR2004
        # No need for further processing, we already know these can't be
        # merged.
        return np.empty(0, dtype=int), list(positions), None

    tree = STRtree(group_geoms)
    left, right = tree.query(group_geoms, predicate="intersects")
    mask = left != right

    left = left[mask]
    right = right[mask]

    intersections = intersection(group_geoms[left], group_geoms[right])
    has_edge_or_overlap = get_dimensions(intersections) > 0  # not a (multi)point

    is_connected = np.zeros(geom_count, dtype=bool)
    is_connected[left[has_edge_or_overlap]] = True

    unmerged_positions = list(positions[~is_connected])
    merged_positions = positions[is_connected]
    union = union_all(group_geoms[is_connected]) if len(merged_positions) > 0 else None

    return merged_positions, unmerged_positions, union


def dissolve_and_inherit_attributes(  # noqa: PLR0915
    input_gdf: GeoDataFrame,
    by_column: str | list[str] | None = None,
    old_ids_column: str = "old_ids",
    *,
    grouping_strategy: Literal["iterative", "bulk"] = "iterative",
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
        grouping_strategy: `iterative` performs a BFS directly on the spatial
            index, whereas `bulk` constructs an graph and finds connected
            components. Generally speaking `bulk` may be faster with sparser
            datasets with fewer intersecting geometries, whereas `iterative` may be
            faster with dense datasets with lots of intersections.

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

    if by_column is None:
        attribute_group_indexes = [np.arange(len(gdf))]
    else:
        attribute_group_indexes = [
            idx
            for idx in gdf.groupby(by=by_column, sort=False).indices.values()
            if len(idx) > 0
        ]

    if not attribute_group_indexes:
        return copy_gdf_as_empty(input_gdf, add_columns={old_ids_column: "object"})

    next_group = 0
    geom_group_ids = []

    for indexes in attribute_group_indexes:
        grouped = group_intersecting_geometries(
            gdf.iloc[indexes],
            strategy=grouping_strategy,
        )

        codes, _ = factorize(grouped["_geometry_group"].to_numpy(), sort=False)
        geom_group_ids.append(codes + next_group)
        next_group += codes.max() + 1

    # We sort features into a contiguous array which allows for fast slicing
    # later on
    feature_positions = np.concatenate(attribute_group_indexes)
    geom_group_ids = np.concatenate(geom_group_ids)

    order = np.argsort(geom_group_ids, kind="stable")
    feature_positions = feature_positions[order]
    geom_group_ids = geom_group_ids[order]

    geoms = gdf.geometry.to_numpy()[feature_positions]
    index_values = gdf.index.to_numpy()[feature_positions]

    # Geometry group slice boundaries
    geom_group_sizes = np.bincount(geom_group_ids)
    geom_group_offsets = np.concatenate(([0], np.cumsum(geom_group_sizes)))

    single_geom_groups = np.flatnonzero(geom_group_sizes == 1)
    single_geom_positions = geom_group_offsets[single_geom_groups]

    representative_feature_positions: list[int] = []
    out_geoms: list = []
    out_old_ids: list[tuple] = []
    out_geom_groups: list[int] = []

    # Union geometry groups with more than one member
    for group_id in np.flatnonzero(geom_group_sizes > 1):
        start, end = geom_group_offsets[group_id], geom_group_offsets[group_id + 1]
        group_geoms = geoms[start:end]

        # group_intersecting_geometries groups purely by intersection. This
        # means there may be polygons intersect another polygon in the group,
        # but only at points (no shared edge or overlap), which means we cannot
        # dissolve the group into a single polygon. Thus we keep track of what
        # geometries at what indices we can actually merge.
        merged_positions = np.arange(start, end)
        unmerged_positions: list[int] = []

        union = union_all(group_geoms)

        # Union returns MultiPolygon -> we need to find which polygons touch
        # another such that the whole group can't be dissolved
        if isinstance(union, MultiPolygon):
            merged_positions, unmerged_positions, union = (
                _separate_point_touching_polygons(
                    group_geoms,
                    merged_positions,
                )
            )

        if len(merged_positions) > 0:
            areas = area(geoms[merged_positions])
            ranking = np.argsort(-areas, kind="stable")
            ranked = merged_positions[ranking]

            representative_feature_positions.append(ranked[0])

            out_geoms.append(union)
            out_old_ids.append(tuple(index_values[ranked]))
            out_geom_groups.append(int(group_id))

        for pos in unmerged_positions:
            representative_feature_positions.append(pos)

            out_geoms.append(geoms[pos])
            out_old_ids.append((index_values[pos],))
            out_geom_groups.append(int(group_id))

    all_representative_feature_positions = np.concatenate(
        [
            single_geom_positions,
            np.asarray(representative_feature_positions, dtype="int64"),
        ],
    )
    all_geom_groups = np.concatenate(
        [
            single_geom_groups,
            np.asarray(out_geom_groups, dtype="int64"),
        ],
    )

    all_geoms = np.empty(len(all_representative_feature_positions), dtype=object)
    all_geoms[: len(single_geom_positions)] = geoms[single_geom_positions]
    all_geoms[len(single_geom_positions) :] = out_geoms
    all_old_ids = [(idx,) for idx in index_values[single_geom_positions]] + out_old_ids

    final = np.argsort(all_geom_groups, kind="stable")
    final_rep_feature_positions = all_representative_feature_positions[final]

    result = gdf.iloc[feature_positions[final_rep_feature_positions]].copy()
    result[old_ids_column] = Series(
        [all_old_ids[i] for i in final],
        index=result.index,
        dtype="object",
    )
    result = result.set_geometry(
        GeoSeries(all_geoms[final], index=result.index, crs=gdf.crs),
        crs=gdf.crs,
    )
    result.index = Index(index_values[final_rep_feature_positions], name=old_index_name)

    return result


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
