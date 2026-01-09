#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, cast

from geopandas import GeoDataFrame, GeoSeries
from numpy import nan
from pandas import concat
from pygeoops import centerline

from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.core.geometry import (
    centerline_length,
    remove_line_segments_at_wide_sections,
    remove_small_parts,
)

if TYPE_CHECKING:
    from shapely import LineString, MultiLineString


def thin_polygon_sections_to_lines(  # noqa: PLR0913
    input_gdf: GeoDataFrame,
    threshold: float,
    min_line_length: float,
    min_new_section_length: float,
    min_new_section_area: float,
    *,
    width_check_distance: float = 10.0,
    old_ids_column: str | None = None,
) -> GeoDataFrame:
    """Convert thin sections of polygon features to linestrings.

    New line features inherit attributes from largest intersecting polygon.

    Args:
    ----
        input_gdf: GeoDataFrame containing polygons
        threshold: Width threshold for a section to be considered thin.
        min_line_length: Minimum length for a new line feature.
        min_new_section_length: Minimum length of a new section which is
            created in initial polygon-to-line conversion. If section's length is
            under this, it too will be transformed into a line.
        min_new_section_area: Minimum area for newly created section.
        width_check_distance: Controls the distance at which width is checked.
        old_ids_column: Optionally define a column name, to which all
            intersecting polygon IDs will be saved for new line features.

    Returns:
    -------
        GeoDataFrame containing new line features and remaining polygon
        features, with thin sections removed.

    """
    gdf = input_gdf.copy()

    # We need to calculate centerline of union of input geodataframe.
    # Otherwise if it has shared edges the centerlines will not form properly.
    input_union = input_gdf.union_all()
    full_centerline = cast(
        "LineString | MultiLineString",
        centerline(
            input_union.segmentize(1),  # segmentize to get less jagged centerline
            simplifytolerance=0.5,
        ),
    )

    lines = remove_line_segments_at_wide_sections(
        full_centerline.segmentize(width_check_distance),
        input_union,
        threshold,
    )

    lines = remove_small_parts(lines, min_line_length)

    if lines.is_empty:
        if old_ids_column is not None:
            gdf[old_ids_column] = nan

        return gdf

    # Create a polygonal mask from lines which have been determined to be long
    # enough. This is used to remove thin sections from the polygons.
    mask = lines.buffer(threshold, cap_style="flat")

    # Apply mask, find out which geometries were modified and apply processing
    # function to them
    masked_geoms = gdf.geometry.difference(mask)
    gdf["modified_by_mask__"] = ~gdf.geometry.geom_equals(masked_geoms)
    gdf.geometry = masked_geoms

    geom_column_name = str(gdf.geometry.name)
    for row in gdf.itertuples():
        if not row.modified_by_mask__:
            continue

        geom = getattr(row, geom_column_name)

        new_geom = remove_small_parts(geom, min_new_section_area)

        # Check length from exterior centerline. This gives better results
        # cartographically, because if the polygon has a bunch of holes,
        # the centerline will weave around them and consist of many parts
        # which will add to the total calculated length, even though here
        # we want to only consider the "length" of the polygon as it
        # appears on a map.
        new_geom = remove_small_parts(
            geom,
            min_new_section_length,
            size_function=lambda geom: centerline_length(geom, exterior_only=True),
        )

        gdf.loc[row.Index, geom_column_name] = new_geom

    gdf = gdf.drop("modified_by_mask__", axis=1)
    gdf = gdf.loc[~gdf.geometry.is_empty]

    # Extract new lines from original, non-segmentized centerline
    new_lines = full_centerline.difference(gdf.geometry.union_all())

    if new_lines.is_empty:
        if old_ids_column is not None:
            gdf[old_ids_column] = nan

        return gdf

    new_line_features = GeoDataFrame(
        geometry=GeoSeries(
            new_lines,
        ).explode(ignore_index=True),
        crs=input_gdf.crs,
    )

    new_line_features = inherit_attributes_from_largest(
        input_gdf,
        new_line_features,
        old_ids_column,
    )

    return concat([new_line_features, gdf])
