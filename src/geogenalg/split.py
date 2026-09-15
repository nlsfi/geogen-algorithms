#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from typing import TYPE_CHECKING

from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype

from geogenalg.core.geometry import split_line_at_distances
from geogenalg.identity import hash_duplicate_indexes
from geogenalg.utility.dataframe_processing import combine_gdfs, copy_gdf_as_empty

if TYPE_CHECKING:
    from shapely import LineString


def explode_and_hash_id(
    data: GeoDataFrame,
    hash_prefix: str,
) -> GeoDataFrame:
    """Explode any multigeometries and set their index as a hash value.

    It is required that the input has a string index.

    If a feature is of a multigeometry type but has only single part, it
    will be changed to a single geometry and the original ID retained.

    Hashing is done with SHA256 and its input is the concatenation of the hash
    prefix, the original index and the WKT of the part's geometry.

    Args:
    ----
        data: GeoDataFrame to be processed.
        hash_prefix: Prefix to use in hash function input.

    Returns:
    -------
        GeoDataFrame with unchanged (if any) and exploded (if any) features.

    Raises:
    ------
        ValueError: If input GeoDataFrame does not have a string index.

    """
    if not is_string_dtype(data.index):
        msg = "GeoDataFrame must have a string index."
        raise ValueError(msg)

    return hash_duplicate_indexes(data.explode(), hash_prefix)


def split_lines_by_points(
    lines_gdf: GeoDataFrame,
    points_gdf: GeoDataFrame,
    max_distance: float = 1.0,
) -> GeoDataFrame:
    """Split line geometries at locations nearest to nearby points.

    Points within max_distance from a line are projected onto the line and
    used as split locations.

    Args:
    ----
        lines_gdf: GeoDataFrame containing LineString geometries.
        points_gdf: GeoDataFrame containing Point geometries.
        max_distance: Maximum snapping distance from a point to the line.

    Returns:
    -------
        GeoDataFrame containing split LineString geometries.

    """
    points_sindex = points_gdf.sindex

    result_rows = []

    for i, row in enumerate(lines_gdf.itertuples()):
        line: LineString = getattr(row, lines_gdf.geometry.name)

        candidate_idx = list(
            points_sindex.intersection(
                line.buffer(max_distance).bounds,
            )
        )

        if not candidate_idx:
            result_rows.append(lines_gdf.iloc[[i]])
            continue

        candidate_points = points_gdf.iloc[candidate_idx]

        split_distances: list[float] = []

        for point in candidate_points.geometry:
            if line.distance(point) >= max_distance:
                continue

            distance_along = line.project(point)
            split_distances.append(distance_along)

        if not split_distances:
            result_rows.append(lines_gdf.iloc[[i]])
            continue

        split_distances = sorted(set(split_distances))

        segments = split_line_at_distances(line, split_distances)

        for segment in segments:
            new_row = lines_gdf.iloc[[i]].copy()
            new_row.geometry = [segment]
            result_rows.append(new_row)

    if not result_rows:
        return copy_gdf_as_empty(lines_gdf)

    result = combine_gdfs(result_rows, ignore_index=False)
    return GeoDataFrame(result, geometry=lines_gdf.geometry.name, crs=lines_gdf.crs)
