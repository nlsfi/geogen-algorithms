#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from itertools import pairwise
from math import isclose

import pandas as pd
from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype
from shapely import LineString

from geogenalg.identity import hash_duplicate_indexes


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

    for row in lines_gdf.itertuples():
        line: LineString = row.geometry

        candidate_idx = list(
            points_sindex.intersection(
                line.buffer(max_distance).bounds,
            )
        )

        if not candidate_idx:
            new_row = lines_gdf.loc[[row.Index]]  # keep as single-row DataFrame
            result_rows.append(new_row)
            continue

        candidate_points = points_gdf.iloc[candidate_idx]

        split_distances: list[float] = []

        for point in candidate_points.geometry:
            if line.distance(point) >= max_distance:
                continue

            distance_along = line.project(point)
            split_distances.append(distance_along)

        if not split_distances:
            new_row = lines_gdf.loc[[row.Index]]  # keep as single-row DataFrame
            result_rows.append(new_row)
            continue

        split_distances = sorted(set(split_distances))

        segments = split_line_at_distances(line, split_distances)

        for segment in segments:
            new_row = lines_gdf.loc[[row.Index]].copy()  # single-row DataFrame
            new_row["geometry"] = [segment]
            result_rows.append(new_row)

    if not result_rows:
        return lines_gdf.iloc[0:0].copy()  # empty GeoDataFrame with correct schema

    result = pd.concat(result_rows, ignore_index=True)
    return GeoDataFrame(result, crs=lines_gdf.crs)


def split_line_at_distances(
    line: LineString,
    distances: list[float],
) -> list[LineString]:
    """Split a LineString at distances measured along the line.

    Returns
    -------
        List of LineString segments resulting from the split.

    """
    if not distances:
        return [line]

    coords = list(line.coords)

    result_segments: list[LineString] = []

    current_coords = [coords[0]]

    distance_iter = iter(sorted(set(distances)))
    next_split = next(distance_iter, None)

    accumulated = 0.0

    for start_coord, end_coord in pairwise(coords):
        segment = LineString([start_coord, end_coord])

        segment_length = segment.length
        segment_end_distance = accumulated + segment_length

        while (
            next_split is not None and accumulated < next_split < segment_end_distance
        ):
            relative_distance = next_split - accumulated

            split_point = segment.interpolate(relative_distance)

            split_coord = (split_point.x, split_point.y)

            current_coords.append(split_coord)

            result_segments.append(LineString(current_coords))

            current_coords = [split_coord]

            next_split = next(distance_iter, None)

        if next_split is not None and isclose(next_split, segment_end_distance):
            current_coords.append(end_coord)

            result_segments.append(LineString(current_coords))

            current_coords = [end_coord]

            next_split = next(distance_iter, None)

        else:
            current_coords.append(end_coord)

        accumulated = segment_end_distance

    if len(current_coords) > 1:
        result_segments.append(LineString(current_coords))

    return result_segments
