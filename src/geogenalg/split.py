#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import numpy as np
from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype
from shapely import (
    LineString,
    get_coordinates,
    line_interpolate_point,
    line_locate_point,
)
from shapely.geometry import MultiLineString

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


def split_lines_by_points(  # noqa: PLR0915, PLR0914
    lines_gdf: GeoDataFrame,
    points_gdf: GeoDataFrame,
    max_distance: float,
) -> GeoDataFrame:
    """Split line geometries at locations nearest to nearby points.

    Points within max_distance from a line are projected onto the line and
    used as split locations.

    Split geometries are collected and returned as a MultiLineString.

    Args:
    ----
        lines_gdf: GeoDataFrame containing LineString geometries.
        points_gdf: GeoDataFrame containing Point geometries.
        max_distance: Maximum snapping distance from a point to the line.

    Returns:
    -------
        GeoDataFrame containing split LineString geometries.

    """
    epsilon = 1e-8  # No splits under this length of endpoints.
    new_geoms = lines_gdf.geometry.to_numpy(copy=True)

    point_indexes, line_spatial_indexes = lines_gdf.sindex.query(
        points_gdf.geometry,
        predicate="dwithin",
        distance=max_distance,
    )

    if len(line_spatial_indexes) == 0:
        return lines_gdf.copy()

    # Group points which are within the tolerance to lines
    order = np.argsort(line_spatial_indexes)
    sorted_line_ids = line_spatial_indexes[order]
    sorted_point_ids = point_indexes[order]

    unique_line_ids, split_at = np.unique(sorted_line_ids, return_index=True)
    grouped_point_indexes = np.split(sorted_point_ids, split_at[1:])

    line_geoms = lines_gdf.geometry.to_numpy()[unique_line_ids]
    line_count = len(line_geoms)

    all_coords, line_indexes = get_coordinates(line_geoms, return_index=True)

    # Calculate segment lengths
    segment_vectors = np.diff(all_coords, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)

    # Mark "fake" segments between linestrings in the flat array as having zero
    # length.
    same_line = line_indexes[1:] == line_indexes[:-1]
    segment_lengths[~same_line] = 0.0

    # Insert 0 at cumulative segment lengths to match vertex count
    cumulative_all = np.insert(np.cumsum(segment_lengths), 0, 0.0)

    line_start_point_indexes = np.searchsorted(line_indexes, np.arange(line_count))
    line_end_point_indexes = (
        np.searchsorted(
            line_indexes,
            np.arange(line_count),
            side="right",
        )
        - 1
    )

    for i in range(line_count):
        line_id = unique_line_ids[i]
        line_geom = line_geoms[i]
        points_on_line_indexes = grouped_point_indexes[i]
        points_on_line = points_gdf.geometry.to_numpy()[points_on_line_indexes]

        # Extract line coordinates and reset cumulative distances to start at 0.0
        start_point_id = line_start_point_indexes[i]
        end_point_id = line_end_point_indexes[i]

        line_coords = all_coords[start_point_id : end_point_id + 1]
        line_cumulative_lengths = (
            cumulative_all[start_point_id : end_point_id + 1]
            - cumulative_all[start_point_id]
        )
        line_length = line_cumulative_lengths[-1]

        # Calculate distances where we need to split the line at.
        distances = line_locate_point(line_geom, points_on_line)
        distances = distances[
            (distances > epsilon) & (distances < line_length - epsilon)
        ]
        distances = np.unique(distances)

        if len(distances) == 0:
            continue

        split_points = line_interpolate_point(line_geom, distances)
        split_coords = get_coordinates(split_points)

        # Combine the split distances and point onto the line.
        all_distances = np.concatenate((line_cumulative_lengths, distances))
        all_coords_line = np.vstack((line_coords, split_coords))

        # Mark which points we split at.
        is_split_point = np.concatenate(
            (
                np.zeros(len(line_cumulative_lengths), dtype=bool),
                np.ones(len(distances), dtype=bool),
            )
        )

        # Sort line coordinates according to new distances.
        sort_order = np.argsort(all_distances, kind="stable")
        sorted_coords = all_coords_line[sort_order]
        sorted_distances = all_distances[sort_order]
        sorted_is_split = is_split_point[sort_order]

        # Remove duplicates (f.e. split point is directly on a vertex).
        mask = np.zeros(len(sorted_distances), dtype=bool)
        mask[1:] = np.diff(sorted_distances) < epsilon

        if np.any(mask):
            sorted_is_split[:-1] |= (mask & sorted_is_split)[1:]

            # Drop duplicates.
            keep_mask = ~mask
            sorted_coords = sorted_coords[keep_mask]
            sorted_is_split = sorted_is_split[keep_mask]

        # Go over each coordinate which has been marked as a split point.
        split_indexes = np.where(sorted_is_split)[0]
        line_split_parts = []
        start_idx = 0
        for split_idx in split_indexes:
            # Extract coordinates of the split linestring part
            split_part_coords = sorted_coords[start_idx : split_idx + 1]
            if len(split_part_coords) >= 2:  # noqa: PLR2004
                line_split_parts.append(LineString(split_part_coords))
            start_idx = split_idx

        remaining_coords = sorted_coords[start_idx:]
        if len(remaining_coords) >= 2:  # noqa: PLR2004
            line_split_parts.append(LineString(remaining_coords))

        new_geoms[line_id] = (
            MultiLineString(line_split_parts)
            if len(line_split_parts) > 1
            else line_split_parts[0]
        )

    out = lines_gdf.copy()
    out.geometry = new_geoms

    return out
