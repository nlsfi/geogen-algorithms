#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from geogenalg.core.exceptions import GeometryTypeError


def displace_points(
    input_gdf: gpd.GeoDataFrame,
    displace_threshold: float,
    iterations: int,
) -> gpd.GeoDataFrame:
    """Move points apart based on the given threshold.

    Note:
        If more than two points are located close to each other, the exact
        displace_threshold may not be achieved. Increasing the number of iterations
        will bring the minimum distances closer to the threshold.

        The Z coordinates of the original points are preserved, but they are not taken
        into account during displacement. If a point does not have a Z coordinate, it is
        set to 0.0.

    Args:
    ----
        input_gdf: A GeoDataFrame with point geometries
        displace_threshold: The minimum allowed distance between points
        iterations: The number of times to repeat displacement loop

    Returns:
    -------
        A GeoDataFrame with displaced points.

    Raises:
    ------
        GeometryTypeError: If the input GeoDataFrame contains other than point features

    """
    if input_gdf.geometry.type.unique().tolist() != ["Point"]:
        msg = "displace_points only supports Point geometries."
        raise GeometryTypeError(msg)

    result_gdf = input_gdf.copy()

    for _ in range(iterations):
        original_coordinates = np.array(
            [
                [geom.x, geom.y, geom.z if geom.has_z else 0.0]
                for geom in result_gdf.geometry
            ]
        )
        number_of_coordinates = len(original_coordinates)

        # Initialize displacement vectors for each point
        # Only work with XY for displacement
        xy_coordinates = original_coordinates[:, :2]
        displacement_vectors = np.zeros_like(xy_coordinates)

        for i in range(number_of_coordinates):
            for j in range(number_of_coordinates):
                if i == j:
                    continue

                dx, dy = xy_coordinates[i] - xy_coordinates[j]
                distance = np.hypot(dx, dy)

                # If points are too close, calculate how much to move them apart
                if 0 < distance < displace_threshold:
                    move_distance = (displace_threshold - distance) / 2
                    unit_vector = np.array([dx, dy]) / distance
                    move_vector = unit_vector * move_distance

                    # Accumulate the displacement for the current point
                    displacement_vectors[i] += move_vector

        new_xy_coords = xy_coordinates + displacement_vectors

        # Rebuild points with preserved Z
        new_geoms = [
            Point(x, y, z)
            for (x, y), (_, _, z) in zip(
                new_xy_coords, original_coordinates, strict=False
            )
        ]
        result_gdf.geometry = new_geoms

    return result_gdf
