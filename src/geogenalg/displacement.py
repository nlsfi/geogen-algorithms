#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from itertools import starmap

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
            [[geom.x, geom.y] for geom in result_gdf.geometry]
        )
        number_of_coordinates = len(original_coordinates)

        # Initialize displacement vectors for each point
        displacement_vectors = np.zeros((number_of_coordinates, 2))

        for i in range(number_of_coordinates):
            for j in range(number_of_coordinates):
                if i == j:
                    continue

                dx = original_coordinates[i][0] - original_coordinates[j][0]
                dy = original_coordinates[i][1] - original_coordinates[j][1]
                distance = np.hypot(dx, dy)

                # If points are too close, calculate how much to move them apart
                if distance < displace_threshold and distance != 0:
                    move_distance = (displace_threshold - distance) / 2
                    unit_vector = np.array([dx, dy]) / distance
                    move_vector = unit_vector * move_distance

                    # Accumulate the displacement for the current point
                    displacement_vectors[i] += move_vector

        # Apply displacement vectors to coordinates
        new_coordinates = original_coordinates + displacement_vectors
        result_gdf.geometry = list(starmap(Point, new_coordinates))

    return result_gdf
