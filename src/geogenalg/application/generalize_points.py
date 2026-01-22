#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame

from geogenalg.application import supports_identity
from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.displacement import displace_points


@supports_identity
@dataclass(frozen=True)
class GeneralizePoints(GeneralizePointClustersAndPolygonsToCentroids):
    """Generalizes point features by clustering and displacing them.

    Reference data is not used for this algorithm.

    Output is generalized data, which contains the newly
    created centroids and all input features which were not turned into
    centroids. These can be distinguished by the value in the column of
    the name set by the feature_type_column attribute.

    Note:
        If more than two points are located close to each other, the exact
        `displace_threshold` may not be achieved. Increasing the number of iterations
        for displacing points will bring the minimum distances closer to the threshold.

    The algorithm does the following steps:
    - Reduces amount of points by clustering (uses parent algorithm
        `GeneralizePointClustersAndPolygonsToCentroids`)
    - Moves points close to each other apart

    """

    displace_threshold: float = 70.0
    """Minimum allowed distance between points after displacement."""
    displace_points_iterations: int = 10
    """The number of times to repeat displacement loop."""
    polygon_min_area: int = -1
    """Parameter NOT used for this algorithm (inherited from parent algorithm)."""

    # Inherited from GeneralizePointClustersAndPolygonsToCentroids, override default
    cluster_distance: float = 20.0

    valid_input_geometry_types: ClassVar = {"Point"}

    def _execute(
        self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing point geometries to generalize.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            A GeoDataFrame containing the generalized points.

        """
        clustered_points = super()._execute(data, reference_data)

        return displace_points(
            clustered_points, self.displace_threshold, self.displace_points_iterations
        )
