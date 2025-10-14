#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import geopandas as gpd

from geogenalg import cluster, displacement
from geogenalg.application import BaseAlgorithm


@dataclass(frozen=True)
class GeneralizePoints(BaseAlgorithm):
    """Generalizes point features by reducing and displacing them.

    Note:
        If more than two points are located close to each other, the exact
        `displace_threshold` may not be achieved. Increasing the number of iterations
        for displacing points will bring the minimum distances closer to the threshold.

    """

    reduce_threshold: float
    """Distance used for buffering and clustering points."""
    displace_threshold: float
    """Minimum allowed distance between points after displacement."""
    displace_points_iterations: int
    """The number of times to repeat displacement loop."""
    unique_key_column: str
    """Name of the column containing unique identifiers."""
    cluster_members_column: str
    """Name of the column that lists the points included in each cluster."""

    def execute(
        self,
        data: gpd.GeoDataFrame,
        reference_data: dict[str, gpd.GeoDataFrame] | None = None,
    ) -> gpd.GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing point geometries to generalize.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            A GeoDataFrame containing the generalized points.

        """
        if reference_data is None:
            reference_data = {}
        clustered_points = cluster.reduce_nearby_points_by_clustering(
            data,
            self.reduce_threshold,
            self.unique_key_column,
            self.cluster_members_column,
        )

        return displacement.displace_points(
            clustered_points, self.displace_threshold, self.displace_points_iterations
        )
