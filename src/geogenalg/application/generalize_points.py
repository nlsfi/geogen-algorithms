#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import override

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm
from geogenalg.cluster import reduce_nearby_points_by_clustering
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.displacement import displace_points
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizePoints(BaseAlgorithm):
    """Generalizes point features by reducing and displacing them.

    Reference data is not used for this algorithm.

    Output contains created cluster centroids and single points which
    were not turned into centroids. The `cluster_members_column`
    attribute lists all points that were clustered or None if the
    point is original point from input.

    Note:
        If more than two points are located close to each other, the exact
        `displace_threshold` may not be achieved. Increasing the number of iterations
        for displacing points will bring the minimum distances closer to the threshold.

    The algorithm does the following steps:
    - Reduces amount of points by clustering
    - Moves points close to each other apart

    """

    reduce_threshold: float = 10.0
    """Distance used for buffering and clustering points."""
    displace_threshold: float = 70.0
    """Minimum allowed distance between points after displacement."""
    displace_points_iterations: int = 10
    """The number of times to repeat displacement loop."""
    unique_key_column: str = "mtk_id"
    """Name of the column containing unique identifiers."""
    cluster_members_column: str = "cluster_members"
    """Name of the column that lists the points included in each cluster."""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing point geometries to generalize.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            A GeoDataFrame containing the generalized points.

        Raises:
        ------
            GeometryTypeError: If `data` contains non-point geometries.

        """
        if not check_gdf_geometry_type(data, ["Point"]):
            msg = "GeneralizePoints works only with Point geometries."
            raise GeometryTypeError(msg)

        clustered_points = reduce_nearby_points_by_clustering(
            data,
            self.reduce_threshold,
            self.unique_key_column,
            self.cluster_members_column,
        )

        return displace_points(
            clustered_points, self.displace_threshold, self.displace_points_iterations
        )
