#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from itertools import chain
from typing import Any, ClassVar

from geopandas import GeoDataFrame
from pandas import Series

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.cluster import get_cluster_centroids
from geogenalg.displacement import displace_points
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
@dataclass(frozen=True)
class GeneralizePoints(BaseAlgorithm):
    """Generalizes point features by clustering and (optionally) displacing them.

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
    - Reduces amount of points by clustering
    - If specified, moves points close to each other apart

    """

    displace: bool = False
    """Whether points should be displaced in addition to clustering."""
    displace_threshold: float = 70.0
    """Minimum allowed distance between points after displacement."""
    displace_points_iterations: int = 10
    """The number of times to repeat displacement loop."""
    cluster_distance: float = 20.0
    """Points within this distance of each other will be clustered."""
    aggregation_functions: dict[str, Callable[[Series], Any] | str] | None = None
    """Dictionary containing keys corresponding to a column in a GeoDataFrame
    and a function which will aggregate the column's values when creating
    centroid from multiple points. If the function is given as a string, it
    must correspond to Pandas's aggregation function names. If no function is
    given "first" will be used by default."""
    is_cluster_column: str | None = "is_cluster"
    """Name of column indicating whether a point in the output is a cluster or
    an unchanged point."""

    valid_input_geometry_types: ClassVar = {"Point"}
    required_projected_crs: ClassVar = False

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
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
        index_name = data.index.name

        gdf = data.copy()
        clusters = get_cluster_centroids(
            gdf,
            self.cluster_distance,
            aggregation_functions=self.aggregation_functions,
        )

        clusters = clusters.set_index(
            clusters["old_ids"].apply(
                lambda ids: sha256(
                    b"pointcluster" + "".join(sorted(ids)).encode()
                ).hexdigest(),
            ),
        )
        if self.is_cluster_column is not None:
            clusters[self.is_cluster_column] = True
        ids_to_remove = list(chain.from_iterable(clusters["old_ids"]))
        clusters = clusters.drop("old_ids", axis=1)

        gdf = gdf.loc[~gdf.index.isin(ids_to_remove)]

        if self.is_cluster_column is not None:
            gdf[self.is_cluster_column] = False

        gdf = combine_gdfs([clusters, gdf])

        if self.displace:
            gdf = displace_points(
                gdf,
                self.displace_threshold,
                self.displace_points_iterations,
            )

        gdf.index.name = index_name

        return gdf
