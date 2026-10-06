#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar, cast

from geopandas import GeoDataFrame
from numpy import any as np_any  # Avoids shadowing Python 'any'
from numpy import column_stack, random, zeros
from pydantic import Field
from scipy.spatial import KDTree

from geogenalg.application import BaseAlgorithm, supports_identity


@supports_identity
class GeneralizeBlockFields(BaseAlgorithm):
    """Generalizes block field points.

    No reference data used.

    Performs spatial thinning to block field points.
    - Samples points in random (but deterministic) order
    - Retains a point if no points have been retained closer than
    `min_distance` so far
    """

    min_distance: float = Field(150.0, gt=0)
    """Retained points will be at least this distance from each other."""

    valid_input_geometry_types: ClassVar = {"Point"}
    requires_projected_crs: ClassVar = True

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing block field point geometries to generalize.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            A GeoDataFrame containing the generalized block field points.

        """
        return _distance_based_subsample(data, self.min_distance)


def _distance_based_subsample(
    gdf: GeoDataFrame, min_distance: float, random_seed: int = 0
) -> GeoDataFrame:
    if gdf.empty:
        return gdf.copy()

    coords = column_stack((gdf.geometry.x, gdf.geometry.y))
    n_points = len(coords)

    rng = random.default_rng(random_seed)
    sample_order = rng.permutation(n_points)

    tree = KDTree(coords)
    retained_mask = zeros(n_points, dtype=bool)
    for idx in sample_order:
        if retained_mask[idx]:
            continue

        # Find other points within min_distance
        neighbors = tree.query_ball_point(coords[idx], r=min_distance)

        # Retain this point if no other points too near to it have been retained yet
        if not np_any(retained_mask[neighbors]):
            retained_mask[idx] = True

    return cast("GeoDataFrame", gdf.iloc[retained_mask].copy().reset_index(drop=True))
