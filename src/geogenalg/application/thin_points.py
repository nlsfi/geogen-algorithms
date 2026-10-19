#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar

import numpy as np
from geopandas import GeoDataFrame
from numpy import column_stack, zeros
from pydantic import Field
from scipy.spatial import KDTree

from geogenalg.application import BaseAlgorithm, supports_identity


@supports_identity
class ThinPoints(BaseAlgorithm):
    """Generalizes points by spatial thinning.

    No reference data used.

    This algorithm can be used to reduce points for e.g. block fields. The
    generalization method can be described as spatial thinning or distance-based
    subsampling of points.

    The algorithm does the following:
    - Samples points in random (but deterministic) order
    - Retains a point if no points have been retained closer than
    `min_distance` so far
    """

    min_distance: float = Field(75.0, gt=0)
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
            data: GeoDataFrame containing point geometries to generalize.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            A GeoDataFrame containing the generalized points.

        """
        if data.empty:
            return data.copy()

        coords = column_stack((data.geometry.x, data.geometry.y))
        tree = KDTree(coords)
        retained_mask = zeros(len(coords), dtype=bool)

        # Iterate points in a stable index-based order
        for idx in data.index.argsort(kind="stable"):
            if retained_mask[idx]:
                continue

            # Find other points within min_distance
            neighbors = tree.query_ball_point(coords[idx], r=self.min_distance)

            # Retain this point if no other points too near to it have been retained yet
            if not np.any(retained_mask[neighbors]):
                retained_mask[idx] = True

        return data.iloc[retained_mask].copy()
