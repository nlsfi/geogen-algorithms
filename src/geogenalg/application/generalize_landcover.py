#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import geopandas as gpd
from shapelysmooth import chaikin_smooth

from geogenalg import attributes, selection
from geogenalg.application import BaseAlgorithm


@dataclass(frozen=True)
class GeneralizeLandcover(BaseAlgorithm):
    """Generalize the polygon layer representing a land cover class.

    This algorithm can process only one polygon layer at a time, as generalization
    parameters need to be adjusted between different classes and scales to achieve
    a good cartographic result.
    """

    buffer_constant: float
    """Constant used for buffering polygons."""
    simplification_tolerance: float
    """Tolerance used for geometry simplification."""
    area_threshold: float
    """Minimum polygon area to retain."""
    hole_threshold: float
    """Minimum area of holes to retain."""
    smoothing: bool
    """If True, polygons will be smoothed."""

    def execute(
        self,
        data: gpd.GeoDataFrame,
        reference_data: dict[str, gpd.GeoDataFrame] | None = None,
    ) -> gpd.GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: A GeoDataFrame containing the land cover polygons.
            reference_data: Not used.

        Returns:
        -------
            Generalized land cover polygons.

        """
        if reference_data is None:
            reference_data = {}
        result_gdf = data.copy()

        # Create a buffer of size buffer_constant to close narrow gaps between polygons
        result_gdf.geometry = result_gdf.geometry.buffer(
            self.buffer_constant, cap_style="square", join_style="bevel"
        )
        result_gdf = result_gdf.dissolve(as_index=False)

        # Create a double negative buffer to remove narrow polygon parts
        result_gdf.geometry = result_gdf.geometry.buffer(
            -2 * self.buffer_constant, cap_style="square", join_style="bevel"
        )

        # Restore polygons to their original size with a positive buffer
        result_gdf.geometry = result_gdf.geometry.buffer(
            self.buffer_constant, cap_style="square", join_style="bevel"
        )

        result_gdf = result_gdf.dissolve(as_index=False)
        result_gdf = result_gdf.explode(index_parts=False)

        # Simplify the polygons
        result_gdf.geometry = result_gdf.geometry.simplify(
            self.simplification_tolerance, preserve_topology=True
        )

        # Smooth the polygons if smoothing is wanted
        if self.smoothing:
            geometries = list(result_gdf.geometry)
            smoothed_geometries = [chaikin_smooth(geom) for geom in geometries]
            result_gdf.geometry = smoothed_geometries

        # Remove polygons smaller than the area_threshold
        result_gdf = selection.remove_small_polygons(result_gdf, self.area_threshold)

        # Remove holes smaller than the hole_threshold and return
        result_gdf = selection.remove_small_holes(result_gdf, self.hole_threshold)

        return attributes.inherit_attributes(data, result_gdf)
