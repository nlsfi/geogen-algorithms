#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import override

from geopandas import GeoDataFrame
from shapelysmooth import chaikin_smooth

from geogenalg import attributes, selection
from geogenalg.application import BaseAlgorithm
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type


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

    @override
    def execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: A GeoDataFrame containing the land cover polygons.
            reference_data: Not used for this algorithm.

        Returns:
        -------
            Generalized land cover polygons.

        Raises:
        ------
            GeometryTypeError: If data contains something other than polygon
            geometries.

        """
        if not check_gdf_geometry_type(data, ["Polygon"]):
            msg = "GeneralizeLandcover works only with Polygon geometries."
            raise GeometryTypeError(msg)

        result_gdf = data.copy()

        def _buffer(gdf: GeoDataFrame, distance: float) -> None:
            gdf.geometry = gdf.geometry.buffer(
                distance, cap_style="square", join_style="bevel"
            )

        # Create a buffer of size buffer_constant to close narrow gaps between polygons
        _buffer(result_gdf, self.buffer_constant)
        result_gdf = result_gdf.dissolve(as_index=False)

        # Create a double negative buffer to remove narrow polygon parts
        _buffer(result_gdf, -2 * self.buffer_constant)

        # Restore polygons to their original size with a positive buffer
        _buffer(result_gdf, self.buffer_constant)

        result_gdf = result_gdf.dissolve(as_index=False)
        result_gdf = result_gdf.explode(index_parts=False)

        # Simplify the polygons
        result_gdf.geometry = result_gdf.geometry.simplify(
            self.simplification_tolerance, preserve_topology=True
        )

        # Smooth the polygons if smoothing is wanted
        if self.smoothing:
            result_gdf.geometry = result_gdf.geometry.apply(chaikin_smooth)

        # Remove polygons smaller than the area_threshold
        result_gdf = selection.remove_small_polygons(result_gdf, self.area_threshold)

        # Remove holes smaller than the hole_threshold and return
        result_gdf = selection.remove_small_holes(result_gdf, self.hole_threshold)

        # Inherit attributes from source gdf
        return attributes.inherit_attributes(data, result_gdf)
