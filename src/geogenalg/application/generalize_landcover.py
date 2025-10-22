#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from geopandas import GeoDataFrame
from shapelysmooth import chaikin_smooth

from geogenalg.application import BaseAlgorithm
from geogenalg.attributes import inherit_attributes
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.selection import remove_small_holes, remove_small_polygons
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizeLandcover(BaseAlgorithm):
    """Generalize polygon data representing a land cover class.

    Input should be a polygon dataset. This algorithm can process only one
    dataset at a time, as generalization parameters need to be adjusted
    between different classes and scales to achieve a good cartographic result.

    Reference data is not used in this algorithm.

    Output is a GeoDataFrame with generalized land cover polygons.

    This algorithm performs the following steps to generalize landcover polygons:
    - Merges polygons with narrow gaps by buffering geometries
    - Removes narrow polygon parts by negatively buffering geometries
    - Simplifies polygons using `simplification_tolerance`
    - Optionally smooths polygons
    - Removes areas with area under given `area_threshold`
    - Removes holes with area under given `hole_threshold`
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

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
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
        result_gdf = remove_small_polygons(result_gdf, self.area_threshold)

        # Remove holes smaller than the hole_threshold and return
        result_gdf = remove_small_holes(result_gdf, self.hole_threshold)

        # Assign nearst z values from source gdf
        result_gdf = assign_nearest_z(data, result_gdf)

        # Inherit attributes from source gdf
        return inherit_attributes(data, result_gdf)
