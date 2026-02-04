#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame
from shapelysmooth import chaikin_smooth

from geogenalg.application import BaseAlgorithm
from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.selection import remove_small_holes, remove_small_polygons


@dataclass(frozen=True)
class GeneralizeLandcover(BaseAlgorithm):
    """Generalize polygon data representing a land cover class.

    Input should be a polygon dataset. This algorithm can process only one
    dataset at a time, as generalization parameters need to be adjusted
    between different classes and scales to achieve a good cartographic result.

    Reference data is not used in this algorithm.

    Output is a GeoDataFrame with generalized land cover polygons.

    The algorithm does the following steps:
    - Merges polygons with narrow gaps by buffering geometries
    - Removes narrow polygon parts by negatively buffering geometries
    - Simplifies polygons using `simplification_tolerance`
    - Optionally smooths polygons
    - Removes areas with area under given `area_threshold`
    - Removes holes with area under given `hole_threshold`
    """

    positive_buffer: float = 10.0
    """Buffer to close narrow gaps."""
    negative_buffer: float = -10.0
    """Negative buffer to remove narrow parts."""
    simplification_tolerance: float = 5.0
    """Tolerance used for geometry simplification."""
    area_threshold: float = 2500.0
    """Minimum polygon area to retain."""
    hole_threshold: float = 2500.0
    """Minimum area of holes to retain."""
    smoothing: bool = False
    """If True, polygons will be smoothed."""
    group_by: str | list[str] | None = None
    """Column(s) whose values define the groups to be dissolved."""

    valid_input_geometry_types: ClassVar = {"Polygon"}

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        result_gdf = data.copy()

        def _buffer(gdf: GeoDataFrame, distance: float) -> None:
            gdf.geometry = gdf.geometry.buffer(
                distance, cap_style="square", join_style="bevel"
            )

        # Create a positive_buffer to close narrow gaps between polygons
        _buffer(result_gdf, self.positive_buffer)
        result_gdf = result_gdf.dissolve(as_index=False, by=self.group_by)

        # Create a negative buffer to restore polygons back to their original size and
        # remove narrow polygon parts
        negative_buffer = -abs(self.negative_buffer)
        total_negative_buffer = negative_buffer - self.positive_buffer
        _buffer(result_gdf, total_negative_buffer)

        # Restore polygons to their original size with a positive buffer
        _buffer(result_gdf, -negative_buffer)

        result_gdf = result_gdf.dissolve(as_index=False, by=self.group_by)
        result_gdf = result_gdf.explode(index_parts=False)
        result_gdf = inherit_attributes_from_largest(data, result_gdf, "old_ids")

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
        return assign_nearest_z(data, result_gdf)

        # TODO: Ensure polygons do not overlap and there are no narrow gaps between them
        # (e.g. using GEOS CoverageCleaner)
