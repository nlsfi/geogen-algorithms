#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame
from shapely import Point
from shapely.geometry import Polygon

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.geometry import mean_z


@supports_identity
@dataclass(frozen=True)
class GeneralizePolygonsToPoints(BaseAlgorithm):
    """Reduces polygons below an area threshold to a point on surface.

    Reference data is not used for this algorithm.

    Point is formed as a point on surface.

    Output contains both unchanged polygon and changed point features.
    """

    polygon_min_area: float = 4000.0
    """Polygons with an area smaller than this will be turned into a point."""

    valid_input_geometry_types: ClassVar = {"Polygon"}

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],  # noqa: ARG002
    ) -> GeoDataFrame:
        gdf = data.copy()

        def _to_point(geom: Polygon) -> Point | Polygon:
            if geom.area >= self.polygon_min_area:
                return geom

            point = geom.point_on_surface()
            if geom.has_z:
                z = mean_z(geom)
                point = Point(point.x, point.y, z)

            return point

        gdf.geometry = gdf.geometry.apply(_to_point)

        return gdf
