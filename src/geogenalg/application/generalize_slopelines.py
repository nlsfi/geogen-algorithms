#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar

from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)


@supports_identity
class GeneralizeSlopeLines(BaseAlgorithm):
    """Filter slope line points based on generalized contours.

    Input should be slope line points.

    Reference data should represent generalized contour lines.

    Output is a GeoDataFrame containing only slope line points that still intersect
    the generalized contour network within the given tolerance.
    """

    tolerance: float = Field(1.0, ge=0)
    """Maximum allowed distance from a contour line."""
    reference_key: str = Field("contours")
    """Reference LineString data key for generalized contours."""

    valid_input_geometry_types: ClassVar = {"Point"}

    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={"LineString"},
        ),
    }

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        contours_gdf = reference_data[self.reference_key]

        if gdf.empty or contours_gdf.empty:
            return gdf

        contours_gdf = contours_gdf.geometry.union_all()

        # Retain only points located on or near contour lines
        mask = gdf.geometry.apply(
            lambda geom: geom.distance(contours_gdf) <= self.tolerance
        )

        return gdf[mask].copy()
