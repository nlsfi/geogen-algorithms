#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import override

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.utility.validation import check_gdf_geometry_type


@supports_identity
@dataclass(frozen=True)
class RemoveOverlap(BaseAlgorithm):
    """Remove overlapping areas from input polygon data.

    Identifies which areas in input data overlap with areas in mask data
    and returns a copy of input data with overlapping areas removed.

    Keeps attributes and IDs returned geometries intact. If a polygon
    is split by the mask, a MultiPolygon is created. New vertices introduced
    by cuts get Z value via linear interpolation along the affected edges.
    """

    reference_key: str = "mask"
    """Reference data key to use as a mask layer for overlap detection."""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame with (multi)polygons to be overlaid.
            reference_data: Should contain a GeoDataFrame with
                (multi)polygons with which to overlay input data.

        Returns:
        -------
            A copy of `input_data` where areas overlapping with geometries of
            `mask_data` have been removed.

        Raises:
        ------
            GeometryTypeError: If data or mask data have non-polygon geometries.
            MissingReferenceError: If reference data is not found.

        """
        if not check_gdf_geometry_type(data, ["Polygon"]):
            msg = "Remove overlap works only with Polygon geometries."
            raise GeometryTypeError(msg)

        if self.reference_key not in reference_data:
            msg = "Reference data with mask polygons is mandatory."
            raise MissingReferenceError(msg)

        mask_data = reference_data[self.reference_key]

        if not check_gdf_geometry_type(mask_data, ["Polygon"]):
            msg = "Mask data should include only Polygon geometries."
            raise GeometryTypeError(msg)

        return data.overlay(mask_data, how="difference")
