#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.identity import hash_duplicate_indexes


@supports_identity
@dataclass(frozen=True)
class KeepIntersection(BaseAlgorithm):
    """Keep intersecting areas from input polygon data.

    Identifies which areas in input data intersect with areas in mask data
    and returns a copy of input data with only intersecting areas remaining.

    If feature in input data has zero intersection with mask, it will be
    removed.

    Keeps attributes and IDs returned geometries intact. If a polygon
    is split by the mask, a MultiPolygon is created. New vertices introduced
    by cuts get Z value via linear interpolation along the affected edges.
    """

    reference_key: str = "mask"
    """Reference data key to use as a mask layer for overlap detection."""

    valid_input_geometry_types: ClassVar = {
        "LineString",
        "Polygon",
        "MultiLineString",
        "MultiPolygon",
        "Point",
        "MultiPoint",
        "LinearRing",
        "GeometryCollection",
    }
    valid_reference_geometry_types: ClassVar = {"Polygon", "MultiPolygon"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame to be overlaid.
            reference_data: Should contain a GeoDataFrame with
                (multi)polygons with which to overlay input data.

        Returns:
        -------
            A copy of `input_data` where areas outside geometries of
            `mask_data` have been removed.

        Raises:
        ------
            MissingReferenceError: If reference data is not found.

        """
        if self.reference_key not in reference_data:
            msg = "Reference data with mask polygons is mandatory."
            raise MissingReferenceError(msg)

        mask_data = reference_data[self.reference_key]

        # Overlay does not keep index here, so copy it to a separate column,
        # and set back as index later
        index_name = data.index.name

        # Create GeoDataFrame with only geometries for mask so its attributes
        # are not inherited.
        mask_gdf = GeoDataFrame(mask_data.geometry)

        gdf = data.copy()
        gdf["__index"] = gdf.index
        result = gdf.overlay(mask_gdf, how="intersection")
        result = result.set_index("__index")
        result.index.name = index_name

        return hash_duplicate_indexes(result, "keepintersection")
