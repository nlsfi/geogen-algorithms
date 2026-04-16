#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.split import explode_and_hash_id


@supports_identity
@dataclass(frozen=True)
class RemoveOverlap(BaseAlgorithm):
    """Remove overlapping sections from input data.

    Identifies which sections in input data overlap with  in mask data
    and returns a copy of input data with overlapping sections removed.

    If feature in input data is wholly contained by the mask, it will be
    removed.

    Keeps attributes intact. If a feature is split by the mask, its parts will
    be turned to new features and their IDs hashed. New vertices introduced by
    cuts get Z value via linear interpolation along the affected edges.
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
    }

    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={
                "Polygon",
                "MultiPolygon",
            },
        )
    }
    requires_projected_crs: ClassVar = False

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        mask_data = reference_data[self.reference_key]

        # Overlay does not keep index here, so copy it to a separate column,
        # and set back as index later
        index_name = data.index.name

        # Create GeoDataFrame with only geometries for mask so its attributes
        # are not inherited.
        mask_gdf = GeoDataFrame(
            {data.geometry.name: mask_data.geometry},
            geometry=data.geometry.name,
            crs=data.crs,
        )

        gdf = data.copy()
        gdf["__index"] = gdf.index
        result = gdf.overlay(mask_gdf, how="difference")
        result = result.set_index("__index")
        result.index.name = index_name

        # Sometimes gdf.overlay results in multiple single geometries, sometimes
        # in multigeometries. Therefore explode and hash.
        return explode_and_hash_id(result, "removeoverlap")
