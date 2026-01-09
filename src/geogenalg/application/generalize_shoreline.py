#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass
from itertools import combinations
from typing import override

from geopandas import GeoDataFrame, overlay

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.utility.validation import check_gdf_geometry_type

logger = logging.getLogger(__name__)


@supports_identity
@dataclass(frozen=True)
class GeneralizeShoreline(BaseAlgorithm):
    """Extracts new shorelines from generalized water areas.

    Converts generalized water areas to linestrings and divides them into features,
    matching the original shoreline linestrings and their attributes.

    Requires a polygon dataset (generalized water areas) passed as reference
    data.
    """

    buffer_distance: float = 7.5
    """Buffer distance used to identify shoreline features. A suitable value is
    well above the maximum distance which the generalize water areas algorithm
    is expected shift geometries. This is likely at least the buffer distance
    used for exaggeration."""
    reference_key: str = "areas"
    "Reference data key to use as the source of the generalized shoreline."

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Raises:
            GeometryTypeError: If input GeoDataFrames have incorrect geometry types.
            MissingReferenceError: If reference data is not found.

        Returns:
            New shoreline extracted from generalized water areas.

        """
        if not check_gdf_geometry_type(data, ["LineString"]):
            msg = "Input data must contain only LineStrings."
            raise GeometryTypeError(msg)

        if self.reference_key in reference_data:
            water_areas_gdf = reference_data[self.reference_key]

            if not check_gdf_geometry_type(
                water_areas_gdf, ["Polygon", "MultiPolygon"]
            ):
                msg = "Reference data must contain only (Multi)Polygons."
                raise GeometryTypeError(msg)
        else:
            msg = "Reference data is mandatory."
            raise MissingReferenceError(msg)

        new_shoreline = water_areas_gdf.geometry.boundary.to_frame()
        new_shoreline = new_shoreline.explode()

        buffered_old_shoreline = data.copy()
        buffered_old_shoreline.geometry = buffered_old_shoreline.geometry.buffer(
            self.buffer_distance
        )

        # Remove overlap from intersecting buffered polygons to stop new
        # shorelines overlapping each other
        buffered_geom = buffered_old_shoreline.geometry
        for point_1_idx, point_2_idx in combinations(buffered_geom.index, 2):
            if buffered_geom.loc[point_1_idx].intersects(
                buffered_geom.loc[point_2_idx]
            ):
                buffered_geom.loc[point_2_idx] -= buffered_geom.loc[point_1_idx]

        # GeoDataFrame index is not inherited with the identity overlay, but it
        # needs to remain unchanged for the new shoreline, so copy index as a
        # column to set as index later.
        buffered_old_shoreline["__duplicated_index"] = buffered_old_shoreline.index

        # There may be small artifact features left over from the identity
        # overlay. These do not inherit attributes, so set an always not null
        # column beforehand so artifacts can be identified from having this be
        # null.
        buffered_old_shoreline["__identification_successful"] = "true"

        new_shoreline = overlay(new_shoreline, buffered_old_shoreline, how="identity")
        new_shoreline = new_shoreline.loc[
            ~new_shoreline["__identification_successful"].isna()
        ]
        new_shoreline = new_shoreline.drop("__identification_successful", axis=1)
        new_shoreline = new_shoreline.set_index("__duplicated_index")
        new_shoreline.index.name = data.index.name

        new_shoreline.geometry = new_shoreline.geometry.remove_repeated_points()
        new_shoreline.geometry = new_shoreline.geometry.line_merge()

        new_shoreline.index.name = data.index.name

        if not check_gdf_geometry_type(new_shoreline, ["LineString"]):
            logger.warning("WARNING: not all new features are LineStrings!")

        return new_shoreline
