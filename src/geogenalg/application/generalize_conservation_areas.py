#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame

from geogenalg.analyze import calculate_edge_adjacency
from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.identity import hash_index_from_old_ids
from geogenalg.merge import dissolve_and_inherit_attributes
from geogenalg.utility.dataframe_processing import combine_gdfs

WATER_PROXIMITY_DISTANCE = 20
WATER_RATIO_THRESHOLD = 0.5
WATER_RATIO_COLUMN = "water_ratio"


@supports_identity
@dataclass(frozen=True)
class GeneralizeConservationAreas(BaseAlgorithm):
    """Generalize polygon data representing conservation or wilderness areas.

    Input should be a polygon dataset.

    Reference data should represent water areas (sea and lake parts).

    Output is a GeoDataFrame with generalized polygons.

    The algorithm does the following steps:
        1. Classify input areas into coastal and inland groups based on
           adjacency to water.
        2. Generalize coastal areas using coastal-specific parameters.
        3. Generalize inland areas using inland-specific parameters.
        4. Merge the two groups back together and dissolve within the feature class.
    """

    positive_buffer_coastal_areas: float = 25.0
    """Positive buffer to close narrow gaps in coastal areas."""
    negative_buffer_coastal_areas: float = -5.0
    """Negative buffer to remove narrow parts in coastal areas."""
    positive_buffer_inland_areas: float = 5.0
    """Positive buffer to close narrow gaps in inland areas."""
    negative_buffer_inland_areas: float = -5.0
    """Negative buffer to remove narrow parts in inland areas."""
    simplification_tolerance: float = 3.0
    """Tolerance used for geometry simplification."""
    area_threshold: float = 1000.0
    """Minimum polygon area to retain."""
    hole_threshold: float = 2000.0
    """Minimum area of holes to retain."""
    smoothing: bool = False
    """If True, polygons will be smoothed."""
    group_by: frozenset[str] = frozenset()
    """Column(s) whose values define the groups to be dissolved."""
    reference_key: str = "water_areas"
    """Reference data key for water area data."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={
                "Polygon",
                "MultiPolygon",
            },
        ),
    }

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        reference_gdf = reference_data[self.reference_key]

        # Compute water adjacency ratio for each conservation area
        gdf = calculate_edge_adjacency(
            gdf,
            reference_gdf,
            WATER_PROXIMITY_DISTANCE,
            WATER_RATIO_COLUMN,
        )

        # Split conservation areas into coastal and inland groups
        coastal_areas_mask = gdf[WATER_RATIO_COLUMN] > WATER_RATIO_THRESHOLD
        coastal_areas_gdf = gdf[coastal_areas_mask].copy()
        inland_areas_gdf = gdf[~coastal_areas_mask].copy()

        # Generalize coastal polygons
        coastal_generalize = GeneralizeLandcover(
            positive_buffer=self.positive_buffer_coastal_areas,
            negative_buffer=self.negative_buffer_coastal_areas,
            simplification_tolerance=self.simplification_tolerance,
            area_threshold=self.area_threshold,
            hole_threshold=self.hole_threshold,
            smoothing=self.smoothing,
            group_by=self.group_by,
        )
        coastal_areas_gdf = coastal_generalize.execute(data=coastal_areas_gdf)

        # Generalize inland polygons
        inland_generalize = GeneralizeLandcover(
            positive_buffer=self.positive_buffer_inland_areas,
            negative_buffer=self.negative_buffer_inland_areas,
            simplification_tolerance=self.simplification_tolerance,
            area_threshold=self.area_threshold,
            hole_threshold=self.hole_threshold,
            smoothing=self.smoothing,
            group_by=self.group_by,
        )
        inland_areas_gdf = inland_generalize.execute(data=inland_areas_gdf)

        # Merge coastal and inland areas and allow dissolving within the feature class
        result_gdf = combine_gdfs([inland_areas_gdf, coastal_areas_gdf])
        result_gdf = result_gdf.drop(columns=[WATER_RATIO_COLUMN], errors="ignore")

        result_gdf = dissolve_and_inherit_attributes(
            input_gdf=result_gdf,
            by_column=None if not self.group_by else list(self.group_by),
        )

        return hash_index_from_old_ids(result_gdf, "landcover", "old_ids")
