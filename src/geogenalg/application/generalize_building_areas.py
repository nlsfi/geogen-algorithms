#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from pathlib import Path
from typing import override

import geopandas as gpd
import pandas as pd
from geopandas import GeoDataFrame

from geogenalg import analyze, merge, selection
from geogenalg.application import BaseAlgorithm, remove_overlap, supports_identity
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.core.geometry import remove_close_line_segments, remove_short_lines
from geogenalg.utility.validation import check_gdf_geometry_type

# from shapely.geometry import Polygon, MultiPolygon


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreas(BaseAlgorithm):
    """Generalize polygons representing buildings and property parcels.

    Output contains generalized building-area polygons.

    The algorithm does the following steps:
    - Filters out large non-residential buildings
    - Calculates building coverage per parcel
    - Selects parcels with high coverage
    - Merges nearby parcels from the selection

    Note:
        This version (Feb 2026) is still a draft. The resulting building areas
        are not yet fully aligned with the intended final output, and testing
        has so far been limited to data from Helsinki. Future tasks will refine
        and complete the algorithm.

    """

    building_area_threshold: float = 4000
    """to determine large (non-residential) buildings"""
    coverage_threshold: float = 5
    """to determine parcels with "high" building density"""
    buffer_distance: float = 20
    """buffer_distance in meters"""
    attribute_for_building_type: str = "building_function_id"
    """building use type attribute, in MTK buildings"""
    building_type_to_select: int = 1
    """used to select type of building, residential for example"""
    attribute_for_building_coverage = "building_coverage"
    """standard name for new attribute, no need to change"""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data:
                A GeoDataFrame of building polygons to be generalized.
            reference_data:
                Auxiliary GeoDataFrame of proporty parcel polygons.

        Returns:
        -------
            GeoDataFrame containing the generalized building-area polygons.

        Raises:
        ------
            GeometryTypeError:
                If `data` contains non-polygon geometries or any polygon layer
                in `reference_data` contains unsupported geometry types.
            KeyError:
                If required attributes for computing building coverage or
                parcel selection are missing from `data`.

        """
        result_gdf = data.copy()
        parcels_gdf = reference_data["parcels"]
        """landcover_gdf = reference_data["landcovers"]"""

        # 1 - Remove large non-residential buildings

        # 1.1 - Select non-residential buildings only
        non_residential_buildings = result_gdf[
            result_gdf[self.attribute_for_building_type] != self.building_type_to_select
        ].copy()

        # 1.2 - Remove large buildings from the non-residential buildings
        filtered_non_residential = selection.remove_large_polygons(
            input_gdf=non_residential_buildings,
            area_threshold=self.building_area_threshold,
        )

        # 1.3 - Combine residential buildings with filtered non-residential buildings
        filtered_buildings = pd.concat(
            [
                result_gdf[
                    result_gdf[self.attribute_for_building_type]
                    == self.building_type_to_select
                ],
                filtered_non_residential,
            ],
            ignore_index=True,
        )

        # 2 - Calculate building coverage per parcel

        parcels_with_coverage = analyze.calculate_coverage(
            overlay_features=filtered_buildings,
            base_features=parcels_gdf,
            coverage_attribute=self.attribute_for_building_coverage,
        )

        # 3 - Select parcels with high building coverage

        high_coverage_mask = (
            parcels_with_coverage[self.attribute_for_building_coverage]
            > self.coverage_threshold
        )

        high_coverage_parcels = parcels_with_coverage[high_coverage_mask].copy()

        # 4 - Buffer and merge high-coverage parcels into building areas
        building_areas = merge.buffer_and_merge_polygons(
            high_coverage_parcels,
            buffer_distance=self.buffer_distance,
        )
        final_building_areas = building_areas

        """
        # Additional steps 5 and 6 were considered unnecessary in this segment
        # Similar steps can be excecuted elsewhere in the broader generalization process

        # 5 - Dissolve land cover layers into one polygon layer
        if not landcover_gdf:
            landcover_union = None
        else:
            landcover_union = merge.dissolve_polygon_layers(input_gdfs=landcover_gdf)

        # 6 Remove overlaps between building areas and landcover layers
        if landcover_union is not None:
            final_building_areas = remove_overlap(
                target_gdf=building_areas,
                cut_gdf=landcover_union,
            )
        else:
            final_building_areas = building_areas
        """

        return final_building_areas
