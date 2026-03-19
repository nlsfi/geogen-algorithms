#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.analyze import calculate_coverage
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.identity import hash_index_from_geometry
from geogenalg.merge import buffer_and_merge_polygons
from geogenalg.selection import remove_large_polygons
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreasByParcel(BaseAlgorithm):
    """Generalize polygons representing buildings.

    Generalization is done according to property parcels the buildings reside in.

    Output contains generalized building-area polygons.

    The algorithm does the following steps:
    - Filters out large non-residential buildings
    - Calculates building coverage per parcel
    - Selects parcels with high coverage
    - Merges nearby parcels from the selection

    """

    building_area_threshold: float = 4000
    """To determine large (non-residential) buildings."""
    coverage_threshold: float = 5
    """To determine parcels with "high" building density."""
    buffer_distance: float = 20
    """Buffer distance used to merge parcels."""
    building_type_column: str = "building_function_id"
    """Column containing attributes describing the intended use of a building."""
    building_type_to_select: int = 1
    """Used to select type of building, residential for example."""
    building_coverage_column: str = "building_coverage"
    """Name of column containing coverage percentage values."""
    reference_key: str = "parcels"
    """Reference data key for parcel data."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    valid_reference_geometry_types: ClassVar = {"Polygon"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if self.reference_key not in reference_data:
            raise MissingReferenceError
        result_gdf = data.copy()
        parcels_gdf = reference_data[self.reference_key]
        """landcover_gdf = reference_data["landcovers"]"""

        # 1 - Remove large non-residential buildings

        # 1.1 - Select non-residential buildings only
        non_residential_buildings = result_gdf[
            result_gdf[self.building_type_column] != self.building_type_to_select
        ].copy()

        # 1.2 - Remove large buildings from the non-residential buildings
        filtered_non_residential = remove_large_polygons(
            input_gdf=non_residential_buildings,
            area_threshold=self.building_area_threshold,
        )

        # 1.3 - Combine residential buildings with filtered non-residential buildings
        filtered_buildings = combine_gdfs(
            [
                result_gdf[
                    result_gdf[self.building_type_column]
                    == self.building_type_to_select
                ],
                filtered_non_residential,
            ],
            ignore_index=True,
        )

        # 2 - Calculate building coverage per parcel

        parcels_with_coverage = calculate_coverage(
            overlay_features=filtered_buildings,
            base_features=parcels_gdf,
            coverage_attribute=self.building_coverage_column,
        )

        # 3 - Select parcels with high building coverage

        high_coverage_mask = (
            parcels_with_coverage[self.building_coverage_column]
            > self.coverage_threshold
        )

        high_coverage_parcels = parcels_with_coverage[high_coverage_mask].copy()

        # 4 - Buffer and merge high-coverage parcels into building areas

        gdf = buffer_and_merge_polygons(
            high_coverage_parcels,
            buffer_distance=self.buffer_distance,
        ).explode(as_index=False)

        gdf = hash_index_from_geometry(gdf, "buildingareasparcel")
        return assign_nearest_z(data, gdf)
