#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import ClassVar, override

from geopandas.geodataframe import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.application.generalize_building_areas_by_geometry import (
    GeneralizeBuildingAreasByGeometry,
)
from geogenalg.application.generalize_building_areas_by_parcel import (
    GeneralizeBuildingAreasByParcel,
)
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.identity import hash_index_from_geometry
from geogenalg.selection import remove_small_holes
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreas(BaseAlgorithm):
    """Generalize polygons representing buildings.

    This algorithm utilizes two other algorithms,
    GeneralizeBuildingAreasByGeometry and GeneralizeBuildingAreasByParcel. It
    runs them both, combines the results and performs further post-processing.

    Output contains generalized building-area polygons.

    """

    parcel_building_size_threshold: float = 4000.0
    """For parcel algorithm: to determine large (non-residential buildings."""
    parcel_coverage_threshold: float = 5
    """For parcel algorithm: To determine parcels with "high" building density."""
    parcel_buffer_distance: float = 20
    """For parcel algorithm: Buffer distance used to merge parcels."""
    parcel_building_type_column: str = "building_function_id"
    """For parcel algorithm: Column containing attributes describing the
    intended use of a building."""
    parcel_building_type_to_select: int = 1
    """For parcel algorithm: Used to select type of building, residential for
    example."""
    geometry_simplify_tolerance_single_buildings: float = 10.0
    """For geometry algorithm: Tolerance for building simplification
    (Douglas-Peucker)."""
    geometry_simplify_tolerance_building_areas: float = 10.0
    """For geometry algorithm: Tolerance for building simplification
    (Douglas-Peucker)."""
    geometry_network_buffer_distance: float = 10.0
    """How large a section will be removed close to roads from building areas."""
    threshold_building_area_hole: float = 500.0
    """Minimum size for holes inside newly generated building areas."""
    threshold_building_area_far: float = 20000.0
    """Minimum size for newly generated building areas far from other areas."""
    threshold_building_area_near: float = 4000.0
    """Minimum size for newly generated building areas near other areas."""
    near_area_distance: float = 50.0
    """Distance for areas to be considered as near each other, affecting
    threshold used to filter small areas."""
    gap_removal_buffer_distance: float = 5
    """Distance used to remove small gaps left after combining parcel and
    geometry algorithm results."""
    reference_key_parcels: str = "parcels"
    """Reference data key for road data."""
    reference_key_roads: str = "roads"
    """Reference data key for parcel data."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    valid_reference_geometry_types: ClassVar = {"Polygon", "LineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if self.reference_key_parcels not in reference_data:
            raise MissingReferenceError

        parcels_result = GeneralizeBuildingAreasByParcel(
            building_area_threshold=self.parcel_building_size_threshold,
            coverage_threshold=self.parcel_coverage_threshold,
            buffer_distance=self.parcel_buffer_distance,
            building_type_column=self.parcel_building_type_column,
            building_type_to_select=self.parcel_building_type_to_select,
            reference_key=self.reference_key_parcels,
        ).execute(
            data,
            reference_data={
                self.reference_key_parcels: reference_data[self.reference_key_parcels],
            },
        )

        geometry_result = GeneralizeBuildingAreasByGeometry(
            threshold_building_area=self.threshold_building_area_near,
            threshold_building_area_hole=self.threshold_building_area_hole,
            simplify_tolerance_building_areas=self.geometry_simplify_tolerance_building_areas,
            simplify_tolerance_single_buildings=self.geometry_simplify_tolerance_single_buildings,
            network_buffer_distance=self.geometry_network_buffer_distance,
            reference_key=self.reference_key_roads,
        ).execute(
            data,
            reference_data={}
            if reference_data.get(self.reference_key_roads) is None
            else {self.reference_key_roads: reference_data[self.reference_key_roads]},
        )

        merged = combine_gdfs(
            [
                parcels_result,
                geometry_result,
            ]
        )

        merged = merged.dissolve()
        merged = merged.explode(as_index=False)
        merged = remove_small_holes(merged, self.threshold_building_area_hole)
        merged = merged.sjoin_nearest(
            merged.loc[merged.geometry.area > self.threshold_building_area_far],
            distance_col="is_near",
            exclusive=True,
        ).drop("index_right", axis=1)
        merged = merged.drop_duplicates()
        merged["distance_to_nearest"] = merged.is_near
        merged.is_near = merged.is_near <= self.near_area_distance
        merged.is_far = ~merged.is_near

        merged = merged.loc[
            (
                merged.is_near
                & (merged.geometry.area > self.threshold_building_area_near)
            )
            | (
                merged.is_far
                & (merged.geometry.area > self.threshold_building_area_far)
            )
        ]

        merged.geometry = merged.geometry.buffer(
            self.gap_removal_buffer_distance,
            join_style="bevel",
        ).buffer(
            -self.gap_removal_buffer_distance,
            join_style="bevel",
        )

        merged = assign_nearest_z(data, merged)
        return hash_index_from_geometry(merged, "buildingareas")
