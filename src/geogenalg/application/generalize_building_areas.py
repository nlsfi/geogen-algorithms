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
from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.identity import hash_index_from_geometry
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreas(BaseAlgorithm):
    """Generalize polygons representing buildings.

    This algorithm utilizes two other algorithms,
    GeneralizeBuildingAreasByGeometry and GeneralizeBuildingAreasByParcel. It
    runs them both, combines the results and performs further post-processing.

    Output contains generalized building-area polygons.

    The algorithm does the following steps:
    - Runs the parcel-based building areas algorithm
    - Runs the geometry-based building areas algorithm
    - Combines the results
    - Removes sections inside buffered road reference data (if provided)
    - Post-processes building areas using the GeneralizeLandcover algorithm


    """

    # parameters for GeneralizeBuildingAreasByParcel
    parcel_building_size_threshold: float = 4000.0
    """For parcel algorithm: to determine large (non-residential) buildings."""
    parcel_coverage_threshold: float = 5
    """For parcel algorithm: to determine parcels with "high" building density."""
    parcel_buffer_distance: float = 20
    """For parcel algorithm: buffer distance used to merge parcels."""
    parcel_building_type_column: str = "building_function_id"
    """For parcel algorithm: column containing attributes describing the
    intended use of a building."""
    parcel_building_type_to_select: int = 1
    """For parcel algorithm: used to select type of building, residential for
    example."""

    # parameters for GeneralizeBuildingAreasByGeometry
    geometry_buildings_simplify_tolerance: float = 10.0
    """For geometry algorithm: Tolerance for building simplification
    (Douglas-Peucker)."""
    geometry_boffet_area_buffer: float = 10.0
    """For geometry algorithm: The buffer size used to merge buildings that are
    close from each other."""
    geometry_boffet_area_erosion: float = 10.0
    """For geometry algorithm: The erosion size to avoid the building area to
    expand too far from the buildings located on the edge."""

    # parameters for filtering building areas
    near_area_distance: float = 50.0
    """Distance for building areas to be considered as near each other, affecting
    threshold used to filter small areas."""
    threshold_building_area_far: float = 20000.0
    """Minimum size for newly generated building areas far from other areas."""
    threshold_building_area_near: float = 4000.0
    """Minimum size for newly generated building areas near other areas."""

    roads_buffer_distance: float = 10.0
    """How large a section will be removed close to roads from building areas."""

    # parameters for GeneralizeLandCover
    positive_buffer: float = 10
    """Buffer to close narrow gaps."""
    negative_buffer: float = -10.0
    """Negative buffer to remove narrow parts."""
    simplification_tolerance: float = 4.0
    """Tolerance for simplifying building areas."""
    hole_threshold: float = 7500
    """Area threshold for removing holes from building areas."""

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

        reference_roads = (
            reference_data[self.reference_key_roads]
            if self.reference_key_roads in reference_data
            else GeoDataFrame(geometry=[], crs=data.crs)
        )

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
            buildings_simplify_tolerance=self.geometry_buildings_simplify_tolerance,
            boffet_area_erosion=self.geometry_boffet_area_erosion,
            boffet_area_buffer=self.geometry_boffet_area_buffer,
        ).execute(
            data,
            reference_data={}
            if reference_data.get(self.reference_key_roads) is None
            else {self.reference_key_roads: reference_data[self.reference_key_roads]},
        )

        gdf = combine_gdfs(
            [
                parcels_result,
                geometry_result,
            ]
        )

        gdf = gdf.dissolve().explode(as_index=False).reset_index(drop=True)

        gdf = GeneralizeLandcover(
            positive_buffer=self.positive_buffer,
            negative_buffer=self.negative_buffer,
            simplification_tolerance=self.simplification_tolerance,
            hole_threshold=self.hole_threshold,
            smoothing=False,
            # skip filtering here, because we need to consider if area is close
            # or far from other areas
            area_threshold=0.0,
        ).execute(gdf)

        # Remove sections from building areas which are too close to the
        # reference roads
        buffered_network = reference_roads.geometry.buffer(
            self.roads_buffer_distance
        ).to_frame()
        gdf = gdf.overlay(buffered_network, how="difference")

        gdf = gdf.dissolve().explode(as_index=False).reset_index(drop=True)

        # Calculate distance to nearest building area
        gdf = gdf.sjoin_nearest(
            gdf.loc[gdf.geometry.area > self.threshold_building_area_far],
            distance_col="is_near",
            exclusive=True,
        ).drop("index_right", axis=1)
        gdf = gdf.drop_duplicates()
        # Drop small building areas with different threshold for areas which
        # are close to other areas and areas which are far from other areas.
        is_near = gdf["is_near"] <= self.near_area_distance
        is_far = ~is_near
        area = gdf.geometry.area
        gdf = gdf.loc[
            (is_near & (area > self.threshold_building_area_near))
            | (is_far & (area > self.threshold_building_area_far))
        ]

        gdf = gdf.drop("is_near", axis=1)

        gdf = assign_nearest_z(data, gdf)
        return hash_index_from_geometry(gdf, "buildingareas")
