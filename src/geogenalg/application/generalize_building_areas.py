#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import ClassVar, override

from cartagen.enrichment.urban.urban_areas import boffet_areas
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries

from geogenalg.analyze import calculate_coverage
from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.application.generalize_landcover import GeneralizeLandcover
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.identity import hash_index_from_geometry
from geogenalg.merge import buffer_and_merge_polygons
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreas(BaseAlgorithm):
    """Generalize polygons representing buildings.

    Input building data can be filtered out as follows

    Optionally parcel data can be passed as reference. Areas which are covered
    by enough parcels are turned to building areas.

    Output contains generalized building-area polygons.

    The algorithm does the following steps:
    - Creates building areas from building geometries (boffet areas).
    - If passed, creates building areas from parcels with enough building coverage.
    - Combines the results
    - Removes sections inside buffered road reference data (if provided)
    - Post-processes building areas using the GeneralizeLandcover algorithm

    """

    parcel_coverage_threshold: float = 5
    """To determine parcels with "high" building density."""
    parcel_buffer_distance: float = 20
    """Buffer distance used to merge parcels."""
    building_size_filter_threshold: float = 4000.0
    """Buildings which a) belong to a filtered class and b) are larger than
    this threshold are filtered out."""
    building_filter_column: str = "building_function_id"
    """Name of the column which contains attributes used for filtering."""
    classes_for_filtering: frozenset[int | str] = frozenset()
    """Buildings which a) have one of these values and b) are larger than the
    size threshold are filtered out."""
    buildings_simplify_tolerance: float = 10.0
    """Tolerance for building simplification (Douglas-Peucker)."""
    boffet_area_buffer: float = 10.0
    """The buffer size used to merge buildings that are close from each
    other."""
    boffet_area_erosion: float = 10.0
    """The erosion size to avoid the building area to expand too far from the
    buildings located on the edge."""
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
    reference_data_schema: ClassVar = {
        "reference_key_parcels": ReferenceDataInformation(
            required=False,
            valid_geometry_types={
                "Polygon",
            },
        ),
        "reference_key_roads": ReferenceDataInformation(
            required=False,
            valid_geometry_types={
                "LineString",
            },
        ),
    }

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        reference_roads = (
            reference_data[self.reference_key_roads]
            if self.reference_key_roads in reference_data
            else GeoDataFrame(geometry=[], crs=data.crs)
        )

        copy = data.copy()
        gdf = copy.loc[
            ~(
                (copy[self.building_filter_column].isin(self.classes_for_filtering))
                & (copy.geometry.area > self.building_size_filter_threshold)
            )
        ]

        gdf.geometry = gdf.simplify(self.buildings_simplify_tolerance)
        gdf = GeoDataFrame(
            geometry=GeoSeries(
                boffet_areas(
                    gdf.geometry.to_list(),
                    self.boffet_area_buffer,
                    self.boffet_area_erosion,
                ),
            ),
            crs=data.crs,
        )

        if self.reference_key_parcels in reference_data:
            parcels_gdf = calculate_coverage(
                copy, reference_data[self.reference_key_parcels], "coverage"
            )
            parcels_gdf = parcels_gdf.loc[
                parcels_gdf["coverage"] > self.parcel_coverage_threshold
            ]

            parcels_gdf = buffer_and_merge_polygons(
                parcels_gdf,
                self.parcel_buffer_distance,
            ).explode(as_index=False)

            gdf = combine_gdfs(
                [
                    parcels_gdf,
                    gdf,
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

        if gdf.empty:
            # GeneralizeLandCover may have eroded all areas away, which
            # will cause overlay() to fail later -> return early
            return gdf

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
