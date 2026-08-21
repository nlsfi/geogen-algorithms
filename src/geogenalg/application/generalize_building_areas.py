#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import ClassVar, override

from cartagen.enrichment.urban.urban_areas import boffet_areas
from geopandas.geodataframe import GeoDataFrame
from geopandas.geoseries import GeoSeries
from pydantic import Field

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
from geogenalg.utility.dataframe_processing import combine_gdfs, copy_gdf_as_empty


@supports_identity
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

    parcel_coverage_threshold: float = Field(7.5, ge=0)
    """To determine parcels with "high" building density."""
    parcel_buffer_distance: float = Field(20, gt=0)
    """Buffer distance used to merge parcels."""
    building_size_filter_threshold: float = Field(4000.0, gt=0)
    """Buildings which a) belong to a filtered class and b) are larger than
    this threshold are filtered out."""
    building_filter_column: str = "building_function_id"
    """Name of the column which contains attributes used for filtering."""
    classes_for_filtering: frozenset[int | str] = frozenset()
    """Buildings which a) have one of these values and b) are larger than the
    size threshold are filtered out."""
    buildings_simplify_tolerance: float = Field(10.0, ge=0)
    """Tolerance for building simplification (Douglas-Peucker)."""
    boffet_area_buffer: float = Field(10.0, gt=0)
    """The buffer size used to merge buildings that are close from each
    other."""
    boffet_area_erosion: float = Field(10.0, gt=0)
    """The erosion size to avoid the building area to expand too far from the
    buildings located on the edge."""
    near_area_distance: float = Field(50.0, gt=0)
    """Distance for building areas to be considered as near each other, affecting
    threshold used to filter small areas."""
    threshold_building_area_far: float = Field(20000.0, gt=0)
    """Minimum size for newly generated building areas far from other areas."""
    threshold_building_area_near: float = Field(4000.0, gt=0)
    """Minimum size for newly generated building areas near other areas."""
    roads_buffer_distance: float = Field(10.0, gt=0)
    """How large a section will be removed close to roads from building areas."""

    # parameters for GeneralizeLandCover
    positive_buffer: float = Field(10.0, ge=0)
    """Buffer to close narrow gaps."""
    negative_buffer: float = Field(-10.0, lt=0)
    """Negative buffer to remove narrow parts."""
    simplification_tolerance: float = Field(4.0, ge=0)
    """Tolerance for simplifying building areas."""
    hole_threshold: float = Field(7500, gt=0)
    """Area threshold for removing holes from building areas."""

    reference_key_parcels: str = "parcels"
    """Reference data key for parcel data. This optional reference data is
    intended to be a land parcel or similar polygonal dataset inside which
    buildings in the input data reside in. The parcel polygons are turned into
    building areas if they meet the coverage threshold. If not provided,
    building areas will be formed only by buffering the buildings."""
    reference_key_roads: str = "roads"
    """Reference data key for road data. This optional reference data is used
    to create a buffer area around the roads which is removed out of the
    generated building areas. The road data is intended to already be
    generalized to the target scale, although this is not strictly
    necessary."""

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
            else copy_gdf_as_empty(data)
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
            {
                data.geometry.name: GeoSeries(
                    boffet_areas(
                        gdf.geometry.to_list(),
                        self.boffet_area_buffer,
                        self.boffet_area_erosion,
                    ),
                )
            },
            geometry=data.geometry.name,
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

            if parcels_gdf.geometry.name != gdf.geometry.name:
                parcels_gdf = parcels_gdf.rename_geometry(gdf.geometry.name)

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
            return GeoDataFrame(geometry=[], crs=gdf.crs)

        # Remove sections from building areas which are too close to the
        # reference roads
        buffered_network = reference_roads.geometry.buffer(
            self.roads_buffer_distance
        ).to_frame()
        gdf = gdf.overlay(buffered_network, how="difference")

        gdf = gdf.dissolve().explode(as_index=False).reset_index(drop=True)

        # Calculate distance to nearest building area
        distances = gdf.sjoin_nearest(
            gdf.loc[gdf.geometry.area > self.threshold_building_area_far],
            distance_col="distance_to_nearest",
            exclusive=True,
        )["distance_to_nearest"]

        gdf = gdf.assign(distance_to_nearest=distances)
        gdf = gdf.drop_duplicates()
        # Drop small building areas with different threshold for areas which
        # are close to other areas and areas which are far from other areas.
        is_near = gdf["distance_to_nearest"] <= self.near_area_distance
        is_far = ~is_near
        area = gdf.geometry.area
        gdf = gdf.loc[
            (is_near & (area > self.threshold_building_area_near))
            | (is_far & (area > self.threshold_building_area_far))
        ]

        gdf = gdf.drop("distance_to_nearest", axis=1)

        gdf = assign_nearest_z(data, gdf)
        return hash_index_from_geometry(gdf, "buildingareas")
