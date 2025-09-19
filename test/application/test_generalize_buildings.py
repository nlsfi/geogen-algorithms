#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from geopandas import GeoDataFrame
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.application.generalize_building_areas import (
    AlgorithmOptions,
    generalize_building_areas,
)


@pytest.mark.parametrize(
    (
        "parcel_geoms",
        "building_geoms",
        "coverage_threshold",
        "building_threshold",
        "parcel_buffer_distance",
    ),
    [
        (
            [
                Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                Polygon([(20, 0), (25, 0), (25, 5), (20, 5)]),
            ],
            [
                Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
                Polygon([(20, 0), (23, 0), (23, 3), (20, 3)]),
            ],
            15,
            10,
            1.0,
        ),
    ],
    ids=[
        "complete coverage",
    ],
)
def test_generalize_building_areas(
    building_geoms: list[BaseGeometry],
    parcel_geoms: list[BaseGeometry],
    coverage_threshold: float,
    building_threshold: float,
    parcel_buffer_distance: float,
):
    buildings = GeoDataFrame.from_dict(
        {"id": list(range(len(building_geoms))), "geometry": building_geoms},
        crs="EPSG:3067",
    )
    parcels = GeoDataFrame.from_dict(
        {"id": list(range(len(parcel_geoms))), "geometry": parcel_geoms},
        crs="EPSG:3067",
    )
    options = AlgorithmOptions(
        "id", building_threshold, coverage_threshold, parcel_buffer_distance
    )

    result = generalize_building_areas(buildings, parcels, options)
    expected_count = 2
    assert len(result) == expected_count


@pytest.mark.parametrize(
    (
        "building_crs",
        "parcel_crs",
        "coverage_threshold",
    ),
    [
        (
            "EPSG:4326",
            "EPSG:3067",
            15,
        ),
        (
            "EPSG:3067",
            "EPSG:4326",
            15,
        ),
        (
            "EPSG:3067",
            "EPSG:3067",
            150,
        ),
    ],
    ids=[
        "not projected building crs",
        "not projected parcel crs",
        "invalid coverage_threshold",
    ],
)
def test_generalize_building_areas_parameter_checks(
    building_crs: str,
    parcel_crs: str,
    coverage_threshold: float,
):
    building_geoms = [Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])]
    buildings = GeoDataFrame.from_dict(
        {"id": list(range(len(building_geoms))), "geometry": building_geoms},
        crs=building_crs,
    )
    parcels = GeoDataFrame.from_dict(
        {"id": list(range(len(building_geoms))), "geometry": building_geoms},
        crs=parcel_crs,
    )
    options = AlgorithmOptions("id", 1.0, coverage_threshold)
    with pytest.raises(ValueError):  # noqa: PT011
        generalize_building_areas(buildings, parcels, options)
