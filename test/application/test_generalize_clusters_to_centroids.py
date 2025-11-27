#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re
from pathlib import Path

import pytest
from geopandas import GeoDataFrame
from pandas import concat
from pandas.testing import assert_frame_equal
from shapely import LineString, Point

from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.testing import (
    GeoPackageInput,
    get_alg_results_from_geopackage,
    get_result_and_control,
)
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

UNIQUE_ID_COLUMN = "kmtk_id"


@pytest.mark.parametrize(
    (
        "gpkg_file",
        "input_layer",
    ),
    [
        ("boulder_in_water.gpkg", "boulder_in_water"),
        ("boulders_in_water.gpkg", "boulders_in_water"),
    ],
    ids=[
        "points",
        "polygons",
    ],
)
def test_generalize_boulders_in_water(
    testdata_path: Path,
    gpkg_file: str,
    input_layer: str,
) -> None:
    input_path = testdata_path / gpkg_file
    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name=input_layer),
        GeoPackageInput(input_path, layer_name="control"),
        GeneralizePointClustersAndPolygonsToCentroids(
            cluster_distance=30.0,
            polygon_min_area=1000.0,
            feature_type_column="feature_type",
            aggregation_functions=None,
        ),
        UNIQUE_ID_COLUMN,
    )

    assert_frame_equal(result, control)


def test_invalid_geom_type() -> None:
    with pytest.raises(
        GeometryTypeError,
        match=re.escape("Input data must only contain Polygons or Points."),
    ):
        GeneralizePointClustersAndPolygonsToCentroids().execute(
            GeoDataFrame(
                {"id": [1]},
                geometry=[LineString((Point(0, 0), Point(1, 0)))],
            ),
        )


def test_mixed_geom_types(testdata_path: Path) -> None:
    points_path = testdata_path / "boulder_in_water.gpkg"

    points = read_gdf_from_file_and_set_index(
        points_path,
        UNIQUE_ID_COLUMN,
        layer="boulder_in_water",
    )
    polygons = read_gdf_from_file_and_set_index(
        testdata_path / "boulders_in_water.gpkg",
        UNIQUE_ID_COLUMN,
        layer="boulders_in_water",
    )

    mixed = GeoDataFrame(concat([points, polygons]))

    control_path = testdata_path / "boulders_in_water_mixed.gpkg"
    control = read_gdf_from_file_and_set_index(
        control_path,
        UNIQUE_ID_COLUMN,
        layer="control",
    )

    result = get_alg_results_from_geopackage(
        GeneralizePointClustersAndPolygonsToCentroids(
            cluster_distance=30.0,
            polygon_min_area=1000.0,
            feature_type_column="feature_type",
            aggregation_functions=None,
        ),
        mixed,
        unique_id_column=UNIQUE_ID_COLUMN,
    )

    assert_frame_equal(result, control)


def test_aggregation_functions(testdata_path: Path) -> None:
    input_path = testdata_path / "boulder_in_water.gpkg"

    aggregation_functions = {
        "boulder_in_water_type_id": lambda values: min(values)
        if len(values) == set(values)
        else 2
    }

    result, control = get_result_and_control(
        GeoPackageInput(input_path, layer_name="boulder_in_water"),
        GeoPackageInput(input_path, layer_name="control_aggfunc"),
        GeneralizePointClustersAndPolygonsToCentroids(
            cluster_distance=30.0,
            polygon_min_area=1000.0,
            feature_type_column="feature_type",
            aggregation_functions=aggregation_functions,
        ),
        UNIQUE_ID_COLUMN,
    )

    assert_frame_equal(result, control)
