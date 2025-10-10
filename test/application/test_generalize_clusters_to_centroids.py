#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import re
import tempfile
from pathlib import Path

import pytest
from geopandas import GeoDataFrame, read_file
from pandas import Series, concat
from pandas.testing import assert_frame_equal
from shapely import LineString, Point, box

from geogenalg.application.generalize_clusters_to_centroids import (
    GeneralizePointClustersAndPolygonsToCentroids,
)
from geogenalg.core.exceptions import GeometryTypeError


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
    input_data = read_file(input_path, layer=input_layer)

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=None,
    )

    control = read_file(input_path, layer="control")
    mask = read_file(input_path, layer="lake_part")
    algorithm.execute(input_data, {"mask": mask}).to_file(output_path, layer="result")

    result = read_file(output_path, layer="result")

    assert_frame_equal(control, result)


def test_invalid_geom_type() -> None:
    gdf = GeoDataFrame({"id": [1]}, geometry=[LineString((Point(0, 0), Point(1, 0)))])

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=None,
    )

    with pytest.raises(
        GeometryTypeError,
        match="GeneralizePointClustersAndPolygonsToCentroids works only with Point and Polygon geometries",
    ):
        algorithm.execute(data=gdf, reference_data={})


def test_invalid_mask_geom_type() -> None:
    gdf = GeoDataFrame({"id": [1, 2]}, geometry=[Point(0, 0), box(0, 0, 1, 1)])
    mask_gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=None,
    )

    with pytest.raises(
        GeometryTypeError,
        match=re.escape("mask dataframe must only contain (Multi)Polygons"),
    ):
        algorithm.execute(data=gdf, reference_data={"mask": mask_gdf})


def test_mixed_geom_types(testdata_path: Path) -> None:
    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    points_path = testdata_path / "boulder_in_water.gpkg"
    points = read_file(
        points_path,
        layer="boulder_in_water",
    )
    polygons = read_file(
        testdata_path / "boulders_in_water.gpkg",
        layer="boulders_in_water",
    )

    mask_gdf = read_file(points_path, layer="lake_part")

    mixed = GeoDataFrame(concat([points, polygons]))

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=None,
    )

    control_path = testdata_path / "boulders_in_water_mixed.gpkg"
    control = read_file(control_path, layer="control")

    algorithm.execute(mixed, {"mask": mask_gdf}).to_file(output_path, layer="result")
    result = read_file(output_path, layer="result")

    assert_frame_equal(control, result)


def test_no_mask(testdata_path: Path) -> None:
    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    points_path = testdata_path / "boulder_in_water.gpkg"
    points = read_file(
        points_path,
        layer="boulder_in_water",
    )

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=None,
    )

    control_path = testdata_path / "boulder_in_water.gpkg"
    control = read_file(control_path, layer="control_no_mask")

    algorithm.execute(points, {}).to_file(output_path, layer="result")
    result = read_file(output_path, layer="result")

    assert_frame_equal(control, result)


def test_aggregation_functions(testdata_path: Path) -> None:
    def _aggregate_boulder_in_water_type(values: Series) -> int:
        if len(set(values)) == 1:
            return int(values.to_numpy()[0])

        return 2

    temp_dir = tempfile.TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    points_path = testdata_path / "boulder_in_water.gpkg"
    points = read_file(
        points_path,
        layer="boulder_in_water",
    )

    aggregation_functions = {
        "boulder_in_water_type_id": _aggregate_boulder_in_water_type
    }

    algorithm = GeneralizePointClustersAndPolygonsToCentroids(
        cluster_distance=30.0,
        unique_id_column="kmtk_id",
        polygon_min_area=1000.0,
        feature_type_column="feature_type",
        aggregation_functions=aggregation_functions,
    )

    control_path = testdata_path / "boulder_in_water.gpkg"
    control = read_file(control_path, layer="control_aggfunc")

    algorithm.execute(points, {}).to_file(output_path, layer="result")
    result = read_file(output_path, layer="result")

    assert_frame_equal(control, result)
