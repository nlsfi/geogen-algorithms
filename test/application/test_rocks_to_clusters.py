#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import pytest
from geopandas import GeoDataFrame, read_file
from geopandas.testing import assert_geodataframe_equal

from geogenalg.application.generalize_rocks_to_clusters import (
    GeneralizeRocksToClusters,
    generalize_boulders_in_water,
)


@pytest.mark.parametrize(
    (
        "file_and_layer",
        "control",
    ),
    [
        (("boulders_in_water_10k.gpkg", "boulder_in_water"), "NONE"),
        (("boulders_in_water_no_land.gpkg", "boulders_in_water"), "NONE"),
    ],
    ids=[
        "points",
        "polygons",
    ],
)
def test_generalize_points_and_polygons_to_clusters(
    testdata_path: Path,
    file_and_layer: tuple[str, str],
    control: str,
) -> None:
    input_path = testdata_path / file_and_layer[0]
    input_data = read_file(input_path, layer=file_and_layer[1])

    algorithm = GeneralizeRocksToClusters(
        cluster_distance=10.0,
        unique_id_column="kmtk_id",
        polygon_min_area=4000.0,
    )
    algorithm.execute(input_data, {})

    # FIXME: ASSERT


def test_boulders_in_water_10k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing boulders in water to the 1: 10 000 scale, where
    the polygonal layer should not be generalized at all.
    """
    source_path = testdata_path / "boulders_in_water_10k.gpkg"

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/boulder_in_water.gpkg"

    point_gdf = read_file(source_path, layer="boulder_in_water")
    lake_gdf = read_file(source_path, layer="lake_part")

    results = generalize_boulders_in_water(point_gdf, lake_gdf, cluster_distance=10)

    assert results.get("clusters") is not None
    assert results.get("boulder_points") is not None

    assert results.get("boulder_polygons") is None

    results["clusters"].to_file(output_path, layer="clusters")
    results["boulder_points"].to_file(output_path, layer="points")

    control_clusters: GeoDataFrame = read_file(source_path, layer="control_clusters")
    control_points: GeoDataFrame = read_file(source_path, layer="control_points")

    result_clusters = read_file(output_path, layer="clusters")
    result_points = read_file(output_path, layer="points")

    assert_geodataframe_equal(control_clusters, result_clusters)
    assert_geodataframe_equal(control_points, result_points)


def test_boulders_in_water_no_land(
    testdata_path: Path,
) -> None:
    """
    Test that no newly generated boulder clusters in water end up on
    land.
    """
    source_path = testdata_path / "boulders_in_water_no_land.gpkg"

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/boulder_in_water.gpkg"

    point_gdf = read_file(source_path, layer="boulder_in_water")
    polygon_gdf = read_file(source_path, layer="boulders_in_water")
    lake_gdf = read_file(source_path, layer="lake_part")

    results = generalize_boulders_in_water(
        point_gdf,
        lake_gdf,
        cluster_distance=30,
        polygon_gdf=polygon_gdf,
        polygon_min_area=4000,
    )

    assert results.get("clusters") is not None
    assert results.get("boulder_points") is not None
    assert results.get("boulder_polygons") is not None

    results["clusters"].to_file(output_path, layer="clusters")
    results["boulder_points"].to_file(output_path, layer="points")
    results["boulder_polygons"].to_file(output_path, layer="polygons")

    control_clusters: GeoDataFrame = read_file(source_path, layer="control_clusters")
    control_points: GeoDataFrame = read_file(source_path, layer="control_points")
    control_polygons: GeoDataFrame = read_file(source_path, layer="control_polygons")

    result_clusters = read_file(output_path, layer="clusters")
    result_points = read_file(output_path, layer="points")
    result_polygons = read_file(output_path, layer="polygons")

    assert_geodataframe_equal(control_clusters, result_clusters)
    assert_geodataframe_equal(control_points, result_points)
    assert_geodataframe_equal(control_polygons, result_polygons)
