#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

import geopandas as gpd
import geopandas.testing

from geogenalg.application.boulders_in_water import generalize_boulders_in_water


def test_boulders_in_water_10k(
    boulders_in_water_10k_sourcedata_path: Path,
) -> None:
    """
    Test generalizing boulders in water to the 1: 10 000 scale, where
    the polygonal layer should not be generalized at all.
    """
    source_path = boulders_in_water_10k_sourcedata_path

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/boulder_in_water.gpkg"

    point_gdf = gpd.read_file(source_path, layer="boulder_in_water")
    lake_gdf = gpd.read_file(source_path, layer="lake_part")

    results = generalize_boulders_in_water(point_gdf, lake_gdf, cluster_distance=10)

    assert results.get("clusters") is not None
    assert results.get("boulder_points") is not None

    assert results.get("boulder_polygons") is None

    results["clusters"].to_file(output_path, layer="clusters")
    results["boulder_points"].to_file(output_path, layer="points")

    control_clusters: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="control_clusters"
    )
    control_points: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="control_points"
    )

    result_clusters = gpd.read_file(output_path, layer="clusters")
    result_points = gpd.read_file(output_path, layer="points")

    geopandas.testing.assert_geodataframe_equal(control_clusters, result_clusters)
    geopandas.testing.assert_geodataframe_equal(control_points, result_points)


def test_boulders_in_water_no_land(
    boulders_in_water_no_land_sourcedata_path: Path,
) -> None:
    """
    Test that no newly generated boulder clusters in water end up on
    land.
    """
    source_path = boulders_in_water_no_land_sourcedata_path

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/boulder_in_water.gpkg"

    point_gdf = gpd.read_file(source_path, layer="boulder_in_water")
    polygon_gdf = gpd.read_file(source_path, layer="boulders_in_water")
    lake_gdf = gpd.read_file(source_path, layer="lake_part")

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

    control_clusters: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="control_clusters"
    )
    control_points: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="control_points"
    )
    control_polygons: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="control_polygons"
    )

    result_clusters = gpd.read_file(output_path, layer="clusters")
    result_points = gpd.read_file(output_path, layer="points")
    result_polygons = gpd.read_file(output_path, layer="polygons")

    geopandas.testing.assert_geodataframe_equal(control_clusters, result_clusters)
    geopandas.testing.assert_geodataframe_equal(control_points, result_points)
    geopandas.testing.assert_geodataframe_equal(control_polygons, result_polygons)
