#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from geopandas import GeoDataFrame, read_file
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString, Point

from geogenalg.application.generalize_fences import GeneralizeFences
from geogenalg.core.exceptions import GeometryTypeError


def _instantiate_algorithm() -> GeneralizeFences:
    return GeneralizeFences(
        closing_fence_area_threshold=2000,
        closing_fence_area_with_mast_threshold=8000,
        fence_length_threshold=80,
        fence_length_threshold_in_closed_area=300,
        simplification_tolerance=4,
        gap_threshold=25,
        attribute_for_line_merge="kohdeluokka",
    )


def test_generalize_fences_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing fences with parameters to the 1: 50 000 scale
    """
    source_path = testdata_path / "fences_rovaniemi.gpkg"

    temp_dir = TemporaryDirectory()
    output_path = temp_dir.name + "/generalized_fences.gpkg"

    fences_gdf = read_file(source_path, layer="mtk_fences")
    masts_gdf = read_file(source_path, layer="mtk_masts")

    algorithm = _instantiate_algorithm()

    result = algorithm.execute(fences_gdf, reference_data={"masts": masts_gdf})

    assert result is not None

    result.to_file(output_path, layer="fences_50k")

    control_fences: GeoDataFrame = read_file(source_path, layer="generalized_fences")

    result_fences = read_file(output_path, layer="fences_50k")

    control_fences = control_fences.sort_values("geometry").reset_index(drop=True)
    result_fences = result_fences.sort_values("geometry").reset_index(drop=True)

    assert_geodataframe_equal(control_fences, result_fences, check_index_type=False)


def test_generalize_fences_50k_invalid_geometry_type() -> None:
    fences_gdf = GeoDataFrame({"id": [1]}, geometry=[Point(0, 0)])

    algorithm = _instantiate_algorithm()

    with pytest.raises(
        GeometryTypeError,
        match=r"GeneralizeFences works only with LineString geometries.",
    ):
        algorithm.execute(data=fences_gdf, reference_data={})


def test_generalize_fences_50k_missing_masts_data(testdata_path: Path) -> None:
    source_path = testdata_path / "fences_rovaniemi.gpkg"
    fences_gdf = read_file(source_path, layer="mtk_fences")

    algorithm = _instantiate_algorithm()

    with pytest.raises(
        KeyError,
        match=r"GeneralizeFences requires mast Point GeoDataFrame in reference_data with key 'masts'.",
    ):
        algorithm.execute(data=fences_gdf, reference_data={})


def test_generalize_fences_50k_invalid_geometry_type_masts(testdata_path: Path) -> None:
    source_path = testdata_path / "fences_rovaniemi.gpkg"
    fences_gdf = read_file(source_path, layer="mtk_fences")
    masts_gdf = GeoDataFrame(
        {"id": [1]}, geometry=[LineString((Point(0, 0), Point(1, 0)))]
    )

    algorithm = _instantiate_algorithm()

    with pytest.raises(
        GeometryTypeError, match=r"Masts data should be a Point GeoDataFrame."
    ):
        algorithm.execute(data=fences_gdf, reference_data={"masts": masts_gdf})
