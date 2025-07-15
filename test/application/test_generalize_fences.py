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

from geogenalg.application.generalize_fences import AlgorithmOptions, generalize_fences


def test_generalize_fences_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing fences with parameters to the 1: 50 000 scale
    """
    source_path = testdata_path / "fences_rovaniemi.gpkg"

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/generalized_fences.gpkg"

    fences_gdf = gpd.read_file(source_path, layer="mtk_fences")
    masts_gdf = gpd.read_file(source_path, layer="mtk_masts")

    options = AlgorithmOptions(
        closing_fence_area_threshold=2000,
        closing_fence_area_with_mast_threshold=8000,
        fence_length_threshold=80,
        fence_length_threshold_in_closed_area=300,
        simplification_tolerance=4,
        gap_threshold=25,
        attribute_for_line_merge="kohdeluokka",
    )

    result = generalize_fences(fences_gdf, masts_gdf, options)

    assert result is not None

    result.to_file(output_path, layer="fences_50k")

    control_fences: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="generalized_fences"
    )

    result_fences = gpd.read_file(output_path, layer="fences_50k")

    control_fences = control_fences.sort_values("geometry").reset_index(drop=True)
    result_fences = result_fences.sort_values("geometry").reset_index(drop=True)

    geopandas.testing.assert_geodataframe_equal(
        control_fences, result_fences, check_index_type=False
    )
