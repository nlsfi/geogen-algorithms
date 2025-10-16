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

from geogenalg.application.generalize_landcover import GeneralizeLandcover


def test_generalize_landcover_50k(
    testdata_path: Path,
) -> None:
    """
    Test generalizing landcover with cultivated land class
    """
    source_path = testdata_path / "marshes_small_area.gpkg"

    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/generalized_marshes.gpkg"

    input_gdf = gpd.read_file(source_path, layer="mtk_marshes")

    algorithm = GeneralizeLandcover(
        buffer_constant=20,
        simplification_tolerance=30,
        area_threshold=7500,
        hole_threshold=5000,
        smoothing=True,
    )

    result = algorithm.execute(input_gdf, {})

    assert result is not None

    result.to_file(output_path, layer="marshes_50k")

    control_marshes: gpd.GeoDataFrame = gpd.read_file(
        source_path, layer="generalized_marshes"
    )

    result_marshes = gpd.read_file(output_path, layer="marshes_50k")

    control_marshes = control_marshes.sort_values("geometry").reset_index(drop=True)
    result_marshes = result_marshes.sort_values("geometry").reset_index(drop=True)

    # TODO: figure out how to pass Geojson geometries without rounding errors
    geopandas.testing.assert_geodataframe_equal(
        control_marshes.drop(columns=["sijainti_piste"]),
        result_marshes.drop(columns=["sijainti_piste"]),
        check_index_type=False,
    )
