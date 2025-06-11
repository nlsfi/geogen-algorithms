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

from geogenalg.application.lakes_to_shoreline import (
    extract_shoreline_from_generalized_lakes,
)


def test_extract_shoreline_from_generalized_lakes(
    lakes_to_shoreline_sourcedata_path: Path,
) -> None:
    temp_dir = tempfile.TemporaryDirectory()
    output_path = temp_dir.name + "/shoreline.gpkg"

    original_shoreline = gpd.read_file(
        lakes_to_shoreline_sourcedata_path,
        layer="original_shoreline",
    )
    generalized_lakes = gpd.read_file(
        lakes_to_shoreline_sourcedata_path,
        layer="generalized_lakes",
    )
    control: gpd.GeoDataFrame = gpd.read_file(
        lakes_to_shoreline_sourcedata_path,
        layer="control",
    )

    res = extract_shoreline_from_generalized_lakes(
        original_shoreline,
        generalized_lakes,
    )

    res.to_file(output_path)
    result = gpd.read_file(output_path)

    geopandas.testing.assert_geodataframe_equal(result, control)
