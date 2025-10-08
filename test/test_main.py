#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import tempfile
from pathlib import Path

from geopandas import read_file
from geopandas.testing import assert_geodataframe_equal
from typer.testing import CliRunner

from geogenalg.main import app, geopackage_uri

runner = CliRunner()


def test_geopackage_uri():
    assert geopackage_uri("/path/to/geopackage.gpkg").file == "/path/to/geopackage.gpkg"
    assert geopackage_uri("/path/to/geopackage.gpkg").layer_name is None

    assert (
        geopackage_uri("/path/to/geopackage.gpkg|layer").file
        == "/path/to/geopackage.gpkg"
    )
    assert geopackage_uri("/path/to/geopackage.gpkg|layer").layer_name == "layer"

    assert (
        geopackage_uri('"/path/to/geopackage.gpkg|layer"').file
        == "/path/to/geopackage.gpkg"
    )
    assert geopackage_uri('"/path/to/geopackage.gpkg|layer"').layer_name == "layer"


def test_clusters_to_centroids(testdata_path: Path):
    input_gpkg = testdata_path / "boulder_in_water.gpkg"
    input_geopackage_uri = f'"{input_gpkg}|boulder_in_water"'

    mask_geopackage_uri = f'"{input_gpkg}|lake_part"'

    temp_dir = tempfile.TemporaryDirectory()
    output_geopackage_uri = f"{temp_dir.name}/output.gpkg"
    result = runner.invoke(
        app,
        [
            input_geopackage_uri,
            output_geopackage_uri,
            "--unique-id-column=kmtk_id",
            "--polygon-min-area=1000.0",
            f"--mask-data={mask_geopackage_uri}",
        ],
    )

    assert result.exit_code == 0
    assert Path(output_geopackage_uri).exists()

    control = read_file(input_gpkg, layer="control")
    result = read_file(output_geopackage_uri)

    assert_geodataframe_equal(control, result)
