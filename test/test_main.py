#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
import typer
from typer.testing import CliRunner

from geogenalg.main import geopackage_uri

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

    with pytest.raises(typer.BadParameter, match="Incorrectly formatted GeoPackageURI"):
        assert (
            geopackage_uri("/path/to/geopackage.gpkg||layer").file
            == "/path/to/geopackage.gpkg"
        )
