#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


import pytest
import typer
from typer.testing import CliRunner

from geogenalg.main import (
    app,
    build_app,
    geopackage_uri,
    get_class_attribute_docstrings,
)

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


def test_get_class_attribute_docstrings():
    class TestClass:
        attribute_1: str = "something"
        """Attribute 1 docstring"""
        attribute_2: int = 10
        """Attribute 2 docstring"""
        attribute_3: int = 10  # doesn't have a docstring

    assert get_class_attribute_docstrings(TestClass) == {
        "attribute_1": "Attribute 1 docstring",
        "attribute_2": "Attribute 2 docstring",
    }


def test_main():
    build_app()
    result = runner.invoke(
        app,
        [
            "--help",
        ],
    )

    assert result.exit_code == 0
