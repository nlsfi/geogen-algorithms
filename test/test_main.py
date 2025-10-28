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
    GeoPackageURI,
    NamedGeoPackageURI,
    app,
    build_app,
    geopackage_uri,
    get_class_attribute_docstrings,
    named_geopackage_uri,
)

runner = CliRunner()


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        (
            "/path/to/geopackage.gpkg",
            GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name=None),
        ),
        (
            "/path/to/geopackage.gpkg|layer",
            GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name="layer"),
        ),
        (
            '"/path/to/geopackage.gpkg|layer"',
            GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name="layer"),
        ),
    ],
    ids=[
        "no layer",
        "with layer",
        "quotes",
    ],
)
def test_geopackage_uri(input_string: str, expected: GeoPackageURI):
    assert geopackage_uri(input_string) == expected


@pytest.mark.parametrize(
    ("input_string"),
    [
        ("data.gpkg||layer"),
        ("data.gpk|g|layer"),
        ("data.gpkg|la@yer"),
    ],
)
def test_geopackage_uri_raises(input_string: str):
    with pytest.raises(typer.BadParameter, match="Incorrectly formatted GeoPackageURI"):
        geopackage_uri(input_string)


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        (
            "mask:data.gpkg",
            NamedGeoPackageURI(
                "mask",
                GeoPackageURI(file="data.gpkg", layer_name=None),
            ),
        ),
        (
            "mask:data.gpkg|layer",
            NamedGeoPackageURI(
                "mask",
                GeoPackageURI(file="data.gpkg", layer_name="layer"),
            ),
        ),
        (
            "reference:data.gpkg@layer",
            NamedGeoPackageURI(
                "reference",
                GeoPackageURI(file="data.gpkg", layer_name="layer"),
            ),
        ),
    ],
    ids=[
        "no layer",
        "with layer",
        "@ delimiter",
    ],
)
def test_named_geopackage_uri(input_string: str, expected: NamedGeoPackageURI):
    assert named_geopackage_uri(input_string) == expected


@pytest.mark.parametrize(
    ("input_string"),
    [
        ("data.gpkg|layer"),
        (":data.gpkg|layer"),
    ],
    ids=[
        "no name",
        "empty name",
    ],
)
def test_named_geopackage_uri_raises(input_string: str):
    with pytest.raises(
        typer.BadParameter, match="Incorrectly formatted NamedGeoPackageURI"
    ):
        named_geopackage_uri(input_string)


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
