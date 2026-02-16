#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
import typer
from pygeoops import GeoDataFrame
from typer.testing import CliRunner

from geogenalg.application import BaseAlgorithm
from geogenalg.main import (
    GeoPackageURI,
    NamedGeoPackageURI,
    app,
    build_app,
    geopackage_uri,
    get_basealgorithm_attribute_docstrings,
    get_class_attribute_docstrings,
    int_or_str_list,
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


def test_get_basealgorithm_docstrings():
    class MockAlgParent(BaseAlgorithm):
        parent_attribute: str = "something"
        """Parent attribute."""

        def _execute(
            self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
        ) -> GeoDataFrame:
            pass

    class MockAlgSubclass(MockAlgParent):
        subclass_attribute: str = "something"
        """Subclass attribute."""

        def _execute(
            self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
        ) -> GeoDataFrame:
            pass

    assert get_basealgorithm_attribute_docstrings(MockAlgSubclass) == {
        "parent_attribute": "Parent attribute.",
        "subclass_attribute": "Subclass attribute.",
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


@pytest.mark.parametrize(
    ("input_string", "expected"),
    [
        (
            "100",
            100,
        ),
        (
            "str:100",
            "100",
        ),
        (
            "str:100:",
            "str:100:",
        ),
        (
            "str:100.0",
            "str:100.0",
        ),
        (
            "100.0",
            "100.0",
        ),
        (
            "test_string",
            "test_string",
        ),
    ],
    ids=[
        "str_to_int",
        "special_encoding",
        "special_encoding_but_not_integer",
        "special_encoding_float",
        "float",
        "stays_str",
    ],
)
def test_int_or_str_list(input_string: str, expected: NamedGeoPackageURI):
    assert int_or_str_list(input_string) == expected
