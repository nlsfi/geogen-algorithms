#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import pytest
import typer
from typer.testing import CliRunner

from geogenalg.main import (
    GeoPackageURI,
    NamedGeoPackageURI,
    app,
    build_app,
    geopackage_uri,
    input_geopackage_uris,
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
            "/path/to/geopackage.gpkg+/path/to/other/geopackage.gpkg@layer_name+relative.gpkg|lyr",
            [
                GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name=None),
                GeoPackageURI(
                    file="/path/to/other/geopackage.gpkg", layer_name="layer_name"
                ),
                GeoPackageURI(file="relative.gpkg", layer_name="lyr"),
            ],
        ),
        (
            "/path/to/geopackage.gpkg|layer",
            [GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name="layer")],
        ),
        (
            '"/path/to/geopackage.gpkg|layer"',
            [GeoPackageURI(file="/path/to/geopackage.gpkg", layer_name="layer")],
        ),
    ],
    ids=[
        "multiple",
        "with layer",
        "quotes",
    ],
)
def test_input_geopackage_uris(input_string: str, expected: GeoPackageURI):
    assert input_geopackage_uris(input_string) == expected
