#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import re
from typing import ClassVar

import pytest
from geopandas import GeoDataFrame
from pandas import Index
from pandas.testing import assert_index_equal
from shapely import (
    GeometryCollection,
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import GeometryTypeError, InvalidCRSError


def test_index_reset_without_identity_support():
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=["abc", "def", "ghi"],
        geometry="geom",
        crs="EPSG:3857",
    )

    output_data = MockAlg().execute(input_data, {})

    assert len(output_data.index.intersection(["abc", "def", "ghi"])) == 0


def test_index_not_reset_with_identity_support():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=["abc", "def", "ghi"],
        geometry="geom",
        crs="EPSG:3857",
    )

    output_data = MockAlg().execute(input_data, {})

    assert_index_equal(output_data.index, input_data.index)


def test_input_index_converted_to_strings_for_algorithm_use():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=[111, 222, 333],
        geometry="geom",
        crs="EPSG:3857",
    )

    output_data = MockAlg().execute(input_data, {})

    assert_index_equal(output_data.index, Index(["111", "222", "333"], dtype="string"))


def test_geometry_column_does_not_change():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data.copy().rename_geometry("other_geom")

    input_data = GeoDataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "geom": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        index=[111, 222, 333],
        geometry="geom",
        crs="EPSG:3857",
    )

    output_data = MockAlg().execute(input_data, {})

    assert input_data.geometry.name == "geom"
    assert output_data.geometry.name == "geom"


@pytest.mark.parametrize(
    ("input_data"),
    [
        (GeoDataFrame(geometry=[Point(0, 0), LineString()])),
        (GeoDataFrame(geometry=[Point(0, 0), Polygon()])),
        (GeoDataFrame(geometry=[Point(0, 0), GeometryCollection()])),
        (GeoDataFrame(geometry=[Point(0, 0), LinearRing()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiPolygon()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiPoint()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiLineString()])),
    ],
    ids=[
        "linestring",
        "polygon",
        "geometrycollection",
        "linearring",
        "multipolygon",
        "multipoint",
        "multilinestring",
    ],
)
def test_wrong_geometry_type_input_data(input_data: GeoDataFrame):
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        GeometryTypeError,
        match=r"Input data must contain only geometries of following types: Point.",
    ):
        MockAlg().execute(input_data, {})


@pytest.mark.parametrize(
    ("reference_data"),
    [
        (GeoDataFrame(geometry=[Point(0, 0), LineString()])),
        (GeoDataFrame(geometry=[Point(0, 0), Polygon()])),
        (GeoDataFrame(geometry=[Point(0, 0), GeometryCollection()])),
        (GeoDataFrame(geometry=[Point(0, 0), LinearRing()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiPolygon()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiPoint()])),
        (GeoDataFrame(geometry=[Point(0, 0), MultiLineString()])),
    ],
    ids=[
        "linestring",
        "polygon",
        "geometrycollection",
        "linearring",
        "multipolygon",
        "multipoint",
        "multilinestring",
    ],
)
def test_wrong_geometry_type_reference_data(reference_data: GeoDataFrame):
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        valid_reference_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        GeometryTypeError,
        match=r"Reference data must contain only geometries of following types: Point.",
    ):
        MockAlg().execute(
            GeoDataFrame(geometry=[Point()], crs="EPSG:3857"), {"ref": reference_data}
        )


@pytest.mark.parametrize(
    ("input_data", "error_msg"),
    [
        (
            GeoDataFrame(
                geometry=[],
                crs=None,
            ),
            "Input data has no set coordinate reference system.",
        ),
        (
            GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            "Input data does not have a projected coordinate reference system.",
        ),
    ],
    ids=[
        "no_crs",
        "geographic",
    ],
)
def test_input_invalid_crs(
    input_data: GeoDataFrame,
    error_msg: str,
):
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(InvalidCRSError, match=re.escape(error_msg)):
        MockAlg().execute(input_data)


@pytest.mark.parametrize(
    ("reference_data", "error_msg"),
    [
        (
            GeoDataFrame(
                geometry=[],
                crs=None,
            ),
            'Reference data "ref" does not have a projected coordinate reference system.',
        ),
        (
            GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            'Reference data "ref" does not have a projected coordinate reference system.',
        ),
        (
            GeoDataFrame(
                geometry=[],
                crs="EPSG:3067",
            ),
            'Reference data "ref" and input data have different coordinate reference systems.',
        ),
    ],
    ids=[
        "no_crs",
        "geographic",
        "different",
    ],
)
def test_reference_invalid_crs(
    reference_data: GeoDataFrame,
    error_msg: str,
):
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(InvalidCRSError, match=re.escape(error_msg)):
        MockAlg().execute(
            GeoDataFrame(geometry=[], crs="EPSG:3857"),
            reference_data={"ref": reference_data},
        )
