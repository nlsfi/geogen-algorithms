#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import re
from typing import ClassVar, Literal

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas import Index
from pandas.testing import assert_index_equal
from pydantic import Field, ValidationError
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

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.core.exceptions import (
    GeometryTypeError,
    InvalidCRSError,
    MissingReferenceError,
)


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
        reference_key: str = "ref"

        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar = {
            "reference_key": ReferenceDataInformation(
                required=True,
                valid_geometry_types={"Point"},
            ),
        }

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        GeometryTypeError,
        match=r'Reference data "ref" must contain only geometries of following types: Point.',
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
            "Input data has no coordinate reference system.",
        ),
        (
            GeoDataFrame(
                geometry=[],
                crs="EPSG:4326",
            ),
            "Algorithm requires projected CRS and data does not have one.",
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
            'Reference data "ref" and input data have different coordinate reference systems: None != EPSG:3857.',
        ),
        (
            GeoDataFrame(
                geometry=[],
                crs="EPSG:3067",
            ),
            'Reference data "ref" and input data have different coordinate reference systems: EPSG:3067 != EPSG:3857.',
        ),
    ],
    ids=[
        "no_crs",
        "different",
    ],
)
def test_reference_invalid_crs(
    reference_data: GeoDataFrame,
    error_msg: str,
):
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_key: str = "ref"
        reference_data_schema: ClassVar = {
            "reference_key": ReferenceDataInformation(
                required=True,
                valid_geometry_types={"Point"},
            ),
        }

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(InvalidCRSError, match=re.escape(error_msg)):
        MockAlg().execute(
            GeoDataFrame(geometry=[], crs="EPSG:3857"),
            reference_data={"ref": reference_data},
        )


def test_no_projected_crs_required():
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    # Just check that no error is raised, no explicit asserts required

    MockAlg().execute(GeoDataFrame(geometry=[], crs="EPSG:4326"))


def test_subalgorithm_is_frozen():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        parameter: int = 0

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(ValidationError, match=r"1 validation error for MockAlg"):
        MockAlg().parameter = 5


def test_limit_int_value():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        parameter: int = Field(1, ge=-100, le=100)

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    assert MockAlg().parameter == 1

    # Check no validation error happens
    MockAlg(parameter=0)
    MockAlg(parameter=100)
    MockAlg(parameter=-100)

    with pytest.raises(ValidationError, match=r"1 validation error for MockAlg"):
        MockAlg(parameter=-1000)
    with pytest.raises(ValidationError, match=r"1 validation error for MockAlg"):
        MockAlg(parameter=1000)


def test_algorithm_raises_on_extra_input():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        ValidationError,
        match="Extra inputs are not permitted",
    ):
        MockAlg(extra_argument=10)


def test_algorithm_raises_on_nonexistant_reference_key_member():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {
            "reference_key__": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=True,
            )
        }

        reference_key: str = "reference"
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Attribute named 'reference_key__' is defined in reference data schema but not found in algorithm instance."
        ),
    ):
        MockAlg(
            reference_key="not_reference",
        ).execute(GeoDataFrame(geometry=[], crs="EPSG:4326"))


def test_algorithm_raises_on_required_reference_data_keys():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {
            "reference_key": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=True,
            )
        }

        reference_key: str = "reference"
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        MissingReferenceError,
        match=re.escape(
            "Algorithm has required reference data key(s): 'reference_key', but no reference data passed."
        ),
    ):
        MockAlg(
            reference_key="not_reference",
        ).execute(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            None,
        )

    with pytest.raises(
        MissingReferenceError,
        match=re.escape(
            "Algorithm has required reference data key(s): 'reference_key', but no reference data passed."
        ),
    ):
        MockAlg(
            reference_key="not_reference",
        ).execute(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            {},
        )


def test_algorithm_raises_on_unexpected_key():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {
            "reference_key": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=True,
            )
        }

        reference_key: str = "reference_key"
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        MissingReferenceError,
        match=re.escape(
            "Reference data has unexpected key 'not_reference'. Expected one of 'reference_key'"
        ),
    ):
        MockAlg().execute(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            {"not_reference": GeoDataFrame(geometry=[], crs="EPSG:4326")},
        )


def test_algorithm_raises_on_none_reference_data():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {
            "reference_key": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=True,
            )
        }

        reference_key: str = "reference_key"
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        MissingReferenceError,
        match=re.escape(
            "Reference data by key 'reference_key' is None. Expected a GeoDataFrame"
        ),
    ):
        MockAlg().execute(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            {"reference_key": None},
        )


def test_algorithm_raises_on_missing_key():
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point"}
        reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {
            "reference_key": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=True,
            ),
            "other_ref": ReferenceDataInformation(
                valid_geometry_types={"Point"},
                required=False,
            ),
        }

        reference_key: str = "reference_key"
        other_ref: str = "other_ref"
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    with pytest.raises(
        MissingReferenceError,
        match=re.escape("Reference data contains no mandatory key 'reference_key'."),
    ):
        MockAlg().execute(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            {"other_ref": GeoDataFrame(geometry=[], crs="EPSG:4326")},
        )


@pytest.mark.parametrize(
    (
        "mode",
        "input_gdf",
        "expected_gdf",
    ),
    [
        (
            "no",
            GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (4, 2), (0, 2), (2, 0), (0, 0)]),
                    LineString([(0, 0), (2, 2), (0, 2), (2, 0)]),
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (4, 2), (0, 2), (2, 0), (0, 0)]),
                    LineString([(0, 0), (2, 2), (0, 2), (2, 0)]),
                ]
            ),
        ),
        (
            "keep_largest",
            GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (4, 2), (0, 2), (2, 0), (0, 0)]),
                    LineString([(0, 0), (1, 0), (1, 1), (0.5, 1), (0.5, -1)]),
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Polygon([(4, 2), (1.3333333333, 0.6666666666), (0, 2), (4, 2)]),
                    LineString([(0.5, 0), (1, 0), (1, 1), (0.5, 1), (0.5, 0)]),
                ]
            ),
        ),
        (
            "explode",
            GeoDataFrame(
                geometry=[
                    Polygon([(0, 0), (4, 2), (0, 2), (2, 0), (0, 0)]),
                    LineString([(0, 0), (2, 2), (0, 2), (2, 0)]),
                ]
            ),
            GeoDataFrame(
                geometry=[
                    Polygon([(2, 0), (0, 0), (1.3333333333, 0.6666666666), (2, 0)]),
                    Polygon([(4, 2), (1.3333333333, 0.6666666666), (0, 2), (4, 2)]),
                    LineString([(0, 0), (1, 1)]),
                    LineString([(1, 1), (2, 2), (0, 2), (1, 1)]),
                    LineString([(1, 1), (2, 0)]),
                ],
                index=[
                    "ec9c3e680737d94401aed12fd60205cf9ed3c48f71a0fe9673efb1299c508926",
                    "082d4a6d49ec7271a86d784507acfd0ebe60503019eb8101b0c062ee1acb9f49",
                    "34406e3e2314db2f5e4f5296f1b1ddd4fc98693918c327e99a9139b6b97f9a09",
                    "979fe9ee9765c966b47bae87ff510d294f2e6959c77cebb7bc0caeb3e4a39169",
                    "9b0225e4812210250c2fafb2f29ed57cba4e110893200111f59e3dead047c801",
                ],
            ),
        ),
        (
            "keep_largest",
            GeoDataFrame(geometry=[Point(0, 0)]),
            GeoDataFrame(geometry=[Point(0, 0)]),
        ),
        (
            "explode",
            GeoDataFrame(geometry=[Point(0, 0)]),
            GeoDataFrame(geometry=[Point(0, 0)]),
        ),
    ],
    ids=[
        "no",
        "keep_largest",
        "explode",
        "point_keep_largest",
        "point_explode",
    ],
)
def test_repair_result_geometries(
    mode: Literal["no", "keep_largest", "explode"],
    input_gdf: GeoDataFrame,
    expected_gdf: GeoDataFrame,
):
    @supports_identity
    class MockAlg(BaseAlgorithm):
        valid_input_geometry_types: ClassVar = {"Point", "LineString", "Polygon"}
        requires_projected_crs = False

        def _execute(self, data, reference_data):  # noqa: ANN001, ANN202, ARG002
            return data

    alg = MockAlg(
        repair_result_geometries=mode,
    )

    with pytest.warns(UserWarning, match="contains invalid geometries"):
        result = alg._repair_result_geometries(
            input_gdf.set_index(input_gdf.index.astype("string"))
        )

    assert_geodataframe_equal(
        result,
        expected_gdf.set_index(expected_gdf.index.astype("string")),
        check_less_precise=True,
    )
