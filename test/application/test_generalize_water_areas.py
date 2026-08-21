#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from pathlib import Path

import pytest
from conftest import ExpectedResultColumns, IntegrationTest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from shapely import equals_exact
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.testing import GeoPackagePath

UNIQUE_ID_COLUMN = "kmtk_id"


def test_generalize_water_areas(testdata_path: Path):
    gpkg = GeoPackagePath(testdata_path / "water_areas.gpkg")
    IntegrationTest(
        input_uri=gpkg.to_input("areas"),
        control_uri=gpkg.to_input("control"),
        algorithm=GeneralizeWaterAreas(
            min_area=4000.0,
            area_simplification_tolerance=10.0,
            thin_section_width=20.0,
            thin_section_min_size=200.0,
            thin_section_exaggerate_by=3.0,
            island_min_area=100.0,
            island_min_width=185.0,
            island_min_elongation=0.25,
            island_exaggerate_by=3.0,
            island_simplification_tolerance=10.0,
            smoothing_passes=3,
            reference_key="shoreline",
            preserve_shoreline_sections_column="shoreline_type_id",
            preserve_shoreline_sections_values=frozenset([3]),
        ),
        reference_uris={
            "shoreline": gpkg.to_input("shoreline"),
        },
        unique_id_column=UNIQUE_ID_COLUMN,
        check_missing_reference=False,
        dummy_reference_data_mandatory_columns=frozenset(["shoreline_type_id"]),
        expected_result_columns=ExpectedResultColumns(
            inherit="input",
            inherit_from_reference_key="shoreline",
            acceptable_extra_colums=frozenset(["feature_type"]),
        ),
    ).run()


@pytest.mark.parametrize(
    (
        "unmodified_shoreline",
        "new_unsplit_shoreline",
        "expected",
    ),
    [
        (
            GeoDataFrame(
                geometry=[],
            ),
            MultiLineString(),
            MultiLineString(),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [0, 1],
                            [10, 1],
                        ]
                    ),
                ],
            ),
            MultiLineString(
                [
                    LineString(
                        [
                            [-10, 0],
                            [20, 0],
                        ]
                    ),
                ]
            ),
            MultiLineString(
                [
                    LineString(
                        [
                            [0, 1],
                            [0, 0],
                            [0, -0.00001],
                        ]
                    ),
                    LineString(
                        [
                            [10, 1],
                            [10, 0],
                            [10, -0.00001],
                        ]
                    ),
                ]
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [-100, 1],
                            [1, 1],
                        ]
                    ),
                    LineString(
                        [
                            [1, 1],
                            [2, 1],
                        ]
                    ),
                    LineString(
                        [
                            [2, 1],
                            [100, 1],
                        ]
                    ),
                ],
            ),
            MultiLineString(
                [
                    LineString(
                        [
                            [0, 2],
                            [1, 1],
                        ]
                    ),
                    LineString(
                        [
                            [1, 1],
                            [2, 1],
                        ]
                    ),
                    LineString(
                        [
                            [2, 1],
                            [3, 2],
                        ]
                    ),
                ]
            ),
            MultiLineString(
                [
                    LineString(
                        [
                            [1, 1.00001],
                            [1, 1],
                        ]
                    ),
                    LineString(
                        [
                            [1, 1],
                            [1, 0.99999],
                        ]
                    ),
                    LineString(
                        [
                            [2, 1.00001],
                            [2, 1],
                        ]
                    ),
                    LineString(
                        [
                            [2, 1],
                            [2, 0.99999],
                        ]
                    ),
                ]
            ),
        ),
    ],
    ids=[
        "empty",
        "segment",
        "vertex",
    ],
)
def test_get_shoreline_splitters(
    unmodified_shoreline: GeoDataFrame,
    new_unsplit_shoreline: BaseGeometry,
    expected: GeoDataFrame,
):
    result = GeneralizeWaterAreas._get_shoreline_splitters(
        unmodified_shoreline,
        new_unsplit_shoreline,
    )
    assert equals_exact(result, expected)


@pytest.mark.parametrize(
    (
        "algorithm",
        "data",
        "reference_data",
        "expected_segments",
        "expected_skip_coords",
    ),
    [
        (
            GeneralizeWaterAreas(
                reference_key="shoreline",
            ),
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            [0, 0],
                            [10, 0],
                            [10, 10],
                            [0, 10],
                            [0, 0],
                        ]
                    ),
                ],
            ),
            {
                "shoreline": GeoDataFrame(
                    geometry=[
                        LineString(
                            [
                                [0, 0],
                                [10, 0],
                            ]
                        ),
                        LineString(
                            [
                                [0, 10],
                                [10, 10],
                            ]
                        ),
                    ],
                ),
            },
            GeoDataFrame(
                geometry=[
                    LineString(
                        [
                            [10, 0],
                            [10, 10],
                        ]
                    ),
                    LineString(
                        [
                            [0, 10],
                            [0, 0],
                        ]
                    ),
                ],
            ),
            MultiPoint(
                [
                    Point(0, 0),
                    Point(0, 10),
                    Point(10, 0),
                    Point(10, 10),
                ]
            ),
        ),
        (
            GeneralizeWaterAreas(
                reference_key="shoreline",
            ),
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            [0, 0],
                            [10, 0],
                            [10, 10],
                            [0, 10],
                            [0, 0],
                        ]
                    ),
                ],
            ),
            {
                "shoreline": GeoDataFrame(
                    geometry=[
                        LineString(
                            [
                                [0, 0],
                                [10, 0],
                            ]
                        ),
                        LineString(
                            [
                                [10, 0],
                                [10, 10],
                            ]
                        ),
                        LineString(
                            [
                                [10, 10],
                                [0, 10],
                            ]
                        ),
                        LineString(
                            [
                                [0, 10],
                                [0, 0],
                            ]
                        ),
                    ],
                ),
            },
            GeoDataFrame(
                geometry=[],
            ),
            MultiPoint(),
        ),
        (
            GeneralizeWaterAreas(
                reference_key="shoreline",
                preserve_shoreline_sections_column="preserve",
                preserve_shoreline_sections_values=frozenset([1]),
            ),
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            [0, 0],
                            [10, 0],
                            [10, 10],
                            [0, 10],
                            [0, 0],
                        ]
                    ),
                ],
            ),
            {
                "shoreline": GeoDataFrame(
                    {"preserve": [1, 0, 0, 0]},
                    geometry=[
                        LineString(
                            [
                                [0, 0],
                                [10, 0],
                            ]
                        ),
                        LineString(
                            [
                                [10, 0],
                                [10, 10],
                            ]
                        ),
                        LineString(
                            [
                                [10, 10],
                                [0, 10],
                            ]
                        ),
                        LineString(
                            [
                                [0, 10],
                                [0, 0],
                            ]
                        ),
                    ],
                ),
            },
            GeoDataFrame(
                geometry=[],
            ),
            MultiPoint(
                [
                    Point(0, 0),
                    Point(10, 0),
                ]
            ),
        ),
    ],
    ids=[
        "only_reference_data",
        "no_segments_found",
        "preserve_column",
    ],
)
def test_get_segments_and_skip_coords(
    algorithm: GeneralizeWaterAreas,
    data: GeoDataFrame,
    reference_data: dict[str, GeoDataFrame],
    expected_segments: GeoDataFrame,
    expected_skip_coords: MultiPoint,
):
    result_segments, result_skip_coords = algorithm._get_segments_and_skip_coords(
        data,
        reference_data,
    )

    assert_geodataframe_equal(result_segments, expected_segments)
    assert equals_exact(result_skip_coords, expected_skip_coords)
