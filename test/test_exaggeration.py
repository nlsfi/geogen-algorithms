#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re

import pytest
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_frame_equal
from shapely import equals_exact
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from geogenalg import exaggeration
from geogenalg.core.exceptions import GeometryTypeError


@pytest.mark.parametrize(
    (
        "input_gdf",
        "width_threshold",
        "elongation_threshold",
        "expected_gdf",
    ),
    [
        (
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [[0, 0], [0, 4], [1, 4], [1, 0], [0, 0]]
                    ),  # elongated enough, too wide
                    Polygon(
                        [[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]
                    ),  # thin enough, not elongated enough
                    Polygon(
                        [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]
                    ),  # not elongated enough, too wide
                ],
            ),
            0.55,  # width_threshold
            0.5,  # elongation_threshold
            GeoDataFrame(
                geometry=[
                    Polygon([[0, 0], [0, 4], [1, 4], [1, 0], [0, 0]]),
                    Polygon([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]),
                    Polygon([[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]),
                ],
            ),
        ),
        (
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [[0, 0], [0, 4], [1, 4], [1, 0], [0, 0]]
                    ),  # should exaggerate
                    Polygon(
                        [[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]
                    ),  # not elongated enough
                    Polygon(
                        [[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]
                    ),  # too wide and not elongated enough
                ],
            ),
            2,  # width_threshold
            0.5,  # elongation_threshold
            GeoDataFrame(
                geometry=[
                    Polygon([[-1, -1], [-1, 5], [2, 5], [2, -1], [-1, -1]]),
                    Polygon([[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0], [0, 0]]),
                    Polygon([[0, 0], [0, 4], [4, 4], [4, 0], [0, 0]]),
                ],
            ),
        ),
    ],
    ids=[
        "no exaggerations",
        "one exaggerated",
    ],
)
def test_exaggerate_thin_polygons(
    input_gdf: GeoDataFrame,
    width_threshold: float,
    elongation_threshold: float,
    expected_gdf: GeoDataFrame,
):
    result = exaggeration.exaggerate_thin_polygons(
        input_gdf,
        width_threshold,
        elongation_threshold,
        1.0,
        buffer_options=exaggeration.BufferOptions(
            cap_style="square",
            join_style="mitre",
        ),
    )

    assert_geodataframe_equal(result, expected_gdf)


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "expected_gdf"),
    [
        (
            GeoDataFrame(
                {"id": [1]},
                geometry=[Polygon([(0, 0, 5), (0, 0.4, 5), (2, 0.4, 5), (2, 0, 5)])],
            ),
            1.0,
            GeoDataFrame(
                {"id": [1]},
                geometry=[Polygon([(0, 0, 5), (0, 0.4, 5), (2, 0.4, 5), (2, 0, 5)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [2]},
                geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
            ),
            1.0,
            GeoDataFrame(
                {"id": [2]},
                geometry=[Polygon()],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [3]},
                geometry=[
                    MultiPolygon(
                        [
                            Polygon([(0, 0, 1), (0, 0.4, 2), (2, 0.4, 3), (2, 0, 4)]),
                            Polygon([(3, 0), (5, 0), (5, 3), (3, 3)]),
                        ]
                    )
                ],
            ),
            1.0,
            GeoDataFrame(
                {"id": [3]},
                geometry=[Polygon([(0, 0.4, 2), (2, 0.4, 3), (2, 0, 4), (0, 0, 1)])],
            ),
        ),
        (
            GeoDataFrame(columns=["id", "geometry"], geometry=[]),
            1.0,
            GeoDataFrame(columns=["id", "geometry"], geometry=[]),
        ),
        (
            GeoDataFrame(
                {"id": [4], "name": ["test-area"]},
                geometry=[Polygon([(0, 0), (0, 0.4), (3, 0.4), (3, 0)])],
            ),
            1.0,
            GeoDataFrame(
                {"id": [4], "name": ["test-area"]},
                geometry=[Polygon([(0, 0), (0, 0.4), (3, 0.4), (3, 0)])],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [5]},
                geometry=[
                    Polygon([(0, 0), (4, 0), (4, 2), (2.2, 2), (2.2, 0.4), (0, 0.4)])
                ],
            ),
            0.5,
            GeoDataFrame(
                {"id": [5]},
                geometry=[
                    Polygon(
                        [
                            (0, 0.4),
                            (2.2, 0.4),
                            (2.2, 0),
                            (0, 0),
                        ]
                    )
                ],
            ),
        ),
        (
            GeoDataFrame(
                {"id": [6]},
                geometry=[
                    Polygon(
                        [
                            (0, -2),
                            (0, 0.75),
                            (0.75, 0.75),
                            (0.75, 0),
                            (2.25, 0),
                            (2.25, 0.5),
                            (3, 0.5),
                            (3, -2),
                        ]
                    )
                ],
            ),
            1,
            GeoDataFrame(
                {"id": [6]},
                geometry=[
                    MultiPolygon(
                        [
                            Polygon([(2.25, 0.5), (3, 0.5), (3, 0), (2.25, 0)]),
                            Polygon([(0, 0.75), (0.75, 0.75), (0.75, 0), (0, 0)]),
                        ]
                    )
                ],
            ),
        ),
    ],
    ids=[
        "narrow polygon extracted with z-coordinates",
        "no narrow parts",
        "multipolygon with one narrow polygon with z-coordinates",
        "empty input",
        "attributes should be inherited",
        "polygon with narrow part",
        "input polygon with two separate narrow parts",
    ],
)
def test_extract_narrow_polygon_parts(
    input_gdf: GeoDataFrame, threshold: float, expected_gdf: GeoDataFrame
):
    result_gdf = exaggeration.extract_narrow_polygon_parts(input_gdf, threshold)

    result_sorted = result_gdf.reset_index(drop=True)
    expected_sorted = expected_gdf.reset_index(drop=True)

    # Check geometries
    for _, (result_geometry, expected_geometry) in enumerate(
        zip(
            result_gdf.geometry.reset_index(drop=True),
            expected_gdf.geometry.reset_index(drop=True),
            strict=False,
        )
    ):
        assert equals_exact(result_geometry, expected_geometry, tolerance=1e-3)

    # Check attributes
    result_attrs = result_sorted.drop(columns="geometry")
    expected_attrs = expected_sorted.drop(columns="geometry")
    assert_frame_equal(result_attrs, expected_attrs, check_dtype=False)


def test_extract_narrow_polygon_parts_raises_on_non_polygon_geometry():
    invalid_gdf = GeoDataFrame(
        {"id": [1, 2, 3]},
        geometry=[
            Point(0, 0),
            LineString([(0, 0), (1, 1)]),
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        ],
    )

    with pytest.raises(
        GeometryTypeError,
        match=re.escape(
            "Extract narrow parts only supports Polygon or MultiPolygon geometries."
        ),
    ):
        exaggeration.extract_narrow_polygon_parts(invalid_gdf, 1.0)
