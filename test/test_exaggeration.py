#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
import re

import geopandas as gpd
import pytest
from pandas.testing import assert_frame_equal
from shapely import equals_exact
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from geogenalg import exaggeration
from geogenalg.core.exceptions import GeometryTypeError


@pytest.mark.parametrize(
    ("input_gdf", "threshold", "expected_gdf"),
    [
        (
            gpd.GeoDataFrame(
                {"id": [1]},
                geometry=[Polygon([(0, 0, 5), (0, 0.4, 5), (2, 0.4, 5), (2, 0, 5)])],
            ),
            1.0,
            gpd.GeoDataFrame(
                {"id": [1]},
                geometry=[Polygon([(0, 0, 5), (0, 0.4, 5), (2, 0.4, 5), (2, 0, 5)])],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [2]},
                geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
            ),
            1.0,
            gpd.GeoDataFrame(
                {"id": [2]},
                geometry=[Polygon()],
            ),
        ),
        (
            gpd.GeoDataFrame(
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
            gpd.GeoDataFrame(
                {"id": [3]},
                geometry=[Polygon([(0, 0.4, 2), (2, 0.4, 3), (2, 0, 4), (0, 0, 1)])],
            ),
        ),
        (
            gpd.GeoDataFrame(columns=["id", "geometry"], geometry=[]),
            1.0,
            gpd.GeoDataFrame(columns=["id", "geometry"], geometry=[]),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [4], "name": ["test-area"]},
                geometry=[Polygon([(0, 0), (0, 0.4), (3, 0.4), (3, 0)])],
            ),
            1.0,
            gpd.GeoDataFrame(
                {"id": [4], "name": ["test-area"]},
                geometry=[Polygon([(0, 0), (0, 0.4), (3, 0.4), (3, 0)])],
            ),
        ),
        (
            gpd.GeoDataFrame(
                {"id": [5]},
                geometry=[
                    Polygon([(0, 0), (4, 0), (4, 2), (2.2, 2), (2.2, 0.4), (0, 0.4)])
                ],
            ),
            0.5,
            gpd.GeoDataFrame(
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
            gpd.GeoDataFrame(
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
            gpd.GeoDataFrame(
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
    input_gdf: gpd.GeoDataFrame, threshold: float, expected_gdf: gpd.GeoDataFrame
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
    invalid_gdf = gpd.GeoDataFrame(
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
