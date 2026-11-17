#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import numpy as np
import pytest
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geodataframe_equal
from shapely import LineString, MultiPolygon, Polygon, box, union_all
from shapely.testing import assert_geometries_equal

from geogenalg.transform import boffet_areas, thin_polygon_sections_to_lines
from geogenalg.utility.dataframe_processing import add_columns_to_gdf


@pytest.mark.parametrize(
    ("input_gdf", "expected_lines", "expected_polygons", "threshold"),
    [
        (
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            add_columns_to_gdf(
                GeoDataFrame(
                    columns=["id", "attribute"],
                    geometry=[],
                ),
                {"_old_ids": "object"},
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                    "_olds_ids": [np.nan],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            10,
        ),
        (
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                },
                geometry=[
                    box(0, 0, 100, 20),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1],
                    "attribute": ["attribute"],
                    "_old_ids": [(1,)],
                },
                geometry=[LineString([[10, 10], [90, 10]])],
            ),
            GeoDataFrame(columns=["id", "attribute", "_old_ids"], geometry=[]),
            100,
        ),
        (
            GeoDataFrame(
                {
                    "id": [1, 2, 3],
                    "attribute": [
                        "remains a polygon",
                        "turns to line",
                        "section turns to line",
                    ],
                },
                geometry=[
                    box(0, 0, 100, 20),
                    box(50, 50, 60, 150),
                    union_all(
                        [
                            box(80, 30, 150, 130),
                            box(102.5, 130, 107.5, 240),
                        ]
                    ),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [3, 2],
                    "attribute": [
                        "section turns to line",
                        "turns to line",
                    ],
                    "_old_ids": [
                        (3,),
                        (2,),
                    ],
                },
                geometry=[
                    LineString([[105, 133.49107142857142], [105, 237.5]]),
                    LineString([[55, 55], [55, 145]]),
                ],
            ),
            GeoDataFrame(
                {
                    "id": [1, 3],
                    "attribute": [
                        "remains a polygon",
                        "section turns to line",
                    ],
                    "_old_ids": [
                        None,
                        None,
                    ],
                },
                geometry=[
                    box(0, 0, 100, 20),
                    MultiPolygon(
                        [
                            Polygon(
                                [
                                    [150, 30],
                                    [80, 30],
                                    [80, 130],
                                    [102.5, 130],
                                    [102.5, 133.49107142857142],
                                    [107.5, 133.49107142857142],
                                    [107.5, 130],
                                    [150, 130],
                                    [150, 30],
                                ]
                            )
                        ]
                    ),
                ],
            ),
            15,
        ),
    ],
    ids=[
        "no_thin_sections",
        "only_thin_sections",
        "mixed",
    ],
)
def test_thin_polygon_sections_to_lines(
    input_gdf: GeoDataFrame,
    expected_lines: GeoDataFrame,
    expected_polygons: GeoDataFrame,
    threshold: float,
):
    input_gdf = input_gdf.set_index("id")
    modified_lines, modified_polygons = thin_polygon_sections_to_lines(
        input_gdf,
        threshold=threshold,
        min_line_length=10,
        min_new_section_length=10,
        min_new_section_area=250,
        old_ids_column="_old_ids",
        centerline_segmentize_by=1,
    )
    expected_lines = expected_lines.set_index("id")
    expected_polygons = expected_polygons.set_index("id")

    assert_geodataframe_equal(modified_lines, expected_lines, check_like=True)
    assert_geodataframe_equal(modified_polygons, expected_polygons, check_like=True)


BUILDING_1 = box(0, 0, 10, 10)
BUILDING_2 = box(15, 0, 25, 10)
BUILDING_3 = box(100, 0, 110, 10)


@pytest.mark.parametrize(
    (
        "buildings",
        "buffer_distance",
        "erosion_distance",
        "simplification_distance",
        "expected",
    ),
    [
        pytest.param(
            GeoDataFrame(geometry=[BUILDING_1, BUILDING_2], crs="EPSG:3857"),
            5.0,
            2.0,
            2.0,
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            (0.85, -4.97),
                            (-4.82, 0.53),
                            (-2.05, 14.18),
                            (28.65, 13.01),
                            (27.01, -4.18),
                        ]
                    )
                ],
                crs="EPSG:3857",
            ),
            id="gdf_merges",
        ),
        pytest.param(
            GeoSeries([BUILDING_1, BUILDING_2, BUILDING_3], crs="EPSG:32633"),
            5.0,
            2.0,
            2.0,
            GeoSeries(
                [
                    Polygon(
                        [
                            (100.45, -4.84),
                            (95.15, -0.03),
                            (95.80, 12.03),
                            (111.95, 14.09),
                            (114.09, -1.99),
                        ]
                    ),
                    Polygon(
                        [
                            (0.85, -4.97),
                            (-4.82, 0.53),
                            (-2.05, 14.18),
                            (28.65, 13.01),
                            (27.01, -4.18),
                        ]
                    ),
                ],
                crs="EPSG:32633",
            ),
            id="geoseries_splits",
        ),
        pytest.param(
            np.array([BUILDING_1, BUILDING_2], dtype=object),
            5.0,
            2.0,
            2.0,
            np.array(
                [
                    Polygon(
                        [
                            (0.85, -4.97),
                            (-4.82, 0.53),
                            (-2.05, 14.18),
                            (28.65, 13.01),
                            (27.01, -4.18),
                        ]
                    )
                ],
                dtype=object,
            ),
            id="array_merges",
        ),
        pytest.param(
            GeoDataFrame(
                geometry=[
                    box(0, 0, 30, 10),
                    box(0, 20, 30, 30),
                    box(0, 10, 10, 20),
                    box(20, 10, 30, 20),
                ],
                crs="EPSG:3857",
            ),
            2.0,
            1.0,
            1.0,
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            (0.24, -1.97),
                            (-1.99, 0.41),
                            (-1.43, 30.46),
                            (30.44, 31.39),
                            (31.39, -0.46),
                        ]
                    )
                ],
                crs="EPSG:3857",
            ),
            id="interior_ring_filled",
        ),
        pytest.param(
            GeoDataFrame(geometry=[BUILDING_1, BUILDING_2], crs="EPSG:3857"),
            5.0,
            2.0,
            5.0,
            GeoDataFrame(
                geometry=[
                    Polygon(
                        [
                            (0.76, -4.98),
                            (-1.84, 14.12),
                            (29.09, 11.93),
                            (27.01, -4.18),
                        ]
                    )
                ],
                crs="EPSG:3857",
            ),
            id="simplification",
        ),
        pytest.param(
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            5.0,
            2.0,
            2.0,
            GeoDataFrame(geometry=[], crs="EPSG:4326"),
            id="empty_gdf",
        ),
        pytest.param(
            GeoSeries([], crs="EPSG:4326"),
            5.0,
            2.0,
            2.0,
            GeoSeries([], crs="EPSG:4326"),
            id="empty_geoseries",
        ),
        pytest.param(
            np.empty(0, dtype=object),
            5.0,
            2.0,
            2.0,
            np.empty(0, dtype=object),
            id="empty_array",
        ),
        pytest.param(
            GeoSeries([Polygon(), Polygon()], crs="EPSG:3857"),
            5.0,
            2.0,
            2.0,
            GeoSeries([], crs="EPSG:3857"),
            id="all_empty",
        ),
        pytest.param(
            GeoSeries([BUILDING_1], crs="EPSG:3857"),
            1.0,
            -10.0,
            2.0,
            GeoSeries([], crs="EPSG:3857"),
            id="erodes_to_empty",
        ),
    ],
)
def test_boffet_areas(
    buildings: GeoDataFrame | GeoSeries | np.ndarray,
    buffer_distance: float,
    erosion_distance: float,
    simplification_distance: float,
    expected: GeoDataFrame | GeoSeries | np.ndarray,
) -> None:
    result = boffet_areas(
        buildings=buildings,
        buffer_distance=buffer_distance,
        erosion_distance=erosion_distance,
        simplification_distance=simplification_distance,
    )

    assert type(result) is type(expected)

    if isinstance(expected, (GeoDataFrame, GeoSeries)):
        assert result.crs == expected.crs  # ty: ignore[unresolved-attribute]
        assert_geometries_equal(
            result.geometry.to_numpy(),  # ty: ignore[unresolved-attribute]
            expected.geometry.to_numpy(),
            tolerance=1e-2,
        )
    else:
        assert_geometries_equal(
            result,
            expected,
            tolerance=1e-2,
        )
