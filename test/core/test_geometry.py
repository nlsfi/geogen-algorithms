#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

import pytest
from geopandas import GeoDataFrame, gpd
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from shapely import GeometryCollection, MultiPolygon, box, from_wkt, to_wkt
from shapely.geometry import LineString, MultiLineString, MultiPoint, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.core.exceptions import (
    GeometryOperationError,
    GeometryTypeError,
)
from geogenalg.core.geometry import (
    LineExtendFrom,
    assign_nearest_z,
    elongation,
    explode_line,
    extend_line_to_nearest,
    extract_interior_rings,
    extract_interior_rings_gdf,
    lines_to_segments,
    mean_z,
    move_to_point,
    perforate_polygon_with_gdf_exteriors,
    rectangle_dimensions,
    scale_line_to_length,
)


@pytest.mark.parametrize(
    ("input_geometry", "gdf", "expected_geometry"),
    [
        (
            box(0, 0, 20, 20),
            GeoDataFrame(
                geometry=[
                    box(5, 5, 6, 6),
                    box(7, 7, 8, 8),
                ],
            ),
            Polygon(
                shell=[[0, 0], [0, 20], [20, 20], [20, 0], [0, 0]],
                holes=[
                    [[6, 5], [6, 6], [5, 6], [5, 5], [6, 5]],
                    [[8, 7], [8, 8], [7, 8], [7, 7], [8, 7]],
                ],
            ),
        ),
        (
            box(0, 0, 20, 20),
            GeoDataFrame(
                geometry=[
                    # Feature within ring
                    # Feature with ring
                    Polygon(
                        shell=[[1, 1], [1, 10], [10, 10], [10, 1], [1, 1]],
                        holes=[[[2, 2], [2, 9], [9, 9], [9, 2], [2, 2]]],
                    ),
                    Polygon(
                        shell=[[3, 3], [3, 8], [8, 8], [8, 3], [3, 3]],
                        holes=[[[4, 4], [4, 7], [7, 7], [7, 4], [4, 4]]],
                    ),
                    ##
                    # Another feature within ring
                    # Feature with ring
                    Polygon(
                        shell=[[11, 11], [11, 19], [19, 19], [19, 11], [11, 11]],
                        holes=[[[12, 12], [12, 18], [18, 18], [18, 12], [12, 12]]],
                    ),
                    # Feature within ring
                    Polygon(
                        shell=[[13, 13], [13, 17], [17, 17], [17, 13], [13, 13]],
                        holes=[[[14, 14], [14, 16], [16, 16], [16, 14], [14, 14]]],
                    ),
                    ##
                    # Regular standalone feature
                    box(1, 11, 9, 19),
                ],
            ),
            Polygon(
                shell=[[0, 0], [0, 20], [20, 20], [20, 0]],
                holes=[
                    [[1, 10], [1, 1], [10, 1], [10, 10], [1, 10]],
                    [[1, 11], [9, 11], [9, 19], [1, 19], [1, 11]],
                    [[19, 19], [11, 19], [11, 11], [19, 11], [19, 19]],
                ],
            ),
        ),
    ],
    ids=[
        "two exteriors",
        "recursive features",
    ],
)
def test_perforate_polygon_with_gdf_exteriors(
    input_geometry: Polygon,
    gdf: GeoDataFrame,
    expected_geometry: Polygon,
):
    assert (
        perforate_polygon_with_gdf_exteriors(input_geometry, gdf) == expected_geometry
    )


@pytest.mark.parametrize(
    ("input_geometry", "expected_geometry"),
    [
        (
            Polygon(
                shell=[[0, 0, 4.0], [0, 2, 5.5], [2, 2, 6.5], [2, 0, 1.5]],
                holes=[
                    [
                        [0.5, 0.5, 4.0],
                        [0.5, 1, 1.0],
                        [1, 1, 2.3],
                        [1, 0.5, 0.5],
                    ]
                ],
            ),
            MultiPolygon(
                [
                    Polygon(
                        shell=[
                            [0.5, 0.5, 4.0],
                            [0.5, 1, 1.0],
                            [1, 1, 2.3],
                            [1, 0.5, 0.5],
                        ],
                    )
                ]
            ),
        ),
        (
            Polygon(
                shell=[[0, 0], [0, 20], [20, 20], [20, 0]],
                holes=[
                    [
                        [0.5, 0.5],
                        [0.5, 1],
                        [1, 1],
                        [1, 0.5],
                    ],
                    [
                        [6, 6],
                        [6, 8],
                        [8, 8],
                        [8, 6],
                    ],
                ],
            ),
            MultiPolygon(
                [
                    Polygon(
                        shell=[
                            [0.5, 0.5],
                            [0.5, 1],
                            [1, 1],
                            [1, 0.5],
                        ],
                    ),
                    Polygon(
                        shell=[
                            [6, 6],
                            [6, 8],
                            [8, 8],
                            [8, 6],
                        ],
                    ),
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        shell=[[0, 0], [0, 20], [20, 20], [20, 0]],
                        holes=[
                            [[0.5, 0.5], [0.5, 1], [1, 1], [1, 0.5]],
                            [[6.5, 6.5], [6.5, 7.5], [7.5, 7.5], [7.5, 6.5]],
                        ],
                    )
                ]
            ),
            MultiPolygon(
                [
                    Polygon(
                        shell=[[0.5, 0.5], [0.5, 1], [1, 1], [1, 0.5]],
                    ),
                    Polygon(
                        shell=[[6.5, 6.5], [6.5, 7.5], [7.5, 7.5], [7.5, 6.5]],
                    ),
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        shell=[[0, 0], [0, 20], [20, 20], [20, 0]],
                        holes=[
                            [[0.5, 0.5], [0.5, 1], [1, 1], [1, 0.5]],
                        ],
                    ),
                    Polygon(
                        shell=[[100, 100], [100, 110], [110, 110], [110, 100]],
                        holes=[[[105, 105], [105, 106], [106, 106], [106, 105]]],
                    ),
                    box(-10, -10, -5, -5),
                ]
            ),
            MultiPolygon(
                [
                    Polygon(
                        shell=[[0.5, 0.5], [0.5, 1], [1, 1], [1, 0.5]],
                    ),
                    Polygon(
                        shell=[[105, 105], [105, 106], [106, 106], [106, 105]],
                    ),
                ]
            ),
        ),
        (
            Polygon(),
            MultiPolygon(),
        ),
        (
            box(10, 10, 20, 20),
            MultiPolygon(),
        ),
    ],
    ids=[
        "polygon, single ring",
        "polygon, two rings",
        "multipolygon, two rings",
        "multipolygon, three parts",
        "empty polygon",
        "no rings",
    ],
)
def test_extract_interior_rings(
    input_geometry: Polygon | MultiPolygon,
    expected_geometry: MultiPolygon,
):
    assert extract_interior_rings(input_geometry) == expected_geometry


@pytest.mark.parametrize(
    ("geometry", "expected_mean"),
    [
        (Point(0, 0, 1.0), 1.0),
        (MultiPoint([[0, 0, 0], [1.0, 1.0, 1.0]]), 0.5),
        (LineString([[0, 0, 10.0], [1.0, 1.0, 24.0]]), 17.0),
        (
            MultiLineString(
                [
                    LineString(
                        [
                            [0, 0, 10.0],
                            [1.0, 1.0, 24.0],
                        ]
                    ),
                    LineString(
                        [
                            [20.0, 40.0, 5.0],
                            [4.0, 60.0, 2.0],
                        ]
                    ),
                ]
            ),
            10.25,
        ),
        (
            Polygon(
                shell=[
                    [0, 0, 4.0],
                    [0, 2, 5.5],
                    [2, 2, 6.5],
                    [2, 0, 1.5],
                ],
                holes=[
                    [
                        [0.5, 0.5, 4.0],
                        [0.5, 1, 1.0],
                        [1, 1, 2.3],
                        [1, 0.5, 0.5],
                    ]
                ],
            ),
            3.1625,
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        shell=[
                            [0, 0, 0],
                            [0, 2, 0],
                            [2, 2, 0],
                            [2, 0, 0],
                        ],
                        holes=[
                            [
                                [0.5, 0.5, 0],
                                [0.5, 1, 0],
                                [1, 1, 0],
                                [1, 0.5, 0],
                            ]
                        ],
                    ),
                    Polygon(
                        shell=[
                            [0, 0, 1.0],
                            [0, 2, 2.0],
                            [2, 2, 3.0],
                            [2, 0, 5.0],
                        ],
                        holes=[
                            [
                                [0.5, 0.5, 1.0],
                                [0.5, 1, 9.0],
                                [1, 1, 2.3],
                                [1, 0.5, 4.5],
                            ]
                        ],
                    ),
                ]
            ),
            1.7375,
        ),
        (
            Polygon(
                shell=[
                    [0, 0, 4.0],
                    [0, 2, 5.5],
                    [2, 2, 6.6],
                    [2, 0, 1.0],
                ],
            ),
            4.275,
        ),
        (MultiPoint([[0, 0, -999.999], [1.0, 1.0, 1.0]]), 1.0),
    ],
    ids=[
        "point",
        "multipoint",
        "linestring",
        "multilinestring",
        "polygon",
        "multipolygon",
        "polygon, no hole",
        "nodata value",
    ],
)
def test_mean_z(geometry: BaseGeometry, expected_mean: float):
    assert mean_z(geometry) == expected_mean


@pytest.mark.parametrize(
    ("geom"),
    [
        (Point(0, 0)),
        (LineString([[0, 0], [0, 1]])),
    ],
    ids=[
        "point",
        "linestring",
    ],
)
def test_mean_z_no_z_geometry(geom: BaseGeometry):
    with pytest.raises(
        GeometryTypeError, match="Geometry does not include z coordinate!"
    ):
        mean_z(geom)


def test_mean_z_no_geometrycollection():
    with pytest.raises(
        GeometryTypeError,
        match="mean z calculation not implemented for GeometryCollection",
    ):
        mean_z(GeometryCollection([Point(0, 0, 0), Point(0, 0, 1)]))


def test_rectangle_dimensions():
    rect = from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 0, 0 0))")

    assert rectangle_dimensions(rect).width == 1.0
    assert rectangle_dimensions(rect).height == 4.0

    rect2 = from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 5, 1 0, 0 0))")
    with pytest.raises(GeometryOperationError):
        rectangle_dimensions(rect2)


def test_elongation():
    rect = from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 0, 0 0))")

    assert elongation(rect) == 4.0


def test_extract_interior_rings_gdf():
    polygon_with_holes = from_wkt(
        """POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0),
        (2 3, 3 3, 3 2, 2 2, 2 3),
        (4 5, 5 5, 5 4, 4 4, 4 5),
        (7 8, 8 8, 8 7, 7 7, 7 8))"""
    )

    gdf = gpd.GeoDataFrame(
        geometry=[polygon_with_holes],
        crs="EPSG:3857",
    )

    extracted_holes = extract_interior_rings_gdf(gdf)

    assert len(extracted_holes.index) == 3
    assert extracted_holes.crs == "EPSG:3857"

    assert extracted_holes.iloc[0].geometry.wkt == "POLYGON ((2 3, 3 3, 3 2, 2 2, 2 3))"
    assert extracted_holes.iloc[1].geometry.wkt == "POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))"
    assert extracted_holes.iloc[2].geometry.wkt == "POLYGON ((7 8, 8 8, 8 7, 7 7, 7 8))"


def test_explode_line_to_segments():
    line = from_wkt("LINESTRING (0 0, 0 1, 0 3, 0 6, 1 8, 2 9)")

    segments = explode_line(line)

    assert len(segments) == 5

    assert segments[0].wkt == "LINESTRING (0 0, 0 1)"
    assert segments[1].wkt == "LINESTRING (0 1, 0 3)"
    assert segments[2].wkt == "LINESTRING (0 3, 0 6)"
    assert segments[3].wkt == "LINESTRING (0 6, 1 8)"
    assert segments[4].wkt == "LINESTRING (1 8, 2 9)"


def test_lines_to_segments():
    lines = gpd.GeoSeries(
        [
            from_wkt("LINESTRING (0 0, 0 1)"),
            from_wkt("LINESTRING (10 0, 10 15)"),
            from_wkt("LINESTRING (20 20, 21 21, 22 22, 23 23)"),
        ]
    )

    segments = lines_to_segments(lines)

    assert len(segments) == 5

    assert segments[0].wkt == "LINESTRING (0 0, 0 1)"
    assert segments[1].wkt == "LINESTRING (10 0, 10 15)"
    assert segments[2].wkt == "LINESTRING (20 20, 21 21)"
    assert segments[3].wkt == "LINESTRING (21 21, 22 22)"
    assert segments[4].wkt == "LINESTRING (22 22, 23 23)"

    invalid_series = gpd.GeoSeries(
        [
            from_wkt("LINESTRING (0 0, 0 1)"),
            from_wkt("LINESTRING (10 0, 10 15)"),
            from_wkt("LINESTRING (20 20, 21 21, 22 22, 23 23)"),
            from_wkt("POINT (10 10)"),
        ]
    )

    with pytest.raises(GeometryTypeError):
        lines_to_segments(invalid_series)


def test_scale_line_to_length():
    line1 = from_wkt("LINESTRING (0 0, 0 4)")
    scaled_line1 = scale_line_to_length(line1, 4)

    line2 = from_wkt("LINESTRING (10 10, 14 14)")
    scaled_line2 = scale_line_to_length(line2, 10)

    line3 = from_wkt("LINESTRING (5 5, 3 -1)")
    scaled_line3 = scale_line_to_length(line3, 13)

    line4 = from_wkt("LINESTRING (0 0, 0 1, 1 1, 1 5, 1 7)")
    scaled_line4 = scale_line_to_length(line4, 2)

    assert scaled_line1.length == 4
    assert scaled_line2.length == 10
    assert scaled_line3.length == 13
    assert scaled_line4.length == 2

    # check with lower precision to help readability
    assert to_wkt(scaled_line1, 3) == "LINESTRING (0 0, 0 4)"
    assert to_wkt(scaled_line2, 3) == "LINESTRING (8.464 8.464, 15.536 15.536)"
    assert to_wkt(scaled_line3, 3) == "LINESTRING (6.055 8.166, 1.945 -4.166)"
    assert (
        to_wkt(scaled_line4, 3)
        == "LINESTRING (0.375 2.625, 0.375 2.875, 0.625 2.875, 0.625 3.875, 0.625 4.375)"
    )


def test_move_to_point():
    line = from_wkt("LINESTRING (10 10, 14 14, 16 16, 19 20)")
    moved_line = move_to_point(line, Point(0, 0))

    point = from_wkt("POINT (10 200)")
    moved_point = move_to_point(point, Point(0, 0))

    poly = from_wkt("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    moved_poly = move_to_point(poly, Point(0, 0))

    multiline = from_wkt("MULTILINESTRING ((20 21, 21 24), (30 30, 31 31))")
    moved_multiline = move_to_point(multiline, Point(0, 0))

    assert (
        to_wkt(moved_line, 3)
        == "LINESTRING (-4.668 -4.854, -0.668 -0.854, 1.332 1.146, 4.332 5.146)"
    )
    assert to_wkt(moved_point, 3) == "POINT (0 0)"
    assert (
        to_wkt(moved_poly, 3)
        == "POLYGON ((-0.5 -0.5, -0.5 0.5, 0.5 0.5, 0.5 -0.5, -0.5 -0.5))"
    )
    assert (
        to_wkt(moved_multiline, 3)
        == "MULTILINESTRING ((-3.59 -3.972, -2.59 -0.972), (6.41 5.028, 7.41 6.028))"
    )


def test_extend_line_to_nearest():
    line = from_wkt("LINESTRING (0 0, 1 1)")

    point = from_wkt("POINT (-10 -3)")
    linestring = from_wkt("LINESTRING (10 10, 10 2)")
    polygon = from_wkt("POLYGON ((0 2, 0 3, 2 3, 2 2, 0 2))")

    extended_from_end1 = extend_line_to_nearest(line, point, LineExtendFrom.END)
    extended_from_end2 = extend_line_to_nearest(line, linestring, LineExtendFrom.END)
    extended_from_end3 = extend_line_to_nearest(line, polygon, LineExtendFrom.END)

    assert extended_from_end1.wkt == "LINESTRING (0 0, 1 1, -10 -3)"
    assert extended_from_end2.wkt == "LINESTRING (0 0, 1 1, 10 2)"
    assert extended_from_end3.wkt == "LINESTRING (0 0, 1 1, 1 2)"

    extended_from_start1 = extend_line_to_nearest(line, point, LineExtendFrom.START)
    extended_from_start2 = extend_line_to_nearest(
        line, linestring, LineExtendFrom.START
    )
    extended_from_start3 = extend_line_to_nearest(line, polygon, LineExtendFrom.START)

    assert extended_from_start1.wkt == "LINESTRING (-10 -3, 0 0, 1 1)"
    assert extended_from_start2.wkt == "LINESTRING (1 1, 0 0, 10 2)"
    assert extended_from_start3.wkt == "LINESTRING (0 2, 0 0, 1 1)"

    extended_from_both1 = extend_line_to_nearest(line, point, LineExtendFrom.BOTH)
    extended_from_both2 = extend_line_to_nearest(line, linestring, LineExtendFrom.BOTH)
    extended_from_both3 = extend_line_to_nearest(line, polygon, LineExtendFrom.BOTH)

    assert extended_from_both1.wkt == "LINESTRING (-10 -3, 0 0, 1 1, -10 -3)"
    assert extended_from_both2.wkt == "LINESTRING (0 0, 1 1, 10 2, 0 0)"
    assert extended_from_both3.wkt == "LINESTRING (0 2, 0 0, 1 1, 1 2)"


@pytest.fixture
def assign_nearest_z_source_gdf() -> GeoDataFrame:
    # Corner points of a 3x3 square
    src_points = [
        Point(0.0, 0.0, 1.0),
        Point(3.0, 0.0, 2.0),
        Point(3.0, 3.0, 3.0),
        Point(0.0, 3.0, 4.0),
    ]
    return GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=src_points)


def _coords_df(gdf: GeoDataFrame, include_z: bool = True) -> DataFrame:
    return gdf.get_coordinates(include_z=include_z)


def test_assign_nearest_z_points_overwrite(assign_nearest_z_source_gdf: GeoDataFrame):
    # Define Z for some points to test that pre-existing Zs don't matter
    target_points = [
        Point(1.0, 1.0, 0.0),
        Point(2.0, 1.0, 0.0),
        Point(2.0, 2.0),
        Point(1.0, 2.0),
    ]
    target = GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=target_points)

    expected_points = [
        Point(1.0, 1.0, 1.0),
        Point(2.0, 1.0, 2.0),
        Point(2.0, 2.0, 3.0),
        Point(1.0, 2.0, 4.0),
    ]
    expected = GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=expected_points)

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=True)
    assert_frame_equal(_coords_df(out), _coords_df(expected))


def test_assign_nearest_z_points_add_only(assign_nearest_z_source_gdf: GeoDataFrame):
    # Define Z for some points to test that pre-existing Zs don't matter
    target_points = [
        Point(1.0, 1.0, 0.0),
        Point(2.0, 1.0, 0.0),
        Point(2.0, 2.0),
        Point(1.0, 2.0),
    ]
    target = GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=target_points)

    expected_points = [
        Point(1.0, 1.0, 0.0),
        Point(2.0, 1.0, 0.0),
        Point(2.0, 2.0, 3.0),
        Point(1.0, 2.0, 4.0),
    ]
    expected = GeoDataFrame({"id": [1, 2, 3, 4]}, geometry=expected_points)

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=False)
    assert_frame_equal(_coords_df(out), _coords_df(expected))


def test_assign_nearest_z_linestrings_overwrite(
    assign_nearest_z_source_gdf: GeoDataFrame,
):
    target_lines = [
        LineString([Point(1.0, 1.0, 0.0), Point(2.0, 1.0, 0.0)]),
        LineString([Point(2.0, 2.0), Point(1.0, 2.0)]),
    ]
    target = GeoDataFrame({"id": [1, 2]}, geometry=target_lines)

    expected_lines = [
        LineString([(1.0, 1.0, 1.0), (2.0, 1.0, 2.0)]),
        LineString([(2.0, 2.0, 3.0), (1.0, 2.0, 4.0)]),
    ]
    expected = GeoDataFrame({"id": [1, 2]}, geometry=expected_lines)

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=True)
    assert_frame_equal(_coords_df(out), _coords_df(expected))


def test_assign_nearest_z_linestrings_add_only(
    assign_nearest_z_source_gdf: GeoDataFrame,
):
    target_lines = [
        LineString([(1.0, 1.0), (2.0, 1.0)]),
        LineString([(2.0, 2.0), (1.0, 2.0)]),
    ]
    target = GeoDataFrame({"id": [1, 2]}, geometry=target_lines)

    expected_lines = [
        LineString([(1.0, 1.0, 1.0), (2.0, 1.0, 2.0)]),
        LineString([(2.0, 2.0, 3.0), (1.0, 2.0, 4.0)]),
    ]
    expected = GeoDataFrame({"id": [1, 2]}, geometry=expected_lines)

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=False)
    assert_frame_equal(_coords_df(out), _coords_df(expected))


def test_assign_nearest_z_polygon_add_only(assign_nearest_z_source_gdf: GeoDataFrame):
    target_polygon = Polygon(
        [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0), (1.0, 1.0)]
    )
    target = GeoDataFrame({"id": [1]}, geometry=[target_polygon])

    expected_polygon = Polygon(
        [
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 2.0),
            (2.0, 2.0, 3.0),
            (1.0, 2.0, 4.0),
            (1.0, 1.0, 1.0),
        ]
    )
    expected = GeoDataFrame({"id": [1]}, geometry=[expected_polygon])

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=False)
    assert_frame_equal(_coords_df(out), _coords_df(expected))


def test_assign_nearest_z_polygon_overwrite(assign_nearest_z_source_gdf: GeoDataFrame):
    target_polygon = Polygon(
        [
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
            (2.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (1.0, 1.0, 0.0),
        ]
    )
    target = GeoDataFrame({"id": [1]}, geometry=[target_polygon])

    expected_polygon = Polygon(
        [
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 2.0),
            (2.0, 2.0, 3.0),
            (1.0, 2.0, 4.0),
            (1.0, 1.0, 1.0),
        ]
    )
    expected = GeoDataFrame({"id": [1]}, geometry=[expected_polygon])

    out = assign_nearest_z(assign_nearest_z_source_gdf, target, overwrite_z=True)
    assert_frame_equal(_coords_df(out), _coords_df(expected))
