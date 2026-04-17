#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
import math
import re
from collections.abc import Callable
from typing import Literal

import pytest
from geopandas import GeoDataFrame, GeoSeries
from geopandas.testing import assert_geoseries_equal
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from shapely import (
    GeometryCollection,
    MultiPolygon,
    box,
    equals_exact,
    from_wkt,
    length,
    to_wkt,
    unary_union,
)
from shapely.geometry import (
    LinearRing,
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry, BaseMultipartGeometry

from geogenalg.core.exceptions import (
    GeometryOperationError,
    GeometryTypeError,
)
from geogenalg.core.geometry import (
    Dimensions,
    LineExtendFrom,
    add_topological_point,
    add_topological_points,
    assign_nearest_z,
    centerline_length,
    chaikin_smooth_keep_topology,
    chaikin_smooth_skip_coords,
    elongation,
    equalize_z,
    explode_line,
    extend_line_by,
    extend_line_to_nearest,
    extract_interior_rings,
    extract_interior_rings_gdf,
    get_topological_points,
    insert_vertex,
    largest_part,
    lines_to_segments,
    mean_z,
    move_to_point,
    oriented_envelope_dimensions,
    perforate_polygon_with_gdf_exteriors,
    point_on_line,
    polygon_rings_to_multilinestring,
    ramer_douglas_peucker_simplify_keep_coords,
    remove_line_segments_at_wide_sections,
    remove_small_parts,
    scale_line_to_length,
    segment_direction,
    smooth_around_connection_point_of_two_lines,
    smooth_around_ring_closing_vertex,
    snap_to_closest_vertex_or_segment,
    split_linear_geometry,
)


@pytest.mark.parametrize(
    ("input_geoseries", "extra_skip_coords", "expected_geoseries"),
    [
        (
            GeoSeries(),
            [],
            GeoSeries(),
        ),
        (
            GeoSeries(
                [
                    box(0, 0, 2, 2),
                    box(0, 2, 2, 4),
                    box(0, 4, 2, 5),
                ]
            ),
            None,
            GeoSeries(
                [
                    Polygon(
                        [
                            [2, 0.5],
                            [2, 2],
                            [0, 2],
                            [0, 0.5],
                            [0.5, 0],
                            [1.5, 0],
                            [2, 0.5],
                        ]
                    ),
                    box(0, 2, 2, 4),
                    Polygon(
                        [
                            [2, 4],
                            [2, 4.75],
                            [1.5, 5],
                            [0.5, 5],
                            [0, 4.75],
                            [0, 4],
                            [2, 4],
                        ]
                    ),
                ]
            ),
        ),
    ],
    ids=[
        "empty series",
        "middle box not smoothed",
    ],
)
def test_chaikin_smooth_keep_topology(
    input_geoseries: GeoSeries,
    extra_skip_coords: list[Point] | MultiPoint | None,
    expected_geoseries: GeoSeries,
):
    assert_geoseries_equal(
        chaikin_smooth_keep_topology(
            input_geoseries,
            1,
            extra_skip_coords=extra_skip_coords,
        ),
        expected_geoseries,
    )


@pytest.mark.parametrize(
    ("input_geometry", "skip_coords", "iterations", "expected_geometry"),
    [
        (
            LineString(
                [
                    [0, 0],
                    [1, 1],
                    [1, 2],
                ]
            ),
            [],
            1,
            LineString(
                [
                    [0, 0],
                    [0.25, 0.25],
                    [0.75, 0.75],
                    [1, 1.25],
                    [1, 1.75],
                    [1, 2],
                ]
            ),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [1, 1],
                    [1, 2],
                ]
            ),
            [Point(1, 1)],
            1,
            LineString(
                [
                    [0, 0],
                    [0.25, 0.25],
                    [1, 1],
                    [1, 1.75],
                    [1, 2],
                ]
            ),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [1, 1],
                    [1, 2],
                ]
            ),
            [Point(1, 1)],
            2,
            LineString(
                [
                    [0, 0],
                    [0.0625, 0.0625],
                    [0.1875, 0.1875],
                    [0.4375, 0.4375],
                    [1, 1],
                    [1, 1.5625],
                    [1, 1.8125],
                    [1, 1.9375],
                    [1, 2],
                ]
            ),
        ),
        (
            box(0, 0, 1, 1),
            [],
            1,
            Polygon(
                [
                    [1, 0.25],
                    [1, 0.75],
                    [0.75, 1],
                    [0.25, 1],
                    [0, 0.75],
                    [0, 0.25],
                    [0.25, 0],
                    [0.75, 0],
                    [1, 0.25],
                ]
            ),
        ),
        (
            box(0, 0, 1, 1),
            [Point(0, 0)],
            1,
            Polygon(
                [
                    [1, 0.25],
                    [1, 0.75],
                    [0.75, 1],
                    [0.25, 1],
                    [0, 0.75],
                    [0, 0],
                    [0.75, 0],
                    [1, 0.25],
                ]
            ),
        ),
        (
            box(0, 0, 1, 1),
            MultiPoint([Point(0, 0), Point(1, 1)]),
            1,
            Polygon(
                [
                    [1, 0.25],
                    [1, 1],
                    [0.25, 1],
                    [0, 0.75],
                    [0, 0],
                    [0.75, 0],
                    [1, 0.25],
                ]
            ),
        ),
    ],
    ids=[
        "three point linestring, no skip",
        "three point linestring, skip one",
        "three point linestring, skip one, two iterations",
        "square polygon, no skip",
        "square polygon, skip one",
        "square polygon, skip two",
    ],
)
def test_chaikin_smooth_skip_coords(
    input_geometry: LineString | Polygon,
    skip_coords: list[Point] | MultiPoint,
    iterations: int,
    expected_geometry: LineString | Polygon,
):
    assert (
        chaikin_smooth_skip_coords(
            input_geometry,
            skip_coords,
            iterations=iterations,
        )
        == expected_geometry
    )


@pytest.mark.parametrize(
    ("input_geoseries", "expected_points"),
    [
        (
            GeoSeries(),
            [],
        ),
        (
            GeoSeries(Point(0, 0)),
            [],
        ),
        (
            GeoSeries(
                [
                    LineString([[0, 0], [0, 1]]),
                    LineString([[0, 0], [0, -1]]),
                ]
            ),
            [Point(0, 0)],
        ),
        (
            GeoSeries(
                [
                    LineString([[0, 0], [0, 1]]),
                    LineString([[0, 0], [0, -1]]),
                    LineString([[0, 0], [-1, 0]]),
                ]
            ),
            [Point(0, 0)],
        ),
        (
            GeoSeries(
                [
                    box(0, 0, 2, 2),
                    box(0, 2, 2, 4),
                    box(0, 4, 2, 5),
                ]
            ),
            [
                Point(0, 2),
                Point(0, 4),
                Point(2, 2),
                Point(2, 4),
            ],
        ),
    ],
    ids=[
        "empty series",
        "no points",
        "two geoms, one shared point",
        "three geoms, one shared point",
        "polygons, many shared points",
    ],
)
def test_get_topological_points(
    input_geoseries: GeoSeries,
    expected_points: list[Point],
):
    assert get_topological_points(input_geoseries) == expected_points


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


@pytest.mark.parametrize(
    ("geom", "expected_dimensions"),
    [
        (box(0, 0, 1, 1), Dimensions(width=1, height=1)),
        (box(0, 1, 2, 2), Dimensions(width=1.0, height=2.0)),
        (
            Polygon(
                [
                    [0, 0],
                    [2, 0],
                    [2, 1],
                    [4, 1],
                    [4, -2],
                    [2, -2],
                    [2, -1],
                    [0, -1],
                    [0, 0],
                ],
            ),
            Dimensions(width=3.0, height=4.0),
        ),
    ],
    ids=[
        "square",
        "rectangle",
        "irregular polygon",
    ],
)
def test_oriented_envelope_dimensions(geom: Polygon, expected_dimensions: Dimensions):
    assert oriented_envelope_dimensions(geom) == expected_dimensions


@pytest.mark.parametrize(
    ("polygon", "expected_elongation"),
    [
        (Polygon([[0, 0], [0, 4], [1, 4], [1, 0], [0, 0]]), 0.25),
        (Polygon([[0, 0], [0, 2], [2, 2], [2, 0], [0, 0]]), 1.0),
        (
            Polygon(
                [
                    [0, 0],
                    [2, 0],
                    [2, 1],
                    [4, 1],
                    [4, -2],
                    [2, -2],
                    [2, -1],
                    [0, -1],
                    [0, 0],
                ]
            ),
            0.75,
        ),
        (Polygon([[0, 0], [0, 19], [1, 19], [1, 0], [0, 0]]), 0.052631579),
    ],
    ids=[
        "elongated",
        "square",
        "irregular shape",
        "very elongated",
    ],
)
def test_elongation(polygon: Polygon, expected_elongation: float):
    assert math.isclose(elongation(polygon), expected_elongation, rel_tol=1e-08)


def test_extract_interior_rings_gdf():
    polygon_with_holes = from_wkt(
        """POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0),
        (2 3, 3 3, 3 2, 2 2, 2 3),
        (4 5, 5 5, 5 4, 4 4, 4 5),
        (7 8, 8 8, 8 7, 7 7, 7 8))"""
    )

    gdf = GeoDataFrame(
        geometry=[polygon_with_holes],
        crs="EPSG:3857",
    )

    extracted_holes = extract_interior_rings_gdf(gdf)

    assert len(extracted_holes.index) == 3
    assert extracted_holes.crs == "EPSG:3857"

    assert extracted_holes.iloc[0].geometry.wkt == "POLYGON ((2 3, 3 3, 3 2, 2 2, 2 3))"
    assert extracted_holes.iloc[1].geometry.wkt == "POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))"
    assert extracted_holes.iloc[2].geometry.wkt == "POLYGON ((7 8, 8 8, 8 7, 7 7, 7 8))"


@pytest.mark.parametrize(
    ("input_line", "expected_line"),
    [
        (
            LineString(
                [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [3, 0],
                ]
            ),
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0]]),
                    LineString([[1, 0], [2, 0]]),
                    LineString([[2, 0], [3, 0]]),
                ]
            ),
        ),
        (
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0], [2, 0]]),
                    LineString([[4, 4], [5, 4], [6, 4]]),
                ]
            ),
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0]]),
                    LineString([[1, 0], [2, 0]]),
                    LineString([[4, 4], [5, 4]]),
                    LineString([[5, 4], [6, 4]]),
                ]
            ),
        ),
    ],
    ids=[
        "single linestring",
        "multilinestring",
    ],
)
def test_explode_line(
    input_line: LineString | MultiLineString,
    expected_line: MultiLineString,
):
    assert explode_line(input_line) == expected_line


def test_lines_to_segments():
    lines = GeoSeries(
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

    invalid_series = GeoSeries(
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


@pytest.mark.parametrize(
    ("input_line", "extend_to", "extend_from", "tolerance", "expected_line"),
    [
        (
            LineString([[0, 0], [1, 1]]),
            Point(5, 5),
            LineExtendFrom.END,
            0.0,  # tolerance
            LineString([[0, 0], [1, 1], [5, 5]]),  # expected
        ),
        (
            LineString([[0, 0], [1, 1]]),
            Point(150, 150),
            LineExtendFrom.END,
            10.0,  # tolerance
            LineString([[0, 0], [1, 1]]),  # expected
        ),
        (
            LineString([[0, 0], [1, 1]]),
            Point(-1, -1),
            LineExtendFrom.START,
            0.0,  # tolerance
            LineString([[-1, -1], [0, 0], [1, 1]]),  # expected
        ),
        (
            LineString([[0, 0], [1, 0]]),
            Point(0.5, 0.5),
            LineExtendFrom.BOTH,
            0.0,  # tolerance
            LineString([[0, 0], [1, 0], [0.5, 0.5], [0, 0]]),  # expected
        ),
        (
            LineString([[0, 0], [1, 0], [0.5, 0.5], [0, 0]]),
            Point(0.5, 5),
            LineExtendFrom.START,
            0.0,  # tolerance
            LineString(
                [[0, 0], [1, 0], [0.5, 0.5], [0, 0]]
            ),  # expected (shouldn't change)
        ),
        (
            LineString([[0.5, 11], [0.5, 10]]),
            LineString([[0, 0], [1, 0]]),
            LineExtendFrom.END,
            0.0,  # tolerance
            LineString([[0.5, 11], [0.5, 10], [0.5, 0]]),  # expected
        ),
        (
            LineString([[0.5, 11], [0.5, 10]]),
            box(0, 0, 1, -1),
            LineExtendFrom.END,
            0.0,  # tolerance
            LineString([[0.5, 11], [0.5, 10], [0.5, 0]]),  # expected
        ),
        (
            LineString([[0, 0], [1, 0], [1, 1], [0.5, 1]]),
            LineString([[0, -1], [1, -1]]),
            LineExtendFrom.END,
            0.0,  # tolerance
            LineString(
                [[0, 0], [1, 0], [1, 1], [0.5, 1]]
            ),  # expected (shouldn't change)
        ),
        (
            LineString([[0.5, 0.5], [1, 0.5], [1, 0]]),
            Point(0.5, 0.5),
            LineExtendFrom.END,
            0.0,  # tolerance
            LineString([[0.5, 0.5], [1, 0.5], [1, 0], [0.5, 0.5]]),  # expected
        ),
        (
            LineString([[1, 0.5], [1, 0], [0.5, 0.5]]),
            Point(0.5, 0.5),
            LineExtendFrom.START,
            0.0,  # tolerance
            LineString([[0.5, 0.5], [1, 0.5], [1, 0], [0.5, 0.5]]),  # expected
        ),
    ],
    ids=[
        "line_to_point-end-no_tolerance",
        "line_to_point-end-tolerance",
        "line_to_point-start-no_tolerance",
        "line_to_point-both-no_tolerance",
        "linear_ring",
        "line_to_line",
        "line_to_polygon",
        "crosses",
        "extend_to_form_ring-end",
        "extend_to_form_ring-start",
    ],
)
def test_extend_line_to_nearest(
    input_line: LineString,
    extend_to: BaseGeometry,
    extend_from: LineExtendFrom,
    tolerance: float,
    expected_line: LineString,
):
    assert (
        extend_line_to_nearest(
            input_line,
            extend_to,
            extend_from,
            tolerance,
        )
        == expected_line
    )


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


@pytest.mark.parametrize(
    ("input_polygon", "expected_length", "exterior_only"),
    [
        (box(0, 0, 10, 200), 190.0, False),
        (box(0, 0, 10, 200), 190.0, True),
        (
            Polygon(
                shell=[
                    [0, 0],
                    [10, 0],
                    [10, 200],
                    [0, 200],
                    [0, 0],
                ],
                holes=[
                    [
                        [1, 1],
                        [9, 1],
                        [9, 199],
                        [1, 199],
                        [1, 1],
                    ]
                ],
            ),
            416.0,
            False,
        ),
        (
            Polygon(
                shell=[
                    [0, 0],
                    [10, 0],
                    [10, 200],
                    [0, 200],
                    [0, 0],
                ],
                holes=[
                    [
                        [1, 1],
                        [9, 1],
                        [9, 199],
                        [1, 199],
                        [1, 1],
                    ]
                ],
            ),
            190.0,
            True,
        ),
        (
            Polygon(),
            0.0,
            True,
        ),
    ],
    ids=[
        "polygon, no holes, no exterior only",
        "polygon, no holes, exterior only",
        "polygon, holes, no exterior only",
        "polygon, holes, exterior only",
        "empty polygon",
    ],
)
def test_centerline_length(
    input_polygon: Polygon,
    expected_length: float,
    exterior_only: bool,
):
    assert (
        centerline_length(input_polygon, exterior_only=exterior_only) == expected_length
    )


@pytest.mark.parametrize(
    ("input_geometry", "expected_geometry", "size_function"),
    [
        (
            MultiLineString(
                [
                    LineString([[0, 0], [0, 1]]),
                    LineString([[50, 50], [1000, 50]]),
                ]
            ),
            LineString([[50, 50], [1000, 50]]),
            None,
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 2, 2),
                ]
            ),
            box(0, 0, 2, 2),
            None,
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 10, 10),
                    box(20, 0, 21, 20),  # smaller area, larger perimeter
                ]
            ),
            box(20, 0, 21, 20),
            length,
        ),
        (
            box(0, 0, 10, 10),
            box(0, 0, 10, 10),
            None,
        ),
    ],
    ids=[
        "multiline_default_size_function",
        "multipoly_default_size_function",
        "multipoly_different_size_function",
        "single_poly",
    ],
)
def test_largest_part(
    input_geometry: BaseMultipartGeometry,
    expected_geometry: BaseGeometry,
    size_function: Callable[[BaseGeometry], float] | None,
):
    assert (
        largest_part(input_geometry, size_function=size_function) == expected_geometry
    )


@pytest.mark.parametrize(
    ("geom", "error", "match"),
    [
        (MultiPoint(), TypeError, re.escape("Can't reduce (Multi)Point to largest.")),
        (Point(), TypeError, re.escape("Can't reduce (Multi)Point to largest.")),
        (
            GeometryCollection(),
            NotImplementedError,
            re.escape("Not implemented for geometry collection."),
        ),
    ],
    ids=[
        "type_error_mp",
        "type_error_point",
        "notimplemented",
    ],
)
def test_largest_part_raises(
    geom: BaseGeometry,
    error: type[Exception],
    match: str,
):
    with pytest.raises(error, match=match):
        largest_part(geom)


@pytest.mark.parametrize(
    ("input_geometry", "threshold", "expected_geometry", "size_function"),
    [
        (
            MultiLineString(
                [
                    LineString([[0, 0], [0, 1]]),
                    LineString([[50, 50], [1000, 50]]),
                ]
            ),
            50,
            MultiLineString(
                [
                    LineString([[50, 50], [1000, 50]]),
                ]
            ),
            None,
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 1, 1),
                    box(0, 0, 2, 2),
                ]
            ),
            1.5,
            MultiPolygon([box(0, 0, 2, 2)]),
            None,
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 10, 10),
                    box(20, 0, 21, 20),  # smaller area, larger perimeter
                ]
            ),
            40.5,
            MultiPolygon([box(20, 0, 21, 20)]),
            length,
        ),
        (
            box(0, 0, 1, 1),
            0.5,
            box(0, 0, 1, 1),
            None,
        ),
        (
            box(0, 0, 1, 1),
            5,
            Polygon(),
            None,
        ),
        (
            LineString([[0, 0], [0, 1]]),
            0.5,
            LineString([[0, 0], [0, 1]]),
            None,
        ),
        (
            LineString([[0, 0], [0, 1]]),
            5,
            LineString(),
            None,
        ),
    ],
    ids=[
        "multiline, default size function",
        "multipoly, default size function",
        "multipoly, different size function",
        "single poly, over threshold",
        "single poly, under threshold",
        "single line, over threshold",
        "single line, under threshold",
    ],
)
def test_remove_small_parts(
    input_geometry: BaseGeometry,
    threshold: float,
    expected_geometry: BaseGeometry,
    size_function: Callable[[BaseGeometry], float] | None,
):
    assert (
        remove_small_parts(
            input_geometry,
            threshold,
            size_function=size_function,
        )
        == expected_geometry
    )


@pytest.mark.parametrize(
    ("line", "mask", "thinness_threshold", "expected"),
    [
        (
            LineString([[0, 0], [1, 0], [2, 0], [3, 0]]),
            unary_union(
                [
                    box(0, -0.5, 1, 0.5),
                    box(1, -1, 2, 1),
                    box(2, -0.25, 3, 0.25),
                ]
            ),
            0.75,
            LineString([[2, 0], [3, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0], [3, 0]]),
            unary_union(
                [
                    box(0, -0.5, 1, 0.5),
                    box(1, -0.25, 2, 0.25),
                    box(2, -0.25, 3, 0.25),
                ]
            ),
            0.75,
            LineString([[1, 0], [2, 0], [3, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0], [3, 0]]),
            unary_union(
                [
                    box(0, -0.25, 1, 0.25),
                    box(1, -0.5, 2, 0.5),
                    box(2, -0.25, 3, 0.25),
                ]
            ),
            0.75,
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0]]),
                    LineString([[2, 0], [3, 0]]),
                ]
            ),
        ),
    ],
    ids=[
        "keep one segment",
        "keep two segments",
        "keep disjoint segments",
    ],
)
def test_remove_line_segments_at_wide_sections(
    line: LineString | MultiLineString,
    mask: Polygon | MultiPolygon,
    thinness_threshold: float,
    expected: LineString | MultiLineString,
):
    assert (
        remove_line_segments_at_wide_sections(
            line,
            mask,
            thinness_threshold,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (
            LineString([[1, 0], [0, 0]]),
            90.0,
        ),
        (
            LineString([[1, 1], [0, 0]]),
            45.0,
        ),
        (
            LineString([[0, 1], [0, 0]]),
            0.0,
        ),
        (
            LineString([[-1, 1], [0, 0]]),
            315.0,
        ),
        (
            LineString([[-1, 0], [0, 0]]),
            270.0,
        ),
        (
            LineString([[-1, -1], [0, 0]]),
            225.0,
        ),
        (
            LineString([[0, -1], [0, 0]]),
            180.0,
        ),
        (
            LineString([[1, -1], [0, 0]]),
            135.0,
        ),
        (
            LineString([[600, 500], [500, 500]]),
            90.0,
        ),
        (
            LineString([[600, 500, 100], [500, 500, 24]]),
            90.0,
        ),
    ],
    ids=[
        "90 degrees",
        "45 degrees",
        "0 degrees",
        "315 degrees",
        "270 degrees",
        "215 degrees",
        "180 degrees",
        "135 degrees",
        "90 degrees, different location and scale",
        "segment with Z",
    ],
)
def test_segment_direction(geom: LineString, expected: float):
    assert segment_direction(geom) == expected


@pytest.mark.parametrize(
    ("geom", "msg"),
    [
        (
            LineString([[1, 0], [0, 0], [-1, 0]]),
            "Input geometry must have two vertexes",
        ),
        (
            LineString(),
            "Input geometry must have two vertexes",
        ),
        (
            LineString([[0, 0], [0, 0]]),
            "Segment has duplicate vertexes.",
        ),
    ],
    ids=[
        "empty",
        "three vertices",
        "duplicate",
    ],
)
def test_segment_direction_raises(geom: LineString, msg: str):
    with pytest.raises(GeometryOperationError, match=re.escape(msg)):
        segment_direction(geom)


@pytest.mark.parametrize(
    ("geom", "method", "expected"),
    [
        (
            Point(0, 0, 1),
            "min",
            Point(0, 0, 1),
        ),
        (
            MultiPoint(
                [
                    Point(0, 0, 1),
                    Point(2, 2, 2),
                    Point(2, 4, 6),
                ]
            ),
            "min",
            MultiPoint(
                [
                    Point(0, 0, 1),
                    Point(2, 2, 1),
                    Point(2, 4, 1),
                ]
            ),
        ),
        (
            LineString([[0, 0, 1], [1, 0, 0]]),
            "min",
            LineString([[0, 0, 0], [1, 0, 0]]),
        ),
        (
            MultiLineString(
                [
                    LineString([[0, 0, 1], [1, 0, 0]]),
                    LineString([[5, 5, 10], [6, 6, 12]]),
                ]
            ),
            "max",
            MultiLineString(
                [
                    LineString([[0, 0, 12], [1, 0, 12]]),
                    LineString([[5, 5, 12], [6, 6, 12]]),
                ]
            ),
        ),
        (
            Polygon([[0, 0, 0], [1, 0, 1], [1, 1, 2], [0, 1, 3]]),
            "max",
            Polygon([[0, 0, 3], [1, 0, 3], [1, 1, 3], [0, 1, 3]]),
        ),
        (
            Polygon(
                shell=[[0, 0, 0], [1, 0, 1], [1, 1, 2], [0, 1, 3]],
                holes=[
                    [
                        [0.25, 0.25, 4],
                        [0.75, 0.25, 5],
                        [0.75, 0.75, 6],
                        [0.25, 0.75, 7],
                    ]
                ],
            ),
            "max",
            Polygon(
                shell=[[0, 0, 7], [1, 0, 7], [1, 1, 7], [0, 1, 7]],
                holes=[
                    [
                        [0.25, 0.25, 7],
                        [0.75, 0.25, 7],
                        [0.75, 0.75, 7],
                        [0.25, 0.75, 7],
                    ]
                ],
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        [
                            [0, 0, 0],
                            [1, 0, 1],
                            [1, 1, 2],
                            [0, 1, 3],
                        ]
                    ),
                    Polygon(
                        [
                            [10, 10, 3],
                            [11, 10, 4],
                            [11, 11, 5],
                            [10, 11, 6],
                        ]
                    ),
                ]
            ),
            "max",
            MultiPolygon(
                [
                    Polygon(
                        [
                            [0, 0, 6],
                            [1, 0, 6],
                            [1, 1, 6],
                            [0, 1, 6],
                        ]
                    ),
                    Polygon(
                        [
                            [10, 10, 6],
                            [11, 10, 6],
                            [11, 11, 6],
                            [10, 11, 6],
                        ]
                    ),
                ]
            ),
        ),
        (
            LinearRing([[0, 0, 0], [1, 0, 1], [1, 1, 2], [0, 1, 3]]),
            "min",
            LinearRing([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        ),
        (
            LinearRing([[0, 0], [1, 0], [1, 1], [0, 1]]),
            "min",
            LinearRing([[0, 0], [1, 0], [1, 1], [0, 1]]),
        ),
    ],
    ids=[
        "point",
        "multipoint",
        "linestring",
        "multilinestring",
        "polygon",
        "polygon_with_hole",
        "multipolygon",
        "linearring",
        "2d",
    ],
)
def test_equalize_z(
    geom: BaseGeometry,
    method: Literal["min", "max"],
    expected: BaseGeometry,
):
    assert equalize_z(geom, method=method) == expected


def test_equalize_z_raises():
    with pytest.raises(
        NotImplementedError,
        match=re.escape("Function not implemented for geom type: GeometryCollection"),
    ):
        equalize_z(GeometryCollection([Point(0, 0, 1)]), method="min")


@pytest.mark.parametrize(
    ("geom", "expected"),
    [
        (
            box(0, 0, 1, 1),
            MultiLineString(
                [
                    box(0, 0, 1, 1).exterior,
                ]
            ),
        ),
        (
            Polygon(
                shell=box(-2, -2, 2, 2).exterior.coords,
                holes=[
                    box(0, 0, 1, 1).exterior.coords,
                ],
            ),
            MultiLineString(
                [
                    box(-2, -2, 2, 2).exterior,
                    box(0, 0, 1, 1).exterior,
                ]
            ),
        ),
        (
            Polygon(
                shell=box(-20, -20, 20, 20).exterior.coords,
                holes=[
                    box(0, 0, 1, 1).exterior.coords,
                    box(5, 5, 8, 8).exterior.coords,
                ],
            ),
            MultiLineString(
                [
                    box(-20, -20, 20, 20).exterior,
                    box(0, 0, 1, 1).exterior,
                    box(5, 5, 8, 8).exterior,
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 1, 1),
                ]
            ),
            MultiLineString(
                [
                    box(0, 0, 1, 1).exterior,
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    box(0, 0, 1, 1),
                    box(5, 5, 6, 6),
                ]
            ),
            MultiLineString(
                [
                    box(0, 0, 1, 1).exterior,
                    box(5, 5, 6, 6).exterior,
                ]
            ),
        ),
        (
            MultiPolygon(
                [
                    Polygon(
                        shell=box(-20, -20, 20, 20).exterior.coords,
                        holes=[
                            box(0, 0, 1, 1).exterior.coords,
                        ],
                    ),
                    Polygon(
                        shell=box(50, 50, 60, 60).exterior.coords,
                        holes=[
                            box(55, 55, 57, 57).exterior.coords,
                        ],
                    ),
                ]
            ),
            MultiLineString(
                [
                    box(-20, -20, 20, 20).exterior.coords,
                    box(0, 0, 1, 1).exterior.coords,
                    box(50, 50, 60, 60).exterior.coords,
                    box(55, 55, 57, 57).exterior.coords,
                ]
            ),
        ),
    ],
    ids=[
        "single_polygon_no_holes",
        "single_polygon_one_hole",
        "single_polygon_multiple_holes",
        "multipolygon_single_part",
        "multipolygon_multiple_parts",
        "multipolygon_with_holes",
    ],
)
def test_polygon_rings_to_multilinestring(
    geom: Polygon | MultiLineString,
    expected: MultiLineString,
):
    assert polygon_rings_to_multilinestring(geom) == expected


@pytest.mark.parametrize(
    ("geom", "split_with", "expected"),
    [
        (  # split_line_with_point_at_segment
            LineString([[0, 0], [1, 0]]),
            Point(0.5, 0),
            MultiLineString(
                [
                    LineString([[0, 0], [0.5, 0]]),
                    LineString([[0.5, 0], [1, 0]]),
                ]
            ),
        ),
        (  # split_line_with_line_at_segment
            LineString([[0, 0], [1, 0]]),
            LineString([[0.5, -1], [0.5, 1]]),
            MultiLineString(
                [
                    LineString([[0, 0], [0.5, 0]]),
                    LineString([[0.5, 0], [1, 0]]),
                ]
            ),
        ),
        (  # split_line_with_line_at_vertex
            LineString([[0, 0], [0.5, 0], [1, 0]]),
            LineString([[0.5, -1], [0.5, 1]]),
            MultiLineString(
                [
                    LineString([[0, 0], [0.5, 0]]),
                    LineString([[0.5, 0], [1, 0]]),
                ]
            ),
        ),
        (  # split_ring_with_single_point_at_segment_no_change
            LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            Point(0.5, 0),
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
                ]
            ),
        ),
        (  # split_ring_with_line_at_segment
            LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            LineString([[0.5, -5], [0.5, 5]]),
            MultiLineString(
                [
                    LineString([[0.5, 1], [0, 1], [0, 0], [0.5, 0]]),
                    LineString([[0.5, 0], [1, 0], [1, 1], [0.5, 1]]),
                ]
            ),
        ),
        (  # split_ring_with_line_at_vertexes
            LineString([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]),
            LineString([[-5, -5], [5, 5]]),
            MultiLineString(
                [
                    LineString([[0, 0], [1, 0], [1, 1]]),
                    LineString([[1, 1], [0, 1], [0, 0]]),
                ]
            ),
        ),
        (  # split_ring_with_polygon_at_segment
            LineString([[0, 0], [10, 0]]),
            box(2, -2, 6, 6),
            MultiLineString(
                [
                    LineString([[0, 0], [2, 0]]),
                    LineString([[2, 0], [6, 0]]),
                    LineString([[6, 0], [10, 0]]),
                ]
            ),
        ),
    ],
    ids=[
        "split_line_with_point_at_segment",
        "split_line_with_line_at_segment",
        "split_line_with_line_at_vertex",
        "split_ring_with_single_point_at_segment_no_change",
        "split_ring_with_line_at_segment",
        "split_ring_with_line_at_vertexes",
        "split_ring_with_polygon_at_segment",
    ],
)
def test_split_linear_geometry(
    geom: LineString | LinearRing,
    split_with: BaseGeometry,
    expected: MultiLineString,
):
    assert split_linear_geometry(geom, split_with) == expected


@pytest.mark.parametrize(
    ("line", "distance", "expected"),
    [
        (LineString([[0, 0], [1, 0]]), 10, Point(11, 0)),
        (LineString([[0, 0], [1, 0]]), -10, Point(-10, 0)),
        (LineString([[0, 0], [1, 1]]), -1.5, Point(-1.0606, -1.0606)),
    ],
    ids=[
        "from_end",
        "from_start",
        "diagonal",
    ],
)
def test_point_on_line(
    line: LineString,
    distance: float,
    expected: Point,
):
    assert equals_exact(point_on_line(line, distance), expected, tolerance=0.5)


def test_point_on_line_raises():
    with pytest.raises(ValueError, match=re.escape("Line must only have two vertices")):
        point_on_line(
            LineString([[0, 0], [1, 0], [2, 0]]),
            10,
        )


@pytest.mark.parametrize(
    ("line", "extend_by", "extend_from", "expected"),
    [
        (
            LineString([[0, 0], [1, 0]]),
            1.0,
            LineExtendFrom.END,
            LineString([[0, 0], [1, 0], [2, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            1.0,
            LineExtendFrom.START,
            LineString([[-1, 0], [0, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            1.0,
            LineExtendFrom.BOTH,
            LineString([[-1, 0], [0, 0], [1, 0], [2, 0]]),
        ),
    ],
    ids=[
        "end",
        "start",
        "both",
    ],
)
def test_extend_line_by(
    line: LineString,
    extend_by: float,
    extend_from: LineExtendFrom,
    expected: LineString,
):
    assert extend_line_by(line, extend_by, extend_from) == expected


def test_extend_line_by_raises():
    with pytest.raises(
        ValueError, match=re.escape("Extension distance must be above zero.")
    ):
        extend_line_by(
            LineString(),
            -10,
            extend_from=LineExtendFrom.START,
        )

    with pytest.raises(
        GeometryOperationError, match=re.escape("Can't extend line with no length.")
    ):
        extend_line_by(
            LineString(),
            10,
            extend_from=LineExtendFrom.START,
        )


@pytest.mark.parametrize(
    ("geom", "vertex", "index", "expected"),
    [
        (
            LineString([[0, 0], [1, 0]]),
            Point(0.5, 0),
            1,
            LineString([[0, 0], [0.5, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            Point(2.0, 0),
            2,
            LineString([[0, 0], [1, 0], [2, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [1, 1], [2, 3]]),
            Point(1.75, 3),
            3,
            LineString([[0, 0], [1, 0], [1, 1], [1.75, 3], [2, 3]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            Point(0.5, 0, 5),
            1,
            LineString([[0, 0], [0.5, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0, 1], [1, 0, 2]]),
            Point(0.5, 0, 5),
            1,
            LineString([[0, 0, 1], [0.5, 0, 5], [1, 0, 2]]),
        ),
        (
            LineString([[0, 0, 1], [1, 0, 2]]),
            Point(0.5, 0),
            1,
            LineString([[0, 0, 1], [0.5, 0, 1.5], [1, 0, 2]]),
        ),
        (
            LineString([[0, 0, 1], [1, 0, 2]]),
            Point(0.5, 0),
            0,
            LineString([[0.5, 0, 1], [0, 0, 1], [1, 0, 2]]),
        ),
        (
            LineString([[0, 0, 1], [1, 0, 2]]),
            Point(1, 1),
            2,
            LineString([[0, 0, 1], [1, 0, 2], [1, 1, 2]]),
        ),
    ],
    ids=[
        "one_segment",
        "at_end",
        "multiple_segments",
        "vertex_has_z_line_does_not",
        "vertex_has_z_so_does_line",
        "line_has_z_vertex_does_not",
        "line_has_z_vertex_does_not_start",
        "line_has_z_vertex_does_not_end",
    ],
)
def test_insert_vertex(
    geom: LineString,
    vertex: Point,
    index: int,
    expected: LineString,
):
    assert insert_vertex(geom, vertex, index) == expected


@pytest.mark.parametrize(
    ("geom", "point", "expected"),
    [
        (
            LineString([[0, 0], [1, 0]]),
            Point(0.50, 0),
            LineString([[0, 0], [0.5, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0]]),
            Point(0.50, 0),
            LineString([[0, 0], [0.5, 0], [1, 0], [2, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0]]),
            Point(1.5, 0),
            LineString([[0, 0], [1, 0], [1.5, 0], [2, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]),
            Point(2.0, 0.5),
            LineString([[0, 0], [1, 0], [2, 0], [2.0, 0.5], [2, 1], [2, 2]]),
        ),
        (
            LineString([[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]]),
            Point(2.0, 1.25),
            LineString([[0, 0], [1, 0], [2, 0], [2, 1], [2.0, 1.25], [2, 2]]),
        ),
        (
            LineString([[2, 2], [2, 1], [2, 0], [1, 0], [0, 0]]),
            Point(2.0, 1.25),
            LineString([[2, 2], [2.0, 1.25], [2, 1], [2, 0], [1, 0], [0, 0]]),
        ),
        (
            LineString([[1, 0], [0, 0]]),
            Point(0.50, 0),
            LineString([[1, 0], [0.5, 0], [0, 0]]),
        ),
        (
            LineString([[1, 0], [0, 0]]),
            Point(0.20, 0),
            LineString([[1, 0], [0.2, 0], [0, 0]]),
        ),
        (
            LineString([[1, 0], [0, 0]]),
            Point(0.20, 0, 5),
            LineString([[1, 0], [0.2, 0], [0, 0]]),
        ),
        (
            LineString([[1, 0, 1], [0, 0, 1]]),
            Point(0.20, 0, 5),
            LineString([[1, 0, 1], [0.2, 0, 5], [0, 0, 1]]),
        ),
        (
            LineString([[1, 0], [0, 0]]),
            Point(1, 0),
            LineString([[1, 0], [0, 0]]),
        ),
        (
            LineString([[1, 0], [0, 0]]),
            Point(5, 5),
            LineString([[1, 0], [0, 0]]),
        ),
    ],
    ids=[
        "one_segment",
        "two_segments_add_to_first",
        "two_segments_add_to_second",
        "four_segments_add_to_third",
        "four_segments_add_to_last",
        "four_segments_add_to_last_reversed",
        "one_segment_reversed",
        "one_segment_reversed_not_exactly_in_middle",
        "z_in_point_not_in_geom",
        "z_in_point_and_in_geom",
        "at_vertex",
        "not_within",
    ],
)
def test_add_topological_point(
    geom: LineString,
    point: Point,
    expected: LineString,
):
    assert add_topological_point(geom, point) == expected


@pytest.mark.parametrize(
    ("geom", "points", "expected"),
    [
        (
            LineString([[0, 0], [1, 0]]),
            [Point(0.50, 0)],
            LineString([[0, 0], [0.5, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            [Point(0.50, 0), Point(0.1234, 0)],
            LineString([[0, 0], [0.1234, 0], [0.5, 0], [1, 0]]),
        ),
        (
            LineString([[2, 2], [2, 1], [2, 0], [1, 0], [0, 0]]),
            [Point(0.5, 0), Point(2, 1.26), Point(2, 2), Point(2, 1.789)],
            LineString(
                [
                    [2, 2],
                    [2.0, 1.789],
                    [2, 1.26],
                    [2, 1],
                    [2, 0],
                    [1, 0],
                    [0.5, 0],
                    [0, 0],
                ]
            ),
        ),
    ],
    ids=[
        "one_point",
        "two_points",
        "many_points_to_different_segments",
    ],
)
def test_add_topological_points(
    geom: LineString,
    points: list[Point],
    expected: LineString,
):
    assert add_topological_points(geom, points) == expected


@pytest.mark.parametrize(
    ("point", "snap_to", "tolerance", "z_behavior", "expected"),
    [
        (Point(0, 0.01), LineString([[0, 0], [1, 0]]), 0, "inherit", Point(0, 0)),
        (Point(0, 2), LineString([[0, 0], [1, 0]]), 1, "inherit", Point(0, 0)),
        (Point(0, 2), LineString([[0, 0], [1, 0]]), 3, "inherit", Point(0, 2)),
        (Point(0, 0.01), box(0, 0, 1, -1), 0, "inherit", Point(0, 0)),
        (Point(0, 0), Point(1, 1), 0, "inherit", Point(1, 1)),
        (Point(0, 0, 5), Point(1, 1), 0, "inherit", Point(1, 1, 5)),
        (Point(0, 0, 5), Point(1, 1), 0, "exclude", Point(1, 1)),
    ],
    ids=[
        "snap_to_line",
        "outside_tolerance",
        "within_tolerance",
        "snap_to_polygon",
        "snap_to_point",
        "z_inherit",
        "z_exclude",
    ],
)
def test_snap_to_closest_vertex_or_segment(
    point: Point,
    snap_to: BaseGeometry,
    tolerance: float,
    z_behavior: Literal["inherit", "exclude"],
    expected: Point,
):
    assert (
        snap_to_closest_vertex_or_segment(
            point,
            snap_to,
            tolerance,
            z_behavior,
        )
        == expected
    )


def test_add_topological_point_raises():
    with pytest.raises(ValueError, match=r"Tolerance should be 0 or above."):
        add_topological_point(
            Point(),
            Point(),
            -1,
        )


def test_add_topological_points_raises():
    with pytest.raises(ValueError, match=r"Tolerance should be 0 or above."):
        add_topological_points(
            Point(),
            Point(),
            -1,
        )


@pytest.mark.parametrize(
    ("geom", "tolerance", "keep_cords", "expected"),
    [
        (
            LineString(
                [
                    [0, 0],
                    [5, 0],
                    [10, 2],
                    [15, 0],
                    [20, 4],
                    [25, 8],
                    [20, 10],
                    [20, 15],
                    [24, 20],
                ]
            ),
            2,
            set(),
            LineString(
                [
                    [0, 0],
                    [15, 0],
                    [25, 8],
                    [20, 10],
                    [24, 20],
                ]
            ),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [5, 0],
                    [10, 2],
                    [15, 0],
                    [20, 4],
                    [25, 8],
                    [20, 10],
                    [20, 15],
                    [24, 20],
                ]
            ),
            2,
            {Point(-5, -5)},
            LineString(
                [
                    [0, 0],
                    [15, 0],
                    [25, 8],
                    [20, 10],
                    [24, 20],
                ]
            ),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [5, 0],
                    [10, 2],
                    [15, 0],
                    [20, 4],
                    [25, 8],
                    [20, 10],
                    [20, 15],
                    [24, 20],
                ]
            ),
            2,
            {
                Point(5, 0),
                Point(10, 2),
            },
            LineString(
                [
                    [0, 0],
                    [5, 0],
                    [10, 2],
                    [15, 0],
                    [25, 8],
                    [20, 10],
                    [24, 20],
                ]
            ),
        ),
    ],
    ids=[
        "no_keep_cords",
        "keep_cord_not_in_line",
        "has_keep_coords",
    ],
)
def test_ramer_douglas_peucker_simplify_keep_coords(
    geom: LineString,
    tolerance: float,
    keep_cords: set[Point],
    expected: LineString,
):
    assert (
        ramer_douglas_peucker_simplify_keep_coords(
            geom,
            tolerance,
            keep_cords,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("line", "spline_subdivisions", "expected"),
    [
        (
            LineString([[0, 0], [1, 0]]),
            10,
            LineString([[0, 0], [1, 0]]),
        ),
        (
            LineString([[0, 0], [1, 0], [0.5, 1], [0.5, -1]]),
            10,
            LineString([[0, 0], [1, 0], [0.5, 1], [0.5, -1]]),
        ),
        (
            LineString(),
            10,
            LineString(),
        ),
        (
            LineString([[0, 0], [1, 1], [3, 1], [2, 0], [0, 0]]),
            2,
            LineString(
                [
                    [0, 0],
                    [0.3079449839416705, 0.5000000000000001],
                    [1, 1],
                    [3, 1],
                    [2, 0],
                    [0.8385016254650557, -0.161498374534944],
                    [0, 0],
                ]
            ),
        ),
    ],
    ids=[
        "not_closed",
        "not_valid",
        "empty",
        "ring",
    ],
)
def test_smooth_around_ring_closing_vertex(
    line: LineString,
    spline_subdivisions: int,
    expected: LineString,
):
    assert equals_exact(
        smooth_around_ring_closing_vertex(
            line, spline_subdivisions=spline_subdivisions
        ),
        expected,
        tolerance=0.000000001,
    )


@pytest.mark.parametrize(
    ("line_1", "line_2", "point", "spline_subdivisions", "expected_1", "expected_2"),
    [
        (
            LineString(),
            LineString(),
            Point(),
            10,
            LineString(),
            LineString(),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            LineString([[5, 5], [6, 5]]),
            Point(0, 0),
            10,
            LineString([[0, 0], [1, 0]]),
            LineString([[5, 5], [6, 5]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            LineString([[1, 0], [1, 1]]),
            Point(5, 5),
            10,
            LineString([[0, 0], [1, 0]]),
            LineString([[1, 0], [1, 1]]),
        ),
        (
            LineString([[0, 0], [1, 0]]),
            LineString([[1, 0], [1, 1]]),
            Point(0, 0),
            10,
            LineString([[0, 0], [1, 0]]),
            LineString([[1, 0], [1, 1]]),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [1.25, 0.25],
                    [1.75, 1],
                    [2.5, 2],
                    [4, 3],
                ]
            ),
            LineString(
                [
                    [4, 3],
                    [5, 2],
                    [6, 4.25],
                    [5.5, 4.75],
                    [5, 6],
                ]
            ),
            Point(4, 3),
            3,
            LineString(
                [
                    [0, 0],
                    [1.25, 0.25],
                    [1.75, 1],
                    [2.5, 2],
                    [2.9661258867648237, 2.433208736292763],
                    [3.5049204064058337, 2.842054416166005],
                    [4, 3],
                ]
            ),
            LineString(
                [
                    [4, 3],
                    [4.363927764355153, 2.711489274491867],
                    [4.693562719333668, 2.2261563204196673],
                    [5, 2],
                    [6, 4.25],
                    [5.5, 4.75],
                    [5, 6],
                ]
            ),
        ),
        (
            LineString(
                [
                    [0, 0],
                    [1.25, 0.25],
                    [1.75, 1],
                    [2.5, 2],
                    [4, 3],
                ]
            ),
            LineString(
                [
                    [4, 3],
                    [5, 2],
                    [6, 4.25],
                    [5.5, 4.75],
                    [5, 6],
                    [0, 6],
                    [-2, 3],
                    [0, 0],
                ]
            ),
            Point(4, 3),
            3,
            LineString(
                [
                    [0, 0],
                    [1.25, 0.25],
                    [1.75, 1],
                    [2.5, 2],
                    [2.9661258867648237, 2.433208736292763],
                    [3.5049204064058337, 2.842054416166005],
                    [4, 3],
                ]
            ),
            LineString(
                [
                    [4, 3],
                    [4.363927764355153, 2.711489274491867],
                    [4.693562719333668, 2.2261563204196673],
                    [5, 2],
                    [6, 4.25],
                    [5.5, 4.75],
                    [5, 6],
                    [0, 6],
                    [-2, 3],
                    [0, 0],
                ]
            ),
        ),
    ],
    ids=[
        "empty_1",
        "lines_disjoint",
        "point_disjoint",
        "point_disjoint_other",
        "should_smooth",
        "makes_ring",
    ],
)
def test_smooth_around_connection_point_of_two_lines(
    line_1: LineString,
    line_2: LineString,
    point: Point,
    spline_subdivisions: int,
    expected_1: LineString,
    expected_2: LineString,
):
    result_1, result_2 = smooth_around_connection_point_of_two_lines(
        line_1, line_2, point, spline_subdivisions=spline_subdivisions
    )

    assert equals_exact(result_1, expected_1, tolerance=0.000000001)

    assert equals_exact(result_2, expected_2, tolerance=0.000000001)
