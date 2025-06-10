import pytest
import shapely
from geopandas import gpd
from shapely.geometry import Point
from geogenalg.core.exceptions import (
    GeometryOperationError,
    GeometryTypeError,
)
from geogenalg.core.geometry import (
    LineExtendFrom,
    elongation,
    explode_line,
    extend_line_to_nearest,
    extract_interior_rings,
    lines_to_segments,
    move_to_point,
    rectangle_dimensions,
    scale_line_to_length,
)


def test_rectangle_dimensions():
    rect = shapely.from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 0, 0 0))")

    assert rectangle_dimensions(rect).width == 1.0
    assert rectangle_dimensions(rect).height == 4.0

    rect2 = shapely.from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 5, 1 0, 0 0))")
    with pytest.raises(GeometryOperationError):
        rectangle_dimensions(rect2)


def test_elongation():
    rect = shapely.from_wkt("POLYGON ((0 0, 0 4, 1 4, 1 0, 0 0))")

    assert elongation(rect) == 4.0


def test_extract_interior_rings():
    polygon_with_holes = shapely.from_wkt(
        "POLYGON ((0 0, 0 10, 10 10, 10 0, 0 0), "
        "(2 3, 3 3, 3 2, 2 2, 2 3), "
        "(4 5, 5 5, 5 4, 4 4, 4 5), "
        "(7 8, 8 8, 8 7, 7 7, 7 8))"
    )

    gdf = gpd.GeoDataFrame(
        geometry=[polygon_with_holes],
        crs="EPSG:3857",
    )

    extracted_holes = extract_interior_rings(gdf)

    assert len(extracted_holes.index) == 3
    assert extracted_holes.crs == "EPSG:3857"

    assert extracted_holes.iloc[0].geometry.wkt == "POLYGON ((2 3, 3 3, 3 2, 2 2, 2 3))"
    assert extracted_holes.iloc[1].geometry.wkt == "POLYGON ((4 5, 5 5, 5 4, 4 4, 4 5))"
    assert extracted_holes.iloc[2].geometry.wkt == "POLYGON ((7 8, 8 8, 8 7, 7 7, 7 8))"


def test_explode_line_to_segments():
    line = shapely.from_wkt("LINESTRING (0 0, 0 1, 0 3, 0 6, 1 8, 2 9)")

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
            shapely.from_wkt("LINESTRING (0 0, 0 1)"),
            shapely.from_wkt("LINESTRING (10 0, 10 15)"),
            shapely.from_wkt("LINESTRING (20 20, 21 21, 22 22, 23 23)"),
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
            shapely.from_wkt("LINESTRING (0 0, 0 1)"),
            shapely.from_wkt("LINESTRING (10 0, 10 15)"),
            shapely.from_wkt("LINESTRING (20 20, 21 21, 22 22, 23 23)"),
            shapely.from_wkt("POINT (10 10)"),
        ]
    )

    with pytest.raises(GeometryTypeError):
        lines_to_segments(invalid_series)


def test_scale_line_to_length():
    line1 = shapely.from_wkt("LINESTRING (0 0, 0 4)")
    scaled_line1 = scale_line_to_length(line1, 4)

    line2 = shapely.from_wkt("LINESTRING (10 10, 14 14)")
    scaled_line2 = scale_line_to_length(line2, 10)

    line3 = shapely.from_wkt("LINESTRING (5 5, 3 -1)")
    scaled_line3 = scale_line_to_length(line3, 13)

    line4 = shapely.from_wkt("LINESTRING (0 0, 0 1, 1 1, 1 5, 1 7)")
    scaled_line4 = scale_line_to_length(line4, 2)

    assert scaled_line1.length == 4
    assert scaled_line2.length == 10
    assert scaled_line3.length == 13
    assert scaled_line4.length == 2

    # check with lower precision to help readability
    assert shapely.to_wkt(scaled_line1, 3) == "LINESTRING (0 0, 0 4)"
    assert shapely.to_wkt(scaled_line2, 3) == "LINESTRING (8.464 8.464, 15.536 15.536)"
    assert shapely.to_wkt(scaled_line3, 3) == "LINESTRING (6.055 8.166, 1.945 -4.166)"
    assert (
        shapely.to_wkt(scaled_line4, 3)
        == "LINESTRING (0.375 2.625, 0.375 2.875, 0.625 2.875, 0.625 3.875, 0.625 4.375)"
    )


def test_move_to_point():
    line = shapely.from_wkt("LINESTRING (10 10, 14 14, 16 16, 19 20)")
    moved_line = move_to_point(line, Point(0, 0))

    point = shapely.from_wkt("POINT (10 200)")
    moved_point = move_to_point(point, Point(0, 0))

    poly = shapely.from_wkt("POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))")
    moved_poly = move_to_point(poly, Point(0, 0))

    multiline = shapely.from_wkt("MULTILINESTRING ((20 21, 21 24), (30 30, 31 31))")
    moved_multiline = move_to_point(multiline, Point(0, 0))

    assert (
        shapely.to_wkt(moved_line, 3)
        == "LINESTRING (-4.668 -4.854, -0.668 -0.854, 1.332 1.146, 4.332 5.146)"
    )
    assert shapely.to_wkt(moved_point, 3) == "POINT (0 0)"
    assert (
        shapely.to_wkt(moved_poly, 3)
        == "POLYGON ((-0.5 -0.5, -0.5 0.5, 0.5 0.5, 0.5 -0.5, -0.5 -0.5))"
    )
    assert (
        shapely.to_wkt(moved_multiline, 3)
        == "MULTILINESTRING ((-3.59 -3.972, -2.59 -0.972), (6.41 5.028, 7.41 6.028))"
    )


def test_extend_line_to_nearest():
    line = shapely.from_wkt("LINESTRING (0 0, 1 1)")

    point = shapely.from_wkt("POINT (-10 -3)")
    linestring = shapely.from_wkt("LINESTRING (10 10, 10 2)")
    polygon = shapely.from_wkt("POLYGON ((0 2, 0 3, 2 3, 2 2, 0 2))")

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
