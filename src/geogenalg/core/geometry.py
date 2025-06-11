from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

import shapely.ops
from geopandas import gpd
from shapely import (
    LineString,
    MultiLineString,
    MultiPoint,
    Point,
    Polygon,
    affinity,
    count_coordinates,
    polygonize,
    shortest_line,
    union,
)

from ..core.exceptions import (  # noqa: TID252
    GeometryOperationError,
    GeometryTypeError,
    InvalidGeometryError,
)

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


class LineExtendFrom(Enum):
    END = -1
    START = 0
    BOTH = 1


class Dimensions(NamedTuple):
    width: float
    height: float


def rectangle_dimensions(rect: Polygon) -> Dimensions:
    """Returns the width and height of a rectangular polygon.

    If input polygon is invalid or not a rectangle, raises an error.
    """
    required_coords = 5
    if count_coordinates(rect) != required_coords or not math.isclose(  # noqa: SC200
        rect.area,
        rect.oriented_envelope.area,
        rel_tol=1e-06,  # noqa: SC200
    ):
        msg = f"Not a rectangle: {rect}"
        raise GeometryOperationError(msg)

    if not rect.is_valid:
        msg = "Rectangle not valid"
        raise InvalidGeometryError(msg)

    x, y = rect.exterior.coords.xy

    point_1 = Point(x[0], y[0])
    point_2 = Point(x[1], y[1])
    point_3 = Point(x[2], y[2])

    lengths = (point_1.distance(point_2), point_2.distance(point_3))

    return Dimensions(min(lengths), max(lengths))


def elongation(polygon: Polygon) -> float:
    """Returns the elongation of a polygon which is the ratio of the height
    (longest side) to the width (shorter side) of its minimum bounding oriented
    envelope.
    """
    dim = rectangle_dimensions(polygon.oriented_envelope)
    return dim.height / dim.width


def extract_interior_rings(areas: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Extracts the interior rings of a polygon geodataframe.

    Returns a new geodataframe containing the interior rings.
    """
    if not all(
        geom_type in {"Polygon", "MultiPolygon"}
        for geom_type in areas.geometry.geom_type.values
    ):
        msg = "Cannot extract interior rings from a geodataframe with non-polygon geometries"
        raise GeometryTypeError(msg)

    interiors = areas

    # interiors are given as a list of LinearRings, polygonize
    interiors.geometry = interiors.geometry.interiors.apply(polygonize)

    # polygonize gives a GeometryCollection, filter out empty ones
    interiors = interiors.loc[~interiors.geometry.is_empty]

    # explode GeometryCollection to individual features
    return interiors.explode()


def explode_line(line: LineString) -> list[LineString]:
    """Explodes a line geometry and returns all its segments as a list
    of LineStrings.
    """
    return [
        LineString((a, b)) for a, b in zip(line.coords, line.coords[1:], strict=False)
    ]


def lines_to_segments(lines: gpd.GeoSeries) -> gpd.GeoSeries:
    """Returns a GeoSeries of line geometries containing all line segments
    from the input GeoSeries.
    """
    if not all(geom_type == "LineString" for geom_type in lines.geom_type.values):
        msg = "All geometries must be LineStrings."
        raise GeometryTypeError(msg)

    return gpd.GeoSeries(
        [segment for line in lines for segment in explode_line(line)],
        crs=lines.crs,
    )


def scale_line_to_length(
    geom: LineString,
    length: float,
) -> LineString:
    """Returns a version of the input geometry which has been scaled to the
    given length.
    """
    if geom.length <= 0:
        msg = "Geometry has no length"
        raise ValueError(msg)

    if length <= 0:
        msg = "Length must be above zero"
        raise ValueError(msg)

    scaling_factor = length / geom.length

    if scaling_factor == 1:
        return LineString(geom)

    return affinity.scale(geom, xfact=scaling_factor, yfact=scaling_factor)  # noqa: SC200


def move_to_point(geom: BaseGeometry, point: Point) -> BaseGeometry:
    """Returns a version of the input geometry which has been moved to the
    given point position.
    """
    dx = point.x - geom.centroid.x
    dy = point.y - geom.centroid.y

    return affinity.translate(geom, xoff=dx, yoff=dy)  # noqa: SC200


def extend_line_to_nearest(
    line: LineString,
    extend_to: BaseGeometry,
    extend_from: LineExtendFrom,
    tolerance: float = 0,
) -> LineString:
    """Returns a version of the input line which has been extended to the
    nearest point of another geometry. The extension can be done from
    the end or start of the line, or alternatively both ends.
    """
    # TODO: test tolerance

    if extend_from != LineExtendFrom.BOTH:
        point = Point(line.coords[extend_from.value])
        new_segment = shortest_line(point, extend_to)

        if tolerance > 0 and new_segment.length >= tolerance:
            return line

        combined = union(line, new_segment)
    else:
        start = Point(line.coords[0])
        end = Point(line.coords[-1])

        new_segment_1 = shortest_line(start, extend_to)
        new_segment_2 = shortest_line(end, extend_to)

        # TODO: clean this up maybe

        if tolerance > 0:
            combined = MultiLineString([line])
            if new_segment_1.length < tolerance:
                combined = union(combined, new_segment_1)

            if new_segment_2.length < tolerance:
                combined = union(combined, new_segment_2)
        else:
            combined = union(union(line, new_segment_1), new_segment_2)

    if not isinstance(combined, MultiLineString):
        msg = f"Union result was not a MultiLineString {combined.wkt}"
        raise GeometryOperationError(msg)

    merged = shapely.ops.linemerge(combined)

    if not isinstance(merged, LineString):
        msg = f"Merge result was not a LineString: {merged}"
        raise GeometryOperationError(msg)

    return merged


def point_on_line(line: LineString, distance: float) -> Point:
    """Returns a point which is the given distance away and collinear to either
    the first or last segment of the input line. If distance is > 0 the point
    is calculated from the last segment and if < 0 the first segment.
    """
    # TODO: doesn't have a test

    required_coords = 2

    if len(line.coords) != required_coords:
        msg = "Line must only have two vertices"
        raise ValueError(msg)

    if distance == 0:
        return Point(line.coords[-1])

    from_start: bool = distance < 0

    if from_start:
        start_x = line.coords[0][0]
        start_y = line.coords[0][1]

        end_x = line.coords[1][0]
        end_y = line.coords[1][1]
    else:
        start_x = line.coords[1][0]
        start_y = line.coords[1][1]

        end_x = line.coords[0][0]
        end_y = line.coords[0][1]

    t = abs(distance) / line.length
    t *= -1

    px = ((1 - t) * start_x) + (t * end_x)
    py = ((1 - t) * start_y) + (t * end_y)

    return Point(px, py)


def extend_line_by(
    line: LineString,
    extend_by: float,
    extend_from: LineExtendFrom,
) -> LineString:
    """Returns a version of the input line which has been extended by the given distance
    either from its start or end segment, or both.
    """
    # TODO: doesn't have a test

    if extend_by <= 0:
        msg = "Extension distance must be above zero."
        raise ValueError(msg)

    first_segment = LineString((line.coords[0], line.coords[1]))
    last_segment = LineString((line.coords[-2], line.coords[-1]))

    if extend_from == LineExtendFrom.END:
        point = point_on_line(last_segment, extend_by)

        return extend_line_to_nearest(line, point, extend_from)

    if extend_from == LineExtendFrom.START:
        point = point_on_line(first_segment, -extend_by)

        return extend_line_to_nearest(line, point, extend_from)

    point_1 = point_on_line(first_segment, -extend_by)
    point_2 = point_on_line(last_segment, extend_by)

    multi_point = MultiPoint((point_1, point_2))

    return extend_line_to_nearest(line, multi_point, extend_from)
