#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from enum import Enum
from statistics import mean
from typing import NamedTuple

from geopandas import GeoDataFrame, GeoSeries
from numpy import column_stack, ndarray, vstack  # noqa: SC200
from scipy.spatial import KDTree  # noqa: SC200
from shapely import (
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    affinity,
    force_2d,
    get_coordinates,
    polygonize,
    shortest_line,
    union,
)
from shapely.coords import CoordinateSequence
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge

from geogenalg.core.exceptions import (
    GeometryOperationError,
    GeometryTypeError,
    InvalidGeometryError,
)


class LineExtendFrom(Enum):
    """An enumeration describing from which end a linestring should be extended."""

    END = -1
    START = 0
    BOTH = 1


class Dimensions(NamedTuple):
    """Simple named tuple describing the dimensions of a rectangle."""

    width: float
    height: float


def chaikin_smooth_skip_coords(
    geom: LineString | Polygon,
    skip_coords: list[Point] | MultiPoint,
    *,
    iterations: int = 1,
) -> LineString | Polygon:
    """Smooth input linestring or polygon.

    This is an implementation of Chaikin's corner cutting line smoothing
    algorithm, with the addition of being able to skip the smoothing of
    specific points.

    Args:
    ----
        geom: Geometry to smooth.
        iterations: Number of iterations i.e. smoothing passes. Each pass
            doubles the number of vertices (minus the skipped coordinates) i.e.
            increase with caution, growth is exponential.
        skip_coords: List of points to skip. If a corresponding coordinate exists
            in the input geometry, it is guaranteed to remain unchanged.

    Returns:
    -------
        Smoothed geometry.

    Note:
    ----
        If skipping of coordinates is not required, prefer shapelysmooth's
        function as it is likely to be more efficient.


    """
    if isinstance(skip_coords, MultiPoint):
        skip_coords_ = [point.coords[0] for point in skip_coords.geoms]
    else:
        skip_coords_ = [point.coords[0] for point in skip_coords]

    # TODO: allow processing 2.5D geometries and multigeometries?

    def _process_coord_sequence(
        seq: CoordinateSequence,
        output: list[tuple[float, ...]],
    ) -> list[tuple[float, ...]]:
        for i in range(len(seq) - 1):
            current_coord = seq[i]
            next_coord = seq[i + 1]

            current_x = current_coord[0]
            current_y = current_coord[1]
            next_x = next_coord[0]
            next_y = next_coord[1]

            q = ((0.75 * current_x + 0.25 * next_x), (0.75 * current_y + 0.25 * next_y))
            r = ((0.25 * current_x + 0.75 * next_x), (0.25 * current_y + 0.75 * next_y))

            if current_coord in skip_coords_:
                output.append(current_coord)
                if next_coord not in skip_coords_:
                    output.append(r)
                continue

            output.append(q)

            if next_coord not in skip_coords_:
                output.append(r)

        return output

    def _process_linestring(geom: LineString) -> LineString:
        coords = [geom.coords[0]]

        processed_coords = _process_coord_sequence(geom.coords, coords)
        processed_coords.append(geom.coords[-1])

        return LineString(processed_coords)

    def _process_polygon(geom: Polygon) -> Polygon:
        exterior = _process_coord_sequence(geom.exterior.coords, [])
        interiors = [
            _process_coord_sequence(interior.coords, []) for interior in geom.interiors
        ]

        return Polygon(exterior, interiors)

    process_function = (
        _process_linestring if isinstance(geom, LineString) else _process_polygon
    )

    result = geom
    for _ in range(iterations):
        result = process_function(result)

    return result


def get_topological_points(geoseries: GeoSeries) -> list[Point]:
    """Find all topological points in a GeoSeries.

    Topological point referring to a point which is shared by two or more
    geometries in the GeoSeries.

    Args:
    ----
        geoseries: The GeoSeries to find topological points in.

    Returns:
    -------
        List of all the topological points (if any).

    Raises:
    ------
        GeometryOperationError: If union could not be performed on unique points
        in the GeoSeries.

    """
    if geoseries.empty:
        return []

    unique_points = geoseries.force_2d().extract_unique_points().union_all()

    if isinstance(unique_points, Point):
        unique_points = MultiPoint([unique_points])

    if not isinstance(unique_points, MultiPoint):
        msg = "Unique points not a Point or a MultiPoint"
        raise GeometryOperationError(msg)

    # TODO: maybe there's a faster way to do this
    topo_points: list[Point] = []
    for point in unique_points.geoms:
        intersections = geoseries.intersects(point)

        # If there are more than 1 intersections
        if len(intersections.loc[intersections]) > 1:
            topo_points.append(point)

    return topo_points


def chaikin_smooth_keep_topology(
    geoseries: GeoSeries,
    iterations: int = 3,
    *,
    extra_skip_coords: list[Point] | MultiPoint | None,
) -> GeoSeries:
    """Apply smoothing algorithm while keeping topological points unchanged.

    Args:
    ----
        geoseries: GeoSeries to be smoothed.
        iterations: Number of smoothing passes, increase for a smoother result.
            Note that each pass roughly doubles the number of vertices, so
            growth will be exponential. Anything above 6-7 is unlikely to
            produce cartographically meaningful differences and therefore
            unnecessarily increase vertex count.
        extra_skip_coords: Any additional coordinates in addition to
            topological points which should not be smoothed.

    Returns:
    -------
        GeoSeries with smoothed geometries, with shared topological points
        unchanged.

    Note:
    ----
        If the input has 3D geometries, they will be changed to 2D geometries.

    """
    copy = geoseries.copy()

    skipped_coords = get_topological_points(copy)

    if isinstance(extra_skip_coords, MultiPoint):
        extra_skip_coords = list(extra_skip_coords.geoms)

    if extra_skip_coords is not None:
        skipped_coords += extra_skip_coords

    return copy.apply(
        lambda geom: chaikin_smooth_skip_coords(
            force_2d(geom),
            iterations=iterations,
            skip_coords=skipped_coords,
        ),
    )


def perforate_polygon_with_gdf_exteriors(
    geometry: Polygon,
    gdf: GeoDataFrame,
) -> Polygon:
    """Add the exteriors of a polygon GeoDataFrame as holes to a Polygon.

    The purpose of this function over a simple difference operation is to
    handle cases where there are polygon(s) inside interior rings of the input
    geometry and you need to add holes only for the exteriors. This is useful
    f.e. for recursive islands and lakes. With difference, the result would be
    a multipolygon with all the contained geometries as parts.

    Args:
    ----
        geometry: A Polygon to which holes will be added
        gdf: A GeoDataFrame whose geometry exteriors will be added as holes

    Returns:
    -------
        A new Polygon with the holes added.

    """
    features_within_geom = gdf.geometry.loc[gdf.geometry.within(geometry)]
    exteriors = features_within_geom.apply(lambda geom: Polygon(geom.exterior))

    return geometry.difference(exteriors.union_all())


def extract_interior_rings(geometry: Polygon | MultiPolygon) -> MultiPolygon:
    """Extract interior rings (holes) of a geometry as a new geometry.

    Returns
    -------
        Interior rings as a MultiPolygon.

    """
    if isinstance(geometry, Polygon):
        polygons = [Polygon(interior) for interior in geometry.interiors]
    else:
        polygons = []
        for geom in geometry.geoms:
            for interior in geom.interiors:
                polygons.append(Polygon(interior))

    return MultiPolygon(polygons)


def mean_z(geom: BaseGeometry, nodata_value: float = -999.999) -> float:
    """Calculate the mean of z values in a geometry's vertices.

    Returns
    -------
        The calculated mean.

    Raises
    ------
        GeometryTypeError: If geometry is a collection or does not have z values.

    """
    if not geom.has_z:
        msg = "Geometry does not include z coordinate!"
        raise GeometryTypeError(msg)

    if geom.geom_type == "GeometryCollection":
        msg = "mean z calculation not implemented for GeometryCollection"
        raise GeometryTypeError(msg)

    # shapely.get_coordinates doesn't work with polygons in this use case, since
    # it also returns the bounding vertex for each ring in the polygon, which
    # can skew the results especially if there are multiple interior rings.
    if "Polygon" not in geom.geom_type:
        z_values = [
            float(coords[2])
            for coords in get_coordinates(geom, include_z=True)
            if coords[2] != nodata_value
        ]
    else:
        # Turn polygon into to a (Multi)LineString and drop the last vertex(es).
        boundary = geom.boundary

        if boundary.geom_type == "LineString":
            return mean_z(LineString(boundary.coords[:-1]))

        return mean_z(
            MultiLineString([LineString(line.coords[:-1]) for line in boundary.geoms])
        )

    return mean(z_values)


def oriented_envelope_dimensions(geom: Polygon) -> Dimensions:
    """Calculate the dimensions of a geometry from its oriented envelope.

    Returns
    -------
        The width and height of the oriented envelope.

    Raises
    ------
        InvalidGeometryError: If input polygon is invalid.

    """
    if not geom.is_valid:
        msg = "Rectangle not valid"
        raise InvalidGeometryError(msg)

    envelope = geom.oriented_envelope

    x, y = envelope.exterior.coords.xy

    point_1 = Point(x[0], y[0])
    point_2 = Point(x[1], y[1])
    point_3 = Point(x[2], y[2])

    lengths = (point_1.distance(point_2), point_2.distance(point_3))

    return Dimensions(width=min(lengths), height=max(lengths))


def elongation(polygon: Polygon) -> float:
    """Calculate the elongation of a polygon.

    Returns
    -------
        The ratio of the height (longest side) to the width (shorter side)
        of its minimum bounding oriented envelope.

    """
    dimensions = oriented_envelope_dimensions(polygon)
    return dimensions.width / dimensions.height


def extract_interior_rings_gdf(areas: GeoDataFrame) -> GeoDataFrame:
    """Extract the interior rings of a polygon geodataframe.

    Returns
    -------
        A new geodataframe containing the interior rings.

    Raises
    ------
        GeometryTypeError: If geometries are not polygons or multipolygons.

    """
    if not all(
        geom_type in {"Polygon", "MultiPolygon"}
        for geom_type in areas.geometry.geom_type.to_numpy()
    ):
        msg = """Cannot extract interior rings from
        a geodataframe with non-polygon geometries"""
        raise GeometryTypeError(msg)

    interiors = areas

    # interiors are given as a list of LinearRings, polygonize
    interiors.geometry = interiors.geometry.interiors.apply(polygonize)

    # polygonize gives a GeometryCollection, filter out empty ones
    interiors = interiors.loc[~interiors.geometry.is_empty]

    # explode GeometryCollection to individual features
    return interiors.explode()


def explode_line(line: LineString) -> list[LineString]:
    """Explode a line geometry to all its segments.

    Returns
    -------
      All segments of a line geometry.

    """
    return [
        LineString((a, b)) for a, b in zip(line.coords, line.coords[1:], strict=False)
    ]


def lines_to_segments(lines: GeoSeries) -> GeoSeries:
    """Convert lines to segments.

    Returns
    -------
        A GeoSeries of line geometries containing all line segments
        from the input GeoSeries.

    Raises
    ------
        GeometryTypeError: if param `lines` contains other geometry
        types than lines.

    """
    if not all(geom_type == "LineString" for geom_type in lines.geom_type.to_numpy()):
        msg = "All geometries must be LineStrings."
        raise GeometryTypeError(msg)

    return GeoSeries(
        [segment for line in lines for segment in explode_line(line)],
        crs=lines.crs,
    )


def scale_line_to_length(
    geom: LineString,
    length: float,
) -> LineString:
    """Scale line to given length.

    Returns
    -------
        A version of the input geometry which has been scaled to the
        given length.

    Raises
    ------
        ValueError: If param `geom` has no length.
        ValueError: If param `length` is less than or equal to 0.

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
    """Move geometry to given point position.

    Returns
    -------
        A version of the input geometry which has been moved to the
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
    """Extend line to nearest point.

    Returns
    -------
        A version of the input line which has been extended to the
        nearest point of another geometry. The extension can be done from
        the end or start of the line, or alternatively both ends.

    Raises
    ------
        GeometryOperationError: If invalid result got from line union or merge
        operations.

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

    merged = linemerge(combined)

    if not isinstance(merged, LineString):
        msg = f"Merge result was not a LineString: {merged}"
        raise GeometryOperationError(msg)

    return merged


def point_on_line(line: LineString, distance: float) -> Point:
    """Get point along line.

    Returns
    -------
        A point which is the given distance away and collinear to either
        the first or last segment of the input line. If distance is > 0 the point
        is calculated from the last segment and if < 0 the first segment.

    Raises
    ------
        ValueError: If line does not have exactly 2 vertices.

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
    """Extend line by given distance.

    Returns
    -------
        Version of the input line which has been extended by the given distance
        either from its start or end segment, or both.

    Raises
    ------
        ValueError: if extend_from is less or equal than 0

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


def _geometry_with_z_from_kd_tree(  # noqa: PLR0911, SC200
    geometry: BaseGeometry,
    kd_tree: KDTree,  # noqa: SC200
    source_z: ndarray,  # noqa: SC200
) -> BaseGeometry:
    """Create new geometry using nearest Z and given KDTree.

    This function is meant to be used by `assign_nearest_z`.

    Works for Polygon, LineString, and Point geometries and their
    multi versions.

    Args:
    ----
        geometry: Geometry to attach Z value to.
        kd_tree: KDTree created from XY coordinates. `source_z`
            should be from same dataset.
        source_z: Array of Z values.

    Returns:
    -------
        New geometry that matches input `geometry` with nearest Z
        value added from `source_z`.

    """
    if geometry is None or geometry.is_empty:
        return geometry

    def _coords_with_z(coords: CoordinateSequence) -> list[tuple[float, float, float]]:
        x_values, y_values = coords.xy  # Ignore possible pre-existing Z
        _, idx = kd_tree.query(column_stack((x_values, y_values)))  # noqa: SC200
        z_values = source_z[idx]
        return [
            (x, y, float(z))
            for x, y, z in zip(x_values, y_values, z_values, strict=True)
        ]

    def _polygon_with_z(polygon: Polygon) -> Polygon:
        exterior_with_z = _coords_with_z(polygon.exterior.coords)
        interiors_with_z = [_coords_with_z(ring.coords) for ring in polygon.interiors]
        return Polygon(shell=exterior_with_z, holes=interiors_with_z)

    if geometry.geom_type == "Polygon":
        return _polygon_with_z(geometry)

    if geometry.geom_type == "MultiPolygon":
        return MultiPolygon(_polygon_with_z(polygon) for polygon in geometry.geoms)

    if geometry.geom_type == "LineString":
        return LineString(_coords_with_z(geometry.coords))

    if geometry.geom_type == "MultiLineString":
        return MultiLineString(
            [LineString(_coords_with_z(line.coords)) for line in geometry.geoms]
        )

    if geometry.geom_type == "Point":
        return Point(_coords_with_z(geometry.coords)[0])

    if geometry.geom_type == "MultiPoint":
        return MultiPoint(
            [Point(_coords_with_z(point.coords)[0]) for point in geometry.geoms]
        )

    return geometry


def assign_nearest_z(
    source_gdf: GeoDataFrame,
    target_gdf: GeoDataFrame,
    overwrite_z: bool = False,  # noqa: FBT001, FBT002
) -> GeoDataFrame:
    """Copy nearest Z values from source to target geometries.

    Attaches Z values to all geometries in `target_gdf` by
    finding nearest vertex in the source geometries and using
    its Z value. Uses KDTree from SciPy to perform the lookup.

    Works for Polygon, LineString, and Point geometries and their
    multi versions.

    Args:
    ----
        source_gdf: GeoDataFrame with Z values.
        target_gdf: GeoDataFrame without Z values to upgrade.
        overwrite_z: Whether to overwrite possible Z values in
            target_gdf geometries.

    Returns:
    -------
        A copy of `target_gdf` with Z values or the input `target_gdf`
        if no Z values where found in `source_gdf`.

    """
    # No Z values, return unchanged
    if not any(geom.has_z for geom in source_gdf.geometry if geom is not None):
        return target_gdf

    # Find all coordinates
    coords_list = []
    for geom in source_gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        arr = get_coordinates(geom, include_z=True)
        # Check that coordinates are 3D
        if arr.shape[1] == 3:  # noqa: PLR2004
            coords_list.append(arr)

    # Return unchanged if no 3D coordinates were found
    if not coords_list:
        return target_gdf

    source_coords = vstack(coords_list)  # noqa: SC200

    # Build KD-tree
    kd_tree = KDTree(source_coords[:, :2])  # noqa: SC200
    source_z = source_coords[:, 2]

    # Create output
    result_gdf = target_gdf.copy()
    result_gdf.geometry = result_gdf.geometry.apply(
        lambda geom: _geometry_with_z_from_kd_tree(geom, kd_tree, source_z)  # noqa: SC200
        if (overwrite_z or not geom.has_z)
        else geom
    )
    return result_gdf
