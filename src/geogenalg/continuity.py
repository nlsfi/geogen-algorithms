#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from collections import defaultdict
from collections.abc import Callable
from typing import Literal, cast

from geopandas import GeoDataFrame
from pandas import concat
from shapely import force_2d
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from geogenalg.core.geometry import (
    LineExtendFrom,
    extend_line_to_nearest,
    get_topological_points,
)


def find_all_endpoints(
    lines: list[LineString | MultiLineString],
) -> list[tuple[Point, int, int]]:
    """Find all endpoints (start and end points) of line features.

    Args:
    ----
        lines: A list of shapely LineString or MultiLineString geometries

    Returns:
    -------
        A list of tuples in the format (endpoint, line_index, occurrence_count), where
            occurrence_count indicates how many lines share the same endpoint location
            up to 5 decimals.

    """
    endpoint_map = defaultdict(list)
    min_coordinates_for_line = 2

    # Iterate over each geometry and extract its start and end points
    for idx, geom in enumerate(lines):
        if geom.is_empty:
            continue
        if isinstance(geom, LineString):
            parts = [geom]
        elif isinstance(geom, MultiLineString):
            parts = [
                geometry for geometry in geom.geoms if isinstance(geometry, LineString)
            ]

        for part in parts:
            coords = list(part.coords)
            if len(coords) >= min_coordinates_for_line:
                for endpoint in [Point(coords[0]), Point(coords[-1])]:
                    # Round coordinates to five decimal places
                    key = (round(endpoint.x, 5), round(endpoint.y, 5))
                    endpoint_map[key].append((endpoint, idx))

    endpoints = []

    # Flatten the dictionary into a list of (point, line index, count of lines)
    for entries in endpoint_map.values():
        for point, idx in entries:
            endpoints.append((point, idx, len(entries)))

    return endpoints


def connect_nearby_endpoints(
    input_gdf: GeoDataFrame, gap_threshold: float
) -> GeoDataFrame:
    """Generate helper lines to close small gaps between line endpoints.

    Args:
    ----
        input_gdf: A GeoDataFrame containing the original line network as LineStrings
            or MultiLineStrings
        gap_threshold: Maximum distance between endpoints to be connected

    Returns:
    -------
        A new GeoDataFrame containing only the generated helper lines

    """
    new_lines = []
    used_ends: set[int] = set()

    lines = list(input_gdf.geometry)

    endpoints = find_all_endpoints(lines)

    for i, (point_1, _, count_1) in enumerate(endpoints):
        if count_1 != 1 or (i in used_ends):
            continue

        closest_distance = float("inf")
        best_point = None

        # Search for the nearest other endpoint that is within the gap_threshold
        for j, (point_2, _, _) in enumerate(endpoints):
            if i == j:
                continue

            distance = point_1.distance(point_2)
            if distance < gap_threshold and distance < closest_distance:
                closest_distance = distance
                best_point = point_2

        if best_point:
            new_line = LineString([point_1, best_point])
            new_lines.append(new_line)
            used_ends.add(i)

    # Return a new GeoDataFrame containing only the helper lines
    return GeoDataFrame(geometry=new_lines, crs=input_gdf.crs)


def check_line_connections(
    gdf: GeoDataFrame,
    threshold_distance: float,
    connection_info_column: str = "is_connected",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Check the connections of linestrings with respect to lines on the GeoDataFrame.

    Args:
    ----
        gdf (GeoDataframe): input GeoDataFrame consisting of Linestrings.
        threshold_distance (float): the max gap allowed for lines to
            be considered connected.
        connection_info_column (str): name of the attribute column where the info
            about connectedness is stored as a boolean

    Returns:
    -------
        A GeoDataframe of lines with an attribute that indicate the
        connectedness with respect to the linestrings in the input
        GeoDataFrame.

    """
    results = []

    for idx, row in gdf.iterrows():
        line = row[gdf.geometry.name]

        first_point = Point(line.coords[0])
        last_point = Point(line.coords[-1])

        # Create buffers at the first and last points
        first_buffer = first_point.buffer(threshold_distance)
        last_buffer = last_point.buffer(threshold_distance)

        # Check for intersections with other lines
        other_lines = gdf[gdf.index != idx]  # Exclude the current line
        first_intersects = other_lines.intersects(first_buffer).any()
        last_intersects = other_lines.intersects(last_buffer).any()

        # Record whether the line is connected at either end
        results.append(first_intersects or last_intersects)

    gdf[connection_info_column] = results
    connected_lines = gdf.loc[
        gdf[connection_info_column] == True  # noqa: E712
    ]
    unconnected_lines = gdf.loc[
        gdf[connection_info_column] == False  # noqa: E712
    ]
    return connected_lines, unconnected_lines


def check_reference_line_connections(  # noqa: SC200
    gdf: GeoDataFrame,
    threshold_distance: float,
    reference_gdf_list: list[GeoDataFrame],
    connection_info_column: str = "is_connected",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Check the connections of linestrings with respect to lines on the GeoDataFrame.

    Args:
    ----
        gdf (GeoDataframe): input GeoDataFrame consisting of Linestrings.
        threshold_distance (float): the max gap allowed for lines to
            be considered connected.
        reference_gdf_list (list): list of reference GeoDataFrames to which the
        connection_info_column (str): name of the attribute column where the info
            about connectedness is stored as a boolean

    Returns:
    -------
        A GeoDataframe of lines with an attribute that indicate the
        connectedness with respect to the linestrings in the input
        GeoDataFrame.

    """
    results = []
    other_lines = concat(reference_gdf_list)
    other_lines = GeoDataFrame(other_lines)

    for _idx, row in gdf.iterrows():
        line = row[gdf.geometry.name]

        # Extract first and last points of the line
        first_point = Point(line.coords[0])
        last_point = Point(line.coords[-1])

        # Create buffers around the first and last points
        first_buffer = first_point.buffer(threshold_distance)
        last_buffer = last_point.buffer(threshold_distance)

        # Check for intersections with other lines
        first_intersects = other_lines.intersects(first_buffer).any()
        last_intersects = other_lines.intersects(last_buffer).any()

        results.append(first_intersects or last_intersects)

    gdf[connection_info_column] = results

    connected_lines = gdf.loc[
        gdf[connection_info_column] == True  # noqa: E712
    ]
    unconnected_lines = gdf.loc[
        gdf[connection_info_column] == False  # noqa: E712
    ]

    return connected_lines, unconnected_lines


def detect_dead_ends(
    gdf: GeoDataFrame,
    threshold_distance: float,
    dead_end_info_column: str = "dead_end",
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Check the connections of linestrings with respect to lines on the GeoDataFrame.

    Args:
    ----
        gdf (GeoDataframe): input GeoDataFrame consisting of Linestrings.
        threshold_distance (float): the max gap allowed for lines to
            be considered connected.
        dead_end_info_column (str): Column name for the dead end information.

    Returns:
    -------
        A GeoDataframe of lines with an attribute that indicate the
        connectedness with respect to the linestrings in the input
        GeoDataFrame.

    """
    dead_ends = []
    first_point_connected = []
    last_point_connected = []

    for idx, row in gdf.iterrows():
        line = row[gdf.geometry.name]

        first_point = Point(line.coords[0])
        last_point = Point(line.coords[-1])

        first_buffer = first_point.buffer(threshold_distance)
        last_buffer = last_point.buffer(threshold_distance)

        # Check for intersections with other lines
        other_lines = gdf[gdf.index != idx]  # Exclude the current line
        first_intersects = other_lines.intersects(first_buffer).any()
        last_intersects = other_lines.intersects(last_buffer).any()

        dead_ends.append(first_intersects ^ last_intersects)
        first_point_connected.append(first_intersects)
        last_point_connected.append(last_intersects)

    # Add the results to the GeoDataFrame as a new column
    gdf[dead_end_info_column] = dead_ends
    gdf["first_intersects"] = first_point_connected
    gdf["last_intersects"] = last_point_connected
    normal_roads = gdf.loc[
        gdf[dead_end_info_column] == False  # noqa: E712
    ]
    dead_end_roads = gdf.loc[
        gdf[dead_end_info_column] == True  # noqa: E712
    ]

    return normal_roads, dead_end_roads


def inspect_dead_end_candidates(
    gdf: GeoDataFrame,
    threshold_distance: float,
    reference_gdf_list: list[GeoDataFrame],
    dead_end_conn_info_column: str = "dead_end_connects_to_ref_gdf",
) -> GeoDataFrame:
    """Check the connections of linestrings with respect to lines on the GeoDataFrame.

    Args:
    ----
        gdf (GeoDataframe): input GeoDataFrame consisting of Linestrings.
        threshold_distance (float): the max gap allowed for lines to
            be considered connected.
        reference_gdf_list (list): list of reference GeoDataFrames to which the input
            GeoDataFrame is compared for connections.
        dead_end_conn_info_column (str): name of the column where the connectedness of
            dead end linestrings are stored

    Returns:
    -------
        A GeoDataframe of lines with an attribute that indicate the
        connectedness with respect to the linestrings in the input
        GeoDataFrame.

    """
    new_connection = []
    other_lines = GeoDataFrame(concat(reference_gdf_list))
    gdf = GeoDataFrame(gdf, geometry=gdf.geometry.name)

    for _idx, row in gdf.iterrows():
        line = row[gdf.geometry.name]

        # Extract first and last points of the line
        if row["first_intersects"]:
            point = Point(line.coords[-1])
            buffer = point.buffer(threshold_distance)

        else:
            point = Point(line.coords[0])
            buffer = point.buffer(threshold_distance)

        # Check for intersections with other lines
        new_intersects = other_lines.intersects(buffer).any()

        new_connection.append(new_intersects)

    # Add the results to the GeoDataFrame as a new column
    gdf[dead_end_conn_info_column] = new_connection
    return gdf


def get_paths_along_roads(
    gdf_input: GeoDataFrame,
    gdf_reference: GeoDataFrame,
    detection_distance: float = 25.0,
) -> tuple[GeoDataFrame, GeoDataFrame]:
    """Flag lines in gdf_input that fall within a buffer around reference roads.

    Args:
    ----
        gdf_input: GeoDataFrame with LineString geometries to check.
        gdf_reference: GeoDataFrame with reference LineStrings.
        detection_distance: Distance to buffer around reference roads.

    Returns:
    -------
        tuple of GeoDataFrames divided by the attribute "along_ref", i.e.,
        whether they reside completely within detection_distance of the reference
        geometries.

    """
    # Single geometry for the reference roads buffer
    buffered_union = gdf_reference.geometry.buffer(detection_distance)
    buffered_union = buffered_union.union_all()

    gdf_input["along_ref"] = gdf_input.geometry.within(buffered_union)

    lines_along_ref = gdf_input.loc[gdf_input["along_ref"]]
    lines_independent_of_ref = gdf_input.loc[~gdf_input["along_ref"]]

    return lines_along_ref, lines_independent_of_ref


def flag_connections(
    input_gdf: GeoDataFrame,
    *,
    start_connected_column: str = "__start_connected",
    end_connected_column: str = "__end_connected",
) -> GeoDataFrame:
    """Flag which end of an input line is connected to dataset.

    Args:
    ----
        input_gdf: GeoDataFrame with LineString geometries to check.
        start_connected_column: Name of boolean Series which will be added to output,
            tells whether first vertex of an input geometry is connected to reference
            data.
        end_connected_column: Name of boolean Series which will be added to output,
            tells whether last vertex of an input geometry is connected to reference
            data.

    Returns:
    -------
        GeoDataFrame with boolean Series added, telling which ends each line is
        connected to the reference dataset, if any.

    """
    topological_points = get_topological_points(input_gdf.geometry)

    gdf = input_gdf.copy()
    gdf[start_connected_column] = gdf.geometry.apply(
        lambda geom: force_2d(Point(geom.coords[0]))
    )
    gdf[end_connected_column] = gdf.geometry.apply(
        lambda geom: force_2d(Point(geom.coords[-1]))
    )
    gdf[start_connected_column] = gdf[start_connected_column].apply(
        lambda geom: geom in topological_points,
    )
    gdf[end_connected_column] = gdf[end_connected_column].apply(
        lambda geom: geom in topological_points,
    )

    return gdf


def flag_connections_to_reference(
    input_gdf: GeoDataFrame,
    reference_gdf: GeoDataFrame,
    *,
    start_connected_column: str = "__start_connected",
    end_connected_column: str = "__end_connected",
) -> GeoDataFrame:
    """Flag which end of an input line is connected to reference dataset.

    Args:
    ----
        input_gdf: GeoDataFrame with LineString geometries to check.
        reference_gdf: GeoDataFrame with reference LineStrings.
        start_connected_column: Name of boolean Series which will be added to output,
            tells whether first vertex of an input geometry is connected to reference
            data.
        end_connected_column: Name of boolean Series which will be added to output,
            tells whether last vertex of an input geometry is connected to reference
            data.

    Returns:
    -------
        GeoDataFrame with boolean Series added, telling which ends each line is
        connected to the reference dataset, if any.

    """
    reference_union = reference_gdf.union_all()

    def _is_start_connected(geom: LineString) -> bool:
        return Point(geom.coords[0]).intersects(reference_union)

    def _is_end_connected(geom: LineString) -> bool:
        return Point(geom.coords[-1]).intersects(reference_union)

    gdf = input_gdf.copy()
    gdf[start_connected_column] = gdf.geometry.apply(lambda geom: Point(geom.coords[0]))
    gdf[end_connected_column] = gdf.geometry.apply(lambda geom: Point(geom.coords[-1]))
    gdf[start_connected_column] = gdf[start_connected_column].intersects(
        reference_union
    )
    gdf[end_connected_column] = gdf[end_connected_column].intersects(reference_union)

    return gdf


def connect_lines_to_polygon_centroids(
    source_lines: GeoDataFrame,
    reference_polygons: GeoDataFrame,
) -> GeoDataFrame:
    """Extend LineStrings in a GeoDataFrame to touching polygons centroid.

    Line will be extended either from its start or end point if either
    touch a polygon in the reference dataset.

    If the line does not touch a polygon at all, it will not change.

    Args:
    ----
        source_lines: Input data containing LineStrings.
        reference_polygons: Input data containing Polygons.

    Returns:
    -------
        GeoDataFrame with extended lines.

    """
    gdf = source_lines.copy()

    def _extend_line(line: LineString) -> LineString:
        start = Point(line.coords[0])
        end = Point(line.coords[-1])

        start_polys = reference_polygons.loc[
            reference_polygons.geometry.intersects(start.buffer(0.1))
        ]
        end_polys = reference_polygons.loc[
            reference_polygons.geometry.intersects(end.buffer(0.1))
        ]

        # Handle special case of line beginning and ending at the same polygon.
        if (start_polys.shape[0] == 1 and end_polys.shape[0] == 1) and (
            start_polys.geometry.to_numpy()[0] is end_polys.geometry.to_numpy()[0]
        ):
            # Extend from both ends, this has to be done instead of doing it in
            # steps (as below), because shapely.linemerge might change the
            # direction of the line, preventing the other end from being
            # extended.
            return extend_line_to_nearest(
                line,
                start_polys.centroid.union_all(),
                LineExtendFrom.BOTH,
            )

        if not start_polys.empty:
            # Because in theory the start/end point might touch two different
            # polygons, extend to the nearest point in the MultiPoint union of
            # all their centroids.
            line = extend_line_to_nearest(
                line,
                start_polys.centroid.union_all(),
                LineExtendFrom.START,
            )

        if not end_polys.empty:
            line = extend_line_to_nearest(
                line,
                end_polys.centroid.union_all(),
                LineExtendFrom.END,
            )

        return line

    gdf.geometry = gdf.geometry.apply(_extend_line)

    return gdf


def flag_polygon_centerline_connections(
    data: GeoDataFrame,
    reference_gdf: GeoDataFrame,
    polygon_geometry_column: str,
    *,
    start_connected_column: str = "__start_connected",
    end_connected_column: str = "__end_connected",
) -> GeoDataFrame:
    """Flag which "end" of a polygon is connected to reference data.

    Input data should contain LineStrings in the geometry column, formed
    as the centerline of a polygon, which should be in its own column.

    Args:
    ----
        data: Input GeoDataFrame, should have LineStrings in the active
            geometry column, and corresponding polygons in another column.
        reference_gdf: Reference GeoDataFrame to which connections are checked
            against.
        polygon_geometry_column: Name of geometry column containing polygons
            from which the centerline geometry in the active geometry column was
            formed.
        start_connected_column: Name of boolean Series which will be added to
            output, tells whether the polygon edge close to the first vertex of the
            linestring is connected to reference data.
        end_connected_column: Name of boolean Series which will be added to
            output, tells whether the polygon edge close to the last vertex of the
            linestring is connected to reference data.

    Returns:
    -------
        GeoDataFrame with boolean Series added, telling which ends each polygon is
        connected to the reference dataset, if either.

    """
    gdf = data.copy()

    reference_union = reference_gdf.union_all()

    def _process(
        centerline: LineString,
        original_polygon: Polygon,
        end: Literal[0, -1],
    ) -> bool:
        point = Point(centerline.coords[end])
        buffered = point.buffer(50)

        polygon = original_polygon.intersection(buffered)

        return polygon.intersects(reference_union)

    gdf[start_connected_column] = gdf[
        [gdf.geometry.name, polygon_geometry_column]
    ].apply(
        lambda columns: _process(
            columns[gdf.geometry.name],
            columns[polygon_geometry_column],
            end=0,
        ),
        axis=1,
    )
    gdf[end_connected_column] = gdf[[gdf.geometry.name, polygon_geometry_column]].apply(
        lambda columns: _process(
            columns[gdf.geometry.name],
            columns[polygon_geometry_column],
            end=-1,
        ),
        axis=1,
    )

    return gdf


def process_lines_and_reconnect(
    data: GeoDataFrame,
    process_function: Callable[[GeoDataFrame], GeoDataFrame],
    reconnect_to: GeoDataFrame | BaseGeometry,
    *,
    length_tolerance: float = 0.0,
) -> GeoDataFrame:
    """Do something to lines and if connections break reconnect them to reference data.

    Args:
    ----
        data: Input GeoDataFrame with LineStrings.
        process_function: Function which will be called to modify the `data`
            GeoDataFrame.
        reconnect_to: Geometry or GeoDataFrame to which lines whose connection
            was broken will be reconnected to.
        length_tolerance: If the would-be reconnected line segment is above
            this length it will not be reconnected.

    Returns:
    -------
        Processed and reconnected GeoDataFrame.

    """
    gdf = cast("GeoDataFrame", data.copy())
    boundaries = gdf.boundary.union_all()
    gdf = flag_connections(
        gdf,
        start_connected_column="__start_connected_before",
        end_connected_column="__end_connected_before",
    )

    gdf = process_function(gdf)

    gdf = flag_connections(
        gdf,
        start_connected_column="__start_connected_after",
        end_connected_column="__end_connected_after",
    )

    gdf["__start_was_cut"] = gdf.geometry.apply(
        lambda geom: force_2d(Point(geom.coords[0]))
    )
    gdf["__end_was_cut"] = gdf.geometry.apply(
        lambda geom: force_2d(Point(geom.coords[-1]))
    )
    data_union = data.union_all().buffer(0.1)
    gdf["__start_was_cut"] = gdf["__start_was_cut"].intersects(data_union) & gdf[
        "__start_was_cut"
    ].disjoint(boundaries)
    gdf["__end_was_cut"] = gdf["__end_was_cut"].intersects(data_union) & gdf[
        "__end_was_cut"
    ].disjoint(boundaries)

    # Effectively this checks which connections have been broken
    # in the process function.
    gdf["__extend_start"] = (
        gdf["__start_connected_before"] != gdf["__start_connected_after"]
    ) | gdf["__start_was_cut"]

    gdf["__extend_end"] = (
        gdf["__end_connected_before"] != gdf["__end_connected_after"]
    ) | gdf["__end_was_cut"]

    connect_to = (
        reconnect_to
        if isinstance(reconnect_to, BaseGeometry)
        else reconnect_to.union_all()
    )

    if not gdf.empty:
        gdf.geometry = gdf[["__extend_start", "__extend_end", gdf.geometry.name]].apply(
            lambda columns: extend_line_to_nearest(
                columns[gdf.geometry.name],
                connect_to,
                LineExtendFrom.from_bools(
                    extend_start=columns["__extend_start"],
                    extend_end=columns["__extend_end"],
                ),
                length_tolerance,
            ),
            axis=1,
        )

    return gdf.drop(
        [
            "__start_connected_before",
            "__end_connected_before",
            "__start_connected_after",
            "__end_connected_after",
            "__extend_start",
            "__extend_end",
            "__start_was_cut",
            "__end_was_cut",
        ],
        axis=1,
    )
