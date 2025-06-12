#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def generalize_road_links(
    road_links_input: gpd.GeoDataFrame,
    shared_path_links_input: gpd.GeoDataFrame,
    path_links_input: gpd.GeoDataFrame,
    threshold_distance=10,
    threshold_length=75,
    shared_path_parallel_distance=25,
) -> dict[str, gpd.GeoDataFrame]:
    """Generalizes road links for the scale 1:50 000 by removing short and
    unconnected or dead end roads. The connectivity of the road links
    must be considered also with respect to the shared paths and paths
    layers. Shared paths that go along bigger roads are removed.

    Parameters
    ----------
        roads_input (GeoDataFrame): layer of road links linestrings
        shared_paths_input (GeoDataFrame): layer of shared path linestrings
        paths_input (GeoDataFrame): layer of paths linestrings
        threshold_distance (float): size of the buffer at the end nodes of the
        road link linestrings. If the buffer does not intersect with some
        other linestrings, the line will be considered unconnected on that end.
        threshold_length (float): Max. length of unconnected/dead end
        linestring that is removed.
        shared_path_parallel_distance (float): Max. distance of the shared path
        from road_link that gets removed if it goes parallel with it.

    """
    # The algorithm consists of the following steps:
    # 1) Check if the road link geometries are connected
    # to some other geometry within road layer
    # 2) For the unconnected geometries in step 1, see if
    # they are connected to some path or shared_path geometry
    # If so, mark them as connected. For those, which remain
    # unconnected, check if they are shorter than threshold length
    # and remove the short ones.
    # 3) Check the roads for dead ends, again first within the road
    # layer itself, and then compare the found dead ends to path and
    # shared path layers (as with connectedness). Remove short dead
    # ends.
    # 4) Additional step to remove the shared paths that go closely
    # along generalized roads

    road_links = road_links_input.copy()
    path_links = path_links_input.copy()
    shared_path_links = shared_path_links_input.copy()

    # Step 1
    road_links_1 = check_line_connections_original(road_links, threshold_distance)

    connected_part = road_links_1[road_links_1["is_connected"]]
    unconnected_candidates = road_links_1[~road_links_1["is_connected"]]

    # Step 2
    inspected_candidates = check_line_connections_multilayer(  # noqa: SC200
        unconnected_candidates,
        threshold_distance,
        [path_links, shared_path_links],
    )

    roads_to_keep_1 = inspected_candidates[inspected_candidates["is_connected"]]
    removed_short_unconnected = inspected_candidates[
        (~inspected_candidates["is_connected"])
        & (inspected_candidates["geometry"].length > threshold_length)
    ]

    road_links_2 = gpd.GeoDataFrame(
        pd.concat([connected_part, roads_to_keep_1, removed_short_unconnected]),
        geometry="geometry",
        crs=road_links.crs,
    )

    # Step 3
    road_links_3 = detect_dead_ends(road_links_2, threshold_distance)

    roads_regular = road_links_3[~road_links_3["dead_end"]]
    roads_dead_end = road_links_3[road_links_3["dead_end"]]

    inspected_dead_end_candidates = inspect_dead_end_candidates(
        roads_dead_end, threshold_distance, [path_links, shared_path_links]
    )

    road_links_4 = gpd.GeoDataFrame(
        pd.concat([roads_regular, inspected_dead_end_candidates]),
        geometry="geometry",
        crs=road_links.crs,
    )

    roads_to_keep = road_links_4[
        (road_links_4["dead_end"] is False)
        | (road_links_4["dead_end_connects_to_path"] is True)
    ]
    roads_to_generalize = road_links_4[
        (road_links_4["dead_end"] is True)
        & (road_links_4["dead_end_connects_to_path"] is False)
    ]

    remove_short_dead_ends = roads_to_generalize[
        roads_to_generalize["geometry"].length > threshold_length
    ]

    road_links_generalization_output = gpd.GeoDataFrame(
        pd.concat([roads_to_keep, remove_short_dead_ends]),
        geometry="geometry",
        crs=road_links.crs,
    )

    # Step 4
    roadside_shared_paths = detect_shared_paths_along_roads(
        shared_path_links_input,
        road_links_generalization_output,
        shared_path_parallel_distance,
    )

    shared_paths_generalization_output = roadside_shared_paths[
        ~roadside_shared_paths["roadside"]
    ]

    return {
        "road_links": road_links_generalization_output,
        "shared_paths": shared_paths_generalization_output,
        "paths": path_links,
    }


def check_line_connections_original(
    gdf: gpd.GeoDataFrame, threshold_distance: float
) -> gpd.GeoDataFrame:
    """Record as an attribute whether the lines of a GeoDataFrame are connected
    to any other line on the same GeoDataFrame. Threshold_distance is the
    max gap allowed for lines to be considered connected.
    """
    results = []

    for idx, row in gdf.iterrows():
        line = row.geometry

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

    gdf["is_connected"] = results
    return gdf


def check_line_connections_multilayer(  # noqa: SC200
    gdf: gpd.GeoDataFrame,
    threshold_distance: float,
    other_gdfs_list: list[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Record as an attribute whether the lines of a GeoDataFrame are connected
    to any other line on the given GeoDataFrames. Threshold_distance is the
    max gap allowed for lines to be considered connected.
    """
    results = []
    other_lines = pd.concat(other_gdfs_list)
    other_lines = gpd.GeoDataFrame(other_lines)

    for _idx, row in gdf.iterrows():
        line = row.geometry

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

    gdf["is_connected"] = results
    return gdf


def detect_dead_ends(
    gdf: gpd.GeoDataFrame,
    threshold_distance: float,
) -> gpd.GeoDataFrame:
    dead_ends = []
    first_point_connected = []
    last_point_connected = []

    for idx, row in gdf.iterrows():
        line = row.geometry

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
    gdf["dead_end"] = dead_ends
    gdf["first_intersects"] = first_point_connected
    gdf["last_intersects"] = last_point_connected
    return gdf


def inspect_dead_end_candidates(
    gdf: gpd.GeoDataFrame,
    threshold_distance: float,
    other_gdfs_list: list[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    new_connection = []
    other_lines = gpd.GeoDataFrame(pd.concat(other_gdfs_list))
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry")

    for _idx, row in gdf.iterrows():
        line = row.geometry

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
    gdf["dead_end_connects_to_path"] = new_connection
    return gdf


def detect_shared_paths_along_roads(
    gdf_shared_path: gpd.GeoDataFrame, gdf_road: gpd.GeoDataFrame, detection_distance=25
) -> gpd.GeoDataFrame:
    results = []

    # optionally add to buffer, cap_style="flat"
    buffered_roads = gdf_road.buffer(detection_distance)

    for _idx, row in gdf_shared_path.iterrows():
        line = row.geometry
        roadside = line.within(buffered_roads).any()

        results.append(roadside)

    # Add the results to the GeoDataFrame as a new column
    gdf_shared_path["roadside"] = results
    return gdf_shared_path
