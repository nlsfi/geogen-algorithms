#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from __future__ import annotations

from math import cos, radians, sin
from typing import TYPE_CHECKING, ClassVar, override
from uuid import uuid4

from cartagen.enrichment.network.strokes import (
    strokes_roads,
)
from geopandas import GeoDataFrame
from networkx import connected_components
from pydantic import Field
from shapely import union_all
from shapely.geometry import LineString, Point
from shapely.ops import linemerge

from geogenalg.analyze import (
    polygonize_parallel_lines,
    split_polygons_to_sections_on_approximate_width,
)
from geogenalg.application import BaseAlgorithm
from geogenalg.continuity import (
    add_contiguous_lines_information,
    flag_connections,
    gdf_to_networkx_graph,
)
from geogenalg.core.geometry import (
    angle_difference,
    line_average_direction,
)
from geogenalg.utility.dataframe_processing import combine_gdfs

if TYPE_CHECKING:
    from networkx.classes.graph import Graph


# This is is not used, but might be useful to have. Definitely in the wrong module though
def remove_overlaps(input_gdf: GeoDataFrame):
    if input_gdf.index.has_duplicates:
        msg = "Input cannot have duplicate indexes."
        raise ValueError(msg)

    gdf = input_gdf.copy()
    gdf["length"] = gdf.geometry.length
    gdf = gdf.sort_values("length", ascending=False)

    for idx, row in gdf.iterrows():
        geometry = row[gdf.geometry.name]

        intersecting_geoms = input_gdf.loc[
            (input_gdf.index != idx) & (input_gdf.geometry.overlaps(geometry))
        ]

        gdf.at[idx, gdf.geometry.name] = geometry.difference(
            intersecting_geoms.union_all()
        )

    return gdf


def cluster_graph(graph: Graph, threshold: float):
    # This clusters nodes by proximity and connectivity in a graph.
    # It is meant to implement the node clustering part of Touya & Savino's algorithm.

    g = graph.copy()
    for u, v, data in graph.edges(data=True):
        if data["length"] >= threshold:
            g.remove_edge(u, v)

    return list(connected_components(g))


class GeneralizeRailroads(BaseAlgorithm):
    """Keep intersecting sections from input data.

    Identifies which sections in input data intersect with areas in mask data
    and returns a copy of input data with only intersecting sections remaining.

    If feature in input data has zero intersection with mask, it will be
    removed.

    Keeps attributes intact. If a feature is split by the mask, its parts will
    be turned to new features and their IDs hashed. New vertices introduced by
    cuts get Z value via linear interpolation along the affected edges.
    """

    parallel_distance: float = Field(12.0, ge=0.0)
    """..."""
    section_width_bins: frozenset[int] = frozenset([0, 18, 40, 100, 10000000])
    """..."""
    max_dead_end_length: float = Field(100, ge=0.0)
    """..."""
    use_cache: bool = False
    """..."""

    valid_input_geometry_types: ClassVar = {
        "LineString",
    }

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        # FIXME: this and many other functions save intermediate GeoPackages for inspection.
        # Those bits have to be removed, of course, look for "to_file" calls.

        # First we identify so called "fan tracks" as described in Touya & Savino's paper.
        # The identification works a bit differently from the paper, we just find dead ends
        # which are short enough and parallelize them to find if they are in a group or
        # potentially just single dead ends

        # TODO: check if popping to networkx is faster?
        fan_candidates = add_contiguous_lines_information(gdf)
        fan_candidates = fan_candidates.loc[
            (fan_candidates["contiguous_length"] < 400)
            & fan_candidates["contiguous_dead_end"]
        ]

        fan_areas = polygonize_parallel_lines(
            fan_candidates,
            6,
        )

        fan_areas.geometry = fan_areas.buffer(-4).buffer(4)

        fan_areas.to_file("fan_areas.gpkg")

        # This adds an attribute to each potential fan track based on the area it intersects
        fan_candidates["fan_area_id"] = -1
        fan_tracks = fan_candidates.copy()
        for i, area in enumerate(fan_areas.geometry):
            in_area = fan_candidates.loc[fan_candidates.geometry.intersects(area)]

            if in_area.empty:
                continue

            fan_tracks.loc[fan_tracks.index.isin(in_area.index), "fan_area_id"] = i

        fan_tracks = fan_tracks.loc[fan_tracks["fan_area_id"] != -1]
        pack_candidates = gdf.loc[~gdf.index.isin(fan_tracks.index)]

        # This prunes the fan track areas by projecting a perpendicular line to the average
        # direction of the tracks and keeps only the border lines. This is very similar to
        # how packs are pruned later on, and should definitely be refactored to a function
        # or similar. It could also probably be changed to keep more of the lines instead of
        # the extremes
        for fan_area in fan_tracks["fan_area_id"].unique():
            if fan_area in {"free_track", "cluster_track"}:
                continue

            fan_area_lines = fan_tracks.loc[fan_tracks["fan_area_id"] == fan_area]

            centroid = fan_area_lines.union_all().centroid
            direction = line_average_direction(fan_area_lines.union_all()) + 90
            direction_radians = radians(direction)

            dx = cos(direction_radians) * 10000
            dy = sin(direction_radians) * 10000

            perpendicular_line = LineString(
                [
                    (centroid.x - dx, centroid.y - dy),
                    (centroid.x + dx, centroid.y + dy),
                ]
            )

            # Mark those lines which don't intersect the perpendicular line as
            # not belonging to area and filter them out of the lines currently
            # being processed
            fan_tracks.loc[
                fan_tracks.index.isin(
                    fan_area_lines.loc[
                        fan_area_lines.disjoint(perpendicular_line)
                    ].index
                ),
                "fan_area_id",
            ] = -1
            fan_area_lines = fan_area_lines.loc[
                fan_area_lines.intersects(perpendicular_line)
            ]

            if fan_area_lines.shape[0] <= 2:
                continue

            intersections = fan_area_lines.geometry.intersection(perpendicular_line)
            distances = intersections.distance(Point(perpendicular_line.coords[0]))
            distance_ranks = distances.rank()

            to_keep = (distance_ranks.idxmin(), distance_ranks.idxmax())
            to_remove = fan_area_lines.loc[~fan_area_lines.index.isin(to_keep)].index

            fan_tracks = fan_tracks.loc[~fan_tracks.index.isin(to_remove)]

        fan_tracks = fan_tracks.loc[fan_tracks["fan_area_id"] != -1]
        fan_tracks.to_file("fan_tracks.gpkg")

        # This polygon based business is for convenience in testing. The network based approach as in the article
        # works pretty well for certain areas, but in some other areas it breaks. The current plan is to combine
        # a "polygon" based approach, which polygonizes railtracks which could not be properly typified by the
        # network based approach, and then we use the polygons to identify where to prune tracks in.
        # It probably needs changing though, maybe such that we "travel" a centerline down it and split the
        # polygon into parts, maybe based on abrupt direction change or change in number of parallel lines?
        # That way you could identify areas to prune with a bit more control, potentially
        POLYGON_BASED = False
        if POLYGON_BASED:
            single_lines = pack_candidates.copy()

            # "pack" here is a misnomer, these are not really packs as described in the article
            pack_areas = split_polygons_to_sections_on_approximate_width(
                polygonize_parallel_lines(
                    pack_candidates,
                    20,
                ),
                [0, 20, 40, 100000],
                5,
                direction_precision=90,
            ).drop("centerline", axis=1)

            pack_candidates = (
                GeoDataFrame(
                    geometry=[linemerge(pack_candidates.union_all())],
                    crs=data.crs,
                )
                .explode()
                .reset_index()
            )

            single_lines = single_lines.loc[
                (
                    (
                        single_lines.geometry.intersection(
                            pack_areas.buffer(self.parallel_distance - 1).union_all()
                        ).length
                        / single_lines.geometry.length
                    )
                    < 0.5
                )
            ]
            pack_areas.to_file("split_to_sections.gpkg")

            pack_candidates.to_file("pack_candidates.gpkg")
            gdfs = []

            # This loop is pretty bad, basically it finds lines per polygon section (mislabeled as pack_area),
            # and finds the lines which are on either extreme side (left/right) and leaves lines in the middle of the area
            # out. This is done using single sided buffers to check if each line has any parallel lines on either side.
            # This bit is pretty slow and overly complicated and ideally there would be an alternative approach.
            for area in pack_areas.geometry:
                lines = pack_candidates.loc[
                    pack_candidates.geometry.intersects(
                        area.buffer(self.parallel_distance - 1),
                    )
                ].copy()

                lines.geometry = lines.geometry.intersection(
                    area.buffer(self.parallel_distance - 1)
                )
                lines = lines.explode()
                lines = lines.loc[
                    ~(lines.geometry.is_empty) & (lines.geometry.type == "LineString")
                ]
                lines["orientation"] = lines.geometry.apply(line_average_direction)

                lines["positive_buffer"] = lines.geometry.buffer(
                    self.parallel_distance + 1,
                    single_sided=True,
                ).buffer(-1)
                lines["negative_buffer"] = lines.geometry.buffer(
                    -(self.parallel_distance + 1),
                    single_sided=True,
                ).buffer(-1)

                def _do(a, b) -> float:
                    return angle_difference(a, b)

                for idx, line in lines.iterrows():
                    positive_buffer = line["positive_buffer"]
                    negative_buffer = line["negative_buffer"]

                    angle_diff_in_range = (
                        lines["orientation"].apply(
                            _do,
                            b=line["orientation"],
                        )
                        < 15
                    )
                    positive_intersections = lines.loc[
                        (lines.index != idx)
                        & (angle_diff_in_range)
                        & (lines.intersects(positive_buffer))
                    ]
                    negative_intersections = lines.loc[
                        (lines.index != idx)
                        & (angle_diff_in_range)
                        & (lines.intersects(negative_buffer))
                    ]

                    lines.at[idx, "positive_intersections"] = len(
                        positive_intersections
                    )
                    lines.at[idx, "negative_intersections"] = len(
                        negative_intersections
                    )

                lines = lines.drop(["positive_buffer", "negative_buffer"], axis=1)
                lines = lines.loc[
                    (lines["positive_intersections"] == 0)
                    | (lines["negative_intersections"] == 0)
                ]

                lines = (
                    GeoDataFrame(geometry=[linemerge(lines.union_all())], crs=data.crs)
                    .explode()
                    .reset_index()
                )

                gdfs.append(lines)

            gdf = combine_gdfs(gdfs).explode().reset_index()
            gdf = (
                GeoDataFrame(geometry=[linemerge(gdf.union_all())], crs=data.crs)
                .explode()
                .reset_index()
            )
            gdf = strokes_roads(gdf, attributes=[])
            gdf = gdf.loc[gdf.geometry.length > 100]

            single_lines.to_file("single_lines.gpkg")

            return gdf
        # NETWORK BASED CLUSTER                                noqa: RET505
        pack_candidates = (
            GeoDataFrame(
                geometry=[linemerge(pack_candidates.union_all())],
                crs=data.crs,
            )
            .explode()
            .reset_index()
        )
        graph = gdf_to_networkx_graph(pack_candidates)

        clusters = []
        threshold = 200

        # This commented out piece of code is an alternative approach to clustering nodes in the graph/network.
        # It is a recursive approach, which in practice produces fairly similar results, but I'm leaving it
        # just in case

        # def _ride_graph(
        #     node: tuple[float, float],
        #     add_clusters: list,
        # ):
        #     if node in visited_nodes:
        #         return
        #
        #     visited_nodes.add(node)
        #     add_clusters.append(node)
        #
        #     for neighbor in graph.neighbors(node):
        #         if graph.get_edge_data(node, neighbor)["length"] >= threshold:
        #             continue
        #
        #         _ride_graph(neighbor, add_clusters)
        #
        #
        # for node in graph.nodes():
        #     if node in visited_nodes:
        #         continue
        #
        #     cluster = []
        #     _ride_graph(node, cluster)
        #     clusters.append(cluster)

        # This is the other approach for clustering. Much simpler (and probably faster).
        clusters = cluster_graph(graph, threshold)

        # All this business defines a unique id for each node and each cluster
        for node in graph.nodes():
            graph.nodes[node]["uuid"] = str(uuid4())

        for cluster in clusters:
            cluster_id = str(uuid4()) if len(cluster) > 1 else -1
            for node in cluster:
                graph.nodes[node]["cluster"] = cluster_id

        # This bit is just for development convenience, it saves the clusters to a file which can be inspected.
        # Not relevant for the later parts and ultimately to be deleted.
        c = []
        for i, cluster in enumerate(clusters):
            c.extend(
                {
                    "id": graph.nodes[node]["uuid"],
                    "cluster_id": graph.nodes[node]["cluster"],
                    "geometry": Point(node),
                }
                for node in cluster
            )

        GeoDataFrame(c, geometry="geometry", crs=data.crs).dissolve(
            "id", as_index=False
        ).to_file("clusters.gpkg")

        codes = set()
        packs = {}
        for u, v in graph.edges():
            u_cluster = graph.nodes[u]["cluster"]
            v_cluster = graph.nodes[v]["cluster"]

            # This code forms a "connection code" to each edge, as described in the article.

            u_id = graph.nodes[u]["uuid"] if u_cluster == -1 else u_cluster
            v_id = graph.nodes[v]["uuid"] if v_cluster == -1 else v_cluster
            connection_code = ", ".join(sorted([u_id, v_id]))

            if u_cluster != -1 and v_cluster != -1 and u_cluster == v_cluster:
                connection_code = "cluster_track"

            codes.add(connection_code)

            if packs.get(connection_code) is None:
                packs[connection_code] = [graph.edges[u, v]["geometry"]]
            else:
                packs[connection_code].append(graph.edges[u, v]["geometry"])

        features = []
        for code, geoms in packs.items():
            features.append(
                {
                    "pack_id": code if len(geoms) > 1 else "free_track",
                    "geometry": union_all(geoms),
                }
            )

        # "packs" here is a slight misnomer, because this GDF also includes free and cluster tracks
        packs = (
            GeoDataFrame(
                features,
                geometry="geometry",
                crs=data.crs,
            )
            .explode()
            .reset_index()
        )

        combine_gdfs([packs, fan_tracks]).to_file(
            "packs_before_pruning.gpkg"
        )  # For development, unnecessary

        # This prunes the packs (leaving cluster and free tracks intact). Similary approach as with fan tracks,
        # same comments apply as above.
        for pack in packs["pack_id"].unique():
            if pack in {"free_track", "cluster_track"}:
                continue

            pack_lines = packs.loc[packs["pack_id"] == pack]
            mean_length = pack_lines.geometry.length.mean()

            if mean_length > 1000:
                packs.loc[packs.index.isin(pack_lines.index), "pack_id"] = "free_track"
                continue

            centroid = pack_lines.union_all().centroid
            direction = line_average_direction(pack_lines.union_all()) + 90
            direction_radians = radians(direction)

            dx = cos(direction_radians) * 10000
            dy = sin(direction_radians) * 10000

            perpendicular_line = LineString(
                [
                    (centroid.x - dx, centroid.y - dy),
                    (centroid.x + dx, centroid.y + dy),
                ]
            )

            # Mark those lines which don't intersect the perpendicular line
            # as free tracks and filter them out of the lines currently
            # being processed
            packs.loc[
                packs.index.isin(
                    pack_lines.loc[pack_lines.disjoint(perpendicular_line)].index
                ),
                "pack_id",
            ] = "free_track"
            pack_lines = pack_lines.loc[pack_lines.intersects(perpendicular_line)]

            if pack_lines.shape[0] <= 2:
                continue

            intersections = pack_lines.geometry.intersection(perpendicular_line)
            distances = intersections.distance(Point(perpendicular_line.coords[0]))
            distance_ranks = distances.rank()

            to_keep = (distance_ranks.idxmin(), distance_ranks.idxmax())
            to_remove = pack_lines.loc[~pack_lines.index.isin(to_keep)].index

            packs = packs.loc[~packs.index.isin(to_remove)]

        # Now that a lot of lines from packs are removed, cluster tracks are left as deadends. Continually remove dead ends
        # until none remain.
        packs = combine_gdfs([packs, fan_tracks])
        while True:
            packs = flag_connections(packs)
            cluster_tracks = packs.loc[packs["pack_id"] == "cluster_track"]

            to_remove = cluster_tracks.loc[
                ~(cluster_tracks["_start_connected"] & cluster_tracks["_end_connected"])
            ]

            if to_remove.empty:
                break

            packs = packs.loc[~packs.index.isin(to_remove.index)]

        # Additionally remove any totally unconnected lines
        both_disconnected = (~packs["_start_connected"]) & (~packs["_end_connected"])
        packs = packs.loc[~(both_disconnected)]
        packs.to_file("packs.gpkg")

        # This returned GDF is a bit odd, but really this function should continue from here
        # to further processing.
        return pack_candidates
