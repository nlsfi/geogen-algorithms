#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override
from uuid import uuid4

from geopandas import GeoDataFrame
from networkx import connected_components
from pydantic import Field
from shapely import union_all
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

from geogenalg.analyze import polygonize_parallel_lines
from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.attributes import inherit_attributes_from_largest
from geogenalg.continuity import (
    add_contiguous_lines_information,
    flag_connections,
    gdf_to_networkx_graph,
)
from geogenalg.identity import hash_duplicate_indexes
from geogenalg.selection import rank_parallel_lines
from geogenalg.utility.dataframe_processing import combine_gdfs

if TYPE_CHECKING:
    from networkx.classes.graph import Graph


@supports_identity
class GeneralizeRailroads(BaseAlgorithm):
    """Generalizes linear railroad features.

    This largely based on the algorithm proposed by

    Savino S., & Touya G. (2015) Automatic Structure Detection and
    Generalization of Railway Networks.

    However, this algorithm is not a complete or an exact implementation of the
    one described in the article, most notably in how fan tracks are
    identified.

    The algorithm does the following:
    - Identifies different railway structures and typifies them as either
      fans, packs, clusters or free tracks
    - Fans and packs which are structures with many parallel lines are
      reduced by only keeping the outermost tracks
    - Remaining deadend cluster tracks are removed
    """

    fan_minimum_length: float = Field(400, ge=0.0)
    """Deadend tracks shorter than this are considered potential fan tracks."""
    pack_cluster_length_threshold: float = Field(200, ge=0.0)
    """Distance used to determine tracks which are of the same cluster."""

    valid_input_geometry_types: ClassVar = {
        "LineString",
    }

    @staticmethod
    def _cluster_graph(graph: Graph, threshold: float) -> list:
        g = graph.copy()
        for u, v, data in graph.edges(data=True):
            if data["length"] >= threshold:
                g.remove_edge(u, v)

        return list(connected_components(g))

    @override
    def _execute(  # noqa: C901, PLR0912, PLR0915, PLR0914
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        # First, find potential fan tracks which are sets of parallel deadend
        # tracks The approach used here is different to the one described in
        # the article
        fan_candidates = add_contiguous_lines_information(gdf)
        fan_candidates = fan_candidates.loc[
            (fan_candidates["contiguous_length"] < self.fan_minimum_length)
            & fan_candidates["contiguous_dead_end"]
        ]

        fan_areas = polygonize_parallel_lines(
            fan_candidates,
            6,
        )

        fan_areas.geometry = fan_areas.buffer(-4).buffer(4)

        # This adds an attribute to each potential fan track based on the area
        # it intersects
        fan_candidates["fan_area_id"] = -1
        fan_tracks = fan_candidates.copy()
        for i, area in enumerate(fan_areas.geometry):
            in_area = fan_candidates.loc[fan_candidates.geometry.intersects(area)]

            if in_area.empty:
                continue

            fan_tracks.loc[fan_tracks.index.isin(in_area.index), "fan_area_id"] = i

        fan_tracks = fan_tracks.loc[fan_tracks["fan_area_id"] != -1]

        pack_candidates = gdf.loc[~gdf.index.isin(fan_tracks.index)]

        # Find the indexes of the outermost lines in the identified fan tracks
        remaining_fans = []
        for fan_area in fan_tracks["fan_area_id"].unique():
            if fan_area in {"free_track", "cluster_track"}:
                continue

            fan_area_lines = fan_tracks.loc[fan_tracks["fan_area_id"] == fan_area]
            ranked, _ = rank_parallel_lines(fan_area_lines)
            remaining_fans.extend([ranked.idxmin(), ranked.idxmax()])

        # Keep only the outermost lines.
        fan_tracks = fan_tracks.loc[fan_tracks.index.isin(remaining_fans)]
        fan_tracks = fan_tracks.loc[fan_tracks["fan_area_id"] != -1]

        # Start typifying the remaining track types
        pack_candidates = (
            GeoDataFrame(
                geometry=[
                    linemerge(pack_candidates.union_all())
                    if isinstance(pack_candidates.union_all(), MultiLineString)
                    else pack_candidates.union_all()
                ],
                crs=data.crs,
            )
            .explode()
            .reset_index()
        )
        graph = gdf_to_networkx_graph(pack_candidates)
        clusters = GeneralizeRailroads._cluster_graph(
            graph, self.pack_cluster_length_threshold
        )

        # Define a unique id for each node and each cluster
        for node in graph.nodes():
            graph.nodes[node]["uuid"] = str(uuid4())

        for cluster in clusters:
            cluster_id = str(uuid4()) if len(cluster) > 1 else -1
            for node in cluster:
                graph.nodes[node]["cluster"] = cluster_id

        codes = set()
        tracks_dict: dict[str, list] = {}
        for u, v in graph.edges():
            u_cluster = graph.nodes[u]["cluster"]
            v_cluster = graph.nodes[v]["cluster"]

            # This code forms a "connection code" to each edge, as described in
            # the article.

            u_id = graph.nodes[u]["uuid"] if u_cluster == -1 else u_cluster
            v_id = graph.nodes[v]["uuid"] if v_cluster == -1 else v_cluster
            connection_code = ", ".join(sorted([u_id, v_id]))

            if u_cluster != -1 and v_cluster != -1 and u_cluster == v_cluster:
                connection_code = "cluster_track"

            codes.add(connection_code)

            if tracks_dict.get(connection_code) is None:
                tracks_dict[connection_code] = [graph.edges[u, v]["geometry"]]
            else:
                tracks_dict[connection_code].append(graph.edges[u, v]["geometry"])

        features = []
        for code, geoms in tracks_dict.items():
            features.append(
                {
                    "pack_id": code if len(geoms) > 1 else "free_track",
                    data.geometry.name: union_all(geoms),
                }
            )

        tracks = (
            GeoDataFrame(
                features,
                geometry=data.geometry.name,
                crs=data.crs,
            )
            .explode()
            .reset_index()
        )

        # Tracks are now defined as being either part of a pack, cluster or a free track
        for track in tracks["pack_id"].unique():
            if track in {"free_track", "cluster_track"}:
                continue

            track_lines = tracks.loc[tracks["pack_id"] == track]
            mean_length = track_lines.geometry.length.mean()

            if mean_length > 1000:  # noqa: PLR2004
                tracks.loc[
                    tracks.index.isin(track_lines.index),
                    "pack_id",
                ] = "free_track"
                continue

            rank, disjoint = rank_parallel_lines(track_lines)

            # In case there is an errant track in a pack which is not actually parallel
            # to the other ones, define it as a free track
            tracks.loc[
                tracks.index.isin(disjoint),
                "pack_id",
            ] = "free_track"

            if rank.empty:
                continue

            # We're handling only pack tracks at this point, similar to fan tracks, only
            # keep the outermost ones.
            to_keep = [rank.idxmax(), rank.idxmin(), *disjoint]
            to_remove = track_lines.loc[~track_lines.index.isin(to_keep)].index

            tracks = tracks.loc[~tracks.index.isin(to_remove)]

        # Now that a lot of lines from packs are removed, cluster tracks are
        # left as deadends. Continually remove dead ends until none remain.
        tracks = combine_gdfs([tracks, fan_tracks])
        while True:
            tracks = flag_connections(tracks)
            cluster_tracks = tracks.loc[tracks["pack_id"] == "cluster_track"]

            to_remove = cluster_tracks.loc[
                ~(cluster_tracks["_start_connected"] & cluster_tracks["_end_connected"])
            ]

            if to_remove.empty:
                break

            tracks = tracks.loc[~tracks.index.isin(to_remove.index)]

        # Additionally remove any totally unconnected lines
        both_disconnected = (~tracks["_start_connected"]) & (~tracks["_end_connected"])
        tracks = tracks.loc[~(both_disconnected)]

        tracks = tracks.set_index(tracks.index.astype("string"))
        tracks = inherit_attributes_from_largest(data, tracks)
        return hash_duplicate_indexes(tracks, "railroads")
