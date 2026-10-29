#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import shapely
from geopandas import GeoDataFrame
from networkx import dijkstra_predecessor_and_distance
from pydantic import Field
from shapely import STRtree, union_all
from shapely.geometry.base import BaseGeometry

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import gdf_to_networkx_graph
from geogenalg.core.parallel import in_parallel

if TYPE_CHECKING:
    from collections.abc import Hashable


@supports_identity
class GeneralizeRoadNetwork(BaseAlgorithm):
    """Generalizes a network of (smaller) roads against (larger) reference roads.

    This algorithm does the following steps:
    - Buffers each reference road and creates spatial index of them
    - Iterates each input class and it's feature in descending line length order
    (class at a time, features in order inside the class)
    - Keeps the feature if is long enough outside of all created buffers so far
    - Adds a buffered version of the kept feature to a growing spatial index
    - Restores connectivity for those kept roads that were originally connected to
    the reference roads but lost it in the previous steps
    """

    buffer_size: float = Field(100.0, ge=0.0)
    """Size used to buffer both reference road lines and kept road lines."""
    length_threshold: float = Field(50.0, ge=0.0)
    """The minimum length a road needs to have outside all buffered roads to be kept."""
    road_type_column: str = "road_type"
    """Column name used to select and order target road classes."""
    road_classes: ClassVar = (12141, 12316, 12314, 12313)
    """Road classes in descending processing order."""
    keep_unhandled: bool = False
    """Whether to keep features outside of input `road_classes` in the output."""
    reference_key: str = "roads"
    """Reference key for a mandatory dataset of (large) reference roads."""

    valid_input_geometry_types: ClassVar = {"LineString", "MultiLineString"}
    requires_projected_crs: ClassVar = True
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={"LineString", "MultiLineString"},
        ),
    }

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if data.empty:
            return data

        gdf = data.copy()
        gdf["_simplified"] = gdf.geometry.simplify(1)

        reference_roads = reference_data[self.reference_key]
        reference_road_buffers = (
            in_parallel(reference_roads.geometry.simplify(1))
            .buffer(
                self.buffer_size,
                resolution=4,
            )
            .resolve_parallel()
            .to_numpy()  # TODO: switch to resolve_numpy() once merged
        )
        reference_roads_tree = STRtree(reference_road_buffers)

        kept_indices: list[Hashable] = []
        kept_buffers: list[BaseGeometry] = []
        kept_tree = STRtree(kept_buffers)

        gdf_descending_length = gdf.assign(geom_len=gdf.geometry.length).sort_values(
            by="geom_len", ascending=False, kind="stable"
        )

        for road_class in self.road_classes:
            class_mask = gdf_descending_length[self.road_type_column] == road_class

            for idx, feature in gdf_descending_length.loc[class_mask].iterrows():
                geom = feature[str(gdf.geometry.name)]
                simplified = feature["_simplified"]

                outside_reference = self._difference_with_nearby(
                    geom, reference_roads_tree
                )
                if outside_reference.length <= self.length_threshold:
                    continue

                outside_all = self._difference_with_nearby(
                    outside_reference,
                    kept_tree,
                )
                if outside_all.length > self.length_threshold:
                    kept_indices.append(idx)
                    kept_buffers.append(
                        shapely.buffer(simplified, self.buffer_size, quad_segs=4),
                    )
                    kept_tree = STRtree(kept_buffers)

        # Restore connectivity to reference roads where possible
        kept_indices = list(
            self._preserve_connectivity_to_reference_lines(
                data=gdf,
                reference_roads=reference_roads,
                seed_indices=kept_indices,
            )
        )

        # NOTE: Is this needed?
        if self.keep_unhandled:
            unhandled_mask = ~gdf[self.road_type_column].isin(self.road_classes)
            kept_indices.extend(gdf.loc[unhandled_mask].index)

        gdf = gdf.drop("_simplified", axis=1)
        return gdf.loc[gdf.index.isin(set(kept_indices))]

    @staticmethod
    def _difference_with_nearby(geom: BaseGeometry, tree: STRtree) -> BaseGeometry:
        nearby = tree.query(geom, predicate="intersects")
        if len(nearby) == 0:
            return geom

        return geom.difference(union_all(tree.geometries.take(nearby)))

    def _preserve_connectivity_to_reference_lines(
        self,
        data: GeoDataFrame,
        reference_roads: GeoDataFrame,
        seed_indices: list,
    ) -> set:
        if not seed_indices:
            return set()

        lines = data.geometry.explode(index_parts=False)
        # Sort lines by length so that if parallel connections exist, shortest
        # survives while constructing the graph
        lines = lines.iloc[np.argsort(-lines.length.to_numpy(), kind="stable")]
        graph = gdf_to_networkx_graph(GeoDataFrame(geometry=lines, crs=data.crs))

        if graph.number_of_nodes() == 0:
            return set(seed_indices)

        # NOTE: Now rounding of coordinates, node key construction and processing a
        # line are repeated, might be a good idea to create utilities for them in
        # continuity.py
        kept_nodes = set()
        for line in lines[lines.index.isin(seed_indices)]:
            for point in (shapely.get_point(line, 0), shapely.get_point(line, -1)):
                kept_nodes.add((round(point.x, 3), round(point.y, 3)))

        nodes = list(graph.nodes)
        reference_roads_tree = STRtree(reference_roads.geometry.to_numpy())
        anchor_node_indices, _ = reference_roads_tree.query(
            shapely.points(nodes),
            predicate="dwithin",
            distance=0.001,
        )
        anchor_nodes = {nodes[i] for i in anchor_node_indices}

        if not anchor_nodes:
            return set(seed_indices)

        # Connect all anchor nodes to one network origin node
        network_origin = "network_origin"
        graph.add_edges_from(
            (network_origin, anchor, {"length": 0.0}) for anchor in anchor_nodes
        )

        # Compute the shortest-path predecessor of every node
        predecessors, _ = dijkstra_predecessor_and_distance(
            graph,
            network_origin,
            weight="length",
        )

        all_kept_indices = set(seed_indices)

        # Walk connecting roads from kept raods / nodes to reference roads.
        # Isolated roads that could not be connected but were kept in the previous
        # step are preserved.
        visited = set()
        for target_node in kept_nodes:
            current = target_node
            while (
                current != network_origin
                and current in predecessors
                and current not in visited
            ):
                visited.add(current)
                previous = predecessors[current][0]
                if previous != network_origin:
                    all_kept_indices.add(graph[current][previous]["idx"])
                current = previous

        return all_kept_indices
