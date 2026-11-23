#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar

import numpy as np
import shapely
from geopandas import GeoDataFrame
from networkx import dijkstra_predecessor_and_distance
from pandas import Categorical
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


@supports_identity
class GeneralizeRoadNetwork(BaseAlgorithm):
    """Generalizes a network of (smaller) roads against (larger) reference roads.

    This algorithm does the following steps:
    - Buffers each reference road and creates spatial index of them
    - Iterates each input class and it's feature in descending line length order
    (class at a time, features in order inside the class)
    - Keeps the feature if is long enough outside of all preserved buffered road
    geometries
    - Marks whether to feature was kept for subsequent road length checks
    - Restores connectivity for those kept roads that were originally connected to
    the reference roads but lost it in the previous steps
    """

    buffer_size: float = Field(100.0, ge=0.0)
    """Size used to buffer both reference road lines and kept road lines."""
    length_threshold: float = Field(50.0, ge=0.0)
    """The minimum length a road needs to have outside all buffered roads to be kept."""
    road_type_column: str = "road_type"
    """Column name used to select and order target road classes."""
    road_classes: tuple[int | str, ...] = ()
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

        # 1. Order roads by their class and length
        road_class = Categorical(
            data[self.road_type_column],
            categories=list(self.road_classes),
            ordered=True,
        )
        roads = data.assign(_class=road_class, _length=data.geometry.length)
        roads = roads.loc[roads["_class"].notna()]
        roads = roads.sort_values(
            by=["_class", "_length"], ascending=[True, False], kind="stable"
        )
        roads = roads.drop(columns=["_class", "_length"])

        # 2. Discard smaller roads mostly nearby the reference roads
        reference_roads = reference_data[self.reference_key]
        reference_roads_tree = self._create_tree_from_buffered(reference_roads)
        roads_parts_outside_reference = np.array(
            [
                self._difference_with_nearby(geom, reference_roads_tree)
                for geom in roads.geometry.to_numpy()
            ],
            dtype=object,
        )
        pass_mask = (
            shapely.length(roads_parts_outside_reference) > self.length_threshold
        )
        # Keep only roads and road parts that were long enough outside buffered
        # reference roads for the next steps
        roads = roads.loc[pass_mask]
        roads_parts_outside_reference = roads_parts_outside_reference[pass_mask]

        # 3. Discard smaller roads in greedy fashion by deciding keep vs.
        # discard one road at a time based on whether they are currently long enough
        # outside all buffered geometries. Tracking keep/discard decision separately as
        # bool list makes it possible to avoid rebuilding the tree
        roads_tree = self._create_tree_from_buffered(roads)
        kept = np.zeros(len(roads), dtype=bool)
        for i, geom in enumerate(roads_parts_outside_reference):
            # Subtract the so far kept other buffered road geometries and keep
            # this one if it is still long enough
            remaining_road = self._difference_with_nearby(geom, roads_tree, kept)
            if remaining_road.length > self.length_threshold:
                kept[i] = True
        kept_indices = roads.index[kept].tolist()

        # 4. Restore connectivity to reference roads where possible
        kept_indices = list(
            self._preserve_connectivity_to_reference_roads(
                data=data,
                reference_roads=reference_roads,
                seed_indices=kept_indices,
            )
        )

        # 5. Decide whether to keep or discard unhandled road classes
        if self.keep_unhandled:
            unhandled_mask = ~data[self.road_type_column].isin(self.road_classes)
            kept_indices.extend(data.loc[unhandled_mask].index)

        return data.loc[data.index.isin(set(kept_indices))]

    def _create_tree_from_buffered(self, gdf: GeoDataFrame) -> STRtree:
        return STRtree(
            in_parallel(gdf.geometry.simplify(1))
            .buffer(self.buffer_size, resolution=4)
            .resolve_numpy()
        )

    @staticmethod
    def _difference_with_nearby(
        geom: BaseGeometry,
        tree: STRtree,
        mask: np.ndarray | None = None,
    ) -> BaseGeometry:
        nearby = tree.query(geom, predicate="intersects")
        if mask is not None:
            nearby = nearby[mask[nearby]]
        if len(nearby) == 0:
            return geom

        return geom.difference(union_all(tree.geometries.take(nearby)))

    def _preserve_connectivity_to_reference_roads(
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
