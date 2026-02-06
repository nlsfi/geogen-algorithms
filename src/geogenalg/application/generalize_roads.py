#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame
from pandas import concat

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import (
    check_line_connections,
    check_reference_line_connections,
    detect_dead_ends,
    inspect_dead_end_candidates,
)

CONNECTION_INFO_COLUMN = "is_connected"
DEAD_END_INFO_COLUMN = "dead_end"
DEAD_END_CONN_INFO_COLUMN = "dead_end_connects_to_ref_gdf"


@supports_identity
@dataclass(frozen=True)
class GeneralizeRoads(BaseAlgorithm):
    """Removes short unconnected linestrings from the network.

    By selection generalization, linestrings in the input GeoDataFrame
    are removed if 1.) they are non-connected or partly non-connected (dead-ends)
    and 2.) too short. The connectedness is determined also with respect to
    linestrings in the reference data, if provided.
    """

    threshold_distance: float = 10.0
    """Size of the buffer at the end vertices of the road linestrings. If the buffer
    does not intersect with any other linestrings, the line will be considered
    unconnected on that end."""
    threshold_length: float = 75.0
    """Unconnected/dead-end linestring shorter than this will be removed."""

    valid_input_geometry_types: ClassVar = {"LineString"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        input_gdf_columns = list(gdf.columns)

        connected_roads, unconnected_roads = check_line_connections(
            gdf, self.threshold_distance, CONNECTION_INFO_COLUMN
        )
        if list(reference_data.values()) and unconnected_roads is not None:
            roads_connected_to_reference, completely_unconnected_roads = (
                check_reference_line_connections(
                    unconnected_roads,
                    self.threshold_distance,
                    list(reference_data.values()),
                )
            )

            # save roads that are connected to other roads or reference paths or are
            # long enough by themselves.
            connected_roads = GeoDataFrame(
                concat(
                    [
                        connected_roads,
                        roads_connected_to_reference,
                        completely_unconnected_roads[
                            completely_unconnected_roads.geometry.length
                            > self.threshold_length
                        ],
                    ]
                ),
                geometry=gdf.geometry.name,
                crs=gdf.crs,
            )

        normal_roads, dead_end_roads = detect_dead_ends(
            connected_roads, self.threshold_distance
        )

        inspected_dead_end_candidates = inspect_dead_end_candidates(
            dead_end_roads, self.threshold_distance, list(reference_data.values())
        )
        roads = GeoDataFrame(
            concat([normal_roads, inspected_dead_end_candidates]),
            geometry=gdf.geometry.name,
            crs=gdf.crs,
        )

        # Ensure that attribute is boolean and also default to a save road link
        # if there is no data
        roads[DEAD_END_CONN_INFO_COLUMN] = (
            roads[DEAD_END_CONN_INFO_COLUMN].astype("boolean").fillna(True)  # noqa: FBT003
        )
        roads_to_keep = roads[
            (~roads[DEAD_END_INFO_COLUMN]) | (roads[DEAD_END_CONN_INFO_COLUMN])
        ]

        roads_to_generalize = roads[
            roads[DEAD_END_INFO_COLUMN] & (~roads[DEAD_END_CONN_INFO_COLUMN])
        ]
        long_enough_dead_ends = roads_to_generalize[
            roads_to_generalize.geometry.length > self.threshold_length
        ]

        generalized_roads = concat([roads_to_keep, long_enough_dead_ends])
        generalized_roads = generalized_roads.drop(
            [
                column
                for column in generalized_roads.columns
                if column not in input_gdf_columns
            ],
            axis=1,
        )
        generalized_roads = GeoDataFrame(generalized_roads)
        generalized_roads = generalized_roads.set_geometry(gdf.geometry.name)

        return generalized_roads.set_crs(gdf.crs)
