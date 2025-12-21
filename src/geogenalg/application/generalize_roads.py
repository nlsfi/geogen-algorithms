#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from geopandas import GeoDataFrame
from pandas import concat

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import (
    check_line_connections,
    check_reference_line_connections,
    detect_dead_ends,
    inspect_dead_end_candidates,
)
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type

CONNECTION_INFO_COLUMN = "is_connected"
DEAD_END_INFO_COLUMN = "dead_end"
DEAD_END_CONN_INFO_COLUMN = "dead_end_connects_to_ref_gdf"


@supports_identity
@dataclass(frozen=True)
class GeneralizeRoads(BaseAlgorithm):
    """Removes short unconnected linestrings from the network."""

    """By selection generalization, linestrings in the input GeoDataFrame
    are removed if 1.) they are non-connected or partly non-connected (dead-ends)
    and 2.) too short. The connectedness is determined also with respect to
    linestrings in the reference data, if provided.
    """

    threshold_distance: float = 10.0
    """size of the buffer at the end vertices of the road linestrings. If the buffer
    does not intersect with any other linestrings, the line will be considered
    unconnected on that end."""

    threshold_length: float = 75.0
    """Unconnected/dead-end linestring shorter than this will be removed."""
    connection_info_column: str = "is_connected"
    """Column name for which the connection information is stored as a boolean"""

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if not check_gdf_geometry_type(data, ["LineString"]):
            msg = "GeneralizeRoads works only with LineString geometries"
            raise GeometryTypeError(msg)

        if reference_data:
            for gdf in list(reference_data.values()):
                if not check_gdf_geometry_type(gdf, ["LineString"]):
                    msg = "Reference data must only contain LineStrings"
                    raise GeometryTypeError(msg)

        data = data.copy()

        input_gdf_columns = list(data.columns)

        connected_roads, unconnected_roads = check_line_connections(
            data, self.threshold_distance, CONNECTION_INFO_COLUMN
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
                            completely_unconnected_roads["geometry"].length
                            > self.threshold_length
                        ],
                    ]
                ),
                geometry="geometry",
                crs=data.crs,
            )

        normal_roads, dead_end_roads = detect_dead_ends(
            connected_roads, self.threshold_distance
        )

        inspected_dead_end_candidates = inspect_dead_end_candidates(
            dead_end_roads, self.threshold_distance, list(reference_data.values())
        )
        roads = GeoDataFrame(
            concat([normal_roads, inspected_dead_end_candidates]),
            geometry="geometry",
            crs=data.crs,
        )
        """Ensure that attribute is boolean and also default to a save road link if
        there is no data."""
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
            roads_to_generalize["geometry"].length > self.threshold_length
        ]

        generalized_roads = concat([roads_to_keep, long_enough_dead_ends])

        return GeoDataFrame(
            generalized_roads.drop(
                columns=[
                    column
                    for column in list(generalized_roads.columns)
                    if column not in input_gdf_columns
                ]
            ),
            geometry="geometry",
            crs=data.crs,
        )
