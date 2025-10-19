#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from geopandas import GeoDataFrame
from pandas import concat

from geogenalg.application import BaseAlgorithm
from geogenalg.continuity import (
    check_line_connections,
    check_reference_line_connections,
    detect_dead_ends,
    inspect_dead_end_candidates,
)
from geogenalg.core.exceptions import GeometryTypeError
from geogenalg.utility.validation import check_gdf_geometry_type


@dataclass(frozen=True)
class GeneralizeRoads(BaseAlgorithm):
    """Removes short unconnected linestrings from the network."""

    """By selection generalization, linestrings in the input GeoDataFrame
    are removed if 1.) they are non-connected or partly non-connected (dead-ends)
    and 2.) too short. The connectedness is determined also with respect to
    linestrings in the reference data, if provided.
    """

    """size of the buffer at the end vertices of the road linestrings. If the buffer
    does not intersect with any other linestrings, the line will be considered
    unconnected on that end."""
    threshold_distance: float

    """Unconnected/dead-end linestring shorter than this will be removed."""
    threshold_length: float

    def execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Args:
        ----
            data: GeoDataFrame containing Linestrings.
            reference_data: May contain one or more GeoDataFrames of Linestrings. They
            represent the larger road network that may have to be taken into
            consideration for deciding if the roads are connected.

        Returns:
        -------
            GeoDataFrame with generalized data, which contains

        Raises:
        ------
            GeometryTypeError: If data contains something other than lines.

        """
        # TODO: Should multigeometries be allowed?
        if not check_gdf_geometry_type(data, ["Linestring"]):
            msg = "GeneralizeRoads works only with Linestring geometries"
            raise GeometryTypeError(msg)

        if reference_data:
            for gdf in list(reference_data.values()):
                if not check_gdf_geometry_type(gdf, ["Linestring"]):
                    msg = "Reference data must only contain Linestrings"
                    raise GeometryTypeError(msg)

        # Step 1
        road_links_1 = check_line_connections(data.copy(), self.threshold_distance)

        connected_part = road_links_1[road_links_1["is_connected"]]
        unconnected_candidates = road_links_1[~road_links_1["is_connected"]]

        # Step 2
        # if list(reference_data.values()):

        inspected_candidates = check_reference_line_connections(
            unconnected_candidates,
            self.threshold_distance,
            list(reference_data.values()),
        )

        roads_to_keep_1 = inspected_candidates[inspected_candidates["is_connected"]]
        removed_short_unconnected = inspected_candidates[
            (~inspected_candidates["is_connected"])
            & (inspected_candidates["geometry"].length > self.threshold_length)
        ]

        road_links_2 = GeoDataFrame(
            concat([connected_part, roads_to_keep_1, removed_short_unconnected]),
            geometry="geometry",
            crs=data.crs,
        )

        # Step 3
        road_links_3 = detect_dead_ends(road_links_2, self.threshold_distance)

        roads_regular = road_links_3[~road_links_3["dead_end"]]
        roads_dead_end = road_links_3[road_links_3["dead_end"]]

        inspected_dead_end_candidates = inspect_dead_end_candidates(
            roads_dead_end, self.threshold_distance, list(reference_data.values())
        )

        road_links_4 = GeoDataFrame(
            concat([roads_regular, inspected_dead_end_candidates]),
            geometry="geometry",
            crs=data.crs,
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
            roads_to_generalize["geometry"].length > self.threshold_length
        ]

        road_links_generalization_output = GeoDataFrame(
            concat([roads_to_keep, remove_short_dead_ends]),
            geometry="geometry",
            crs=data.crs,
        )

        if road_links_generalization_output is not None:
            return road_links_generalization_output
        return data
