#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import (
    flag_contiguous_dead_ends,
    flag_disconnected_lines_contiguous_length,
)


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
    reference_key: str = "network"
    """Reference key for other line datasets which are part of the same network."""

    valid_input_geometry_types: ClassVar = {"LineString"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        reference_network = (
            reference_data[self.reference_key]
            if self.reference_key in reference_data
            else GeoDataFrame(geometry=[], crs=data.crs)
        )

        gdf = data.copy()

        gdf = flag_contiguous_dead_ends(
            gdf,
            reference_network,
            minimum_length=self.threshold_length,
        )
        gdf = gdf.loc[~gdf["__part_of_dead_end"]]

        gdf = flag_disconnected_lines_contiguous_length(
            gdf,
            reference_network,
        )
        gdf = gdf.loc[
            (gdf["__part_of_line_length"] != 0.0) & gdf["__part_of_line_length"]
            < self.threshold_length
        ]

        return gdf.drop(
            [column for column in gdf.columns if column not in data.columns], axis=1
        )
