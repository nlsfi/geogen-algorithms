#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import add_contiguous_lines_information


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
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=False,
            valid_geometry_types={
                "LineString",
            },
        ),
    }

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

        gdf = add_contiguous_lines_information(
            gdf,
            reference_network,
        )

        gdf = gdf.loc[
            ~(
                (gdf["contiguous_length"] <= self.threshold_length)
                & (gdf["contiguous_dead_end"] | gdf["contiguous_disconnected"])
            )
        ]

        return gdf.drop(
            [column for column in gdf.columns if column not in data.columns], axis=1
        )
