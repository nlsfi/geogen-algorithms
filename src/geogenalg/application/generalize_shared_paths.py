#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import (
    get_lines_along_reference_lines,
    process_lines_and_reconnect,
)
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import (
    assign_nearest_z,
)


@supports_identity
@dataclass(frozen=True)
class GeneralizeSharedPaths(BaseAlgorithm):
    """Removes lower priority lines which are parallel to higher priority lines.

    Low priority lines which had their original connection to other low
    priority lines broken will be reconnected to the higher priority lines.

    Reference data should contain a line GeoDataFrame with the key
    `reference_key` (default "roads").
    """

    detection_distance: float = 25.0
    """Distance within which the shared paths (or generally network of linestrings of
    lower priority) along roads (or generally network of linestrings of higher priority)
    are removed.
    """
    minimum_percentage: float = 90.0
    """If the percentage of a lower priority line's total length within the
    detection distance of higher priority lines is below this it will be removed."""
    reference_key: str = "roads"
    """Reference data, higher priority layer"""

    valid_input_geometry_types: ClassVar = {"LineString"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Raises
        ------
            MissingReferenceError: if no reference data is provided

        Returns
        -------
        GeoDataFrame of LineStrings that are not contained within
        the detection_distance threshold around reference data LineStrings.

        """
        if self.reference_key in reference_data:
            reference_gdf = reference_data[self.reference_key]
        else:
            raise MissingReferenceError

        def _process(geodataframe: GeoDataFrame) -> GeoDataFrame:
            return get_lines_along_reference_lines(
                geodataframe,
                reference_gdf,
                self.detection_distance,
                length_percentage=self.minimum_percentage,
            )[1]

        gdf = process_lines_and_reconnect(
            data,
            _process,
            reference_gdf,
            length_tolerance=self.detection_distance * 1.25,
        )

        return assign_nearest_z(data, gdf)
