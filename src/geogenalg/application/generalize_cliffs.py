#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import ClassVar, override

from geopandas import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.selection import remove_close_line_segments, remove_short_lines


@supports_identity
@dataclass(frozen=True)
class GeneralizeCliffs(BaseAlgorithm):
    """Generalize lines representing cliffs.

    Reference data should contain a Line GeoDataFrame representing
    roads with the key `reference_key`.

    Output contains the generalized cliffs.

    The algorithm does the following steps:
    - Removes cliff line segments too close to roads
    - Removes cliff line features that are too short
    """

    buffer_size: float = 20.0
    """Size for buffering roads and removing cliff segments too close."""
    length_threshold: float = 50.0
    """Minimum length for cliff feature to be preserved."""
    reference_key: str = "roads"
    """Reference data key for roads data."""

    valid_input_geometry_types: ClassVar = {"LineString", "MultiLineString"}
    valid_reference_geometry_types: ClassVar = {"LineString", "MultiLineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Raises:
            GeometryTypeError: If input GeoDataFrames have incorrect geometry types.
            MissingReferenceError: If reference data is not found.

        Returns:
            Generalized cliff lines.

        Raises:
            MissingReferenceError: If reference data is missing.

        """
        if self.reference_key in reference_data:
            roads_data = reference_data[self.reference_key]
        else:
            raise MissingReferenceError

        result = remove_close_line_segments(data, roads_data, self.buffer_size)
        return remove_short_lines(result, self.length_threshold)
