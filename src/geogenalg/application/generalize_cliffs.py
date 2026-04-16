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
from geogenalg.selection import remove_close_line_segments, remove_short_lines
from geogenalg.split import explode_and_hash_id


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
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={
                "LineString",
                "MultiLineString",
            },
        ),
    }
    required_projected_crs: ClassVar = False

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        roads_data = reference_data[self.reference_key]

        result = remove_close_line_segments(data, roads_data, self.buffer_size)
        result = remove_short_lines(result, self.length_threshold)
        return explode_and_hash_id(result, "cliffs")
