#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame
from shapely import LineString

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import (
    flag_connections,
)
from geogenalg.core.exceptions import MissingReferenceError
from geogenalg.core.geometry import LineExtendFrom, extend_line_to_nearest
from geogenalg.selection import remove_close_line_segments
from geogenalg.split import explode_and_hash_id


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
            msg = "Reference data is mandatory."
            raise MissingReferenceError(msg)

        gdf = flag_connections(
            data,
            start_connected_column="__start_connected_before",
            end_connected_column="__end_connected_before",
        )

        gdf = remove_close_line_segments(
            gdf,
            reference_gdf,
            self.detection_distance,
        )
        gdf = explode_and_hash_id(gdf, "sharedpaths")

        gdf = flag_connections(
            gdf,
            start_connected_column="__start_connected_after",
            end_connected_column="__end_connected_after",
        )

        # Effectively this checks which connections have been broken
        # in remove_close_line_segments().
        gdf["__extend_start"] = (
            gdf["__start_connected_before"] != gdf["__start_connected_after"]
        )

        gdf["__extend_end"] = (
            gdf["__end_connected_before"] != gdf["__end_connected_after"]
        )

        union_geom = reference_gdf.union_all()

        def _extend(
            extend_start: bool,  # noqa: FBT001
            extend_end: bool,  # noqa: FBT001
            line: LineString,
        ) -> LineString:
            if extend_start and extend_end:
                extend_from = LineExtendFrom.BOTH
            elif extend_start:
                extend_from = LineExtendFrom.START
            elif extend_end:
                extend_from = LineExtendFrom.END
            else:
                return line

            return extend_line_to_nearest(
                line,
                union_geom,
                extend_from,
                self.detection_distance * 1.25,
            )

        # Connect those shared paths whose connection was broken back into
        # the larger network.
        gdf.geometry = gdf[["__extend_start", "__extend_end", gdf.geometry.name]].apply(
            lambda columns: _extend(*columns),
            axis=1,
        )

        return gdf.drop(
            [column for column in gdf.columns if column not in data.columns],
            axis=1,
        )
