#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar

from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import (
    get_contiguous_lengths,
    smooth_linestring_connections,
)
from geogenalg.core.geometry import (
    assign_z_from_attribute,
    gaussian_smooth,
)
from geogenalg.split import explode_and_hash_id, split_lines_by_points

SNAP_DISTANCE = 1.0


@supports_identity
class GeneralizeContours(BaseAlgorithm):
    """Generalize contour line geometries.

    Input should contain LineString geometries representing contours.

    Reference data should contain Point geometries representing
    positions of slope lines.

    Output is a GeoDataFrame containing generalized contour lines.

    The algorithm does the following steps:
        1. Filter contours based on elevation interval.
        2. Split contours at slope reference points to preserve fixed locations.
        3. Apply Gaussian smoothing to contour geometries.
        4. Smooth connections between adjacent contour segments.
        5. Remove short contour lines.
    """

    interval: float = Field(5, gt=0)
    """Elevation interval used for contour filtering."""
    gaussian_filter_strength: float = Field(8, ge=0)
    """Sigma value used for Gaussian smoothing."""
    length_threshold: float = Field(200, ge=0)
    """Minimum length for contour line."""
    level_attribute: str = Field("elevation_value")
    """Attribute containing contour elevation values."""
    reference_key: str = Field("slope")
    """Reference Point data key for slope line positions. If provided, contour lines
    are split at slope line locations before smoothing, preserving the fixed anchor
    points where slope lines intersect the contour lines."""

    valid_input_geometry_types: ClassVar = {"LineString"}

    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=False,
            valid_geometry_types={"Point"},
        ),
    }

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()
        reference_gdf = reference_data.get(self.reference_key, GeoDataFrame())
        gdf.geometry = gdf.geometry.force_2d()

        # Filter contours by elevation interval
        gdf = gdf[gdf[self.level_attribute] % self.interval == 0]

        if gdf.empty:
            return gdf.copy()

        if not reference_gdf.empty:
            gdf = split_lines_by_points(gdf, reference_gdf, SNAP_DISTANCE)
            gdf = explode_and_hash_id(gdf, "contour")

        gdf.geometry = gaussian_smooth(
            gdf.geometry.to_numpy(),
            sigma=self.gaussian_filter_strength,
        )

        # TODO: Handle potentially intersecting contours after smoothing

        gdf = gdf[get_contiguous_lengths(gdf) >= self.length_threshold]
        gdf = smooth_linestring_connections(gdf, spline_subdivisions=10)

        return assign_z_from_attribute(gdf, self.level_attribute, overwrite_z=True)
        # TODO: reduce the number of vertices
