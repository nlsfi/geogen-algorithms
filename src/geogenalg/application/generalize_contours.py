#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar

from cartagen.utils import gaussian_smoothing
from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.continuity import (
    add_contiguous_lines_information,
    smooth_linestring_connections,
)
from geogenalg.core.geometry import assign_z_from_attribute
from geogenalg.identity import hash_duplicate_indexes
from geogenalg.split import split_lines_by_points

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

        # Split contours at slope line positions
        if not reference_gdf.empty:
            gdf = split_lines_by_points(gdf, reference_gdf, SNAP_DISTANCE)

        # Gaussian smoothing
        gdf.geometry = gdf.geometry.apply(
            lambda geom: gaussian_smoothing(geom, sigma=self.gaussian_filter_strength)
        )

        # Smooth line connections
        gdf = smooth_linestring_connections(gdf)

        # TODO: Handle potentially intersecting contours after smoothing

        # Add information about the total length of the continuous contour
        gdf = add_contiguous_lines_information(
            gdf, GeoDataFrame(geometry=[], crs=gdf.crs)
        )

        # Remove short contours
        gdf = gdf[gdf["contiguous_length"] >= self.length_threshold]

        gdf = hash_duplicate_indexes(gdf, "contour")
        gdf = assign_z_from_attribute(gdf, self.level_attribute, overwrite_z=True)
        return gdf.drop(
            [column for column in gdf.columns if column not in data.columns], axis=1
        )

        # TODO: reduce the number of vertices
