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
from geogenalg.continuity import smooth_linestring_connections
from geogenalg.core.geometry import assign_z_from_attribute
from geogenalg.merge import merge_connecting_lines_by_attribute
from geogenalg.split import split_lines_by_points


@supports_identity
class GeneralizeContours(BaseAlgorithm):
    """Generalize contour line geometries.

    Input should contain LineString or MultiLineString geometries representing contours.

    Reference data should contain Point or MultiPoint geometries representing
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
    level_attribute: str = Field("n60_elevation_value")
    """Attribute containing contour elevation values."""
    reference_key: str = Field("slope")
    """Reference Point or MultiPoint data key for slope lines."""

    valid_input_geometry_types: ClassVar = {"LineString", "MultiLineString"}

    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={"Point", "MultiPoint"},
        ),
    }

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()
        reference_gdf = reference_data[self.reference_key]
        gdf.geometry = gdf.geometry.force_2d()

        # Filter contours by elevation interval
        gdf = gdf[gdf[self.level_attribute] % self.interval == 0]

        if gdf.empty:
            return gdf.copy()

        # Split contours at slope line positions
        gdf = split_lines_by_points(gdf, reference_gdf, 1.0)

        # Gaussian smoothing
        gdf.geometry = gdf.geometry.apply(
            lambda geom: gaussian_smoothing(geom, sigma=self.gaussian_filter_strength)
        )

        # Smooth line connections
        gdf = smooth_linestring_connections(gdf)

        # TODO: Handle potentially intersecting contours after smoothing

        # Merge contour segments back into continuous geometries
        gdf = merge_connecting_lines_by_attribute(gdf, self.level_attribute)

        # Remove short lines
        gdf = gdf[gdf.geometry.length >= self.length_threshold]

        # Assing z-values
        return assign_z_from_attribute(gdf, self.level_attribute, overwrite_z=True)

        # TODO: reduce the number of vertices
