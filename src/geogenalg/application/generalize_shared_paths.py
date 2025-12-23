#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from geopandas import GeoDataFrame
from pandas import concat

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.continuity import get_paths_along_roads
from geogenalg.core.exceptions import GeometryTypeError, MissingReferenceError
from geogenalg.utility.validation import check_gdf_geometry_type


@supports_identity
@dataclass(frozen=True)
class GeneralizeSharedPaths(BaseAlgorithm):
    """Detects paths that go close along bigger roads and removes them if needed."""

    detection_distance: float = 25.0
    """Distance within which the shared paths (or generally network of linestrings of
    lower priority) along roads (or generally network of linestrings of higher priority)
    are removed.
    """
    reference_key: str = "roads"
    """Reference data, higher priority layer"""

    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Execute algorithm.

        Raises
        ------
            GeometryTypeError: if the geometry of input GeoDataFrames are not
            LineStrings
            MissingReferenceError: if no reference data is provided

        Returns
        -------
        GeoDataFrame of LineStrings that are not contained within
        the detection_distance threshold around reference data LineStrings.

        """
        if not check_gdf_geometry_type(data, ["LineString"]):
            msg = "Input data must contain only LineStrings."
            raise GeometryTypeError(msg)

        if self.reference_key in reference_data:
            water_areas_gdf = reference_data[self.reference_key]

            if not check_gdf_geometry_type(water_areas_gdf, ["LineString"]):
                msg = "Reference data must contain only LineStrings."
                raise GeometryTypeError(msg)
        else:
            msg = "Reference data is mandatory."
            raise MissingReferenceError(msg)

        combined_ref_roads = GeoDataFrame(
            concat(reference_data.values(), ignore_index=True),
            crs=next(iter(reference_data.values())).crs,
        )

        _too_close_shared_paths, shared_paths_to_keep = get_paths_along_roads(
            data, combined_ref_roads, self.detection_distance
        )

        return shared_paths_to_keep
