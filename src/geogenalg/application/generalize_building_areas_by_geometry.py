#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar, override

from cartagen.enrichment.urban.urban_areas import boffet_areas
from geopandas import GeoDataFrame
from geopandas.geoseries import GeoSeries

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.core.geometry import assign_nearest_z
from geogenalg.identity import hash_index_from_geometry


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreasByGeometry(BaseAlgorithm):
    """Generalize polygons representing buildings.

    Output contains generalized building-area polygons.

    Forms building areas out of building polygons.

    """

    buildings_simplify_tolerance: float = 10.0
    """Tolerance for building simplification (Douglas-Peucker)."""
    boffet_area_buffer: float = 10.0
    """The buffer size used to merge buildings that are close from each other."""
    boffet_area_erosion: float = 10.0
    """The erosion size to avoid the building area to expand too far from the
    buildings located on the edge."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        gdf = data.copy()

        gdf.geometry = gdf.simplify(self.buildings_simplify_tolerance)

        # Create building areas from building polygons
        gdf = GeoSeries(
            boffet_areas(
                gdf.geometry.to_list(),
                self.boffet_area_buffer,
                self.boffet_area_erosion,
            ),
            crs=data.crs,
        ).to_frame()

        # Dissolve to remove overlapping features and explode to single features
        gdf = gdf.dissolve()
        gdf = gdf.explode()

        gdf = assign_nearest_z(data, gdf)

        return hash_index_from_geometry(gdf, "buildingareasgeometry")
