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
from geogenalg.selection import remove_small_holes
from geogenalg.utility.dataframe_processing import copy_gdf_as_empty


@supports_identity
@dataclass(frozen=True)
class GeneralizeBuildingAreasByGeometry(BaseAlgorithm):
    """Generalize polygons representing buildings.

    Output contains generalized building-area polygons.

    The algorithm does the following steps:
    - Forms building areas out of building polygons
    - Simplifies building areas
    - Removes small building areas
    - Removes sections close to roads from building areas

    """

    threshold_building_area: float = 1000.0
    """Minimum size for newly generated building areas."""
    threshold_building_area_hole: float = 500.0
    """Minimum size for holes inside newly generated building areas."""
    simplify_tolerance_single_buildings: float = 10.0
    """Tolerance for building simplification (Douglas-Peucker)."""
    simplify_tolerance_building_areas: float = 10.0
    """Tolerance for building simplification (Douglas-Peucker)."""
    network_buffer_distance: float = 10.0
    """How large a section will be removed close to roads from building areas."""
    reference_key: str = "roads"
    """Reference key for line data which building areas adapt to."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        reference_network = (
            reference_data[self.reference_key]
            if self.reference_key in reference_data
            else copy_gdf_as_empty(data)
        )

        gdf = data.copy()

        gdf.geometry = gdf.simplify(self.simplify_tolerance_single_buildings)

        # Create building areas from building polygons
        gdf = GeoSeries(
            boffet_areas(gdf.geometry.to_list(), 10, 10),
            crs=data.crs,
        ).to_frame()
        gdf.geometry.name = data.geometry.name
        gdf.geometry = gdf.simplify(self.simplify_tolerance_building_areas)

        # Remove sections around roads
        buffered_network = reference_network.geometry.buffer(
            self.network_buffer_distance
        ).to_frame()
        gdf = gdf.overlay(buffered_network, how="difference")

        # Dissolve to remove overlapping features and explode to single features
        gdf = gdf.dissolve()
        gdf = gdf.explode()
        gdf = gdf.loc[gdf.geometry.area > self.threshold_building_area]

        gdf.geometry = gdf.geometry.buffer(self.network_buffer_distance / 2)
        gdf = remove_small_holes(gdf, self.threshold_building_area_hole)

        gdf = assign_nearest_z(data, gdf)

        return hash_index_from_geometry(gdf, "buildingareasgeometry")
