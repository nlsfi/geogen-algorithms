#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from typing import ClassVar, override

from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)

from .generalize_building_areas import GeneralizeBuildingAreas


@supports_identity
class GeneralizeTallBuildingAreas(BaseAlgorithm):
    """Generalize building areas, but only for multi-storey buildings."""

    valid_input_geometry_types: ClassVar = {"Polygon"}

    height_class_column: str = "building_height_class"
    """Name of the column which defines building height classes."""

    tall_building_classes: frozenset[int | str] = frozenset({2})
    """Values in height_class_column that define tall buildings."""

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame] | None = None,
    ) -> GeoDataFrame:

        # Suodata korkeat rakennukset (täysin geneerinen)
        filtered = data[data[self.height_class_column].isin(self.tall_building_classes)]

        # Delegoi kaikki loppu pääalgoritmille
        base_algo = GeneralizeBuildingAreas()

        return base_algo.execute(
            filtered,
            reference_data=reference_data,
        )
