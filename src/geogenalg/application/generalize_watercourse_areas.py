#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar

from geopandas import GeoDataFrame
from pandas import concat

from geogenalg.application import supports_identity
from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.identity import hash_duplicate_indexes, hash_index_from_old_ids
from geogenalg.split import explode_and_hash_id
from geogenalg.transform import thin_polygon_sections_to_lines


@supports_identity
@dataclass(frozen=True)
class GeneralizeWaterCourseAreas(GeneralizeWaterAreas):
    """Generalize polygonal watercourse areas.

    This algorithms first applies the GeneralizeWaterAreas algorithm, then converts
    thin sections of remaining areas to linestrings.
    """

    line_transform_width: float = 30.0
    """Polygon sections under this width will be considered for transforming to
    a line."""
    line_min_length: float = 200.0
    """Minimum length for new line features."""
    min_new_section_length: float = 300.0
    """After initial conversion of thin sections to lines, this determines the
    minimum length of remaining sections. If under this, it will be converted
    to a line as well."""
    width_check_distance: float = 10.0
    """Controls the distance of width checks. Smaller number means more
    precise checking."""

    # Inherited from GeneralizeWaterAreas, override default
    thin_section_exaggerate_by: float = 0.0

    valid_input_geometry_types: ClassVar = {"Polygon"}
    valid_reference_geometry_types: ClassVar = {"LineString"}

    def _execute(
        self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
    ) -> GeoDataFrame:
        generalized = super()._execute(data, reference_data)

        lines, polygons = thin_polygon_sections_to_lines(
            input_gdf=generalized,
            threshold=self.line_transform_width,
            min_line_length=self.line_min_length,
            min_new_section_length=self.min_new_section_length,
            min_new_section_area=self.min_area,
            width_check_distance=self.width_check_distance,
            old_ids_column="old_ids_temp",
        )

        polygons = explode_and_hash_id(polygons, "watercourseareas").drop(
            "old_ids_temp", axis=1
        )
        lines = hash_index_from_old_ids(
            lines, "watercourseareas", old_ids_column="old_ids_temp"
        )
        lines = hash_duplicate_indexes(lines, "watercourseareas")

        return concat(
            [
                lines,
                polygons,
            ]
        )
