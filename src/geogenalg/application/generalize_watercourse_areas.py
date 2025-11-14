#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

from geopandas import GeoDataFrame

from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.transform import thin_polygon_sections_to_lines


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
    thin_section_width: float = 0.0

    def _execute(
        self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
    ) -> GeoDataFrame:
        generalized = super()._execute(data, reference_data)

        # TODO: ID handling:
        # * handle hashing old ids (use old_ids_column argument in
        #   thin_polygon_sections_to_lines)
        # * thin_polygon_sections can give MultiPolygons. These need
        #   to be exploded into new features and their IDs handled.
        # * do Z values need to be recalculated?

        return thin_polygon_sections_to_lines(
            input_gdf=generalized,
            threshold=self.line_transform_width,
            min_line_length=self.line_min_length,
            min_new_section_length=self.min_new_section_length,
            min_new_section_area=self.min_area,
            width_check_distance=self.width_check_distance,
        )
