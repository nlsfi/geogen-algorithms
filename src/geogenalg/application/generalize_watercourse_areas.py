#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from enum import IntEnum
from typing import ClassVar, override

from geopandas import GeoDataFrame
from pandas.api.types import is_string_dtype
from pydantic import Field
from shapely.geometry import MultiPoint

from geogenalg.application import supports_identity
from geogenalg.application.generalize_water_areas import GeneralizeWaterAreas
from geogenalg.core.geometry import assign_nearest_z, chaikin_smooth_keep_topology
from geogenalg.identity import hash_duplicate_indexes, hash_index_from_old_ids
from geogenalg.split import explode_and_hash_id
from geogenalg.transform import thin_polygon_sections_to_lines
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
class GeneralizeWaterCourseAreas(GeneralizeWaterAreas):
    """Generalize polygonal watercourse areas.

    This algorithms first applies the GeneralizeWaterAreas algorithm, then converts
    thin sections of remaining areas to linestrings.
    """

    class OutputFeatureType(IntEnum):
        """Enum describing output feature types."""

        WATERCOURSE_AREA = 1
        WATERCOURSE_LINE = 2
        SHORELINE = 3

    line_transform_width: float = Field(30.0, gt=0)
    """Polygon sections under this width will be considered for transforming to
    a line."""
    line_min_length: float = Field(200.0, gt=0)
    """Minimum length for new line features."""
    min_new_section_length: float = Field(300.0, gt=0)
    """After initial conversion of thin sections to lines, this determines the
    minimum length of remaining sections. If under this, it will be converted
    to a line as well."""
    width_check_distance: float = Field(10.0, gt=0)
    """Controls the distance of width checks. Smaller number means more
    precise checking."""
    feature_type_column: str = "feature_type"
    """Name of column describing what type of feature each row is, described
    by the OutputFeatureType enum."""

    # Inherited from GeneralizeWaterAreas, override default
    thin_section_exaggerate_by: float = Field(0.0, ge=0)

    valid_input_geometry_types: ClassVar = {"Polygon"}

    @override
    def _post_process(
        self,
        gdf: GeoDataFrame,
        original_data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
        non_shoreline_segments: GeoDataFrame,
        skip_coords: MultiPoint,
    ) -> GeoDataFrame:
        generalized_lines, generalized_areas = thin_polygon_sections_to_lines(
            input_gdf=gdf,
            threshold=self.line_transform_width,
            min_line_length=self.line_min_length,
            min_new_section_length=self.min_new_section_length,
            min_new_section_area=self.min_area,
            width_check_distance=self.width_check_distance,
            old_ids_column="old_ids_temp",
        )

        generalized_areas = explode_and_hash_id(
            generalized_areas, "watercourseareas"
        ).drop("old_ids_temp", axis=1)

        generalized_areas.geometry = chaikin_smooth_keep_topology(
            generalized_areas.geometry,
            iterations=self.smoothing_passes,
            extra_skip_coords=skip_coords,
        )

        generalized_lines = hash_index_from_old_ids(
            generalized_lines, "watercourseareas", old_ids_column="old_ids_temp"
        )

        generalized_areas[self.feature_type_column] = (
            self.OutputFeatureType.WATERCOURSE_AREA
        )
        generalized_lines[self.feature_type_column] = (
            self.OutputFeatureType.WATERCOURSE_LINE
        )

        if self.reference_key in reference_data and not generalized_areas.empty:
            shoreline = self._build_generalized_shoreline(
                generalized_areas,
                reference_data[self.reference_key],
                non_shoreline_segments,
            )

            if not is_string_dtype(shoreline.index.dtype):
                shoreline = shoreline.set_index(shoreline.index.astype("string"))

            shoreline[self.feature_type_column] = self.OutputFeatureType.SHORELINE
            generalized_areas = combine_gdfs([generalized_areas, shoreline])

        out_gdf = combine_gdfs(
            [
                generalized_areas,
                generalized_lines,
            ]
        )

        out_gdf = hash_duplicate_indexes(out_gdf, "watercourseareas")

        return assign_nearest_z(original_data, out_gdf)

    def _execute(
        self, data: GeoDataFrame, reference_data: dict[str, GeoDataFrame]
    ) -> GeoDataFrame:
        return super()._execute(data, reference_data)
