#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import geopandas as gpd

from geogenalg.analyze import calculate_coverage
from geogenalg.cluster import group_touching_or_intersecting_features
from geogenalg.merge import dissolve_and_inherit_attributes


@dataclass
class AlgorithmOptions:
    """Options for generalize building areas algorithm.

    Attributes:
        building_threshold: Minimum gap between buildings (square meters).
        coverage_threshold: Minimun coverage of buidlings/parcel area
            (valid values 0-100).
        parcel_buffer_distance: Buffer distance in meters.

    """

    unique_id_column: str
    building_threshold: float
    coverage_threshold: float
    parcel_buffer_distance: float = 0.0


def generalize_building_areas(
    buildings: gpd.GeoDataFrame,
    parcels: gpd.GeoDataFrame,
    options: AlgorithmOptions,
) -> gpd.GeoDataFrame:
    """Generalize building areas.

    Returns:
        Dataframe containining buildings.

    Raises:
        ValueError: if CRSs are not projected.

    """
    if options.coverage_threshold < 0 or options.coverage_threshold > 100:  # noqa: PLR2004
        error_msg = "coverage_threshold must be between 0 and 100."
        raise ValueError(error_msg)
    if not buildings.crs.is_projected:
        error_msg = "buildings CRS must be projected to calculate area accurately."
        raise ValueError(error_msg)
    if not parcels.crs.is_projected:
        error_msg = "parcels CRS must be projected to calculate area accurately."
        raise ValueError(error_msg)

    coverage = calculate_coverage(buildings, parcels, "_coverage_pct")

    group_by_column = "_group_id"
    building_areas = coverage[coverage["_coverage_pct"] >= options.coverage_threshold]
    group_touching_or_intersecting_features(building_areas, group_by_column)

    return dissolve_and_inherit_attributes(
        building_areas,
        by_column=group_by_column,
        unique_key_column=options.unique_id_column,
    )
