#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import ClassVar, Literal, override

from geopandas.geodataframe import GeoDataFrame

from geogenalg.application import BaseAlgorithm, supports_identity
from geogenalg.identity import hash_index_from_old_ids
from geogenalg.merge import dissolve_and_inherit_attributes


@supports_identity
@dataclass(frozen=True)
class DissolvePolygons(BaseAlgorithm):
    """Dissolve any overlapping polygons.

    Optionally input data can be dissolved by column(s).

    Attributes for dissolved polygons are inherited from the original polygon
    features, by default from the feature which has the most amount of
    intersection with the dissolved polygon.

    If a polygon feature is not dissolved, it retains its index. For dissolved
    polygons, the index is formed as a SHA256 hash, where the input is
    comprised of a hash prefix and all the indexes of intersecting original
    polygon features concatenated together.
    """

    hash_prefix: str = "dissolvepolygons"
    """Prefix used in hash input."""
    by_column: frozenset[str] = frozenset()
    """Column(s) whose values define the groups to be dissolved. If left empty,
    all input data is considered a single group to dissolve."""
    inherit_from: Literal["min_id", "most_intersection"] = "most_intersection"
    """Method for determining which intersecting feature is considered as the
    representative original polygon to inherit attributes from."""

    valid_input_geometry_types: ClassVar = {"Polygon"}
    requires_projected_crs: ClassVar = False

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        dissolved = dissolve_and_inherit_attributes(
            data,
            by_column=list(self.by_column) or None,
            inherit_from=self.inherit_from,
        )

        return hash_index_from_old_ids(
            dissolved,
            self.hash_prefix,
            "old_ids",
            drop_old_ids=True,
        )
