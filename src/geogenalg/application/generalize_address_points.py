#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar, override

from geopandas import GeoDataFrame
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.utility.dataframe_processing import combine_gdfs

BUILDING_CENTROID_COLUMN = "_generalize_address_points_centroid"
BUILDING_DISTANCE_COLUMN = "distance"
TEMPORARY_AREA_COLUMN = "temporary_area_column"


@supports_identity
class GeneralizeAddressPoints(BaseAlgorithm):
    """Filter address points based on building features.

    Reference data should contain a Polygon GeoDataFrame representing
    building part areas.

    Input and reference data are combined at the start of the algorithm with
    an inner join on `join_column` (a persistent building identifier), so
    address points without a matching building polygon are filtered out.

    The algorithm does the following steps:
    - Joins the building polygon area, function type and building-to-point distance to
      each address point using `join_column`
    - Removes points whose `address_number_column` value is invalid
    - Removes points whose distance to the linked building's centroid
      exceeds `max_building_distance`
    - Removes points located within `duplicate_location_tolerance` of
      another point, keeping the one with the greatest `priority_column`
      value
    - For points linked to a building whose function is in
      `secondary_usage_values`, keeps only the point with the largest
      building area among those sharing the same `address_column` value,
      unless a point sharing that same address is linked to a building
      whose usage is in `primary_function_values` within `search_radius`, in
      which case all of the secondary-usage points sharing that address are
      removed

    """

    join_column: str = "building_id"
    """Name of the building identifier column in the input address point data."""
    reference_join_column: str | None = "None"
    """Name of the building identifier column in the reference building data.
    Defaults to `join_column` if not set."""
    address_number_column: str = "address_number"
    """Name of the column containing the number part of address."""
    address_primary_column: str = "address_primary"
    """Name of the primary column containing the address."""
    address_secondary_column: str = "address_secondary"
    """Name of the secondary column containing address,
    f.e. a secondary language address."""
    # TODO: Make 'address_secondary_column' optional and only use it when given
    priority_column: str = "address_number"
    """Name of the column used to resolve duplicate-location conflicts."""
    duplicate_location_tolerance: float = Field(1.0, ge=0)
    """Points within this distance are classified as being in the same location."""
    max_building_distance: float = Field(200.0, gt=0)
    """Maximum allowed distance between an address point and the centroid of
    its linked building polygon."""
    search_radius: float = Field(500.0, gt=0)
    """Search radius used when checking for a matching address on a
    secondary function building."""
    primary_function_values: frozenset[int] = frozenset()
    """Building function values considered primary. A same-address building
    with one of these values found within `search_radius` causes secondary
    function building points to be removed."""
    building_function_column: str = "building_function"
    """Name of the column containing the building function type."""
    invalid_address_numbers: frozenset[int] = frozenset([])
    """Address number values which are considered invalid.
    Points with these will be removed."""
    reference_key: str = "buildings"
    """Reference data key to use as a source of building part area data."""

    valid_input_geometry_types: ClassVar = {"Point"}
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={"Polygon", "MultiPolygon"},
        ),
    }

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        buildings = reference_data[self.reference_key]

        reference_join_column = self.reference_join_column or self.join_column

        if self.join_column not in data.columns:
            msg = f"Join column `{self.join_column}` not found in input GeoDataFrame."
            raise KeyError(msg)
        if reference_join_column not in buildings.columns:
            msg = f"Join column `{reference_join_column}` not found in reference data."
            raise KeyError(msg)

        index_name = data.index.name
        index_col = index_name if index_name is not None else "index"

        building_attributes = buildings[
            [reference_join_column, self.building_function_column]
        ].copy()
        building_attributes[TEMPORARY_AREA_COLUMN] = buildings.geometry.area
        building_attributes[BUILDING_CENTROID_COLUMN] = buildings.geometry.centroid

        # Rows with a missing join id cannot be matched without causing duplicate joins
        building_attributes = building_attributes[
            building_attributes[reference_join_column].notna()
        ]

        data_for_merge = data[data[self.join_column].notna()]

        merged = data_for_merge.reset_index(names=index_col).merge(
            building_attributes,
            left_on=self.join_column,
            right_on=reference_join_column,
            how="inner",
            suffixes=("", "_building"),
        )

        # Drop the reference-side join column if it was named differently
        if reference_join_column != self.join_column:
            merged = merged.drop(columns=[reference_join_column])

        merged = merged.set_index(index_col)
        gdf = GeoDataFrame(merged, geometry=data.geometry.name, crs=data.crs)
        gdf.index.name = index_name

        gdf[BUILDING_DISTANCE_COLUMN] = gdf.geometry.distance(
            gdf[BUILDING_CENTROID_COLUMN]
        )
        gdf = gdf.drop(columns=[BUILDING_CENTROID_COLUMN])

        gdf = gdf[~gdf[self.address_number_column].isin(self.invalid_address_numbers)]

        gdf = gdf[gdf[BUILDING_DISTANCE_COLUMN] <= self.max_building_distance]

        gdf = self._remove_duplicate_locations(gdf)

        # Points linked to a primary-function building are always kept
        is_primary = gdf[self.building_function_column].isin(
            self.primary_function_values
        )
        primary_gdf = gdf[is_primary]
        secondary_gdf = gdf[~is_primary]

        # Filter secondary-function buildings with specified rules
        kept_secondary_gdf = self._filter_secondary_function_addresses(
            secondary_gdf, primary_gdf
        )

        result_gdf = combine_gdfs([primary_gdf, kept_secondary_gdf])
        result_gdf.index.name = index_name

        return result_gdf.drop(
            columns=[
                self.building_function_column,
                TEMPORARY_AREA_COLUMN,
                BUILDING_DISTANCE_COLUMN,
            ]
        )

    def _remove_duplicate_locations(self, gdf: GeoDataFrame) -> GeoDataFrame:
        """Remove duplicate points, keeping the greatest priority value.

        Returns
        -------
        An input gdf without low-priority points in duplicate locations.

        """
        ordered = gdf.sort_values(self.priority_column, ascending=False, kind="stable")
        sindex = gdf.sindex
        removed: set = set()

        for idx, geom in zip(ordered.index, ordered.geometry, strict=True):
            if idx in removed:
                continue

            candidate_positions = sindex.query(
                geom, predicate="dwithin", distance=self.duplicate_location_tolerance
            )
            for other_idx in gdf.index[candidate_positions]:
                if other_idx == idx or other_idx in removed:
                    continue
                if geom.dwithin(
                    gdf.geometry.loc[other_idx],
                    self.duplicate_location_tolerance,
                ):
                    removed.add(other_idx)

        return gdf.drop(index=list(removed))

    def _filter_secondary_function_addresses(
        self,
        secondary_gdf: GeoDataFrame,
        primary_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
        """Apply the address-density rule to secondary-function buildings.

        Address is taken from 'address_primary_column', falling back to
        'address_secondary_column' where the primary column is empty.

        For each group of 'secondary_gdf' points sharing the same address,
        the whole group is removed if any point in it has a same-address
        'primary_gdf' point within 'search_radius'. Otherwise only the point
        with the largest building area is kept.

        Returns
        -------
        The filtered subset of 'secondary_gdf' rows.

        """
        primary = secondary_gdf[self.address_primary_column]
        secondary_address = primary.where(
            primary.notna() & primary.astype(str).str.strip().str.len().gt(0),
            secondary_gdf[self.address_secondary_column],
        )

        primary = primary_gdf[self.address_primary_column]
        primary_address = primary.where(
            primary.notna() & primary.astype(str).str.strip().str.len().gt(0),
            primary_gdf[self.address_secondary_column],
        )

        kept_indices: list = []

        for address, group in secondary_gdf.groupby(secondary_address):
            same_address_primary = primary_gdf[primary_address == address]

            has_primary_nearby = False
            if not same_address_primary.empty:
                for geom in group.geometry:
                    distances = same_address_primary.geometry.distance(geom)
                    if (distances <= self.search_radius).any():
                        has_primary_nearby = True
                        break

            if has_primary_nearby:
                continue

            best_idx = group[TEMPORARY_AREA_COLUMN].idxmax()
            kept_indices.append(best_idx)

        return secondary_gdf.loc[kept_indices]
