#  Copyright (c) 2026 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from typing import ClassVar, override

from geopandas import GeoDataFrame, GeoSeries
from pydantic import Field

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)
from geogenalg.utility.dataframe_processing import combine_gdfs


@supports_identity
class GeneralizeAddressPoints(BaseAlgorithm):
    """Filter address points based on building features.

    Reference data should contain a Polygon GeoDataFrame representing
    building part areas.

    Input and reference data are combined at the start of the algorithm with
    an inner join on `join_column` (a persistent building identifier), so
    address points without a matching building polygon are filtered out.

    Tip:
        When visualizing `result_gdf` on a map, the full address number can
        be shown with an expression such as:

        concat(
            "number_part_of_address_number",
            "subdivision_letter_of_address_number"
        )

    The algorithm does the following steps:
    - Joins the building polygon area, function and building-to-point distance to
      each address point using `join_column`
    - Removes points whose `address_number_column` value is 0
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

    join_column: str = "permanent_building_identifier"
    """Name of the permanent building identifier column in the input
    address point data."""
    reference_join_column: str | None = "vtj_prt"
    """Name of the permanent building identifier column in the reference
    building data. Defaults to `join_column` if not set."""
    address_number_column: str = "number_part_of_address_number"
    """Name of the column containing the number part of address."""
    address_fin_column: str = "address_fin"
    """Name of the column containing the Finnish-language address."""
    address_swe_column: str = "address_swe"
    """Name of the column containing the Swedish-language address."""
    priority_column: str = "address_number"
    """Name of the column used to resolve duplicate-location conflicts."""
    duplicate_location_tolerance: float = Field(1.0, gt=0)
    """Points within this distance are classified as being in the same location."""
    max_building_distance: float = Field(200.0, gt=0)
    """Maximum allowed distance between an address point and the centroid of
    its linked building polygon."""
    search_radius: float = Field(500.0, gt=0)
    """Search radius used when checking for a matching address on a
    secondary function building."""
    primary_function_values: frozenset[int] = frozenset({1, 3})
    """Building function values considered primary. A same-address building
    with one of these values found within `search_radius` causes secondary
    function building points to be removed."""
    building_function_column: str = "building_function_id"
    """Name of the column containing the building function type."""
    building_area_column: str = "area"
    """Name of the column to store the building area in."""
    building_distance_column: str = "distance"
    """Name of the column to store the building-to-address distance in."""
    reference_key: str = "buildings"
    """Reference data key to use as a source of building part area data."""

    valid_input_geometry_types: ClassVar = {"Point"}
    reference_data_schema: ClassVar = {
        "reference_key": ReferenceDataInformation(
            required=True,
            valid_geometry_types={"Polygon", "MultiPolygon"},
        ),
    }

    _building_centroid_column: ClassVar = "_generalize_address_points_centroid"

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        buildings = reference_data[self.reference_key]

        reference_join_column = self.reference_join_column or self.join_column

        if self.join_column not in data.columns:
            msg = (
                "Specified `join_column` "
                + f"({self.join_column}) not found in input GeoDataFrame."
            )
            raise KeyError(msg)
        if reference_join_column not in buildings.columns:
            msg = (
                "Specified `reference_join_column` "
                + f"({reference_join_column}) not found in reference "
                + "building GeoDataFrame."
            )
            raise KeyError(msg)

        index_name = data.index.name
        index_col = index_name if index_name is not None else "index"

        building_attributes = buildings[
            [reference_join_column, self.building_function_column]
        ].copy()
        building_attributes[self.building_area_column] = buildings.geometry.area
        building_attributes[self._building_centroid_column] = (
            buildings.geometry.centroid
        )

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

        centroids = GeoSeries(
            gdf[self._building_centroid_column], index=gdf.index, crs=data.crs
        )
        gdf[self.building_distance_column] = gdf.geometry.distance(centroids)
        gdf = gdf.drop(columns=[self._building_centroid_column])

        # Remove points with no valid address number
        gdf = gdf[gdf[self.address_number_column] != 0]

        # Remove points too far from their linked building's centroid
        gdf = gdf[gdf[self.building_distance_column] <= self.max_building_distance]

        # Remove points sharing the same location, keeping the one with
        # the greatest priority_column value
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
                self.building_area_column,
                self.building_function_column,
                self.building_distance_column,
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
                geom.buffer(self.duplicate_location_tolerance),
                predicate="intersects",
            )
            for other_idx in gdf.index[candidate_positions]:
                if other_idx == idx or other_idx in removed:
                    continue
                if (
                    geom.distance(gdf.geometry.loc[other_idx])
                    <= self.duplicate_location_tolerance
                ):
                    removed.add(other_idx)

        return gdf.drop(index=list(removed))

    def _filter_secondary_function_addresses(
        self,
        secondary_gdf: GeoDataFrame,
        primary_gdf: GeoDataFrame,
    ) -> GeoDataFrame:
        """Apply the address-density rule to secondary-function buildings.

        Address is taken from 'address_fin_column', falling back to
        'address_swe_column' where the Finnish value is empty.

        For each group of 'secondary_gdf' points sharing the same address,
        the whole group is removed if any point in it has a same-address
        'primary_gdf' point within 'search_radius'. Otherwise only the point
        with the largest building area is kept.

        Returns
        -------
        The filtered subset of 'secondary_gdf' rows.

        """
        fin = secondary_gdf[self.address_fin_column]
        secondary_address = fin.where(
            fin.notna() & fin.astype(str).str.strip().str.len().gt(0),
            secondary_gdf[self.address_swe_column],
        )

        fin = primary_gdf[self.address_fin_column]
        primary_address = fin.where(
            fin.notna() & fin.astype(str).str.strip().str.len().gt(0),
            primary_gdf[self.address_swe_column],
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

            best_idx = group[self.building_area_column].idxmax()
            kept_indices.append(best_idx)

        return secondary_gdf.loc[kept_indices]
