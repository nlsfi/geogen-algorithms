#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from collections import deque
from typing import ClassVar, override

import numpy as np
from geopandas import GeoDataFrame
from pandas import Series
from pydantic import Field
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from shapely import STRtree, area, buffer, intersection, linestrings

from geogenalg.application import (
    BaseAlgorithm,
    ReferenceDataInformation,
    supports_identity,
)

MAX_EXHAUSTIVE_GROUP_SIZE = 14
"""
Largest group of conflicting marks whose every rotation combination is tried.
Number of checked rotation combinations and memory usage grow aggressively
with group size. With 3 rotation candidates the number of combinations is
3^MAX_EXHAUSTIVE_GROUP_SIZE. For size 14, that's ~4.8 million and for size 15
~14.3 million.
"""


@supports_identity
class GeneralizeNavigationMarks(BaseAlgorithm):
    """Generalize navigation marks such as buoys.

    The algorithm changes rotations of input navigation marks to avoid
    overlaps with other marks in input data and reference data. Navigation
    mark with rotation != 0 can either keep their rotation, change to mirrored
    rotation or change to 0 rotation. When navigation marks overlap with each
    other or the reference data, the algorithm tries to minimize overlaps by
    both checking individually each mark and rotating it if it helps, and then
    check all configurations of connected groups up to a defined max size (14)
    to reach global optimum where possible. The algorithm can set default
    rotations for features with missing rotation and assume default rotations
    for other navigation marks passed in reference data.
    """

    navigation_mark_length: float = Field(60.0, gt=0)
    """The length used to compute a rectangle footprint for a navigation mark."""
    navigation_mark_width: float = Field(20.0, gt=0)
    """The width used to compute a rectangle footprint for a navigation mark."""
    boulder_radius: float = Field(30.0, gt=0)
    """The radius used to compute a circle footprint for boulder (groups) in water."""
    rotation_column: str = "rotation"
    """Column name of the rotation column."""
    set_default_rotations: bool = False
    """Controls if default rotations are set for navigation marks before changing
    rotations based on overlaps, and whether to assume default rotations for
    other_navigation_marks with missing rotations."""
    class_column: str = "navigational_mark_mtkgml_class_id"
    """Class column used for setting default rotations."""
    default_left_rotation_classes: frozenset[int] = frozenset()
    """Classes which default rotation is left rotation."""
    default_left_rotation_degrees: float = -10.0
    """The default rotation degrees for classes that should have left rotation."""
    default_right_rotation_classes: frozenset[int] = frozenset()
    """Classes which default rotation is right rotation."""
    default_right_rotation_degrees: float = 10.0
    """The default rotation degrees for classes that should have right rotation."""
    other_navigation_marks_reference_key: str = "other_navigation_marks"
    """Reference data key for navigation marks of other types."""
    boulder_in_water_reference_key: str = "boulder_in_water"
    """Reference data key for boulders in water."""
    boulder_in_water_group_reference_key: str = "boulder_group_in_water"
    """Reference data key for boulder groups in water."""

    valid_input_geometry_types: ClassVar = {"Point"}
    reference_data_schema: ClassVar = {
        "other_navigation_marks_reference_key": ReferenceDataInformation(
            required=False, valid_geometry_types={"Point"}
        ),
        "boulder_in_water_reference_key": ReferenceDataInformation(
            required=False, valid_geometry_types={"Point"}
        ),
        "boulder_in_water_group_reference_key": ReferenceDataInformation(
            required=False, valid_geometry_types={"Point"}
        ),
    }
    requires_projected_crs: ClassVar = False

    @override
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        if data.empty:
            return data

        gdf = data.copy()

        if self.set_default_rotations:
            if self.class_column not in gdf.columns:
                msg = f"Class column `{self.class_column}` not found in input data."
                raise KeyError(msg)

            gdf[self.rotation_column] = self._default_rotations(gdf[self.class_column])

        if self.rotation_column not in gdf.columns:
            msg = f"Rotation column `{self.rotation_column}` not found in input data."
            raise KeyError(msg)

        # Skip navigation marks with missing rotations
        has_rotation = gdf[self.rotation_column].notna().to_numpy()
        gdf_filtered = gdf[has_rotation]
        if gdf_filtered.empty:
            return gdf

        coords = np.column_stack((gdf_filtered.geometry.x, gdf_filtered.geometry.y))
        rotations = gdf_filtered[self.rotation_column].to_numpy(dtype=float, copy=True)

        # Candidate rotations prioritized: 1. keep the current, 2. mirrored, 3. upright
        candidate_rotations = np.column_stack(
            (rotations, -rotations, np.zeros_like(rotations))
        )

        # Compute footprints for the 3 candidate rotations
        candidate_footprints = self._navigation_mark_footprints(
            np.repeat(coords, 3, axis=0), candidate_rotations.ravel()
        ).reshape(candidate_rotations.shape)

        # Compute how much each candidate rotation overlaps with reference
        # data and nearby other navigation marks of the same type
        footprint_overlaps = self._overlaps_with_reference_data(
            candidate_footprints, reference_data, coords
        )
        mark_pairs, mark_pair_overlaps = self._mark_pair_overlaps(
            coords, candidate_footprints
        )

        # Solve rotations, first locally by rotating one at a time and then
        # exhaustively for groups up to MAX_EXHAUSTIVE_GROUP_SIZE. In practice
        # the exact solve should handle most if not all cases, but depending on
        # data and mark size parameters groups could end up large and a fallback
        # solver (the local one) + group size limit are needed.
        chosen_rotations = self._choose_rotations_locally(
            footprint_overlaps, mark_pairs, mark_pair_overlaps
        )
        chosen_rotations = self._choose_group_rotations_exactly(
            footprint_overlaps, mark_pairs, mark_pair_overlaps, chosen_rotations
        )

        gdf.loc[has_rotation, self.rotation_column] = candidate_rotations[
            np.arange(len(gdf_filtered)), chosen_rotations
        ]
        return gdf

    def _default_rotations(self, mark_classes: Series) -> np.ndarray:
        rotations = np.zeros(len(mark_classes))
        rotations[mark_classes.isin(self.default_left_rotation_classes).to_numpy()] = (
            self.default_left_rotation_degrees
        )
        rotations[mark_classes.isin(self.default_right_rotation_classes).to_numpy()] = (
            self.default_right_rotation_degrees
        )
        return rotations

    def _navigation_mark_footprints(
        self, coords: np.ndarray, rotations: np.ndarray
    ) -> np.ndarray:
        rotations_radians = np.radians(rotations)
        unit_directions = np.column_stack(
            (np.sin(rotations_radians), np.cos(rotations_radians))
        )
        mark_tips = coords + self.navigation_mark_length * unit_directions
        center_lines = linestrings(np.stack((coords, mark_tips), axis=1))
        return np.asarray(
            buffer(center_lines, self.navigation_mark_width / 2, cap_style="flat")
        )

    def _other_navigation_mark_footprints(
        self, reference_data: dict[str, GeoDataFrame], coords: np.ndarray
    ) -> np.ndarray:
        other_marks = reference_data.get(self.other_navigation_marks_reference_key)
        if other_marks is None:
            return np.empty(0, dtype=object)

        # Filter marks with same locations in reference data (they are same marks
        # as the input data being generalized currently)
        other_coords = np.column_stack((other_marks.geometry.x, other_marks.geometry.y))
        distances, _ = KDTree(coords).query(other_coords, distance_upper_bound=0.01)
        is_other_mark = ~np.isfinite(distances)
        other_marks = other_marks[is_other_mark]
        other_coords = other_coords[is_other_mark]
        if other_marks.empty:
            return np.empty(0, dtype=object)

        rotations = np.full(len(other_marks), np.nan)
        if self.rotation_column in other_marks:
            rotations = other_marks[self.rotation_column].to_numpy(
                dtype=float, na_value=np.nan, copy=True
            )
        missing = np.isnan(rotations)
        if self.set_default_rotations:
            rotations[missing] = self._default_rotations(
                other_marks[self.class_column]
            )[missing]
        else:
            other_coords = other_coords[~missing]
            rotations = rotations[~missing]

        return self._navigation_mark_footprints(other_coords, rotations)

    def _boulders_in_water_footprints(
        self, reference_data: dict[str, GeoDataFrame]
    ) -> np.ndarray:
        boulders = [
            reference_data[key].geometry.to_numpy()
            for key in (
                self.boulder_in_water_reference_key,
                self.boulder_in_water_group_reference_key,
            )
            if key in reference_data
        ]
        if not boulders:
            return np.empty(0, dtype=object)

        buffered_boulders = buffer(
            np.concatenate(boulders),
            self.boulder_radius,
            quad_segs=4,
        )
        return np.asarray(buffered_boulders)

    def _overlaps_with_reference_data(
        self,
        candidate_footprints: np.ndarray,
        reference_data: dict[str, GeoDataFrame],
        coords: np.ndarray,
    ) -> np.ndarray:
        navigation_mark_footprints = candidate_footprints.ravel()
        avoided_footprints = np.concatenate(
            (
                self._boulders_in_water_footprints(reference_data),
                self._other_navigation_mark_footprints(reference_data, coords),
            )
        )
        if len(avoided_footprints) == 0:
            return np.zeros(candidate_footprints.shape)

        navigation_mark_indices, avoided_indices = STRtree(avoided_footprints).query(
            navigation_mark_footprints, predicate="intersects"
        )
        shared_areas = area(
            intersection(
                navigation_mark_footprints[navigation_mark_indices],
                avoided_footprints[avoided_indices],
            )
        )
        overlap_per_footprint = np.bincount(
            navigation_mark_indices,
            weights=shared_areas,
            minlength=len(navigation_mark_footprints),
        )
        return overlap_per_footprint.reshape(candidate_footprints.shape)

    def _mark_pair_overlaps(
        self,
        coords: np.ndarray,
        candidate_footprints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        search_radius = 2 * self.navigation_mark_length + self.navigation_mark_width
        mark_pairs = KDTree(coords).query_pairs(r=search_radius, output_type="ndarray")
        if len(mark_pairs) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0, 3, 3))

        first_mark_footprints = candidate_footprints[mark_pairs[:, 0]][:, :, np.newaxis]
        second_mark_footprints = candidate_footprints[mark_pairs[:, 1]][
            :, np.newaxis, :
        ]
        mark_pair_overlaps = area(
            intersection(first_mark_footprints, second_mark_footprints)
        )
        is_conflicting = mark_pair_overlaps.any(axis=(1, 2))
        return mark_pairs[is_conflicting], mark_pair_overlaps[is_conflicting]

    def _choose_rotations_locally(
        self,
        footprint_overlaps: np.ndarray,
        mark_pairs: np.ndarray,
        mark_pair_overlaps: np.ndarray,
    ) -> np.ndarray:
        marks = np.concatenate((mark_pairs[:, 0], mark_pairs[:, 1]))
        neighbours = np.concatenate((mark_pairs[:, 1], mark_pairs[:, 0]))
        overlaps = np.concatenate(
            (mark_pair_overlaps, mark_pair_overlaps.transpose(0, 2, 1))
        )
        mark_count = len(footprint_overlaps)
        pairs_per_mark = np.split(
            np.argsort(marks, kind="stable"),
            np.cumsum(np.bincount(marks, minlength=mark_count))[:-1],
        )

        chosen_rotations = np.argmin(footprint_overlaps, axis=1)
        marks_to_check = deque(np.unique(marks).tolist())
        is_queued = np.zeros(mark_count, dtype=bool)
        is_queued[marks] = True
        while marks_to_check:
            mark = marks_to_check.popleft()
            is_queued[mark] = False
            mark_pair_indices = pairs_per_mark[mark]
            mark_neighbours = neighbours[mark_pair_indices]

            candidate_costs = footprint_overlaps[mark] + overlaps[
                mark_pair_indices, :, chosen_rotations[mark_neighbours]
            ].sum(axis=0)

            best_rotation = int(np.argmin(candidate_costs))
            if best_rotation == chosen_rotations[mark]:
                continue

            chosen_rotations[mark] = best_rotation
            marks_to_check.extend(mark_neighbours[~is_queued[mark_neighbours]])
            is_queued[mark_neighbours] = True

        return chosen_rotations

    def _choose_group_rotations_exactly(
        self,
        footprint_overlaps: np.ndarray,
        mark_pairs: np.ndarray,
        mark_pair_overlaps: np.ndarray,
        chosen_rotations: np.ndarray,
    ) -> np.ndarray:
        chosen_rotations = chosen_rotations.copy()
        mark_count = len(footprint_overlaps)

        # Marks overlapping each other, directly or through other marks, belong
        # to the same group
        _, group_id_of_mark = connected_components(
            coo_array(
                (np.ones(len(mark_pairs)), (mark_pairs[:, 0], mark_pairs[:, 1])),
                shape=(mark_count, mark_count),
            ),
            directed=False,
        )
        # Both marks of a pair are in the same group
        group_id_of_pair = group_id_of_mark[mark_pairs[:, 0]]

        # Numbers marks 0, 1, 2... within their group, filled below
        number_within_group = np.zeros(mark_count, dtype=int)

        for group_id in np.unique(group_id_of_pair):
            marks_in_group = np.flatnonzero(group_id_of_mark == group_id)
            group_size = len(marks_in_group)
            if group_size > MAX_EXHAUSTIVE_GROUP_SIZE:
                continue

            pairs_in_group = np.flatnonzero(group_id_of_pair == group_id)
            number_within_group[marks_in_group] = np.arange(group_size)
            group_mark_pairs = number_within_group[mark_pairs[pairs_in_group]]

            # combinations[m, c] is the candidate of mark m in combination c
            # int8 is used here to optimize memory usage
            combinations = np.indices((3,) * group_size, dtype=np.int8).reshape(
                group_size, -1
            )

            # Total overlap of each combination, one mark and one pair added at a time
            total_overlaps = np.zeros(combinations.shape[1])
            for mark_number, mark in enumerate(marks_in_group):
                total_overlaps += footprint_overlaps[mark][combinations[mark_number]]
            for (first_mark, second_mark), pair in zip(
                group_mark_pairs, pairs_in_group, strict=True
            ):
                total_overlaps += mark_pair_overlaps[pair][
                    combinations[first_mark], combinations[second_mark]
                ]

            chosen_rotations[marks_in_group] = combinations[
                :, np.argmin(total_overlaps)
            ]

        return chosen_rotations
