#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TypeVar, final

from geopandas import GeoDataFrame, read_file
from pandas.api.types import is_string_dtype

from geogenalg.core.exceptions import (
    GeometryTypeError,
    InvalidCRSError,
    MissingReferenceError,
)
from geogenalg.utility.hash import reset_with_random_hash_index
from geogenalg.utility.validation import (
    ShapelyGeometryTypeString,
    check_gdf_geometry_type,
)

_SUPPORTS_IDENTITY_ATTR = "__supports_identity"


@dataclass(frozen=True)
class ReferenceDataInformation:
    """Struct for defining information about reference data an algorithm accepts."""

    valid_geometry_types: set[ShapelyGeometryTypeString]
    required: bool


class BaseAlgorithm(ABC):
    """Abstract base class for all algorithms."""

    valid_input_geometry_types: ClassVar[set[ShapelyGeometryTypeString]] = set()
    """Set of accepted geometry types for input data. If there is a mismatch,
    GeometryTypeError will be raised."""
    reference_data_schema: ClassVar[dict[str, ReferenceDataInformation]] = {}
    """Dictionary telling what kind of reference data algorithm accepts."""
    requires_projected_crs: ClassVar[bool] = True
    """Tells whether algorithm requires a projected coordinate reference
    system. If True, when data with non-projected CRS is passed to execute(),
    InvalidCRSError is raised."""

    @final
    def execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame] | None = None,
    ) -> GeoDataFrame:
        """Execute the algorithm.

        Args:
        ----
            data: Input data to be generalized
            reference_data: Any additional data required in the algorithm

        Returns:
        -------
            A GeoDataFrame containing generalized data

        """
        # Convert non-string indices to allow algorithm implementations
        # to rely on the assumption that input index is some kind of string
        if not is_string_dtype(data.index.dtype):
            data = data.set_index(data.index.astype("string"))

        if reference_data is None:
            reference_data = {}

        self._validate_data(data, reference_data)

        output = self._execute(data=data, reference_data=reference_data)

        # Ensure the output has the same geometry column name as the input data
        if output.geometry.name != data.geometry.name:
            output = output.rename_geometry(data.geometry.name)

        # Ensure the output has the same index name as the input data
        if output.index.name != data.index.name:
            output.index.name = data.index.name

        if getattr(self, _SUPPORTS_IDENTITY_ATTR, False):
            return output

        return reset_with_random_hash_index(output)

    @abstractmethod
    def _execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> GeoDataFrame:
        """Actual implementation of the algorithm.

        Override this method in subclasses, which is called by the equivalent api
        public wrapper.

        Process the data with possible (read only) reference datas, and return the new
        processed data.

        If the algorithm supports preserving feature identities, the output data
        index is expected to contain valid identity hashes and @supports_identity
        decorator is required to be added on the class.
        """

    def _execute_using_geopackage(
        self,
        input_path: str,
        layer_name: str,
        output_path: str,
    ) -> None:
        if not Path(input_path).resolve().exists():
            raise FileNotFoundError

        data = self.execute(
            data=read_file(input_path, layer=layer_name), reference_data={}
        )

        data.to_file(output_path, layer=layer_name)

    @final
    def _validate_data(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
    ) -> None:
        """Run a series of checks on input and reference data.

        Running this ensures that algorithm subclasses can assume all data passed
        to it
            1) has only geometries of accepted types
            2) has a coordinate reference system
            3) has a projected coordinate reference system
            4) has the same coordinate reference system

        Raises
        ------
            GeometryTypeError: If input or reference data contains invalid
                geometry types, or valid geometry types are not defined for
                subclass.
            InvalidCRSError: If input or reference data has missing, non-projected
                or differing coordinate reference systems.
            MissingReferenceError: If a required reference dataset is missing.

        """
        if not self.valid_input_geometry_types:
            msg = "Valid input geometry types not defined."
            raise GeometryTypeError(msg)

        if not check_gdf_geometry_type(data, self.valid_input_geometry_types):
            types = (
                next(iter(self.valid_input_geometry_types))
                if len(self.valid_input_geometry_types) == 1
                else f"{', '.join(sorted(self.valid_input_geometry_types))}"
            )
            msg = (
                f"Input data must contain only geometries of following types: {types}."
            )
            raise GeometryTypeError(msg)

        if data.crs is None:
            msg = "Input data has no coordinate reference system."
            raise InvalidCRSError(msg)

        if self.requires_projected_crs and not data.crs.is_projected:
            msg = "Algorithm requires projected CRS and data does not have one."
            raise InvalidCRSError(msg)

        for key_attribute, ref in self.reference_data_schema.items():
            key = getattr(self, key_attribute)
            reference = reference_data.get(key)

            if reference is None and ref.required:
                raise MissingReferenceError

            if reference is None:
                continue

            if not check_gdf_geometry_type(
                reference,
                ref.valid_geometry_types,
            ):
                types = (
                    next(iter(ref.valid_geometry_types))
                    if len(ref.valid_geometry_types) == 1
                    else f"{', '.join(sorted(ref.valid_geometry_types))}"
                )
                msg = (
                    f'Reference data "{key}" must contain only '
                    + "geometries of following types: "
                    + f"{types}."
                )
                raise GeometryTypeError(msg)

            # We've already established that the input data has a CRS and
            # either the algorithm a) does not require a projected CRS or b)
            # does require one and the data has one. Therefore it should be
            # enough to just check that the reference data CRS matches the
            # input data CRS.
            if reference.crs != data.crs:
                msg = (
                    f'Reference data "{key}" and input data have different coordinate '
                    + f"reference systems: {reference.crs} != {data.crs}."
                )
                raise InvalidCRSError(msg)


_Alg = TypeVar("_Alg", bound=type[BaseAlgorithm])


def supports_identity(algorithm_class: _Alg) -> _Alg:  # noqa: UP047, bump min python version to 3.12 or use 3.11 as ruff target
    """Add identity support for an algorithm.

    Use this decorator for algorithms that are explicitly checked to
    support correct identity hash generation.

    Requirements for identity support are:
    - Result dataframe has its index set to SHA256 hash hex str based on some parts of
      the input and output data which are relevant for the algorithm in question.
    - Hash generation is not input order specific
    - Hash generation includes a component for the algorithm, for example a short
      identifier, to avoid producing same hashes from two different algorithms for
      the same input data.

    An algorithm that processes single features only without any clustering etc. can
    preserve the original indexes and still be configured as supports identity.

    For example a clustering algorithm may base its hash generation on each clusters
    sorted input hashes, and a splitting algorithm may base its hash generation on each
    split output components original input hash and the output geometry.

    Returns:
        Wrapped algorithm class with internal identity support marker added.

    """
    setattr(algorithm_class, _SUPPORTS_IDENTITY_ATTR, True)
    return algorithm_class
