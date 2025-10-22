#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, final

from geopandas import GeoDataFrame, read_file
from pandas.api.types import is_string_dtype

from geogenalg.utility.hash import reset_with_random_hash_index

_SUPPORTS_IDENTITY_ATTR = "__supports_identity"


class BaseAlgorithm(ABC):
    """Abstract base class for all algorithms."""

    @final
    def execute(
        self,
        data: GeoDataFrame,
        reference_data: dict[str, GeoDataFrame],
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

        output = self._execute(data=data, reference_data=reference_data)

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
