#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod
from pathlib import Path

from geopandas import GeoDataFrame, read_file


class BaseAlgorithm(ABC):
    """Abstract base class for all algorithms."""

    identity_hash_column: str = "identity_hash"

    @abstractmethod
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
