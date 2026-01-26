#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import NamedTuple, cast

from geopandas import GeoDataFrame, overlay
from geopandas.testing import assert_geodataframe_equal
from pandas import concat

from geogenalg.application import BaseAlgorithm
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

logger = getLogger(__name__)


@dataclass(frozen=True)
class GeoPackageInput:
    """Dataclass to specify GeoPackage path and a layer within it."""

    file: Path
    layer_name: str | None


@dataclass(frozen=True)
class GeoPackagePath:
    """Convenience class for more ergonomic integration tests."""

    file: Path

    def to_input(self, layer_name: str) -> GeoPackageInput:
        """Create GeoPackageInput pointing to specific layer.

        Returns
        -------
            GeoPackageInput object.

        """
        return GeoPackageInput(
            file=self.file,
            layer_name=layer_name,
        )


class TestGeoDataFrames(NamedTuple):
    """Convenience class for more ergonomic integration tests."""

    input_data: GeoDataFrame
    reference_data: dict[str, GeoDataFrame]
    result: GeoDataFrame
    control: GeoDataFrame


def assert_gdf_equal_save_diff(
    result: GeoDataFrame,
    control: GeoDataFrame,
    *,
    directory: Path | str | None = None,
    save_control: bool = True,
) -> None:
    """Assert GeoDataFrames are equal and if not, save results.

    Intended mainly for convenience in debugging. In tests, the
    geopandas.testing and pandas.testing modules should be used.

    Args:
    ----
        result: Result GeoDataFrame.
        control: Control GeoDataFrame.
        directory: Specify directory in which results will be saved. If None, a
            temporary directory will be created.
        save_control: Whether to save control dataset to directory as well.

    """
    if directory is None:
        directory = Path(gettempdir()) / "geogenalg_tests"

    if isinstance(directory, str):
        directory = Path(directory)

    if not directory.exists():
        directory.mkdir()

    try:
        assert_geodataframe_equal(result, control)
    except AssertionError:
        result_path = directory / "result.gpkg"
        geometry_difference_path = directory / "geomdiff.gpkg"

        result.to_file(result_path)

        try:
            geom_diff = overlay(result, control, how="difference")
            if not geom_diff.union_all().is_empty:
                geom_diff.to_file(geometry_difference_path)
        # Blindly catching exception and just logging should be fine in this case,
        # as this function is just used for debugging purposes.
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not create geometry difference: %s", e)

        # TODO: support saving dataframe comparison to file as well
        # something like (comparison = result.compare(control))

        if save_control:
            control_path = directory / "control.gpkg"
            control.to_file(control_path)

        logger.info("Test results saved to %s.", directory)

        raise


def get_alg_results_from_geopackage(
    alg: BaseAlgorithm,
    input_data: GeoDataFrame,
    unique_id_column: str,
    reference_data: dict[str, GeoDataFrame] | None = None,
) -> GeoDataFrame:
    """Execute algorithm, write to and read from GeoPackage.

    This is useful because if control GeoDataFrame is read from a GeoPackage it can
    have some small, insignificant differences to one directly created by
    an algorithm.

    Args:
    ----
        alg: Instance of an algorithm class.
        input_data: Input GeoDataFrame to pass to algorithm's execute function
        reference_data: Reference data to pass to algorithm's execute function
        unique_id_column: Column to set as output GeoDataFrame's index.

    Returns:
    -------
        Algorithm's result GeoDataFrame, which is read from a GeoPackage.

    """
    temp_dir = TemporaryDirectory()
    output_path = Path(temp_dir.name) / "output.gpkg"

    alg.execute(input_data, reference_data).to_file(output_path, layer="result")

    return read_gdf_from_file_and_set_index(
        output_path,
        unique_id_column,
        layer="result",
    )


def get_test_gdfs(
    input_uri: GeoPackageInput | list[GeoPackageInput],
    control_uri: GeoPackageInput,
    alg: BaseAlgorithm,
    unique_id_column: str,
    reference_uris: dict[str, GeoPackageInput] | None = None,
) -> TestGeoDataFrames:
    """Get input, reference, result and control GeoDataFrame.

    Args:
    ----
        input_uri: Object pointing to input dataset(s) and the correct layer(s) therein.
        control_uri: Object pointing to control dataset and the correct layer therein.
        alg: Instance of an algorithm to execute.
        unique_id_column: Column to set as GeoDataFrame index.
        reference_uris: Dictionary of GeoPackageURIs, which is used to
            construct matching reference_data dictionary to pass to algorithm's
            execute function.

    Returns:
    -------
        Input, reference, result and control GeoDataFrames, in that order.

    """
    reference_data = {}
    if reference_uris is not None:
        for key, uri in reference_uris.items():
            reference_data[key] = read_gdf_from_file_and_set_index(
                uri.file,
                unique_id_column,
                layer=uri.layer_name,
            )

    if isinstance(input_uri, list):
        gdfs = [
            read_gdf_from_file_and_set_index(
                uri.file,
                unique_id_column,
                layer=uri.layer_name,
            )
            for uri in input_uri
        ]
        input_data = cast("GeoDataFrame", concat(gdfs))
    else:
        input_data = read_gdf_from_file_and_set_index(
            input_uri.file,
            unique_id_column,
            layer=input_uri.layer_name,
        )

    result = get_alg_results_from_geopackage(
        alg,
        input_data,
        unique_id_column,
        reference_data,
    )

    control = read_gdf_from_file_and_set_index(
        control_uri.file,
        unique_id_column,
        layer=control_uri.layer_name,
    )

    return TestGeoDataFrames(
        input_data,
        reference_data,
        result,
        control,
    )
