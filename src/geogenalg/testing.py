#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any, Literal, NamedTuple, cast
from warnings import warn

from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from pandas import DataFrame, concat

from geogenalg.application import BaseAlgorithm
from geogenalg.utility.dataframe_processing import read_gdf_from_file_and_set_index

logger = getLogger(__name__)

AssertFunctionParameter = Literal[
    "check_dtype",
    "check_index_type",
    "check_column_type",
    "check_frame_type",
    "check_like",
    "check_less_precise",
    "check_geom_type",
    "check_crs",
    "normalize",
]


class DiffWarning(UserWarning):  # noqa: D101
    pass


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
    input_data_before: GeoDataFrame
    reference_data: dict[str, GeoDataFrame]
    result: GeoDataFrame
    control: GeoDataFrame


def assert_gdf_equal_save_diff(  # noqa: C901
    result: GeoDataFrame,
    control: GeoDataFrame,
    *,
    assert_function_arguments: dict[AssertFunctionParameter, Any] | None = None,
    directory: Path | str | None = None,
) -> None:
    """Assert GeoDataFrames are equal and if not, save results.

    Intended for convenience in debugging. In actual tests, the
    geopandas.testing and pandas.testing modules should be used.

    Args:
    ----
        result: Result GeoDataFrame.
        control: Control GeoDataFrame.
        assert_function_arguments: Arguments to pass to assert_geodataframe_equal.
        directory: Specify directory in which results will be saved. If None, a
            temporary directory will be created.

    """
    if assert_function_arguments is None:
        assert_function_arguments = {}

    if directory is None:
        directory = Path(gettempdir()) / "geogenalg_tests"

    if isinstance(directory, str):
        directory = Path(directory)

    # No side effects
    result = result.copy()
    control = control.copy()

    try:
        assert_geodataframe_equal(result, control, **assert_function_arguments)
    except:
        if not directory.exists():
            directory.mkdir(parents=True)

        warn(
            "Exception occured while checking GeoDataFrame "
            + f"equality. Saving diff to {directory}",
            category=DiffWarning,
            stacklevel=3,
        )

        result_path = directory / "result.gpkg"
        result.to_file(result_path)

        if result.index.equals(control.index):
            geom_diff_path = directory / "geomdiff.gpkg"
            attribute_diff_path = directory / "attributediff.csv"

            geom_diff = result.copy()
            geom_diff.geometry = geom_diff.geometry.difference(control.geometry)
            geom_diff = geom_diff.loc[~geom_diff.geometry.is_empty]

            attribute_diff = cast(
                "DataFrame",
                result.drop(result.geometry.name, axis=1).compare(
                    control.drop(control.geometry.name, axis=1)
                ),
            )

            if not geom_diff.empty:
                geom_diff.to_file(geom_diff_path)
            if not attribute_diff.empty:
                attribute_diff.to_csv(attribute_diff_path)
        else:
            result_mismatches_path = directory / "result_features_not_in_control.gpkg"
            result_mismatches = result.loc[~result.index.isin(control.index)]

            control_mismatches_path = directory / "control_features_not_in_result.gpkg"
            control_mismatches = control.loc[~control.index.isin(result.index)]

            if not result_mismatches.empty:
                result_mismatches.to_file(result_mismatches_path)

            if not control_mismatches.empty:
                control_mismatches.to_file(control_mismatches_path)

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
        Input, input before, reference, result and control GeoDataFrames, in that order.

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

    input_data_before = input_data.copy()
    # Rename geometry to simulate it not being called "geometry". Reading gdf
    # from file sets it automatically to "geometry", even when it's named
    # differently in a GeoPackage. Algorithms should work regardless of the
    # name of the geometry columns and this allows testing for that.
    input_data = input_data.rename_geometry("geom", inplace=False)

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

    input_data = input_data.rename_geometry("geometry", inplace=False)

    return TestGeoDataFrames(
        input_data,
        input_data_before,
        reference_data,
        result,
        control,
    )
