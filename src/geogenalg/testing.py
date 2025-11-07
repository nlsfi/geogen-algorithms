#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.


from logging import getLogger
from pathlib import Path
from tempfile import gettempdir

from geopandas import GeoDataFrame, overlay
from geopandas.testing import assert_geodataframe_equal

logger = getLogger(__name__)


def assert_gdf_equal_save_diff(
    result: GeoDataFrame,
    control: GeoDataFrame,
    *,
    directory: Path | None = None,
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

    Raises:
    ------
        AssertionError: If GeoDataFrames are different.

    """
    if directory is None:
        directory = Path(gettempdir()) / "geogenalg_tests"

    if not directory.exists():
        directory.mkdir()

    try:
        assert_geodataframe_equal(result, control)
    except AssertionError:
        result_path = directory / "result.gpkg"
        geometry_difference_path = directory / "geomdiff.gpkg"
        directory / "compare.gpkg"

        result.to_file(result_path)

        geom_diff = overlay(result, control, how="difference")
        if not geom_diff.union_all().is_empty:
            geom_diff.to_file(geometry_difference_path)

        # TODO: support saving dataframe comparison to file as well
        # something like (comparison = result.compare(control))

        if save_control:
            control_path = directory / "control.gpkg"
            control.to_file(control_path)

        logger.info("Test results saved to %s.", directory)

        raise
