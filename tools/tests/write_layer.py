#!/usr/bin/env python3

#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

try:
    import typer
except ImportError:
    print("Typer not found. Install by running 'uv sync --extra-cli'")

    import sys

    sys.exit(0)

import sys

from geopandas import read_file

from geogenalg.main import GeoPackageArgument


def write_layer(
    input_layer: GeoPackageArgument,
    output_layer: GeoPackageArgument,
) -> None:
    gdf = read_file(input_layer.file, layer=input_layer.layer_name)
    gdf.to_file(output_layer.file, layer=output_layer.layer_name)


if __name__ == "__main__":
    typer.run(write_layer)
