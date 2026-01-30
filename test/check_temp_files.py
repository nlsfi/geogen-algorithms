#!/usr/bin/env python3
#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import sys
from pathlib import Path


def testdata_path() -> Path:
    return Path(__file__).resolve().parent / "testdata"


def main() -> int:
    temp_files = [
        str(file)
        for file in testdata_path().rglob("*")
        if file.name.endswith(".gpkg-shm") or file.name.endswith(".gpkg-wal")
    ]

    if temp_files:
        sys.stderr.write(
            "Following temporary GeoPackage files found:\n\t"
            + "\n\t".join(temp_files)
            + "\n\n"
        )
        sys.stderr.write(
            """This is likely because the file is open in a program. Edits made to the file might not entirely be saved yet. Make sure the edits are permanently saved to the main GeoPackage file before commiting."""
        )

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(
        main(),
    )
