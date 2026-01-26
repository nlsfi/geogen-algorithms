#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


class GeometryTypeError(Exception):
    pass


class GeometryOperationError(Exception):
    pass


class InvalidGeometryError(Exception):
    pass


class MissingReferenceError(Exception):
    def __init__(
        self,
        msg: str = "Reference data is missing.",
        *args: object,
        **kwargs: object,
    ) -> None:
        """Initialize exception."""
        super().__init__(msg, *args, **kwargs)
