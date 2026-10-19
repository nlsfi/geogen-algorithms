#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT


import shlex
from ast import AnnAssign, Assign, ClassDef, Constant, Expr, Name, parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from inspect import cleandoc, getmro, getsource
from itertools import pairwise
from textwrap import dedent
from typing import Any

from annotated_types import BaseMetadata, Ge, Gt, Le, Lt
from pandas import Series
from pydantic_core import PydanticUndefined

from geogenalg.application import BaseAlgorithm

BASE_ALGORITHM_CLASS_VARS = {
    "valid_input_geometry_types",
    "reference_data_schema",
    "requires_projected_crs",
}


# This should include any parameter types which are unnecessary/too
# complex to support entering in the CLI.
IGNORED_TYPES_FOR_CLI = {
    Mapping[str, Callable[[Series], Any] | str] | None,
}


@dataclass(frozen=True)
class GeoPackageURI:
    """Custom input type for CLI.

    Allows specifying GeoPackage file and picking a specific layer in one argument.
    """

    file: str
    layer_name: str | None


@dataclass(frozen=True)
class NamedGeoPackageURI:
    """Custom input type for CLI.

    Allows specifying a GeoPackageURI by name, to be used as reference data in
    algorithms.
    """

    name: str
    uri: GeoPackageURI


@dataclass(frozen=True)
class TransformedTypeInformation:
    """Class to describe how to transform an unsupported type to a supported one."""

    transformed_type: type
    parser: Callable[[str], Any] | None = None
    custom_default: Any | None = None
    extra_parse_help: str | None = None
    transform_function: Callable[[Any], Any] | None = None


def try_parse_int(value: str) -> int | str:
    """Parse string into an int if possible, keep as is otherwise.

    If value is encoded as f.e. str:10 the returned type will be a string
    "10".

    Args:
    ----
        value: string to parse

    Returns:
    -------
        Parsed int or string.

    """
    if value.startswith("str:"):
        integer_part = value[4:]

        try:
            # This might be redundant, maybe you could do just some kind of nop
            # call here, but I got worried that could be optimized out, which
            # is the reason for the funny looking assignments here.
            integer = int(integer_part)
            return str(integer)
        except ValueError:
            return value

    try:
        return int(value)
    except ValueError:
        return value


def get_class_attribute_docstrings(cls: type[Any]) -> dict[str, str]:
    """Get any docstrings placed after attribute assignments in a class body.

    Adapted from https://davidism.com/attribute-docstrings/ on 2025-10-24 (MIT
    License).

    Returns
    -------
        Dictionary containing docstring for each attribute found in class.

    Raises
    ------
        TypeError: If input object is not a class.

    """
    cls_node = parse(dedent(getsource(cls))).body[0]

    if not isinstance(cls_node, ClassDef):
        msg = "Given object was not a class."
        raise TypeError(msg)

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, Assign | AnnAssign)
            or not isinstance(b, Expr)
            or not isinstance(b.value, Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = cleandoc(b.value.value)

        targets = a.targets if isinstance(a, Assign) else [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, Name):
                continue

            out[target.id] = doc

    return out


def get_basealgorithm_attribute_docstrings(cls: type[BaseAlgorithm]) -> dict[str, str]:
    """Get docstrings of a BaseAlgorithm subclass, including parent algorithm classes.

    Returns
    -------
        Dictionary containing docstring for each attribute found in class.

    """
    output: dict[str, str] = {}
    base_classes = getmro(cls)
    for base_class in base_classes:
        if not issubclass(base_class, BaseAlgorithm):
            continue

        if base_class == cls:
            continue

        output |= get_class_attribute_docstrings(base_class)

    output |= get_class_attribute_docstrings(cls)

    return {k: v for k, v in output.items() if k not in BASE_ALGORITHM_CLASS_VARS}


def basemetadata_to_string(data: BaseMetadata) -> str:
    """Get displayable string out of pydantic metadata object.

    Returns
    -------
        Displayable string.

    Raises
    ------
        NotImplementedError: if function is not implemented for given object type.

    """
    if isinstance(data, Ge):
        return f"greater than or equal to {data.ge}"
    if isinstance(data, Gt):
        return f"greater than {data.gt}"
    if isinstance(data, Le):
        return f"less than or equal to {data.le}"
    if isinstance(data, Lt):
        return f"less than {data.lt}"

    msg = f"Function not implemented for {type(data)}"
    raise NotImplementedError(msg)


def dump_cli_parameters(
    algorithm: BaseAlgorithm,
    *,
    exclude_defaults: bool = False,
) -> str:
    """Return an algorithm's parameters as a string which can be used in the CLI."""
    args: list[str] = []

    for field_name, field_info in type(algorithm).model_fields.items():
        value = getattr(algorithm, field_name)

        if value is None:
            continue

        if (
            exclude_defaults
            and (default := field_info.get_default(call_default_factory=True))
            is not PydanticUndefined
            and isinstance(value, bool) == isinstance(default, bool)
            and value == default
        ):
            continue

        clean_name = field_name.replace("_", "-")

        if isinstance(value, bool):
            args.append(f"--{clean_name}" if value else f"--no-{clean_name}")
        elif isinstance(value, frozenset):
            args.extend(
                [
                    f"--{clean_name}={shlex.quote(str(item))}"
                    for item in sorted(value, key=str)
                ]
            )
        else:
            args.append(f"--{clean_name}={shlex.quote(str(value))}")

    return " ".join(args)
