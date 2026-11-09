#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import numpy as np
import pytest

from geogenalg.core.graph import filter_short_chains, label_chains


@pytest.mark.parametrize(
    ("start_nodes", "end_nodes", "expected_chains", "expected_degrees"),
    [
        pytest.param(
            [],
            [],
            [],
            [],
            id="empty",
        ),
        pytest.param(
            [0],
            [1],
            [0],
            [1, 1],
            id="single_line",
        ),
        pytest.param(
            [0, 0, 0, 3],
            [1, 2, 3, 4],
            [0, 1, 2, 2],
            [3, 3, 3, 2, 1, 1, 2, 1],
            id="junction_with_one_extended_arm",
        ),
        pytest.param(
            [0, 1],
            [1, 0],
            [0, 0],
            [2, 2, 2, 2],
            id="closed_loop",
        ),
    ],
)
def test_label_chains(
    start_nodes: list[int],
    end_nodes: list[int],
    expected_chains: list[int],
    expected_degrees: list[int],
):
    chain_ids, degree, end_lines = label_chains(
        np.array(start_nodes),
        np.array(end_nodes),
    )

    np.testing.assert_array_equal(chain_ids, expected_chains)
    np.testing.assert_array_equal(degree, expected_degrees)
    np.testing.assert_array_equal(
        end_lines,
        np.tile(np.arange(len(start_nodes)), 2),
    )


@pytest.mark.parametrize(
    (
        "start_nodes",
        "end_nodes",
        "lengths",
        "length_threshold",
        "dead_end_exempt",
        "disconnected_exempt",
        "expected",
    ),
    [
        pytest.param([], [], [], 1.0, None, None, [], id="empty"),
        pytest.param(
            [0, 0, 0],
            [1, 2, 3],
            [1, 5, 5],
            2,
            None,
            None,
            [False, True, True],
            id="short_dead_end_removed",
        ),
        pytest.param(
            [0, 0, 0],
            [1, 2, 3],
            [3, 4, 5],
            6,
            None,
            None,
            [False, False, False],
            id="all_dead_ends_short_removes_everything",
        ),
        pytest.param(
            [0, 0, 0, 1],
            [1, 2, 3, 4],
            [1, 5, 5, 1],
            2.5,
            None,
            None,
            [False, True, True, False],
            id="chain_length_is_summed",
        ),
        pytest.param(
            [0, 0, 0, 1],
            [1, 2, 3, 4],
            [1, 5, 5, 1],
            1.5,
            None,
            None,
            [True, True, True, True],
            id="chain_not_short_is_kept",
        ),
        pytest.param(
            [0, 0, 0, 3, 3],
            [1, 2, 3, 4, 5],
            [1, 1, 1, 5, 5],
            2,
            None,
            None,
            [False, False, False, True, True],
            id="removing_branches_exposes_new_dead_end",
        ),
        pytest.param(
            [0],
            [1],
            [1],
            5,
            None,
            None,
            [False],
            id="short_disconnected_line_removed",
        ),
        pytest.param(
            [0],
            [1],
            [1],
            5,
            None,
            [True],
            [True],
            id="disconnected_exempt",
        ),
        pytest.param(
            [0, 0, 0],
            [1, 2, 3],
            [1, 5, 5],
            2,
            [True, False, False],
            None,
            [True, True, True],
            id="dead_end_exempt",
        ),
        pytest.param(
            [0, 1],
            [1, 0],
            [1, 1],
            5,
            None,
            None,
            [False, False],
            id="short_closed_loop_removed_as_disconnected",
        ),
    ],
)
def test_filter_short_chains(
    start_nodes: list[int],
    end_nodes: list[int],
    lengths: list[float],
    length_threshold: float,
    dead_end_exempt: list[bool] | None,
    disconnected_exempt: list[bool] | None,
    expected: list[bool],
):
    result = filter_short_chains(
        np.array(start_nodes, dtype=int),
        np.array(end_nodes, dtype=int),
        np.array(lengths, dtype=float),
        length_threshold,
        dead_end_exempt=(
            None if dead_end_exempt is None else np.array(dead_end_exempt)
        ),
        disconnected_exempt=(
            None if disconnected_exempt is None else np.array(disconnected_exempt)
        ),
    )

    np.testing.assert_array_equal(result, expected)
