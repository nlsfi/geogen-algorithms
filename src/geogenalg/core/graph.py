#  Copyright (c) 2025 National Land Survey of Finland (Maanmittauslaitos)
#
#  This file is part of geogen-algorithms.
#
#  SPDX-License-Identifier: MIT

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def label_chains(
    start_nodes: np.ndarray,
    end_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group lines of a graph into chains.

    Args:
    ----
        start_nodes: Node id of each line's start.
        end_nodes: Node id of each line's end.

    Returns:
    -------
        A tuple of the chain id of each line, the node degree of each line end
        and the index of the line each line end belongs to. The line ends are
        ordered as all start nodes followed by all end nodes.

    """
    line_count = len(start_nodes)
    if line_count == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty

    node_count = int(max(start_nodes.max(), end_nodes.max())) + 1

    line_ends = np.concatenate([start_nodes, end_nodes])
    end_lines = np.tile(np.arange(line_count), 2)
    degree = np.bincount(line_ends, minlength=node_count)[line_ends]

    pass_through = degree == 2  # noqa: PLR2004
    graph_size = line_count + node_count
    adjacency = csr_matrix(
        (
            np.ones(np.count_nonzero(pass_through), dtype=bool),
            (end_lines[pass_through], line_count + line_ends[pass_through]),
        ),
        shape=(graph_size, graph_size),
    )
    _, labels = connected_components(adjacency, directed=False)

    return labels[:line_count], degree, end_lines


def filter_short_chains(  # noqa: PLR0913
    start_nodes: np.ndarray,
    end_nodes: np.ndarray,
    lengths: np.ndarray,
    length_threshold: float,
    *,
    dead_end_exempt: np.ndarray | None = None,
    disconnected_exempt: np.ndarray | None = None,
) -> np.ndarray:
    """Find short chains of lines which are disconnected or dead ends.

    Args:
    ----
        start_nodes: Node id of each line's start.
        end_nodes: Node id of each line's end.
        lengths: Length of each line.
        length_threshold: Chains shorter than this are marked for removal.
        dead_end_exempt: Boolean mask of lines which are never removed as part
            of a dead end chain.
        disconnected_exempt: Boolean mask of lines which are never removed as
            part of a disconnected chain.

    Returns:
    -------
        Boolean mask of lines to keep.

    """
    line_count = len(start_nodes)
    keep = np.ones(line_count, dtype=bool)
    if line_count == 0:
        return keep

    if dead_end_exempt is None:
        dead_end_exempt = np.zeros(line_count, dtype=bool)
    if disconnected_exempt is None:
        disconnected_exempt = np.zeros(line_count, dtype=bool)

    while keep.any():
        alive = np.flatnonzero(keep)

        chain_ids, degree, end_lines = label_chains(
            start_nodes[alive], end_nodes[alive]
        )
        chain_count = int(chain_ids.max()) + 1

        chain_lengths = np.bincount(chain_ids, lengths[alive], minlength=chain_count)
        free_ends = np.bincount(
            chain_ids[end_lines[degree == 1]], minlength=chain_count
        )
        junction_ends = np.bincount(
            chain_ids[end_lines[degree > 2]],  # noqa: PLR2004
            minlength=chain_count,
        )

        short = chain_lengths[chain_ids] < length_threshold
        disconnected = (junction_ends == 0)[chain_ids]
        dead_end = ((junction_ends == 1) & (free_ends == 1))[chain_ids]

        remove = short & (
            (disconnected & ~disconnected_exempt[alive])
            | (dead_end & ~dead_end_exempt[alive])
        )
        if not remove.any():
            break

        keep[alive[remove]] = False

    return keep
