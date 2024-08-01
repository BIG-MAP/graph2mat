import numpy as np

import cython
import cython.cimports.numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
def get_edgelabels_resorting_array(edge_types: cnp.int64_t[:], sizes: cnp.int64_t[:]):
    n_edges = edge_types.shape[0]
    n_edge_types: cython.int = sizes.shape[0]

    edge_type_nlabels: cnp.int64_t[:] = np.zeros(n_edge_types, dtype=int)
    offset: cnp.int64_t[:] = np.zeros(n_edge_types, dtype=int)

    for i_edge in range(n_edges):
        edge_type: cython.int = edge_types[i_edge]

        edge_type_nlabels[edge_type] += sizes[edge_type]

    # Cumsum of edge_type_nlabels
    for edge_type in range(1, n_edge_types):
        offset[edge_type] = offset[edge_type - 1] + edge_type_nlabels[edge_type - 1]

    indices: cnp.int64_t[:] = np.empty(
        offset[n_edge_types - 1] + edge_type_nlabels[n_edge_types - 1], dtype=int
    )

    type_i: cnp.int64_t[:] = np.zeros_like(sizes, dtype=int)
    i: cython.int = 0

    for i_edge in range(n_edges):
        edge_type = edge_types[i_edge]

        block_size: cython.int = sizes[edge_type]
        start: cython.int = offset[edge_type] + type_i[edge_type]

        for j in range(start, start + block_size):
            indices[i] = j
            i += 1

        type_i[edge_type] += block_size

    return np.asarray(indices)
