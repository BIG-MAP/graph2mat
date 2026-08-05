import numpy as np

import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def get_labels_resorting_array(
    types: cython.integral[:],
    shapes: cython.integral[:, :],
    transpose_neg: cython.bint = False,
):
    """
    The problem this function solves is that graph2mat executes edge/node operations
    per edge/node type. In the case where there are 3 types, for example, you end up with
    three arrays:

    labels_0 = [...] (n_0, x_0, y_0)
    labels_1 = [...] (n_1, x_1, y_1)
    labels_2 = [...] (n_2, x_2, y_2)

    Where n_i is the number of edges/nodes of type i, and x_i, y_i are the number of rows
    and columns of the block of type i.

    Since each type has a different block shape, they are always
    raveled and concatenated into a single array:

    labels = np.concatenate([labels_0.ravel(), labels_1.ravel(), labels_2.ravel()])

    The labels array is of course not in the same order as the target labels.

    This function receives the original order of types and then returns the indices
    to apply to the labels array to get the correct order. I.e.:

    sorted_labels = labels[indices]

    Extra complication for edges
    -----------------------------

    An interesting fact is that when grouping the edges by size, there might be edges
    in one direction and edges in the other. E.g.:

    Original basis: A, B, C

    where shape A == shape C != shape B. Then when grouping by size, we have:

    Basis: B, (A, C)

    In the target, you will always have AB and BC edges (never BA or CB). But once you
    group, you have to face the problem that now the order is B(A, C), and therefore
    AB edges have been reversed. That is, the predicted blocks are the transpose of
    the target blocks.

    I think this is only a problem for symmetric matrices where only
    one direction is predicted.
    """
    n_entries = types.shape[0]
    if any(type < 0 for type in types):  # case of edges
        n_types: cython.int = shapes.shape[1]  # number of types, including negative ones
        ntypes_int: cython.int = n_types // 2 + 1  # number of types >=0
    else:  # case of nodes
        ntypes_int: cython.int = shapes.shape[1]  # number of types, excluding negative ones
        n_types: cython.int = ntypes_int*2 - 1  # number of types, including negative ones


    type: cython.int
    rows: cython.int
    cols: cython.int
    jrow: cython.int
    jcol: cython.int

    # Always fill positive and negative types, even if they are not present in the unique types, due to the logic used by the function
    type_nlabels: cython.long[:] = np.zeros(n_types, dtype=int)  
    offset: cython.long[:] = np.zeros(n_types, dtype=int)


    # Compute the sizes for each type
    sizes: cython.int[:] = np.zeros(n_types, dtype=np.int32)

    sizes[ntypes_int-1] = shapes[0, 0] * shapes[1, 0]  # type 0
    for type in range(1, ntypes_int):  # skip type 0, already done
        sizes[type + ntypes_int - 1] = shapes[0, type] * shapes[1, type]

        # The ones for th enegative types are stored in the second half of the sizes array,
        # As we cannot have negative indexes: shape[-type] = shape[ntypes - type]
        # so we access them with ntypes_int - type
        sizes[-type + ntypes_int - 1] = shapes[0, n_types - type] * shapes[1, n_types - type]

    # Count the number of entries of each type
    for i_edge in range(n_entries):
        type: cython.int = types[i_edge]
        type_nlabels[type + ntypes_int - 1] += sizes[type + ntypes_int - 1]

    # Cumsum of type_nlabels to understand where do the labels for
    # each type start.

    prev_type: cython.int = ntypes_int - 1  # Start from the position that corresponds to type 0
    for type in range(1, ntypes_int):  # Here we just range in n_types, but this takes the negatives
        offset[type + ntypes_int - 1] = offset[prev_type] + type_nlabels[prev_type]  # >0
        offset[-type + ntypes_int - 1] = offset[type + ntypes_int - 1] + type_nlabels[type + ntypes_int - 1]  # < 0

        # We have to continue from the negative type, because the next positive type will be after it.
        prev_type = -type + ntypes_int - 1 

    # Initialize the indices array.
    # (for each label value, index of the unsorted array where it is located)
    indices: cython.long[:] = np.empty(
        offset[0] + type_nlabels[0], dtype=int
    )

    type_i: cython.long[:] = np.zeros_like(sizes, dtype=int)
    i: cython.int = 0

    for i_edge in range(n_entries):
        type = types[i_edge]
        abs_type: cython.int = abs(type)

        block_size: cython.int = sizes[type + ntypes_int - 1]
        start: cython.int = offset[type + ntypes_int - 1] + type_i[type + ntypes_int - 1]

        if transpose_neg and type < 0:
            # Get the transposed shape
            cols, rows = shapes[0, abs_type], shapes[1, abs_type]
            for jrow in range(rows):
                for jcol in range(cols):
                    indices[i] = start + jcol * rows + jrow
                    i += 1
        else:
            for j in range(start, start + block_size):
                indices[i] = j
                i += 1

        type_i[type + ntypes_int - 1] += block_size

    return np.asarray(indices)


# HERE IS SOME CODE THAT COULD BE USED IN THE FUTURE TO GET RESORTING
# INDICES WHEN basis_grouping="max". It is not used currently because
# we just compute a mask (see Graph2Mat._get_labels_resort_index)

# offsets = np.cumsum(original_sizes)
# offsets = np.concatenate(([0], offsets))

# abs_original_types = np.abs(original_types)

# n_vals = original_sizes[abs_original_types].sum()

# max_size = self.graph2mat_table.point_block_size[0]

# indices = np.empty(n_vals, dtype=np.int64)
# ival = 0
# ientry = 0
# for type in abs_original_types:
#     # Get the start and end of the block
#     start = offsets[type]
#     end = offsets[type + 1]

#     # Get the indices for this type
#     indices[ival : ival + end - start] = (
#         ientry * max_size + filters[start:end]
#     )
#     ival += end - start
#     ientry += 1

# return indices
