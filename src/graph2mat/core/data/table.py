"""Storage of global basis information for a group of configurations.

In a dataset, different **point types are represented by an integer that
is the type index**. However, **there are moments in which information
of what that type means is required**. E.g. to know what's the size of
the basis for that point or what are its irreps.

All the data for the types that a model might see is stored in an object
that we call a "Table".

In a typical matrix, there will be elements that belong to connections between
different points. Therefore, these tables also need to keep track of edge types.
"""

import itertools
from io import StringIO
from pathlib import Path
from typing import Callable, Generator, List, Literal, Optional, Sequence, Union, Tuple
from itertools import chain

import numpy as np
import sisl

from .basis import BasisConvention, PointBasis, get_change_of_basis


class BasisTableWithEdges:
    """Stores the unique types of points in the system, with their basis and the possible edges.

    It also knows the size of the blocks, and other type dependent variables.

    Its function is to assist in pre and post processing data by providing a centralized
    source of truth for the basis that a model should be able to deal with.

    It allows non-square matrices, in which case the row and column basis are different. In that case,
    the basis should be passed as a dict with keys "row" and "col" and the corresponding basis for each,
    passed as a list of `PointBasis` objects with matching types. The order of the types in the row and
    column basis should be the same (although it handles reordering), and the types should be unique. 

    Parameters
    ----------
    basis:
        List of `PointBasis` objects for types that are (possibly) present in the systems
        of interest. If dict, it should have keys "row" and "col" with the corresponding basis for each.
    get_point_matrix:
        A function that takes a `PointBasis` object and returns the matrix that is a
        constant for that type of point.

        The constant matrix *can* be substracted from the training examples so that the
        models only need to learn the part that is different.

        A sensible choice for this constant matrix would be the matrix of the point if
        it was isolated in the system, because then what the models learn is the
        result of the interaction with other points.

        However, this might not make any sense for your particular problem. In that
        case just don't use this argument.
    """

    #: List of ``PointBasis`` objects that this table knows about.
    row_basis: List[PointBasis]
    col_basis: List[PointBasis]
    #: The spherical harmonics convention used for the basis
    #: (same for all ``PointBasis``).
    basis_convention: BasisConvention
    #: Type identifier for each point basis.
    types: List[Union[str, int]]

    #: The matrix that is constant for each type of point.
    point_matrix: Union[List[np.ndarray], None]
    #: The edge type for each pair of point types.
    #: Array of shape (n_point_types, n_point_types).
    edge_type: np.ndarray
    #: Shape (n_point_types,).
    #: The reach of each point type.
    R: np.ndarray
    #: Shape (n_point_types,).
    #: The number of basis functions for each point type.
    basis_size: np.ndarray
    #: Shape (n_point_types, 2).
    #: The shape of self-interacting matrix blocks of each point type.
    point_block_shape: np.ndarray
    #: Shape (n_point_types,).
    #: The number of elements for self-interacting matrix blocks of each point type.
    point_block_size: np.ndarray
    #: Shape (n_edge_types, 2).
    #: The shape of interaction matrix blocks of each edge type.
    edge_block_shape: np.ndarray
    #: Shape (n_edge_types,).
    #: The number of elements for interaction matrix blocks of each edge type.
    edge_block_size: np.ndarray

    #: Shape (n_edge_types, 2).
    #: For each (positive) edge index, returns the pair of point types that make it.
    #: This performs the inverse operation of ``edge_type``.
    edge_type_to_point_types: np.ndarray

    #: Shape (3, 3).
    #: The change of basis matrix from cartesian to the convention of the basis.
    change_of_basis: np.ndarray
    #: Shape (3, 3).
    #: The change of basis matrix from the convention of the basis to cartesian.
    change_of_basis_inv: np.ndarray

    # These are used for saving the object in a more
    # human readable and portable way than regular pickling.

    #: If the basis was read from files, this might store the names of the files.
    #: For saving/loading purposes.
    file_names: Optional[List[str]]
    #: If the basis was read from files, this might store the contents of the files.
    #: For saving/loading purposes.
    file_contents: Optional[List[str]]
    def __init__(self, basis: Sequence[PointBasis], get_point_matrix: Optional[Callable] = None):
        self._init_args = {"atoms": basis, "get_point_matrix": get_point_matrix}

        # Non-square matrices are allowed, but only if the basis is passed as a dict with keys "row" and "col".
        self.is_square = not isinstance(basis, dict)

        # Define row and col basis
        # The funtion also orders the basis by types, taking the order of the rows basis
        # Also verifyes that len(row)==len(col), and types_row == types_col
        row_basis, col_basis = self._process_basis_rows_cols(basis)
        self.row_basis = row_basis
        self.col_basis = col_basis

        # Get types
        self.types = [point_basis.type for point_basis in self.row_basis]
        assert len(set(self.types)) == len(
            self.row_basis
        ), f"The tag of each basis must be unique. Got {self.types}."
        n_types = len(self.types)

        self.R = self.get_R()


        # Define the point matrix
        if get_point_matrix is None:
            self.point_matrix = None
        else:
            self.point_matrix = [
                get_point_matrix(point_basis_row, point_basis_col)
                for point_basis_row, point_basis_col in zip(self.row_basis, self.col_basis)
            ]

        # Define the basis convention and make sure that all the point basis adhere to that convention.
        for point_basis in chain(self.row_basis, self.col_basis):
            if len(point_basis.basis) > 0:
                basis_convention = point_basis.basis_convention
                break
        else:
            basis_convention = "cartesian"

        all_conventions = [
            point_basis.basis_convention
            for point_basis in chain(self.row_basis, self.col_basis)
            if len(point_basis.basis) > 0
        ]
        if len(all_conventions) > 0:
            assert (
                len(set(all_conventions)) == 1
                and all_conventions[0] == basis_convention
            ), f"All point basis must have the same convention. Requested convention: {basis_convention}. Basis conventions {all_conventions}."

        self.basis_convention = basis_convention

        # For the basis convention, get the matrices to change from cartesian to our convention.
        self.change_of_basis, self.change_of_basis_inv = get_change_of_basis(
            "cartesian", self.basis_convention
        )

        # Array to get the edge type from point types.
        point_types_to_edge_types = np.empty((n_types, n_types), dtype=np.int32)
        edge_type = 0
        for i in range(n_types):
            # The diagonal edge type, always positive
            point_types_to_edge_types[i, i] = edge_type
            edge_type += 1
            # The non diagonal edge types, which are negative for the lower triangular part,
            # to account for the fact that the direction is different.
            for j in range(i + 1, n_types):
                point_types_to_edge_types[i, j] = edge_type
                point_types_to_edge_types[j, i] = -edge_type
                edge_type += 1

            # For non square matrices, we don't have a symmetric lower triangular part,
            # but the change in sign will still be used to indicate the direction of the edge. 

        self.edge_type = point_types_to_edge_types

        # Store the sizes of each point's basis.
        row_basis_size = np.array(
            [basis.basis_size for basis in self.row_basis], dtype=np.int32
        )
        self.row_basis_size = row_basis_size
        col_basis_size = np.array(
            [basis.basis_size for basis in self.col_basis], dtype=np.int32
        )
        self.col_basis_size = col_basis_size

        # And also the sizes of the blocks.
        self.point_block_shape = np.array([row_basis_size, col_basis_size])
        self.point_block_size = row_basis_size * col_basis_size

        point_types_combinations = np.array(
                list(itertools.combinations_with_replacement(range(n_types), 2))).T
    
        # Store the sizes of each point's basis
        # Even if its not square, we take each direction once:
        # we define the inverse shape, where we pass from (Ar, Bc) to (Ac, Br)

        self.edge_type_to_point_types = point_types_combinations.T

        row_type_indices = point_types_combinations[0]   # which row-type each combo refers to
        col_type_indices = point_types_combinations[1]   # which col-type each combo refers to

        # Block shape: (row_basis_size[i], col_basis_size[j]) for each edge type (i, j).
        self.edge_block_shape = np.array([
            row_basis_size[row_type_indices],
            col_basis_size[col_type_indices],
        ])                                          # shape: (2, n_combos)

        # Define the inverse block shape: (row_basis_size[j], col_basis_size[i]) for each edge type (i, j).
        # This is befause for non-square, (Ac, Br) != (Bc, Ar) in general.
        self.edge_block_shape_inv = np.array([
            row_basis_size[col_type_indices],
            col_basis_size[row_type_indices],
        ])                                          # shape: (2, n_combos)

        self.edge_block_size = self.edge_block_shape.prod(axis=0)  # shape: (n_combos,)
        self.edge_block_size_inv = self.edge_block_shape_inv.prod(axis=0)  # shape: (n_combos,)

    @staticmethod
    def _process_basis_rows_cols(basis: Union[List[PointBasis], dict[str, List[PointBasis]]]) \
        -> Tuple[List[PointBasis], List[PointBasis]]:
        """Processes the basis to determine which basis is used for rows and which for columns.

        If the basis has a `matrix_role` attribute, it will be used to determine the role of each basis.
        If not, the first basis will be used for rows and the second for columns.

        Parameters
        ----------
        basis: List[PointBasis] or dict[str, List[PointBasis]]
            In the case of the list, it is a square matrix with the same
            basis for rows and columns. In the case of the dict, it should
            have keys "row" and "col" with the corresponding basis.

        Returns
        -------
        i_basis: List[PointBasis]
            The list of point bases for rows.
        j_basis: List[PointBasis]
            The list of point bases for columns.
        """
        i_basis = []
        j_basis = []
        if isinstance(basis, dict):
            if "row" not in basis or "col" not in basis:
                raise ValueError("If basis is a dict, it must have keys 'row' and 'col'.")
            i_basis = basis["row"]
            j_basis = basis["col"]
        else:
            i_basis = j_basis = basis
        assert len(i_basis) == len(j_basis), "The number of row and column bases must be the same."
        # Make sure order if correct
        # If it is correct, return the basis as is. If not, reorder the basis to match the order of the other basis.
        # Quick check if already correctly ordered
        if all(a.type == b.type for a, b in zip(i_basis, j_basis)):
            return i_basis, j_basis
        print("Reordering basis to match rows and columns.")

        type_to_index = {}
        for idx, elem in enumerate(j_basis):
            if elem.type in type_to_index:
                raise ValueError(f"Duplicate type '{elem.type}' found in j_basis")
            type_to_index[elem.type] = idx
        # Reorder basis: check that the point type is the same for i_basis[k] and j_basis[k]
        ordered_j_basis = []
        for elem in i_basis:
            idx = type_to_index.get(elem.type)
            if idx is None:
                raise ValueError(f"Type '{elem.type}' not found in j_basis")
            ordered_j_basis.append(j_basis[idx])
        return i_basis, ordered_j_basis

    def get_R(self) -> np.ndarray:
        """Returns the cutoff radii in the correct format."""
        # Get also the cutoff radii for each point.
        r_rows = np.array([point_basis.maxR() for point_basis in self.row_basis])
        r_cols = np.array([point_basis.maxR() for point_basis in self.col_basis])
        if not np.allclose(r_rows, r_cols):
            print(f"Warning: Row and column point types have different cutoff radii. Taking the maximum of the two.")
        return np.max([r_rows, r_cols], axis=0)


    def group(
        self, grouping: Literal["basis_shape", "point_type", "max"]
    ) -> tuple["BasisTableWithEdges", np.ndarray, np.ndarray, Optional[np.ndarray]]:
        r"""Groups the basis in this table and creates a new table.

        It also returns useful objects to convert between the ungrouped
        and the grouped basis tables.

        Parameters
        ----------
        grouping :
            Method to group point types. The options are:

                - ``"point_type"``: No grouping.

                - ``"basis_shape"``: Groups all point types that have the same basis shape.
                  In a basis of spherical harmonics, "same basis shape" means that the number
                  of basis functions for each angular momentum :math:`\ell` is the same. Note
                  that the radial functions might differ, but they are not considered when
                  grouping.

                - ``"max"``: Groups all point types into a single group.

        Returns
        -------
        new_table :
            The new table with the grouped point types.
        point_type_conversion :
            Array of shape (n_point_types,) that maps the point types in the old table to the
            point types in the new table. E.g. if one has the original point types they
            can be converted to the new point types by doing:

            .. code-block:: python

                new_point_types = point_type_conversion[old_point_types]

            To save memory, ``point_type_conversion`` is not a real array when
            ``grouping == "point_type"``. It is just a dummy object that
            returns the key when indexed: ``point_type_conversion[key] == key``.
        edge_type_conversion :
            Array of shape (n_edge_types,) that maps the edge types in the old table to the
            edge types in the new table. E.g. if one has the original edge types they
            can be converted to the new edge types by doing:

            .. code-block:: python

                new_edge_types = edge_type_conversion[old_edge_types]

            To save memory, ``edge_type_conversion`` is not a real array when
            ``grouping == "point_type"``. It is just a dummy object that
            returns the key when indexed: ``edge_type_conversion[key] == key``.
        filters :
            This is None unless ``grouping == "max"``.

            In that case, it is an array of shape (n_point_types, dim_new_basis).
            For each original point type (first dimension), it contains a mask
            to select the values of the new basis that correspond to it. E.g.:

            .. code-block:: python

                values = ... # some computation with the grouped basis (dim_new_basis, )
                type0_values = values[filters[0]]
        """

        "TODO: define this for nonsquare in this scheme"
        # SN: fix I did: for non square, raises error if its not point type.
        # Hence, the square grouping takes only row_basis, and does the same as before.

        if not self.is_square and grouping != "point_type":
            raise NotImplementedError(f"Grouping for non-square matrices with grouping {grouping} is not implemented yet.")
        filters = None
        if grouping == "point_type":
            new_table = self

            point_type_conversion = IdentityConversion()
            edge_type_conversion = IdentityConversion()
        elif grouping == "basis_shape":
            # Get all basis sizes:
            basis_sizes = np.zeros((len(self.row_basis), 5), dtype=int)
            for i, point_type_basis in enumerate(self.row_basis):
                for n, l, _ in point_type_basis.basis:
                    basis_sizes[i, l] += n
    
            # Get the unique basis sizes
            unique_sizes, unique_indices, pseudo_types = np.unique(
                basis_sizes, axis=0, return_index=True, return_inverse=True
            )
    
            # Create the new table with the unique basis sizes. Here we just
            # take the first point type that has that basis size, but perhaps
            # it would be better to create a new point type that represents
            # that basis size (e.g. so that the name of the type is not misleading).
            new_table = self.__class__([self.row_basis[i] for i in unique_indices])
            point_type_conversion = pseudo_types
    
            # Get conversions
            old_edgetypes_to_new_point_types = point_type_conversion[
                self.edge_type_to_point_types
            ]
            edge_type_conversion = new_table.edge_type[
                old_edgetypes_to_new_point_types[:, 0],
                old_edgetypes_to_new_point_types[:, 1],
            ]
            # For edge type conversions handle the case in which the edge type
            # is negative.
            edge_type_conversion = np.concatenate(
                [edge_type_conversion, -1 * np.flip(edge_type_conversion[1:])]
            )
        elif grouping == "max":
            # Get all basis sizes:
            basis_sizes = np.zeros((len(self.row_basis), 5), dtype=int)
            for i, point_type_basis in enumerate(self.row_basis):
                for n, l, _ in point_type_basis.basis:
                    basis_sizes[i, l] += n
    
            # Maximum sizes:
            max_sizes = basis_sizes.max(axis=0)
    
            # Build the new point basis
            max_basis = PointBasis(
                "all",
                R=self.maxR(),
                basis=[(int(n), l, (-1) ** l) for l, n in enumerate(max_sizes)],
                basis_convention=self.basis_convention,
            )
            # Create the new table with only one type.
            new_table = self.__class__([max_basis])
    
            # For each original point type, compute a mask that allows us to
            # select the values of the new basis that correspond to that point type.
            # (i.e. discard the values that are not present in that original point type).
            missing_ls = max_sizes - basis_sizes
            filters = np.zeros((len(self.row_basis), max_basis.basis_size), dtype=bool)
            i = 0
            for l, n in enumerate(max_sizes):
                if n == 0:
                    continue
    
                for i_point, point_missing_ls in enumerate(missing_ls[:, l]):
                    filters[i_point, i : i + (n - point_missing_ls) * (2 * l + 1)] = 1
    
                i += (2 * l + 1) * n
    
            # Point and edge type conversions, just map any type to 0.
            point_type_conversion = np.zeros(len(self.row_basis), dtype=int)
            edge_type_conversion = np.zeros(
                len(self.edge_type_to_point_types), dtype=int
            )
        else:
            raise NotImplementedError(f"Grouping by {grouping} is not implemented.")
        
        return  new_table, point_type_conversion, edge_type_conversion, filters
        
            

    def _repr_html_(self):
        def _repr_html_1basis(my_basis):
            table = "<table><tbody>"
            table += f"<tr><th>Index</th><th>Type</th><th>Irreps</th><th>Max R</th></tr>"
    
            def _basis_string(basis1):
                s = ""
    
                for basis_set in basis1:
                    mul, l, parity = basis_set
                    s += f"{mul}x{l}{'e' if parity == 1 else 'o'} + "
                s = s[:-3]
    
                return s
    
            for i, point_basis in enumerate(my_basis):
                table += f"<tr><td>{i}</td><td>{point_basis.type}</td><td>{_basis_string(point_basis.basis)}</td><td>{point_basis.maxR()}</td></tr>"
    
            table += "</tbody></table>"
            return table
    
        if self.is_square:
            return _repr_html_1basis(self.row_basis)
        else:
            table = "<table><tbody>"
            table += f"<tr><th>Row Basis</th><th>Column Basis</th></tr>"
            for i in range(len(self.row_basis)):
                table += f"<tr><td>{self.row_basis[i]}</td><td>{self.col_basis[i]}</td></tr>"
            table += "</tbody></table>"
            return table

    def __repr__(self):
        if self.is_square:
            return f"{self.__class__.__name__}({self.basis_convention}, basis={self.row_basis})"
        else:
            return f"{self.__class__.__name__}({self.basis_convention}, row_basis={self.row_basis}, col_basis={self.col_basis})"


    def index_to_type(self, index: int) -> Union[str, int]:
        """Converts from the index of the point type to the type ID.

        Parameters
        ----------
        index:
            The index of the point type in the table for which the ID is desired.
        """
        return self.types[index]

    

    def get_sisl_atoms(self) -> List[sisl.Atom]:
        """Returns a list of sisl atoms corresponding to the basis.

        If the basis does not contain atoms, `PointBasis` objects are
        converted to atoms.

        WARNING: to sislfunctions only working well for SQUARE matrices. 
        """
        if hasattr(self, "atoms"):
            return self.atoms
        else:
            return [point.to_sisl_atom() for point in self.row_basis]

    

    def __str__(self):
        if self.is_square:
            # Both are the same, print one
            return "\n".join([f"\t- {point_basis}" for point_basis in self.row_basis])
        else:
            str_print = "Row basis:\n"
            str_print += "\n".join([f"\t- {point_basis}" for point_basis in self.row_basis])
            str_print += "\nColumn basis:\n"
            str_print += "\n".join([f"\t- {point_basis}" for point_basis in self.col_basis])
            return str_print

    def __len__(self):
        if len(self.col_basis) != len(self.row_basis):
            raise ValueError("Row and column basis have different lengths.")
        return len(self.row_basis)  # Both have same length


    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        same = self.is_square == other.is_square

        same = all(x == y for x, y in itertools.zip_longest(self.row_basis, other.row_basis))
        same &= all(x == y for x, y in itertools.zip_longest(self.col_basis, other.col_basis))
        same &= all(x == y for x, y in itertools.zip_longest(self.types, other.types))

        if self.point_matrix is None:
            same &= other.point_matrix is None
        else:
            if other.point_matrix is None:
                return False
            same &= all(
                np.allclose(x, y)
                for x, y in itertools.zip_longest(self.point_matrix, other.point_matrix)
            )

        same &= np.allclose(self.edge_type, other.edge_type)
        same &= np.allclose(self.R, other.R)
        same &= np.allclose(self.basis_size, other.basis_size)
        same &= np.allclose(self.point_block_shape, other.point_block_shape)
        same &= np.allclose(self.point_block_size, other.point_block_size)
        same &= np.allclose(self.edge_block_shape, other.edge_block_shape)
        same &= np.allclose(self.edge_block_size, other.edge_block_size)
        return same

    def type_to_index(self, point_type: Union[str, int]) -> int:
        """Converts from the type ID to the index of the point type in the table.

        Parameters
        ----------
        point_type:
            The type ID of the point type for which the index in the table is desired.
        """
        return self.types.index(point_type)

    def types_to_indices(self, types: Sequence) -> np.ndarray:
        """Converts from an array of types IDs to their indices in the basis table.

        Parameters
        ----------
        types:
            The array of types to convert.

        See Also
        --------
        type_to_index: The function used to convert each type.
        """
        # Get the unique types and the inverse indices to reconstruct the original array
        unique_types, inverse_indices = np.unique(types, return_inverse=True)

        # Now convert from types to indices
        unique_indices = np.array(
            [self.type_to_index(unique_type) for unique_type in unique_types]
        )

        # And reconstruct the original array, which is now an array of indices instead of types
        return unique_indices[inverse_indices]

    def point_type_to_edge_type(self, point_type: np.ndarray) -> Union[int, np.ndarray]:
        """Converts pairs of point types to edge types.

        Parameters
        ----------
        point_type:
            Shape (2, n_edges)
            Pair of point types for each edge.
        """
        return self.edge_type[point_type[0], point_type[1]]

    def point_block_pointer(self, point_types: Sequence[int]) -> np.ndarray:
        """Pointers to the beggining of node blocks in a flattened matrix.

        Given a flat array that contains all the elements of the matrix
        corresponing to self-interacting matrix blocks, the indices returned
        here point to the beggining of the values for each block.

        These pointers are useful to recreate the full matrix, for example.

        Parameters
        ----------
        point_types:
            The type indices for the points in the system, in the order in
            which they appear in the flattened matrix.
        """
        pointers = np.zeros(len(point_types) + 1, dtype=np.int32)
        np.cumsum(self.point_block_size[point_types], out=pointers[1:])
        return pointers

    def edge_block_pointer(self, edge_types: Sequence[int]):
        """Pointers to the beggining of edge blocks in a flattened matrix.

        Given a flat array that contains all the elements of the matrix
        corresponing to matrix blocks of interactions between two different
        points, the indices returned here point to the beggining of the values
        for each block.

        These pointers are useful to recreate the full matrix, for example.

        Parameters
        ----------
        edge_types:
            The type indices for the edges in the system, in the order in
            which they appear in the flattened matrix.
        """
        pointers = np.zeros(len(edge_types) + 1, dtype=np.int32)
        if self.is_square:
            np.cumsum(self.edge_block_size[np.abs(edge_types)], out=pointers[1:])
        else:
            # In non-square case, the edge block size is not symmetric, so for 
            # negative values of
            # New conditional selection:
            sizes = np.where(edge_types > 0,
                            self.edge_block_size[edge_types],          # positive → forward size
                            self.edge_block_size_inv[-edge_types])     # non‑positive → backward size (use absolute index)

            np.cumsum(sizes, out=pointers[1:])
        return pointers

    def maxR(self) -> float:
        """Maximum cutoff radius in the basis."""
        return self.R.max()
    


class AtomicTableWithEdges(BasisTableWithEdges):
    """Variant of `BasisTableWithEdges` for the case in which points are atoms.

    This class mostly just adds a few aliases to the methods of `BasisTableWithEdges`
    by replacing "point" to "atom" in the method names. It also provides some methods
    to create a table from atomic basis.

    See also
    --------
    BasisTableWithEdges
        The class that actually does the work.
    """

    atoms: List[sisl.Atom]

    # These are used for saving the object in a more
    # human readable and portable way than regular pickling.
    file_names: Optional[List[str]]
    file_contents: Optional[List[str]]

    def __init__(self, atoms: Sequence[sisl.Atom]):
        from .matrices.physics.density_matrix import get_atomic_DM

        self.atoms = list(
            [
                atom if isinstance(atom, sisl.Atom) else atom.to_sisl_atom(Z=atom.type)
                for atom in atoms
            ]
        )

        basis = [
            PointBasis.from_sisl_atom(atom)
            if not isinstance(atom, PointBasis)
            else atom
            for atom in atoms
        ]

        super().__init__(basis=basis, get_point_matrix=None)
        self._init_args = {"atoms": atoms}

        # Get the point matrix for each type. This is the matrix that a point would
        # have if it was the only one in the system, and it depends only on the type.
        self.point_matrix = [
            get_atomic_DM(atom) if not isinstance(atom, PointBasis) else None
            for atom in self.atoms
        ]

        self.file_names = None
        self.file_contents = None

    @property
    def zs(self):
        return self.types

    def atom_type_to_edge_type(self, atom_type: np.ndarray):
        return self.point_type_to_edge_type(atom_type)

    def atom_block_pointer(self, atom_types: Sequence[int]):
        return self.point_block_pointer(atom_types)

    @property
    def atom_block_shape(self):
        return self.point_block_shape

    @property
    def atom_block_size(self):
        return self.point_block_size

    @property
    def atomic_DM(self):
        return self.point_matrix

    @classmethod
    def from_basis_dir(
        cls,
        basis_dir: str,
        basis_ext: str = "ion.xml",
        no_basis_atoms: Optional[dict] = None,
    ) -> "AtomicTableWithEdges":
        """Generates a table from a directory containing basis files.

        Parameters
        ----------
        basis_dir:
            The directory containing the basis files.
        basis_ext:
            The extension of the basis files.
        """
        basis_path = Path(basis_dir)

        return cls.from_basis_glob(
            basis_path.glob(f"*.{basis_ext}"), no_basis_atoms=no_basis_atoms
        )

    @classmethod
    def from_basis_glob(
        cls, basis_glob: Union[str, Generator], no_basis_atoms: Optional[dict] = None
    ) -> "AtomicTableWithEdges":
        """Generates a table from basis files that match a glob pattern.

        Parameters
        ----------
        basis_glob:
            The glob pattern to match the basis files.
        """
        if isinstance(basis_glob, str):
            basis_glob = Path().glob(basis_glob)

        basis = []
        # file_names = []
        # file_contents = []
        for basis_file in basis_glob:
            # TODO: Find out what to do with binary basis files formats
            # file_names.append(basis_file.name)
            # with open(basis_file, "r") as f:
            #    file_contents.append(f.read())
            basis.append(sisl.get_sile(basis_file).read_basis())

        if no_basis_atoms is not None:
            for k, v in no_basis_atoms.items():
                basis.append(
                    PointBasis(k, R=v["R"], basis_convention="siesta_spherical")
                )

        obj = cls(basis)
        # obj.file_names = file_names
        # obj.file_contents = file_contents
        return obj

    def _set_state_by_atoms(self, atoms: Sequence[sisl.Atom]):
        self.__init__(atoms)

    def _set_state_by_filecontents(
        self, file_names: List[str], file_contents: List[str]
    ):
        assert len(file_names) == len(file_contents)
        atom_list = []
        for fname, fcontents in zip(file_names, file_contents):
            f = StringIO(fcontents)
            sile_class = sisl.get_sile_class(fname)
            with sile_class(f) as sile:
                atom_list.append(sile.read_basis())
        self.__init__(atom_list)
        self.file_names = file_names.copy()
        self.file_contents = file_contents.copy()

    # Create pickling routines
    def __getstate__(self):
        """Return the state of this object"""
        if self.file_names is not None and self.file_contents is not None:
            return {"file_names": self.file_names, "file_contents": self.file_contents}
        else:
            return {"atoms": self.atoms}

    def __setstate__(self, d):
        """Re-create the state of this object"""
        file_names = d.get("file_names")
        file_contents = d.get("file_contents")
        if file_names is not None and file_contents is not None:
            self._set_state_by_filecontents(file_names, file_contents)
        else:
            self._set_state_by_atoms(d["atoms"])

# SN: defined this outside to be able to compare the poijnt type conversions 
# for nonsquare correclty: it should be fulfilled by default, but porsiacaso.
class IdentityConversion:
    def __getitem__(self, key):
        return key
    
    def __eq__(self, other):
        return isinstance(other, IdentityConversion)


# OLD CODE:

# SN: I just copied how to do it for a single basis, so that the changes are minimal,
# and I do not mess up the grouping
def group(self, grouping: Literal["basis_shape", "point_type", "max"]) \
                    -> tuple["BasisTableWithEdges",
                    np.ndarray, np.ndarray, Optional[np.ndarray]]:
    filters = None
    if grouping == "point_type":
        new_table = self

        point_type_conversion = IdentityConversion()
        edge_type_conversion = IdentityConversion()

    elif grouping == "basis_shape":
        # Get all basis sizes:
        basis_sizes = np.zeros((len(self.basis), 5), dtype=int)
        for i, point_type_basis in enumerate(self.basis):
            for n, l, _ in point_type_basis.basis:
                basis_sizes[i, l] += n

        # Get the unique basis sizes
        unique_sizes, unique_indices, pseudo_types = np.unique(
            basis_sizes, axis=0, return_index=True, return_inverse=True
        )

        # Create the new table with the unique basis sizes. Here we just
        # take the first point type that has that basis size, but perhaps
        # it would be better to create a new point type that represents
        # that basis size (e.g. so that the name of the type is not misleading).
        new_table = self.__class__([self.basis[i] for i in unique_indices])
        point_type_conversion = pseudo_types

        # Get conversions
        old_edgetypes_to_new_point_types = point_type_conversion[
            self.edge_type_to_point_types
        ]
        edge_type_conversion = new_table.edge_type[
            old_edgetypes_to_new_point_types[:, 0],
            old_edgetypes_to_new_point_types[:, 1],
        ]
        # For edge type conversions handle the case in which the edge type
        # is negative.
        edge_type_conversion = np.concatenate(
            [edge_type_conversion, -1 * np.flip(edge_type_conversion[1:])]
        )
    elif grouping == "max":
        # Get all basis sizes:
        basis_sizes = np.zeros((len(self.basis), 5), dtype=int)
        for i, point_type_basis in enumerate(self.basis):
            for n, l, _ in point_type_basis.basis:
                basis_sizes[i, l] += n

        # Maximum sizes:
        max_sizes = basis_sizes.max(axis=0)

        # Build the new point basis
        max_basis = PointBasis(
            "all",
            R=self.maxR(),
            basis=[(int(n), l, (-1) ** l) for l, n in enumerate(max_sizes)],
            basis_convention=self.basis_convention,
        )
        # Create the new table with only one type.
        new_table = self.__class__([max_basis])

        # For each original point type, compute a mask that allows us to
        # select the values of the new basis that correspond to that point type.
        # (i.e. discard the values that are not present in that original point type).
        missing_ls = max_sizes - basis_sizes
        filters = np.zeros((len(self.basis), max_basis.basis_size), dtype=bool)
        i = 0
        for l, n in enumerate(max_sizes):
            if n == 0:
                continue

            for i_point, point_missing_ls in enumerate(missing_ls[:, l]):
                filters[i_point, i : i + (n - point_missing_ls) * (2 * l + 1)] = 1

            i += (2 * l + 1) * n

        # Point and edge type conversions, just map any type to 0.
        point_type_conversion = np.zeros(len(self.basis), dtype=int)
        edge_type_conversion = np.zeros(
            len(self.edge_type_to_point_types), dtype=int
        )
    else:
        raise NotImplementedError(f"Grouping by {grouping} is not implemented.")

    return new_table, point_type_conversion, edge_type_conversion, filters
