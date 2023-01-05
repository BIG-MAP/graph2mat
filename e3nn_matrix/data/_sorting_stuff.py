"""Stuff that we were using for sorting atoms in batches but we don't use for now."""

class DatasetDescriptor:

    def __init__(self, unique_atoms):

        n_unique_atoms = len(unique_atoms)
        # Array to get the edge type from atom types. This does not need to be generated here,
        # since it is a property of the whole dataset.
        atom_types_to_edge_type = np.empty((n_unique_atoms, n_unique_atoms), dtype=np.int32)
        edge_type = 0
        for i in range(n_unique_atoms):
            for j in range(i, n_unique_atoms):
                edge_type = n_unique_atoms + j - i
                atom_types_to_edge_type[i, j] = edge_type
                atom_types_to_edge_type[j, i] = edge_type
                edge_type += 1
        
        self.unique_atoms = unique_atoms
        self.n_unique_atoms = n_unique_atoms
        self.edge_types = atom_types_to_edge_type

    @classmethod
    def from_dir(cls, basis_dir: Union[str, Path]):
        basis_dir = Path(basis_dir)

        unique_atoms = [
            sisl.get_sile(ion_file).read_basis() for ion_file in basis_dir.glob(".ion.xml")
        ]

        return cls(unique_atoms=unique_atoms)
    
    def get_global_species(self, atom: sisl.Atom):
        return self.unique_atoms.index(atom)

    def get_atom_translator(self, atoms: Sequence[sisl.Atom]):
        return {i: self.get_global_species(atom) for i, atom in enumerate(self.atoms.atom)}

    def get_global_edge_type(self, atom1_type, atom2_type):
        return 



@dataclass
class DensityMatrix2:
    atom_blocks: Dict[int, np.ndarray]
    edge_blocks: Dict[Tuple[int, int], np.ndarray]
    edge_index: Dict[Tuple[int, int], np.ndarray]
    atoms: sisl.Atoms
    reference: Union[DatasetDescriptor, None] = None

    def to_global_types(self, dataset_descriptor: DatasetDescriptor) -> "DensityMatrix2":
        if self.reference is not None:
            raise ValueError("This DensityMatrix already refers to a data descriptor, the conversion is not implemented yet.")

        atom_translation = 
        edge_translation = {
            edge: dataset_descriptor.edge_types[atom_translation[edge[0]], atom_translation[edge[1]]]
            for edge in self.edge_index.keys()
        }
        
        return self.__class__(
            atom_blocks={atom_translation[k]: v for k, v in self.atom_blocks.items()},
            edge_blocks={edge_translation[k]: v for k, v in self.edge_blocks.items()},
            edge_index={edge_translation[k]: v for k, v in self.edge_index.items()},
            atoms=self.atoms,
            reference=dataset_descriptor
        )

    def to_flat_torch_tensors(self, edge_index: np.ndarray, atom_order:Union[np.ndarray, None]=None
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        if atom_order is None:
            order = np.arange(len(self.orbital_count))
        else:
            order = atom_order
            assert len(order) == len(self.orbital_count)

        blocks = [self.block_dict[i,i].flatten() for i in order]
        sizes = [b.shape[0] for b in blocks]

        atom_labels_ptr = torch.tensor(np.cumsum(sizes))
        atom_labels = torch.tensor(np.concatenate(blocks))

        assert edge_index.shape[0] == 2, "edge_index is assumed to be [2, n_edges]"
        blocks = [self.block_dict[edge[0],edge[1]].flatten() for edge in edge_index.transpose()]
        sizes = [b.shape[0] for b in blocks]
        edge_labels_ptr = torch.tensor(np.cumsum(sizes))
        edge_labels = torch.tensor(np.concatenate(blocks))

        return atom_labels, atom_labels_ptr, edge_labels, edge_labels_ptr

@dataclass
class TypedConfiguration:
    """Same as Configuration, but every atomic property is splitted into the types of atoms.
    
    Atomic properties are stored as a dict where keys are atom types (integers) and values
    are the values for atoms that belong to that atom type (sorted by appearance on the
    geometry)
    """
    # Two extra parameters to handle types.
    atoms: sisl.Atoms
    atom_indices: Dict[int, np.ndarray]

    atomic_numbers: Dict[int, np.ndarray]
    positions: Dict[int, Positions]  # Angstrom
    cell: Optional[Cell] = None
    pbc: Optional[Pbc] = None
    density_matrix: Optional[DensityMatrix] = None

    weight: float = 1.0  # weight of config in loss
    config_type: Optional[str] = DEFAULT_CONFIG_TYPE  # config_type of config

    @classmethod
    def from_configuration(cls, configuration: Configuration, atoms):

        atomic_numbers = configuration.atomic_numbers
        positions = configuration.positions

        atomic_numbers, positions, atom_indices = atomic_data_to_type_dict(
            atomic_numbers, positions, np.arange(len(atomic_numbers)),
            atoms=atoms
        )

        return cls(
            atoms=atoms,
            atom_indices=atom_indices,

            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=configuration.cell,
            pbc=configuration.pbc,
            density_matrix=configuration.density_matrix,

            weight=configuration.weight,
            config_type=configuration.config_type,
        )

def csr_to_type_block_dict(
    spmat: sisl.SparseCSR, atoms: sisl.Atoms
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Creates dictionaries storing all blocks for a certain type of block.

    Since this function loops through all entries of the sparse matrix, we can already
    find all the edges in the sparsity pattern in the same go.

    Parameters
    -----------
    spmat: sisl.SparseCSR
        The sparse matrix that needs to be converted.
    atoms: sisl.Atoms
        The atoms associated with the sparse matrix.

    Returns
    --------
    atom_blocks:
        A dictionary where keys are atom types (an integer), and values are a
        numpy array of shape (natoms, norbs, norbs) storing all blocks for that
        atom type.
    edge_blocks:
        A dictionary where keys are edge types (an integer), and values are a
        numpy array of shape (nedges, norbs, norbs) storing all blocks for that
        edge type. Note that for a pair of atoms `ij` we only store one block, since
        block `ji` is just the transpose of block `ij`.
    edge_index:
        A dictionary where keys are edge types (an integer), and values are a
        numpy array of shape (2, nedges) storing all the atom pairs that correspond
        to an edge of that type. Both directions of the same edge are ALWAYS consecutive 
        in the array.    
    """
    species = atoms.specie
    # This array will store, for each atom type, the number of atoms found in this structure
    atoms_per_specie = np.zeros(atoms.nspecie, dtype=np.int32)
    for specie in range(atoms.nspecie):
        atoms_per_specie[specie] = (species == specie).sum()
        
    # Stores the number of orbitals of each atom type.
    orbitals = atoms.orbitals

    # Initialize the dictionary containing atom blocks.
    atom_blocks = {}
    for atom_type, (natoms, norbitals) in enumerate(zip(atoms_per_specie, orbitals)):
        if natoms == 0:
            continue

        atom_blocks[atom_type] = np.zeros((natoms, norbitals, norbitals), dtype=spmat.dtype)

    # Do the same for the dictionary containing edge blocks.
    # We can't know in advance how many edges there will be of a certain type, therefore
    # in this case we don't initialize the entries. We will keep two dictionaries:
    # edge_blocks: (atom_i, atom_j) -> block (a numpy array)
    # edge_blocks_by_type: (atomi_type, atomj_type) -> list of blocks
    edge_blocks = {}
    edge_blocks_by_type = {}
    # Also, since we will already loop over all entries of the density matrix, take the opportunity
    # to find all the edges and classify them by type. This dictionary stores, for each edge type,
    # a list with the pairs of atom indices that are edges of that type. That is, each value in the
    # dictionary will be [n_edges, 2]
    edge_index = {}
    for edge_type in itertools.product(range(atoms.nspecie), range(atoms.nspecie)):
        edge_blocks_by_type[edge_type] = []
        edge_index[edge_type] = []

    # Here, we have everything set up, so we can start looping over entries 
    # of the matrix and fill the blocks dictionaries.

    # Some variables that will make indexing through the loop easier
    atom_first_orb = atoms.firsto
    n_atoms = atoms.specie.shape[0]
    data = spmat.data[:, 0]
    # Mapping from row/column index to atom type
    rc_to_atom_index = np.concatenate([np.full(o, i, dtype=np.int32) for i, o in enumerate(orbitals)])
    # Keep track of how many atoms we have seen of a certain species. Then
    # we know which item in the atom blocks array to update.
    seen_atoms = np.zeros(atoms.nspecie, dtype=np.int32)
    
    # Loop over atom blocks in the rows.
    for atom_i in range(n_atoms):
        # Get the atom block, in this way we will not need to retreive it
        # each time.
        atomi_type = species[atom_i]
        atomi_block = atom_blocks[atomi_type][seen_atoms[atomi_type]] # atom block
        # Update the counter for atoms of this type already seen
        seen_atoms[atomi_type] += 1

        # Get the orbital limits of the block
        atomi_firsto = atom_first_orb[atom_i]
        atomi_lasto = atom_first_orb[atom_i + 1]
        # And the size of the block
        atomi_norbs = atomi_lasto - atomi_firsto

        # Loop over rows in this atom.
        for orbital_i in range(atomi_norbs):
            # Get the row index
            row = atomi_firsto + orbital_i

            # Find the limits of this row
            row_start = spmat.ptr[row]
            row_end = spmat.ptr[row+1]

            # Then, we will go over all the values in this row.
            # We keep an index to keep track of where we are in the array of values (data).
            current_index = row_start

            # First we will find indices that come before the atom block (are below the block diagonal)
            # We quickly skip over those.
            while spmat.col[current_index] < atomi_firsto:
                current_index += 1
            
            # Then we find the entries for the atom block.
            # Fill all columns, since all elements of the block are nonzero.
            next_index = current_index + atomi_norbs
            atomi_block[orbital_i, :] = data[current_index: next_index]
            current_index = next_index

            # Finally, we will get the entries that belong to edge blocks.
            # We don't know how many of those there are, so we just loop until
            # we get to the end of the row.
            while current_index < row_end:
                # In each iteration of this loop, we are going to fill all columns
                # of a given ij edge block.
                # Get the j atom for this edge block.
                col = spmat.col[current_index]
                atom_j = rc_to_atom_index[col]

                # Get the column limits for the j atom.
                atomj_firsto = atom_first_orb[atom_j]
                atomj_lasto = atom_first_orb[atom_j + 1]

                # Try to get the edge block. If we can't, this means that it hasn't been
                # created yet, so we create it.
                try:
                    edge_block = edge_blocks[atom_i, atom_j]
                except KeyError:
                    # We didn't create an edge for this pair of atoms yet

                    # Get the edge_type
                    atomj_type = species[atom_j]

                    # Create a new block with the appropiate size
                    atomj_norbs = atomj_lasto - atomj_firsto
                    edge_block = np.zeros((atomi_norbs, atomj_norbs), dtype=spmat.dtype)

                    # And set it on the dictionaries
                    edge_blocks[atom_i, atom_j] = edge_block
                    edge_blocks_by_type[atomi_type, atomj_type].append(edge_block)

                    # Also store the new edge (in both directions)
                    edge_index[edge_type].extend(((atom_i, atom_j), (atom_j, atom_i)))

                # Loop until we fall outside of the limits, updating the edge block on
                # the way.
                while col < atomj_lasto:
                    orbital_j = col - atomj_firsto
                    edge_block[orbital_i, orbital_j] = data[current_index]
                    
                    current_index += 1
                    if current_index >= row_end:
                        break
                    col = spmat.col[current_index]
    
    # Now that we have all edges stored, convert the elements of the edges dictionaries
    # from lists to numpy arrays. Delete entries with 0 edges.
    for k in list(edge_blocks_by_type.keys()):
        if len(edge_index[k]) == 0:
            del edge_blocks_by_type[k]
            del edge_index[k]
        else:
            edge_blocks_by_type[k] = np.array(edge_blocks_by_type[k], dtype=spmat.dtype)
            edge_index[k] = np.array(edge_index[k], dtype=np.int32).T
    
    return DensityMatrix2(atom_blocks=atom_blocks, edge_blocks=edge_blocks_by_type, edge_index=edge_index, atoms=atoms)

def atomic_data_to_type_dict(*data, atoms: sisl.Atoms):
    """Converts any atomic data to a dictionary that contains
    the data splitted by atom type.
    """
    type_dicts = tuple([{}] * len(data))
    species = atoms.specie
    
    for specie in range(atoms.nspecie):
        this_specie = (species == specie)
        for i, vals in enumerate(data):
            type_dicts[i][specie] = vals[this_specie]
    
    return type_dicts

def type_block_dict_to_flat(block_dict):
    """Converts from a dictionary of block types to a flat array.
    
    If the dictionary contains numpy arrays, a flat numpy array is
    returned.
    If the dictionary contains torch tensors, a flat torch tensor
    is returned.
    """
    
    types = sorted(block_dict.keys())
    
    tensors = isinstance(block_dict[types[0]], torch.Tensor)
    
    lst = [
        block_dict[block_type].ravel() for block_type in types
    ]
    
    if tensors:
        return torch.concatenate(lst)
    else:
        return np.concatenate(lst)
    
def flat_to_type_block_dict(flat, types_pointer, types_blockshape):
    """Converts from a flat tensor to a dictionary of block types.
    
    It needs some helper arrays.
    
    Parameters
    ----------
    types_pointer: array-like of shape (ntypes + 1, )
        Array indicating, for each type, where the data begins.
    types_blockshape: array-like of shape (ntypes, *block_dims)
        Array indicating, for each type, the shape of blocks.
    """
    
    block_dict = {}
    
    ntypes = types_pointer.shape[0] - 1
    
    for itype in range(ntypes):
        start = types_pointer[itype]
        end = types_pointer[itype]
        
        if start == end:
            continue
        
        block_dict[type] = flat[start:end].reshape(-1, *types_blockshape[itype])
    
    return block_dict

