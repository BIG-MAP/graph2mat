from e3nn import o3

from scipy.sparse import csr_matrix

from mace.modules.utils import get_edge_vectors_and_lengths

from e3nn_matrix.data.configuration import BasisConfiguration

from e3nn_matrix.data.table import BasisTableWithEdges
from e3nn_matrix.data.configuration import BasisConfiguration
from e3nn_matrix.data.processing import MatrixDataProcessor

from e3nn_matrix.torch.data import BasisMatrixTorchData

from e3nn_matrix.torch.modules.mace import StandaloneMACEBasisMatrixReadout

def test_standalone_mace_readout(ABA_basis_configuration: BasisConfiguration, long_A_basis: bool):
    config = ABA_basis_configuration
    basis = ABA_basis_configuration.basis

    input_irreps = o3.Irreps('0e + 1o')

    readout = StandaloneMACEBasisMatrixReadout(
        node_feats_irreps=input_irreps,
        r_max=7,
        num_bessel=10,
        num_polynomial_cutoff=2,
        max_ell=2,
        edge_hidden_irreps=o3.Irreps('0e + 1o'),
        avg_num_neighbors=1.0,
        unique_basis=basis,
        symmetric=True
    )

    # Create the basis table.
    table = BasisTableWithEdges(basis)

    # Initialize the processor.
    processor = MatrixDataProcessor(
        basis_table=table, 
        symmetric_matrix=True,
        sub_point_matrix=False
    )

    data = BasisMatrixTorchData.from_config(config, processor)

    node_state = input_irreps.randn(3, -1, requires_grad=True)

    vectors, lengths = get_edge_vectors_and_lengths(
        positions=data.positions, edge_index=data.edge_index, shifts=data.shifts
    )

    print(data.edge_index)

    node_labels, edge_labels = readout.forward(
        node_feats=node_state,
        node_attrs=data['node_attrs'],
        node_types=data['point_types'],
        edge_index=data['edge_index'],
        edge_types=data['edge_types'],
        edge_vectors=vectors,
        edge_lengths=lengths,
        edge_type_nlabels=data['edge_type_nlabels'],
    )

    matrix = processor.output_to_matrix({'node_labels': node_labels , 'edge_labels': edge_labels}, data)

    assert isinstance(matrix, csr_matrix)
    assert matrix.shape == (5, 5)
    print(matrix.nnz, long_A_basis)
    assert matrix.nnz == 25 if long_A_basis else 23



