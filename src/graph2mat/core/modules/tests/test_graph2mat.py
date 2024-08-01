import pytest

from scipy.sparse import csr_matrix

from graph2mat import (
    BasisTableWithEdges,
    BasisConfiguration,
    MatrixDataProcessor,
    BasisMatrixData,
    Graph2Mat,
    PointBasis,
)


@pytest.fixture(scope="module", params=[False, True])
def B_nobasis(request):
    return request.param


@pytest.fixture
def basis_table(B_nobasis):
    point_1 = PointBasis("A", R=2, basis=[1], basis_convention="cartesian")

    if B_nobasis:
        point_2 = PointBasis("B", R=5)
    else:
        point_2 = PointBasis("B", R=5, basis=[2, 1], basis_convention="cartesian")

    return BasisTableWithEdges([point_1, point_2])


def test_no_operations(basis_table):
    """Can't initialize a Graph2Mat without any block reading operations."""
    with pytest.raises(AssertionError):
        Graph2Mat(unique_basis=basis_table)


# Create a dummy node operation and a dummy edge operation
# that are going to check that everything is initialized correctly.
class DummyNodeOp:
    def __init__(self, i_basis, j_basis, symmetry, symmetric):
        assert isinstance(i_basis, PointBasis)
        assert isinstance(j_basis, PointBasis)
        assert i_basis is j_basis

        # Assert symmetry
        assert symmetry == "ij=ji" if symmetric else "ij", symmetry

    def __call__(self, **kwargs):
        return kwargs


class DummyEdgeOp:
    def __init__(self, i_basis, j_basis, symmetry):
        assert isinstance(i_basis, PointBasis)
        assert isinstance(j_basis, PointBasis)

        # Assert no symmetry
        assert symmetry == "ij", symmetry

    def __call__(self, **kwargs):
        return kwargs


@pytest.mark.parametrize("symmetric", [True, False])
def test_init_graph2mat(basis_table, symmetric, B_nobasis):
    """Test the initialization of a Graph2Mat object"""

    g2m = Graph2Mat(
        unique_basis=basis_table,
        symmetric=symmetric,
        node_operation=DummyNodeOp,
        node_operation_kwargs={"symmetric": symmetric},
        edge_operation=DummyEdgeOp,
    )

    # Check that we have the correct number of self interactions
    assert isinstance(g2m.self_interactions, list)
    assert len(g2m.self_interactions) == len(basis_table.basis)
    assert isinstance(g2m.self_interactions[0].operation, DummyNodeOp)
    if B_nobasis:
        assert g2m.self_interactions[1] is None
    else:
        assert isinstance(g2m.self_interactions[1].operation, DummyNodeOp)

    assert isinstance(g2m.interactions, dict)
    assert (
        len(g2m.interactions) == len(basis_table.edge_block_size)
        if symmetric
        else len(basis_table.basis) ** 2
    )
    if not B_nobasis:
        assert all(
            isinstance(interaction.operation, DummyEdgeOp)
            for interaction in g2m.interactions.values()
        )
    else:
        assert (
            sum(interaction is None for interaction in g2m.interactions.values())
            == len(g2m.interactions) - 1
        )
        print(g2m.interactions)
        assert isinstance(g2m.interactions["(0, 0, 0)"].operation, DummyEdgeOp)


# @pytest.mark.parametrize("symmetric", [True, False])
# def test_init_graph2mat(basis_table, symmetric):
#     """Test the initialization of a Graph2Mat object"""

#     g2m = Graph2Mat(
#         unique_basis=basis_table,
#         symmetric=symmetric,
#         node_operation=DummyNodeOp,
#         node_operation_kwargs={"symmetric": symmetric},
#         edge_operation=DummyEdgeOp,
#     )
# def test_irreps_in(ABA_basis_configuration: BasisConfiguration):
#     config = ABA_basis_configuration
#     basis = ABA_basis_configuration.basis

#     input_irreps = o3.Irreps("0e + 1o")

#     readout = BasisMatrixReadout(
#         unique_basis=basis,
#         node_operation_kwargs={"irreps_in": input_irreps},
#         edge_operation_kwargs={
#             "irreps_in": input_irreps,
#         },
#         symmetric=True,
#     )

#     readout2 = BasisMatrixReadout(
#         unique_basis=basis,
#         irreps_in=input_irreps,
#         symmetric=True,
#     )

#     assert str(readout) == str(readout2)


# def test_readout(ABA_basis_configuration: BasisConfiguration, basis_type: str):
#     config = ABA_basis_configuration
#     basis = ABA_basis_configuration.basis

#     input_irreps = o3.Irreps("0e + 1o")

#     readout = BasisMatrixReadout(
#         unique_basis=basis,
#         node_operation_kwargs={"irreps_in": input_irreps},
#         edge_operation_kwargs={
#             "irreps_in": input_irreps,
#         },
#         symmetric=True,
#     )

#     # Create the basis table.
#     table = BasisTableWithEdges(basis)

#     # Initialize the processor.
#     processor = MatrixDataProcessor(
#         basis_table=table, symmetric_matrix=True, sub_point_matrix=False
#     )

#     data = BasisMatrixTorchData.from_config(config, processor)

#     node_state = input_irreps.randn(3, -1, requires_grad=True)

#     node_labels, edge_labels = readout.forward(
#         node_types=data["point_types"],
#         edge_index=data["edge_index"],
#         edge_types=data["edge_types"],
#         edge_type_nlabels=data["edge_type_nlabels"],
#         node_operation_node_kwargs={
#             "state": node_state,
#         },
#         edge_operation_node_kwargs={
#             "node_state": node_state,
#         },
#     )

#     matrix = processor.matrix_from_data(
#         data,
#         {"node_labels": node_labels, "edge_labels": edge_labels},
#     )

#     assert isinstance(matrix, csr_matrix)
#     assert matrix.shape == (5, 5) if basis_type != "nobasis_A" else (3, 3)
#     assert matrix.nnz == {"normal": 23, "long_A": 25, "nobasis_A": 9}[basis_type]


# def test_readout_filtering(
#     ABA_basis_configuration: BasisConfiguration, basis_type: str
# ):
#     config = ABA_basis_configuration
#     basis = ABA_basis_configuration.basis

#     input_irreps = o3.Irreps("0e + 1o")

#     class EdgeChecker(SimpleEdgeBlock):
#         """Extension of SimpleEdgeBlock that edge kwargs and node kwargs have been correctly filtered."""

#         def forward(self, edge_types, node_types, **kwargs):
#             assert isinstance(edge_types, tuple)
#             assert len(edge_types) == 2
#             assert all(isinstance(x, torch.Tensor) for x in edge_types)
#             assert torch.all(edge_types[0] == edge_types[0][0, 0])
#             assert torch.all(edge_types[0] == -edge_types[1])

#             assert isinstance(node_types, tuple)
#             assert len(node_types) == 2
#             assert all(isinstance(x, torch.Tensor) for x in node_types)
#             assert torch.all(node_types[0] == node_types[0][0, 0])
#             assert torch.all(node_types[1] == node_types[1][0, 0])

#             return super().forward(**kwargs)

#     class NodeChecker(SimpleNodeBlock):
#         """Extension of SimpleEdgeBlock that edge kwargs and node kwargs have been correctly filtered."""

#         def forward(self, node_types, **kwargs):
#             assert isinstance(node_types, torch.Tensor)
#             assert torch.all(node_types == node_types[0, 0])

#             return super().forward(**kwargs)

#     readout = BasisMatrixReadout(
#         unique_basis=basis,
#         node_operation=NodeChecker,
#         node_operation_kwargs={"irreps_in": input_irreps},
#         edge_operation=EdgeChecker,
#         edge_operation_kwargs={
#             "irreps_in": input_irreps,
#         },
#         symmetric=True,
#     )

#     # Create the basis table.
#     table = BasisTableWithEdges(basis)

#     # Initialize the processor.
#     processor = MatrixDataProcessor(
#         basis_table=table, symmetric_matrix=True, sub_point_matrix=False
#     )

#     data = BasisMatrixTorchData.from_config(config, processor)

#     node_state = input_irreps.randn(3, -1, requires_grad=True)

#     edge_types = torch.tensor(
#         [*data["edge_types"]], dtype=torch.get_default_dtype()
#     ).reshape(-1, 1)
#     node_types = torch.tensor(
#         [*data["point_types"]], dtype=torch.get_default_dtype()
#     ).reshape(-1, 1)

#     node_labels, edge_labels = readout.forward(
#         node_types=data["point_types"],
#         edge_index=data["edge_index"],
#         edge_types=data["edge_types"],
#         edge_type_nlabels=data["edge_type_nlabels"],
#         node_operation_node_kwargs={
#             "state": node_state,
#             "node_types": node_types,
#         },
#         edge_operation_node_kwargs={
#             "node_state": node_state,
#             "node_types": node_types,
#         },
#         edge_kwargs={
#             "edge_types": edge_types,
#         },
#     )

#     matrix = processor.matrix_from_data(
#         data,
#         {"node_labels": node_labels, "edge_labels": edge_labels},
#     )

#     assert isinstance(matrix, csr_matrix)
#     assert matrix.shape == (5, 5) if basis_type != "nobasis_A" else (3, 3)
#     assert matrix.nnz == {"normal": 23, "long_A": 25, "nobasis_A": 9}[basis_type]
