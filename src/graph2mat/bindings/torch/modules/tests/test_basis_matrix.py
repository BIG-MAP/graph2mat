# from e3nn import o3

# from scipy.sparse import csr_matrix
# import torch

# from graph2mat.data.configuration import BasisConfiguration

# from graph2mat.torch.modules import (
#     BasisMatrixReadout,
#     SimpleEdgeBlock,
#     SimpleNodeBlock,
# )

# from graph2mat.data.table import BasisTableWithEdges
# from graph2mat.data.configuration import BasisConfiguration
# from graph2mat.data.processing import MatrixDataProcessor

# from graph2mat.torch.data import BasisMatrixTorchData
# from graph2mat.torch.modules import BasisMatrixReadout


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
