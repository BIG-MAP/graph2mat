from e3nn import o3
import torch

from typing import Sequence, Type, Union, Optional, Dict

from graph2mat import PointBasis, MatrixBlock

from graph2mat.bindings.torch import TorchGraph2Mat

from .matrixblock import E3nnIrrepsMatrixBlock
from .node_operations import E3nnSimpleNodeBlock
from .edge_operations import E3nnSimpleEdgeBlock

__all__ = ["E3nnGraph2Mat"]


class E3nnGraph2Mat(TorchGraph2Mat):
    def __init__(
        self,
        unique_basis: Sequence[PointBasis],
        irreps: Dict[str, o3.Irreps],
        preprocessing_nodes: Optional[Type[torch.nn.Module]] = None,
        preprocessing_nodes_kwargs: dict = {},
        preprocessing_edges: Optional[Type[torch.nn.Module]] = None,
        preprocessing_edges_kwargs: dict = {},
        node_operation: Type = E3nnSimpleNodeBlock,
        node_operation_kwargs: dict = {},
        edge_operation: Type = E3nnSimpleEdgeBlock,
        edge_operation_kwargs: dict = {},
        symmetric: bool = False,
        blocks_symmetry: str = "ij",
        self_blocks_symmetry: Union[str, None] = None,
        matrix_block_cls: Type[MatrixBlock] = E3nnIrrepsMatrixBlock,
        **kwargs,
    ):
        preprocessing_nodes_kwargs = {"irreps": irreps, **preprocessing_nodes_kwargs}
        preprocessing_edges_kwargs = {"irreps": irreps, **preprocessing_edges_kwargs}

        node_operation_kwargs = {"irreps": irreps, **node_operation_kwargs}
        edge_operation_kwargs = {"irreps": irreps, **edge_operation_kwargs}

        super().__init__(
            unique_basis=unique_basis,
            preprocessing_nodes=preprocessing_nodes,
            preprocessing_nodes_kwargs=preprocessing_nodes_kwargs,
            preprocessing_edges=preprocessing_edges,
            preprocessing_edges_kwargs=preprocessing_edges_kwargs,
            node_operation=node_operation,
            node_operation_kwargs=node_operation_kwargs,
            edge_operation=edge_operation,
            edge_operation_kwargs=edge_operation_kwargs,
            matrix_block_cls=matrix_block_cls,
            symmetric=symmetric,
            blocks_symmetry=blocks_symmetry,
            self_blocks_symmetry=self_blocks_symmetry,
            **kwargs,
        )

    def _get_readout_irreps(
        self, preprocessor, irreps: Dict[str, o3.Irreps]
    ) -> Dict[str, o3.Irreps]:
        """Possibly updates the irreps if there is a preprocessing step."""
        if preprocessor is not None:
            irreps_out = preprocessor.irreps_out

            if isinstance(irreps_out, o3.Irreps):
                node_feats_irreps = irreps_out
                edge_message_irreps = None
            else:
                # Otherwise it's a tuple with node and edge message irreps
                node_feats_irreps, edge_message_irreps = irreps_out

            irreps = {
                **irreps,
                "node_feats_irreps": node_feats_irreps,
                "edge_message_irreps": edge_message_irreps,
            }

        return irreps

    def _init_self_interactions(
        self, *args, preprocessor=None, irreps: Dict[str, o3.Irreps] = {}, **kwargs
    ):
        readout_irreps = self._get_readout_irreps(preprocessor, irreps)

        return super()._init_self_interactions(
            *args, **kwargs, preprocessor=preprocessor, irreps=readout_irreps
        )

    def _init_interactions(
        self, *args, preprocessor=None, irreps: Dict[str, o3.Irreps] = {}, **kwargs
    ):
        readout_irreps = self._get_readout_irreps(preprocessor, irreps)

        return super()._init_interactions(
            *args, **kwargs, preprocessor=preprocessor, irreps=readout_irreps
        )
