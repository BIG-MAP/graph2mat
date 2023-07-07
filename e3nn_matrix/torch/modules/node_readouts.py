from abc import ABC, abstractmethod

from e3nn import o3
import torch

from ._misc import NonLinearTSQ

__all__ = [
    "NodeBlock",
    "SimpleNodeBlock",
    "SeparateTSQNodeBlock",
    "SeparateTSQLinearNodeBlock",
    "NonLinearTSQNodeBlock",
    "SeparateNonLinearTSQNodeBlock"
]

class NodeBlock(torch.nn.Module, ABC):
    """Base class for computing node blocks of an basis-basis matrix.
    
    Parameters
    -----------
    irreps_in: o3.Irreps
    irreps_out: o3.Irreps
    """
    @abstractmethod
    def forward(self, 
        node_feats: torch.Tensor, 
        node_messages: torch.Tensor,
    ) -> torch.Tensor:
        return node_feats

class SimpleNodeBlock(NodeBlock):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tsq = o3.TensorSquare(irreps_in, irreps_out)

    def forward(self, node_feats: torch.Tensor, node_messages: torch.Tensor) -> torch.Tensor:
        return self.tsq(node_feats + node_messages)

class SeparateTSQNodeBlock(NodeBlock):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tsq1 = o3.TensorSquare(irreps_in, irreps_out)
        self.tsq2 = o3.TensorSquare(irreps_in, irreps_out)

    def forward(self, node_feats: torch.Tensor, node_messages: torch.Tensor) -> torch.Tensor:
        return self.tsq1(node_feats) + self.tsq2(node_messages)

class SeparateTSQLinearNodeBlock(NodeBlock):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tsq1 = o3.TensorSquare(irreps_in, irreps_out)
        self.tsq2 = o3.TensorSquare(irreps_in, irreps_out)

        self.linear = o3.Linear(irreps_out, irreps_out)

    def forward(self, node_feats: torch.Tensor, node_messages: torch.Tensor) -> torch.Tensor:
        return self.linear(self.tsq1(node_feats) + self.tsq2(node_messages))

class NonLinearTSQNodeBlock(NodeBlock):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tsq = NonLinearTSQ(irreps_in, irreps_out)

    def forward(self, node_feats: torch.Tensor, node_messages: torch.Tensor) -> torch.Tensor:
        return self.tsq(node_feats + node_messages)

class SeparateNonLinearTSQNodeBlock(NodeBlock):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.tsq1 = NonLinearTSQ(irreps_in, irreps_out)
        self.tsq2 = NonLinearTSQ(irreps_in, irreps_out)

    def forward(self, node_feats: torch.Tensor, node_messages: torch.Tensor) -> torch.Tensor:
        return self.tsq1(node_feats) + self.tsq2(node_messages)
