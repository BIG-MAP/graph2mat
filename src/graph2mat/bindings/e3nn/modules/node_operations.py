from e3nn import o3
import torch

__all__ = [
    "E3nnSimpleNodeBlock",
    "E3nnSeparateTSQNodeBlock",
]


class E3nnSimpleNodeBlock(torch.nn.Module):
    """Sums all node features and then passes them to a tensor square.

    All node features must have the same irreps.

    Example
    -------
    If we construct a SimpleNodeBlock:

    >>> irreps_in = o3.Irreps("2x0e + 2x1o")
    >>> irreps_out = o3.Irreps("3x2e")
    >>> node_block = SimpleNodeBlock(irreps_in, irreps_out)

    and then use it with 2 different nodewise tensors:

    >>> node_feats = torch.randn(10, irreps_in.dim)
    >>> node_messages = torch.randn(10, irreps_in.dim)
    >>> node_block(node_feats=node_feats, node_messages=node_messages)

    this is equivalent to:

    >>> tsq = o3.TensorSquare(irreps_in, irreps_out)
    >>> output = tsq(node_feats + node_messages)
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (list, tuple)) and not isinstance(
            irreps_in, o3.Irreps
        ):
            assert all(
                irreps == irreps_in[0] for irreps in irreps_in
            ), "All input irreps must be the same."
            irreps_in = irreps_in[0]

        self.tsq = o3.TensorSquare(irreps_in, irreps_out)

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        node_tensors = iter(node_kwargs.values())

        node_feats = next(node_tensors)
        for other_node_feats in node_tensors:
            node_feats = node_feats + other_node_feats

        return self.tsq(node_feats)


class E3nnSeparateTSQNodeBlock(torch.nn.Module):
    """Tensor squares each node features and then sums all outputs.

    Example
    -------
    If we construct a SeparateTSQNodeBlock:

    >>> irreps_in = o3.Irreps("3x0e + 2x1o")
    >>> irreps_out = o3.Irreps("3x2e")
    >>> node_block = SeparateTSQNodeBlock(irreps_in, irreps_out)

    and then use it with 2 different nodewise tensors:

    >>> node_feats = torch.randn(10, irreps_in.dim)
    >>> node_messages = torch.randn(10, irreps_in.dim)
    >>> output = node_block(node_feats=node_feats, node_messages=node_messages)

    this is equivalent to:

    >>> tsq1 = o3.TensorSquare(irreps_in, irreps_out)
    >>> tsq2 = o3.TensorSquare(irreps_in, irreps_out)
    >>> output = tsq1(node_feats) + tsq2(node_messages)
    """

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.tensor_squares = torch.nn.ModuleList(
            [
                o3.TensorSquare(this_irreps_in, irreps_out)
                for this_irreps_in in irreps_in
            ]
        )

    def forward(self, **node_kwargs: torch.Tensor) -> torch.Tensor:
        assert len(node_kwargs) == len(
            self.tensor_squares
        ), f"Number of input tensors ({len(node_kwargs)}) must match number of tensor square operations ({len(self.tensor_squares)})."

        node_tensors = iter(node_kwargs.values())

        node_feats = self.tensor_squares[0](next(node_tensors))
        for i, other_node_feats in enumerate(node_tensors):
            node_feats = node_feats + self.tensor_squares[i + 1](other_node_feats)

        return node_feats
