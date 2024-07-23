"""Torch wrappers for matrix block."""
import torch

from graph2mat import MatrixBlock

__all__ = ["TorchMatrixBlock"]


class TorchMatrixBlock(MatrixBlock, torch.nn.Module):
    """Wrapper for matrix block to make it use torch instead of numpy."""

    numpy = torch
