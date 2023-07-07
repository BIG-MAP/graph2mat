"""Implements the Data class to use in pytorch models."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

import torch

from mace.tools import torch_geometric
from ..data.processing import BasisMatrixData

class BasisMatrixTorchData(BasisMatrixData, torch_geometric.data.Data):
    num_nodes: torch.Tensor
    edge_index: torch.Tensor
    neigh_isc: torch.Tensor
    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    cell: torch.Tensor
    n_supercells: int
    nsc: torch.Tensor
    point_labels: torch.Tensor
    edge_labels: torch.Tensor
    point_label_ptr: torch.Tensor
    edge_label_ptr: torch.Tensor
    point_types: torch.Tensor
    edge_types: torch.Tensor
    edge_type_nlabels: torch.Tensor
    metadata: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        BasisMatrixData.__init__(self, *args, **kwargs)
        torch_geometric.data.Data.__init__(self, **self._data)

    def process_input_array(self, key: str, array: np.ndarray) -> Any:

        if issubclass(array.dtype.type, float):
            return torch.tensor(array, dtype=torch.get_default_dtype())
        else:
            return torch.tensor(array)
    
    def ensure_numpy(self, array: torch.Tensor) -> np.ndarray:
        return array.numpy(force=True)
