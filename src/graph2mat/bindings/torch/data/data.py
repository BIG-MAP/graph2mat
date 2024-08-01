"""Implements the Data class to use in pytorch models."""
from __future__ import annotations

from typing import Dict, Any

import numpy as np

import torch

from torch_geometric.data.data import Data

from graph2mat import BasisMatrixData

__all__ = ["TorchBasisMatrixData"]


class TorchBasisMatrixData(BasisMatrixData, Data):
    """Extension of `BasisMatrixData` to be used within pytorch.

    All this class implements is the conversion of numpy arrays to torch tensors
    and back. The rest of the functionality is inherited from `BasisMatrixData`.

    Please refer to the documentation of `BasisMatrixData` for more information.

    See Also
    --------
    BasisMatrixData
        The class that implements the heavy lifting of the data processing.
    """

    num_nodes: torch.Tensor
    edge_index: torch.Tensor
    neigh_isc: torch.Tensor
    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    cell: torch.Tensor
    n_supercells: torch.Tensor
    nsc: torch.Tensor
    point_labels: torch.Tensor
    edge_labels: torch.Tensor
    point_types: torch.Tensor
    edge_types: torch.Tensor
    edge_type_nlabels: torch.Tensor
    metadata: Dict[str, Any]

    def __init__(self, *args, **kwargs):
        data = BasisMatrixData._sanitize_data(self, **kwargs)
        Data.__init__(self, **data)

    def __getitem__(self, key: str) -> Any:
        return Data.__getitem__(self, key)

    def process_input_array(self, key: str, array: np.ndarray) -> Any:
        if isinstance(array, torch.Tensor):
            return array
        elif issubclass(array.dtype.type, float):
            return torch.tensor(array, dtype=torch.get_default_dtype())
        else:
            return torch.tensor(array)

    def ensure_numpy(self, array: torch.Tensor) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.numpy(force=True)
        else:
            return np.array(array)

    def is_node_attr(self, key: str) -> bool:
        return key in self._node_attr_keys

    def is_edge_attr(self, key: str) -> bool:
        return key in self._edge_attr_keys

    @property
    def _data(self):
        return {**self._store}
