"""File containing modules that compute matrices"""
import torch
from typing import Tuple, List, Dict
from types import ModuleType

from graph2mat import Graph2Mat, MatrixBlock


class TorchMatrixBlock(MatrixBlock, torch.nn.Module):

    numpy = torch

class TorchGraph2Mat(Graph2Mat, torch.nn.Module):

    def __init__(self, 
        *args, 
        numpy: ModuleType = torch,
        self_interactions_list = torch.nn.ModuleList,
        interactions_dict = torch.nn.ModuleDict,
        **kwargs
    ):
        super().__init__(*args, **kwargs, numpy=numpy, self_interactions_list=self_interactions_list, interactions_dict=interactions_dict)

    def _forward_interactions_init_arrays_kwargs(self, edge_types_array):
        return {
            "device": edge_types_array.device,
        }
    