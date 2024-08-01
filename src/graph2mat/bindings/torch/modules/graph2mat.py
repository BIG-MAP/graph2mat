"""Torch wrappers for Graph2Mat."""
import torch
from types import ModuleType

from graph2mat import Graph2Mat

__all__ = ["TorchGraph2Mat"]


class TorchGraph2Mat(Graph2Mat, torch.nn.Module):
    """Wrapper for Graph2Mat to make it use torch instead of numpy.

    It also makes `Graph2Mat` a `torch.nn.Module`,  and it makes it
    store the list of node block functions as a `torch.nn.ModuleList`
    and the dictionary of edge block functions as a `torch.nn.ModuleDict`.

    Parameters
    ----------
    **kwargs:
        Additional arguments passed to the `Graph2Mat` class.

    See Also
    --------
    Graph2Mat
        The class that `TorchGraph2Mat` extends. Its documentation contains a more
        detailed explanation of the inner workings of the class.
    """

    def __init__(
        self,
        *args,
        numpy: ModuleType = torch,
        self_interactions_list=torch.nn.ModuleList,
        interactions_dict=torch.nn.ModuleDict,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
            numpy=numpy,
            self_interactions_list=self_interactions_list,
            interactions_dict=interactions_dict,
        )

    def _forward_interactions_init_arrays_kwargs(self, edge_types_array):
        return {
            "device": edge_types_array.device,
        }
