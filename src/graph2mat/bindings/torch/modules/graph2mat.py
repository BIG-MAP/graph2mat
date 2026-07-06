"""Torch wrappers for Graph2Mat."""
import numpy as np
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

    def _init_center_types(self, basis_grouping):
        super()._init_center_types(basis_grouping)

        for k in ("types_to_graph2mat", "edge_types_to_graph2mat"):
            array = getattr(self, k, None)
            if isinstance(array, np.ndarray):
                # Register the buffer as a torch tensor
                tensor = torch.from_numpy(getattr(self, k))
                delattr(self, k)
                self.register_buffer(k, tensor, persistent=False)

    def _get_labels_resort_index(
        self, types: torch.Tensor, original_types: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Wrapping of the method to use torch instead of numpy."""
        types = types.numpy(force=True)
        original_types = original_types.numpy(force=True)

        indices = super()._get_labels_resort_index(
            types, original_types=original_types, **kwargs
        )

        if self.basis_grouping != "max":
            indices = self.numpy.from_numpy(indices)

        return indices
    
    def _forward_self_interactions(self, *args, **kwargs):
        with torch.profiler.record_function("G2M::self_interactions_forward"):
            return super()._forward_self_interactions(*args, **kwargs)

    def _forward_interactions(self, *args, **kwargs):
        with torch.profiler.record_function("G2M::interactions_forward"):
            return super()._forward_interactions(*args, **kwargs)
