import torch

from e3nn import o3

class FullyConnectedSymmTensorProduct(torch.nn.Module):
    """Fully connected tensor product with symmetric weights"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        kwargs["internal_weights"] = False
        self.tp = o3.FullyConnectedTensorProduct(*args, **kwargs)

        self._tp_weights = torch.nn.ParameterList()
        self._tp_indices = []
        for ins in self.tp.instructions:
            weight = torch.nn.parameter.Parameter(torch.rand(*ins.path_shape))
            self._tp_weights.append(weight)

    def forward(self, x, y):
        full_weights = []
        for weight in self._tp_weights:
            path_weights = (weight + weight.transpose(0, 1)) / 2
            full_weights.append(path_weights.ravel())

        full_weights = torch.concatenate(full_weights)

        return self.tp(x, y, full_weights)