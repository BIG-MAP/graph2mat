"""Pytorch modules for matrices."""

from .graph2mat import TorchMatrixBlock, TorchGraph2Mat

# Import the mace submodule only if MACE is available
try:
    import mace as _
except ModuleNotFoundError:
    pass
else:
    pass
    #from .mace import *
