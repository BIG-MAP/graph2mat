"""Pytorch modules for matrices."""

from .basis_matrix import BasisMatrixReadout
from .node_readouts import *
from .edge_readouts import *

# Import the mace submodule only if MACE is available
try:
    import mace as _
except ModuleNotFoundError:
    pass
else:
    from .mace import *