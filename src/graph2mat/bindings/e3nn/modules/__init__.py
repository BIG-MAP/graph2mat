"""E3nn based functions to use in graph2mat.

In `Graph2Mat`, if you are fitting an equivariant matrix, you might
want to design an equivariant model.

For this reason, this module implements the equivariant functions to
use as the blocks within `Graph2Mat`. It also implements an
`E3nnGraph2Mat` class which is just a subclass of `Graph2Mat` with the
right defaults and an extra argument to pass e3nn's irreps.
"""
from .graph2mat import *
from .matrixblock import *
from .edge_operations import *
from .node_operations import *
from .preprocessing import *
