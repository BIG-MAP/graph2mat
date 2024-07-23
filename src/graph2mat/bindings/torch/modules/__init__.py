"""Wrappers for graph2mat modules in pytorch.

Torch does not add extra functionality to `graph2mat modules`, we just
need to wrap the core functionality to work with `torch` tensors instead
of `numpy`.
"""

from .graph2mat import *
from .matrixblock import *
