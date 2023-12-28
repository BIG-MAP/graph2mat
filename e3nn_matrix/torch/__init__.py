"""Functionality specific to pytorch.

This module contains the data interfaces to pytorch, as well as the
matrix generating functions implemented as torch modules.
"""

from .data import BasisMatrixTorchData
from .dataset import BasisMatrixDataset, InMemoryData
from .modules import *
