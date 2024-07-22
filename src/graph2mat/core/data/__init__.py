"""Tools to create and manipilate data to interact with the models.

**This is the core of `e3nn_matrix`**. It implements the functionality
needed to handle sparse equivariant matrices.

The tools here are generic for any machine learning framework, although
this does not mean that you can straightforwardly use the package with
any framework. Some interfacing work is probably needed. See for example
``e3nn_matrix.torch``, which implements the interface to ``pytorch``.
"""
from .matrices import *

from .basis import PointBasis
from .configuration import BasisConfiguration, OrbitalConfiguration
from .metrics import OrbitalMatrixMetric
from .processing import MatrixDataProcessor, BasisMatrixData
from .table import BasisTableWithEdges, AtomicTableWithEdges
