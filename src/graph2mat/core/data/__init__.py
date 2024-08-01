"""Tools to create and manipilate data to interact with the models.

It implements the functionality needed to handle (sparse) matrices that
are related to a graph. There are several things to take into account
which make the problem of handling the data non-trivial and therefore
this module useful:

- **Matrices are sparse**.
- Matrices are in a basis which is centered around the points in the graph. Therefore
  **elements of the matrix correspond to nodes or edges of the graph**.
- Each point might have more than one basis function, therefore **the matrix is divided
  in blocks (not just single elements)** that correspond to nodes or edges of the graph.
- Different point types might have different basis size, which makes **the different
  blocks in the matrix have different shapes**.
- **The different block sizes and the sparsity of the matrices supose and extra
  challenge when batching** examples for machine learning.

The tools in this submodule are agnostic to the machine learning framework
of choice, and they are based purely on `numpy`, with the extra dependency on `sisl`
to handle the sparse matrices. The `sisl` dependency could eventually be lift off
if needed.
"""
from .matrices import *

from .basis import PointBasis
from .configuration import BasisConfiguration, OrbitalConfiguration
from .metrics import OrbitalMatrixMetric
from . import metrics
from .processing import MatrixDataProcessor, BasisMatrixData
from .table import BasisTableWithEdges, AtomicTableWithEdges
