"""Core functionality of the graph2mat package.

There are two main things that need to be implemented to make
fitting matrices a reality: **data handling and computation**.

This submodule implements all the routines to deal with data
containing graphs and sparse matrices related to graphs, as
well as the skeleton of `Graph2Mat`, the function to convert
graphs to matrices.

For now, this module doesn't implement the functions (modules/blocks)
to be used within `Graph2Mat`, because we have only worked with
equivariant functions and therefore we use the functions defined
in `graph2mat.bindings.e3nn` as the working blocks.
"""

from . import data
from . import modules

from .data import *
from .modules import *
