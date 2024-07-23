"""Interface to use the matrix models with ``pytorch_lightning``.

Pytorch lightning is a very useful to streamline the training and deployment
of machine learning models. It is the primary tool that ``e3nn_matrix`` has
chosen to support for this purpose. Keep in mind that you can use whatever
you want, though!

In this module we implement the data and model classes, which are just
interfaces to the pure ``e3nn_matrix`` classes. We also implement a CLI
that is just ``pytorch_lightning``'s CLI with some tweaks that we think
make its usage smoother for matrix learning.

"""

from .model import LitBasisMatrixModel
from .callbacks import *
from .data import MatrixDataModule
from .cli import OrbitalMatrixCLI, SaveConfigSkipBasisTableCallback
