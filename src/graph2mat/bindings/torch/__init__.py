"""Interface with pytorch.

The interface with `torch` is simple to understand. Instead of
using `numpy` we need to use `torch`.

We wrap the `BasisMatrixData` and `Graph2Mat` to do that, generating
`TorchBasisMatrixData` and `TorchGraph2Mat` respectively. Any framework
that uses `torch` should use these classes instead of the original ones.

Also, if extra bindings are needed for a framework that uses `torch`,
they should take these bindings as a starting point. E.g. if bindings
are implemented for `X`, `XGraph2Mat` should inherit from `TorchGraph2Mat`.

These bindings contain no extra functionality, as all that we do is to
make sure that the core functionality works with `torch` tensors.
"""

from .data import *
from .modules import *
