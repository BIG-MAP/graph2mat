"""Interface to e3nn, as well as functions that use irreps.

`e3nn` uses `torch`.

Therefore, the interface with `e3nn` takes as
a starting point the interface with `torch`, defined in `graph2mat.bindings.torch`.

This interface has two goals:

- **Wrap the core functionality**. The main addition to the core functionality is two thin
  wrappers around ``TorchGraph2Mat`` and ``TorchMatrixBlock`` to handle irreps. **There is no
  need to wrap the data containers for now, as we don't define data in terms of irreps**. In
  the future, one could envision for example training directly on irreps, and that would
  require a wrapper around `TorchBasisMatrixData`.

- **Implement equivariant functions that use** `e3nn` **irreps** and therefore can not be implemented
  in `graph2mat.core` without importing `e3nn`. These functions can be used as the blocks within `Graph2Mat`
  to make the model equivariant.

"""

from .modules import *
