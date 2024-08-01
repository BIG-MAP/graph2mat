Comprehensive description
=========================

In this section we will highlight the most important aspects of the
`graph2mat` API, trying to give a nice high level overview of what
the package has to offer and how it works.

Core functionality
------------------

.. currentmodule:: graph2mat

These are classes that can be imported like:

.. code-block:: python

    from graph2mat import X

    # or

    import graph2mat
    graph2mat.X

They represent the core of `graph2mat`'s functionality, which can be then
extended to suit particular needs.

Graph2Mat
*********

The center of the package is the `Graph2Mat` class.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-class-template.rst

    Graph2Mat

It implements the skeleton to convert graphs to matrices.
The rest of the package revolves around this class, to:

- Help handling data.
- Help defining its architecture.
- Implement functions to be usedd within `Graph2Mat`.
- Provide helpers for training models.
- Ease its use for particular applications.

Basis
********

A 3D point cloud might have points with different basis functions, which results
in blocks of different size and shape in the matrix. Keeping information of the
basis is crucial to determine the architecture of the `Graph2Mat` model,
as well as to process data in the right way.

A unique point type with a certain basis is represented by a `PointBasis` instance,
while basis tables store all the unique point types that appear in the dataset and
therefore the model should be able to handle. Basis tables have helper methods to
aid with reshaping model outputs or disentangling batches of data, for example.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-class-template.rst

    PointBasis
    BasisTableWithEdges
    AtomicTableWithEdges

Data containers
********

These classes are used to store the data of your dataset.

`BasisConfiguration` and `OrbitalConfiguration` are used to store the raw information
of a structure, including its coordinates, atomic numbers and corresponing matrix.

`BasisMatrixData` is a container that stores the configuration in the shape of a graph,
ready to be used by models. It contains, for example, information about the edges. This
class is ready to be batched. It uses `numpy` arrays, therefore to use it in `torch` you
need to use the extension provided in `graph2mat.bindings.torch`.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-class-template.rst

    BasisMatrixData
    BasisConfiguration
    OrbitalConfiguration

Other useful top level modules
*******************************

These are some other modules which contain helper functions that might be useful to you.
FOr example, the `metrics` module contains functions that you can use as loss functions
to train your models.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-module-template.rst

    metrics
    sparse

Bindings
---------

Bindings are essential to use `graph2mat` in combination with other libraries. The core
of `graph2mat` is agnostic to the library you use, and you should choose the bindings
that you need for your specific use case.

Torch
*****

.. currentmodule:: graph2mat.bindings.torch

These are classes that can be imported like:

.. code-block:: python

    from graph2mat.bindings.torch import X

    # or

    import graph2mat.bindings.torch

    graph2mat.bindings.torch.X

Torch bindings implement **extensions of the core data functionality to make it usable in** `torch`.

The `TorchBasisMatrixData` is just a version of `BasisMatrixData` that uses `torch` tensors
instead of `numpy` arrays.

The `TorchBasisMatrixDataset` is a wrapper around `torch.utils.data.Dataset`
that creates `TorchBasisMatrixData` instances for each example in your dataset. It can therefore
be used with `torch_geometric`'s `DataLoader` to create batches of `TorchBasisMatrixData` instances.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-class-template.rst

    TorchBasisMatrixData
    TorchBasisMatrixDataset

The bindings also implement a version of `Graph2Mat` that deals with `torch` tensors
and is ready to be used for training models with `torch`.

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-class-template.rst

    TorchGraph2Mat

E3nn
*****

.. currentmodule:: graph2mat.bindings.e3nn

These are classes that can be imported like:

.. code-block:: python

    from graph2mat.bindings.e3nn import X

    # or

    import graph2mat.bindings.e3nn

    graph2mat.bindings.e3nn.X

Here's a table of the e3nn bindings that are available. There's `E3nnGraph2Mat`, which
is just an extension of the ``Graph2Mat`` model that handles `e3nn`'s irreps. And then
there are implementations of blocks that you might use within your model.

+----------------------------+----------------------+
| Class                      | Type of block        |
+============================+======================+
| `E3nnGraph2Mat`            | Model                |
+----------------------------+----------------------+
| `E3nnInteraction`          | Preprocessing        |
+----------------------------+----------------------+
| `E3nnEdgeMessageBlock`     | Preprocessing (edges)|
+----------------------------+----------------------+
| `E3nnSimpleNodeBlock`      | Node block readout   |
+----------------------------+----------------------+
| `E3nnSeparateTSQNodeBlock` | Node block readout   |
+----------------------------+----------------------+
| `E3nnSimpleEdgeBlock`      | Edge block readout   |
+----------------------------+----------------------+
