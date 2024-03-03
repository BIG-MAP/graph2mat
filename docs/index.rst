
.. title:: e3nn_matrix
.. meta::
   :description: e3nn_matrix is a package for generating sparse equivariant matrices
   :keywords: ML, e3nn, graphs


e3nn_matrix: Sparse equivariant matrices meet machine learning
==============================================================

The aim of ``e3nn_matrix`` is to **pave your way into meaningful science** by providing the tools
to **interface common machine learning frameworks** (``e3nn``, ``pytorch``) **to equivariant matrices**.

Installation
-------------

Using ``pip``, installation is as simple as:

.. code-block:: bash

    pip install e3nn_matrix

I would like to...
------------------

    - **Learn and predict** matrices using built-in models: `Application tutorials <tutorials/applications/index.rst>`_.
    - **Develop** my own matrix-predicting model: `Low-level tutorials <tutorials/low_level/index.rst>`_.
    - **Find documentation** for a particular function/class: `API documentation <api/index.rst>`_.

Background
-----------

We use the term **equivariant matrix** to refer to a matrix whose rows and columns are
representing some basis made of spherical harmonics. The values of this matrix arise
from the interaction of such basis, and therefore follow the equivariance properties
of products of spherical harmonics.

One particular case of equivariant matrices are those in which **rows and columns represent
the same basis**. These matrices usually come up in physics when **atom-centered spherical
harmonics** are used as basis functions. Some examples are Hamiltonian and overlap matrices in
quantum chemistry. By the nature of the basis functions, which usually have a finite
range determined by a radial function, these matrices tend to be sparse.

Dealing with both the **equivariance and the sparsity** of these matrices within a machine
learning framework is not a trivial task. This can easily deter people from implementing
powerful applications that take full advantage of the properties of these matrices. With
``e3nn_matrix``, we hope that people can explore the full potential of these matrices.

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutorials/applications/index
   tutorials/low_level/index

.. autosummary::
    :toctree: api-generated/
    :template: autosummary/custom-module-template.rst
    :recursive:
    :caption: Reference documentation

    e3nn_matrix
