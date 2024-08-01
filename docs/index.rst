
.. title:: graph2mat
.. meta::
   :description: graph2mat is a package for
   :keywords: ML, e3nn, graphs


graph2mat: Equivariant matrices meet machine learning
==============================================================

.. image:: /_static/images/graph2mat_overview.svg
    :align: center



The aim of ``graph2mat`` is to **pave your way into meaningful science** by providing the tools
to **interface to common machine learning frameworks** (``e3nn``, ``pytorch``) **to learn equivariant matrices**.

Installation
-------------

Using ``pip``, installation is as simple as:

.. code-block:: bash

    pip install graph2mat

I would like to...
------------------

    - **Learn and predict** matrices using built-in models: `CLI tutorials <tutorials/cli/index.rst>`_.
    - **Develop** my own matrix-predicting model: `Python API tutorials <tutorials/python_api/index.rst>`_.
    - Get an **overview of the python API**: `API overview <api/description.rst>`_.
    - **Find documentation** for a particular function/class: `API documentation <api/full_api.rst>`_.

Background
-----------

We use the term **equivariant matrix** to refer to a matrix whose rows and columns are
representing some basis made of spherical harmonics. The values of this matrix arise
from the interaction of such basis, and therefore follow the equivariance properties
of products of spherical harmonics.

.. image:: /_static/images/water_equivariant_matrix.png
    :align: center

One particular case of equivariant matrices are those in which **rows and columns represent
the same basis**. These matrices usually come up in physics when **atom-centered spherical
harmonics** are used as basis functions. Some examples are Hamiltonian and overlap matrices in
quantum chemistry. By the nature of the basis functions, which usually have a finite
range determined by a radial function, these matrices tend to be sparse.

Dealing with both the **equivariance and the sparsity** of these matrices within a machine
learning framework is not a trivial task. This can easily deter people from implementing
powerful applications that take full advantage of the properties of these matrices. With
``graph2mat``, we hope that people can explore the full potential of these matrices.

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutorials/cli/index
   tutorials/python_api/index

.. toctree::
   :maxdepth: 3
   :caption: API documentation
   :hidden:

   api/description
   api/full_api
