What is `e3nn_matrix`?
----------------------

It is an e3nn based package that implements output blocks to read out matrices with equivariant blocks from an equivariant neural network. An example of such matrices are orbital-orbital matrices like the **Hamiltonian** of an atomic system.

![e3nn_matrix diagram](https://i.imgur.com/xUQWnhd.png)

It also provides a **set of tools** to facilitate the training and usage of the models created using the package:

- **Training tools**: It contains custom `pytorch_lightning` modules to train, validate and test the orbital matrix models.
- **Server**: A production ready server (and client) to serve predictions of the trained
    models. Implemented using `fastapi`.
- **Siesta**: A set of tools to interface the machine learning models with SIESTA. These include tools for input preparation, analysis of performance...

The package also implements a **command line interface** (CLI): `e3mat`. The aim of this CLI is
to make the usage of `e3nn_matrix`'s tools as simple as possible. It has two objectives:
    - Make life easy for the model developers.
    - Facilitate the usage of the models by non machine learning scientists, who just want
      good predictions for their systems.

Installation
------------

It can be installed with pip. Adding the tools extra will also install all the dependencies
needed to use the tools provided

```
pip install e3nn_matrix[tools]
```

Usage
------

TO BE FILLED...

Package structure
-----------------

The package is structured into the following submodules:

- **e3nn_matrix.data**: Contains all the code implementing the data processing that is demanded to treat block equivariant matrices. This includes treatment of sparse matrices, managing the basis sets, etc...

- **e3nn_matrix.torch**: Contains the binding between the data processing tools in `e3nn_matrix.data` and `pytorch` so that the models
can be used in `pytorch` models.

- **e3nn_matrix.models**: Contains literature models that have been adapted to use `e3nn_matrix`.

- **e3nn_matrix.tools**: Contains tools to facilitate the usage of the package. They are described at the top of this README file.
