What is `e3nn_matrix`?
----------------------

It is an e3nn based package that implements output blocks to read out matrices with equivariant blocks from an equivariant neural network. An example of such matrices are orbital-orbital matrices like the **Hamiltonian** of an atomic system.

![e3nn_matrix diagram](https://i.imgur.com/CCurltj.png)

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

What is an equivariant interaction matrix?
------------------------------

Given two points in space that have each a basis set of spherical harmonics, one can define a matrix of interactions between all the basis functions in the system.

![eq_interaction_matrix](https://i.imgur.com/spAeca6.png)

This matrix is then very naturally splitted into blocks, where each block corresponds to a point-point interaction. In the picture above, these blocks are separated by the orange lines. Rotating one point around the other shouldn't change the interaction between points. However, the contribution of each pair of basis functions to this interaction will change. For example, if we were to move one point on top of the other (vertically), the red basis function should now account for the contribution that the blue and green ones accounted before, and viceversa. In fact, the change in the contributions with rotation is perfectly determined. The matrix is said to be equivariant to rotations in the positions of the points. This means that we don't have to learn any new fact about the interactions of the two points to get the new matrix, we just need to rotate our initial matrix.


Installation
------------

It can be installed with pip. Adding the tools extra will also install all the dependencies
needed to use the tools provided

```
pip install e3nn_matrix[tools]
```

Usage
------

### Server

To serve pretrained models, one can use the command `e3mat serve`. You can do `e3mat serve --help`
to check how it works. Serving models is as simple as:

```bash
e3mat serve some.ckpt other.ckpt
```

This will serve the models in the checkpoint files. However, we recommend that you organize your
checkpoints into folders and then pass the names of the folders instead.

```bash
e3mat serve first_model second_model
```

where `first_model` and `second_model` are folders that contain a `spec.yaml` file looking something like:

```yaml
description: |
    This model predicts single water molecules.
authors:
  - Pol Febrer (pol.febrer@icn2.cat)

files: # All the files related to this model.
  ckpt: best.ckpt
  basis: "*.ion.nc"
  structs: structs.xyz
  sample_metrics: sample_metrics.csv
  database: http://data.com/link/to/your/matrix_database
```

Once your server is running, you will get the url where the server is running, e.g. ttp://localhost:56000.
You can interact with it in multiple ways:
- Through the simple graphical interface included in the package, by opening `http://localhost:56000` in a browser.
- Through the `ServerClient` class in `e3nn_matrix.tools.server.api_client`.
- By simply sending requests to the API of the server. These requests must be sent to `http://localhost:56000/api`. You
can check the documentation for the requests that the server understands under `http://localhost:56000/api/docs`.

Package structure
-----------------

The package is structured into the following submodules:

- **e3nn_matrix.data**: Contains all the code implementing the data processing that is demanded to treat block equivariant matrices. This includes treatment of sparse matrices, managing the basis sets, etc...

- **e3nn_matrix.torch**: Contains the binding between the data processing tools in `e3nn_matrix.data` and `pytorch` so that the models
can be used in `pytorch` models.

- **e3nn_matrix.models**: Contains literature models that have been adapted to use `e3nn_matrix`.

- **e3nn_matrix.tools**: Contains tools to facilitate the usage of the package. They are described at the top of this README file.

The anatomy of an equivariant matrix.
-------
## Orbital-orbital matrix
    
We refer to orbital-orbital matrix in general to any matrix whose rows are basis orbitals
and columns are basis orbitals as well.
Some examples of this in Density Functional Theory (DFT) can be the Hamiltonian (H), the overlap
matrix (S) or the density matrix (DM). 

There can be other matrices in DFT or any other method that processes atoms with basis orbitals
that also follows this structure. This module is meant to be general enough to be applied out
of the box to any such matrix.
## Building it block by block 
The module builds the matrix block by block. We define a block as a region of the matrix
where the rows are all the orbitals of a given atom, and all the columns are the orbitals of another
given atom. This division is nice because then we can still keep atoms as the nodes of our graphs,
as usual. There are then two clearly different types of blocks by their origin, which might also
obey different symmetries:
    -  Self interaction blocks: These are blocks that encode the interactions between orbitals of the
    same atom. These blocks are always square matrices. They are located at the diagonal of the matrix. 
    If the matrix is symmetric, these blocks must also be symmetric.
    -  Interaction blocks: All the rest of blocks, that contain interactions between orbitals from different
    orbitals. Even if the matrix is symmetric, these blocks do not need to be symmetric. For each pair of atoms `ij`, 
    there are two blocks: `ij` and `ji` However, if the matrix is symmetric, one block is the transpose of the other. 
    Therefore, we only need to compute/predict one of them.
## How it is implemented.
The row and column size of each block are defined by the basis size (number of orbitals) of each atom.
Therefore, this module needs to create multiple functions.
    - For self interaction blocks: One function per atom type is required.
    - For interaction blocks: One function per combination of atom types is required.

## The density matrix
    
The density matrix encodes the electronic density in terms of basis orbitals.
Rows represent basis orbitals in the unit cell and columns represent basis orbitals
in the supercell. The supercell is as big as needed to account for all orbital overlaps
that happen inside the unit cell. If the system is not periodic or orbitals from different
cells do not overlap, the supercell is exactly the unit cell. Only in that case, the 
density matrix is square.
The electronic density at some point in space is reproduced from the density matrix by a
sum of the product of all pairs of orbitals that overlap at that point. Each `ij` element
in the density matrix is the coefficient that weights the contribution of the $orbital_i * orbital_j$
product. In a big system, most pairs of orbitals don't overlap, since they are far in space.
For all those pairs of orbitals, we don't need to store any coefficient, since the contribution
is 0 anyways. Therefore, the density matrix is a (potentially) very sparse matrix. We only need
to store, and therefore predict, the coefficients for orbital pairs that overlap at some point of space.
The density matrix is a symmetric object, that is `DM_ij == DM_ji`.
## How it is built.
The implementation for building the matrix is implemented in the `OrbitalMatrixReadout`, see the
documentation there for an extended explanation of how it is built.
