graph2mat: Equivariant matrices meet machine learning
----------------------

![graph2mat_overview](https://raw.githubusercontent.com/BIG-MAP/graph2mat/main/docs/_static/images/graph2mat_overview.svg)

The aim of `graph2mat` is to pave your way into meaningful science by providing the **tools to interface to common machine learning frameworks** (e3nn, pytorch) **to learn equivariant matrices.**

**[Documentation](https://big-map.github.io/graph2mat/)**

It also provides a **set of tools** to facilitate the training and usage of the models created using the package:

- **Training tools**: It contains custom `pytorch_lightning` modules to train, validate and test the orbital matrix models.
- **Server**: A production ready server (and client) to serve predictions of the trained
    models. Implemented using `fastapi`.
- **Siesta**: A set of tools to interface the machine learning models with SIESTA. These include tools for input preparation, analysis of performance...

The package also implements a **command line interface** (CLI): `graph2mat`. The aim of this CLI is
to make the usage of `graph2mat`'s tools as simple as possible. It has two objectives:

- Make life easy for the model developers.
- Facilitate the usage of the models by non machine learning scientists, who just want
  good predictions for their systems.

Installation
------------

It can be installed with pip. Adding the tools extra will also install all the dependencies
needed to use the tools provided.

```
pip install graph2mat[tools]
```

If you want to use `graph2mat` with e3nn you can also ask for the `e3nn` extra dependencies:

```
pip install graph2mat[tools,e3nn]
```

You can also ask for

What is an equivariant matrix?
------------------------------

![water_equivariant_matrix](https://raw.githubusercontent.com/BIG-MAP/graph2mat/main/docs/_static/images/water_equivariant_matrix.png)


Contributions
--------------

We are very open to suggestions, contributions, discussions...

- If you have questions or want do discuss an idea, please [start a discussion](https://github.com/BIG-MAP/graph2mat/discussions)
- If you have a feature suggestion or bug report, please [open an issue](https://github.com/BIG-MAP/graph2mat/issues)

We look forward to your contributions!
