==========================
MD acceleration in SIESTA
==========================

*Approximate time to run the tutorial: 1 hour.*

This tutorial will guide you through all the steps to **train and deploy a model that predicts SIESTA matrices**.

In particular, we will train a model to predict the density matrix of a DZP water molecule. We will:

    -  Run a 50 step MD to **generate the training dataset**.
    -  **Train a model** on that dataset.
    -  Use the model to **predict the matrix** for the next 50 steps, to try to accelerate the SCF cycle.

We have chosen this example because it is something **feasible to do on a laptop, even without a dedicated GPU card**.
So, it is something that everybody can do. However, the procedure described here can be easily adapted to much more
complex and relevant use cases.

.. warning::

    This tutorial assumes that you have ``SIESTA`` installed (ideally with ``lua`` support, or you'll miss some parts!).

    You can install it with ``conda``:

    .. code-block:: bash

        conda install -c conda-forge siesta

    It also assumes that you have installed ``graph2mat`` with
    all its optional dependencies, which can be done with ``pip install graph2mat[tools]``.

Generating a dataset
--------------------

A matrix dataset is simply a directory that contains one folder for each structure in the dataset.
Each folder must contain the files needed to read the structure and the matrix.

Therefore, the first thing you must do is to **choose which SIESTA matrix** you want to train your model for. You have
basically four choices:

    - **Density matrix**: SIESTA outputs it in the ``*.DM`` or ``*.TSDE`` files.
    - **Hamiltonian**: SIESTA outputs it in the ``*.HSX`` or ``*.TSHS`` files.
    - **Energy density matrix**: SIESTA outputs it in the ``*.TSDE`` file.
    - **Overlap matrix**: SIESTA outputs it in the ``*.HSX`` or ``*.TSHS`` files.

Whatever matrix you choose, you must make sure that SIESTA is outputing that matrix at the end of the SCF cycle.
The following inputs must be set on each case:

    - **.DM**: Nothing to do, they are always written.
    - **.HSX**: ``SaveHS t``
    - **.TSDE**: ``TS.DE.Save t``
    - **.TSHS**: ``TS.HS.Save t``

Then, you just have to run SIESTA for all the structures you want to include in your dataset, and keep the matrix files.

**Let's get started!**

Let's create a directory called `graph2mat_tutorial`, where we will create several directories for each step of the process.

.. code-block:: bash

    mkdir graph2mat_tutorial
    cd graph2mat_tutorial

The first directory that we will create there is `dataset`. Here we will store the training dataset.

.. code-block:: bash

    mkdir dataset

If you want to generate a dataset from an MD run and you have SIESTA compiled with lua support,
``graph2mat``'s CLI can automatically generate the needed lua script for you.
We will use this functionality to create the script inside our ``dataset`` directory.

.. code-block:: bash

    cd dataset
    graph2mat siesta md setup-store

You can add ``--help`` to see the available options to tweak the generated script, but the default ones should suffice for us.
If everything went right, you should now have a ``md_store.lua`` file in your current directory. You can explore it to understand
what it will do. Basically, it will create a directory (``MD_steps``) where it will store the matrices for each step of the MD.

Now, all we need is to define the inputs for our water molecule MD!

Store the following content in a file called ``RUN.fdf``:

.. code-block::

    # Run 50 steps of a verlet MD
    MD.TypeOfRun verlet
    MD.Steps 50

    # Use the default Double Zeta Polarized basis.
    PAO.BasisSize DZP

    # Save all matrices
    TS.HS.Save t
    TS.DE.Save t

    # Specify that we want to use our lua script
    Lua.Script md_store.lua

    # ForceAuxCell is not really needed here, but you will need it if you are
    # computing a periodic system only at the Gamma point.
    ForceAuxCell t

    # And then the information about the structure

    # The lattice is just a box big enough so that periodic images don't interact.
    LatticeConstant 1.0 Ang
    %block LatticeVectors
    10.00000000 0.00000000 0.00000000
    0.00000000 10.00000000 0.00000000
    0.00000000 0.00000000 10.00000000
    %endblock LatticeVectors

    # Two species, Oxygen and Hydrogen
    NumberOfSpecies 2
    %block ChemicalSpeciesLabel
    1 8 O
    2 1 H
    %endblock ChemicalSpeciesLabel

    # The coordinates of the water molecule
    NumberOfAtoms 3
    AtomicCoordinatesFormat Ang
    %block AtomicCoordinatesAndAtomicSpecies
    5.00000000  5.00000000  0.11926200 1 # 1: O
    5.00000000  5.76323900 -0.47704700 2 # 2: H
    5.00000000  4.33683900 -0.47704700 2 # 3: H
    %endblock AtomicCoordinatesAndAtomicSpecies

We have the `fdf` file and our `lua` script, the only thing missing are the pseudopotentials.
If you have some for O and H already, you can use them, otherwise we can use the ones from `pseudo-dojo <https://www.pseudo-dojo.org/>`_.
Download the pseudopotential files from there and make sure to copy them as ``O.psml`` and ``H.psml`` in the current directory.

Let's now run SIESTA and see how the dataset is being generated!

.. code-block:: bash

    siesta < RUN.fdf | tee RUN.out

In around 5 minutes, the MD should have ended and you should have a directory called ``MD_steps`` with the matrices
for the 50 steps inside it. Make sure that each step directory contains the ``RUN.fdf``, the ``siesta.XV`` file, which
contains the coordinates for that step, and ``siesta.TSDE`` and ``siesta.TSHS`` files containing the matrices.

You should also check that there is a ``basis`` directory that contains the basis set for each atom.

We have the data and **we are now ready to train a model**!

Training the MACE matrix model
------------------------------

With the data in your hands, you could train whatever model that you wish. In fact, you could design different models and
see which one works best.

However, in this case we just want to train one of the built-in models based on `MACE <https://github.com/ACEsuit/mace>`_.

It's now time to create the ``training`` directory and start training!

.. code-block:: bash

    cd .. # Go to the root (graph2mat_tutorial) directory
    mkdir training
    cd training

The ``graph2mat`` CLI has a ``models`` subcommand where you will be able to access the built-in models. The CLI uses
`pytorch_lightning <https://lightning.ai/pytorch-lightning/>`_ and in paricular the `Lightning CLI <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.cli.LightningCLI.html>`.
Integrated on it you have the three main steps of the training process: ``fit``, ``test`` and ``predict``.

You will just need a ``yaml`` file that specifies all the configuration, split into three sections:
    - ``data``: Specifies which data to use and how to load it.
    - ``model``: Specifies the details of the model (in this case the MACE matrix model).
    - ``trainer``: Specifies the details of the training process.

You can do:

.. code-block:: bash

    graph2mat models mace main fit --help

to get a message with all the available options, but we will start with the simplest thing possible.
This is what our minimal yaml file will look like:

.. code-block:: yaml

    data:
        # We want to fit the density matrix, change to hamiltonian or energy_density_matrix
        # if you want to fit those.
        out_matrix: density_matrix
        # Specify that it is a symmetric matrix (will save operations and predictions will be
        # strictly symmetric)
        symmetric_matrix: True
        # Where to find the basis files (change to *.ion.xml if the basis was not stored
        # in NETCDF format, i.e. nc)
        basis_files: ../dataset/MD_steps/basis/*.ion.nc
        # Where to find the run files. Sisl will attempt to read the matrix from these files.
        train_runs: ../dataset/MD_steps/*/RUN.fdf
        # Data will be split in batches during the training process. Specify how big these
        # batches should be
        batch_size: 10
        # Keep the matrices loaded in memory so that we don't need to read them each time.
        # (This might not be possible for very big datasets)
        store_in_memory: True
    model:
        # We could leave this empty and just use the defaults, but for the sake
        # of learning, we will mention some of the model's most important parameters.
        # FIRST, MACE PARAMETERS
        # Number of times that messages are sent through the graph.
        num_interactions: 1
        # Number that determines how you take into account many-body interactions
        # The higher, the more complex the interactions. 1 means just interact through pairs.
        correlation: 1
        # Maximum order of spherical harmonics used internally by mace.
        # This should at least be as high as your highest order orbital.
        max_ell: 2
        # Size of MACE's internal representation. Here 10 scalars, 10 vectors, and
        # 10 order 2 spherical harmonics. Increasing the number of features will most
        # likely increase the performance if you have enough data.
        hidden_irreps: 10x0e + 10x1o + 10x2e
        # The loss function to use for the optimizer. You can use any of the functions
        # in graph2mat.data.metrics. This is part of the training process, but
        # LightningCLI requires it here for some strange reason.
        loss: graph2mat.metrics.block_type_mae
        # The learning rate for the optimizer. Increasing this might make the learning
        # faster and/or increase performance, but increasing it too much might make
        # the optimizer diverge. It can also make the learning more noisy.
        optim_lr: 0.005
    trainer:
        # Run training on cpu (change to gpu if you have a GPU).
        accelerator: cpu
        # Define how the results of the training process will be logged.
        # Everything will be stored in a lightning_logs/my_first_model directory.
        # Change the name for other models that you train.
        logger:
            class_path: TensorBoardLogger
            init_args:
                name: my_first_model
                save_dir: lightning_logs
        # Number of times the training process goes over the whole dataset (one epoch)
        # We could set it to something very high if we want to stop it manually when we
        # are satisfied.
        max_epochs: 500

Are you ready for your first matrix training? You can now save these contents into a file called ``config.yaml`` and start the training process with:

.. code-block:: bash

    graph2mat models mace main fit -c config.yaml

First, you may see some torch warnings, but don't worry, these are normal! After that, you should see something like:

.. code-block:: bash

    GPU available: False, used: False
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    HPU available: False, using: 0 HPUs

Which tells you the resources you are using for training (in this case the CPU), and then a summary of your model size:

.. code-block:: bash

    | Name  | Type       | Params | Mode
    ---------------------------------------------
    0 | model | MatrixMACE | 41.8 K | train
    ---------------------------------------------
    41.8 K    Trainable params
    0         Non-trainable params
    41.8 K    Total params
    0.167     Total estimated model params size (MB)



This tells you how many parameters can the optimizer tweak in your model to fit the data.
If you play with the config file you should see this changing.

After that, you'll see some progress bar flashing through the epochs. **Congratulations, you are officially training your first matrix model!**

**Track progress**

You can just look at the log output, but you'll hardly get any insight from it.
The best way to track the training progress is to use ``tensorboard``, which you can install with ``pip``:

.. code-block:: bash

    pip install tensorboard

After that, you start tensorboard in the ``training`` directory with (in a separate terminal if training is still running):

.. code-block:: bash

    tensorboard --logdir lightning_logs

It will prompt you to open http://localhost:6006/ in your browser.
If you do that, you'll see a bunch of metrics and their evolution through training.
Probably the most important ones are **the validation metrics**, prefixed with ``val_``.
One useful feature of ``tensorboard`` is that you can pin some graphs to the top of the page, so that you can easily track them at the same time.

By clicking `here <http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_edge_mean%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_edge_max%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_node_mean%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22val_node_max%22%7D%5D#timeseries>`_,
you will get the mean and max absolute errors from nodes and edges pinned.
You will probably also want to set the log scale by clicking on the right-top corner menu.

If you are running 500 epochs on CPU, it should finish after less than 10 minutes.
The mean errors will probably be somewhere around 0.001, and the max errors around 0.01.
Errors will probably still be decreasing, which means that if you continued training you
would get a better model. But this is good enough for us to continue the tutorial.

Using the trained model from the CLI
------------------------------------

We now have a model that is supposedly good at predicting the density matrix of a water molecule.
You can find **a checkpoint files, containing the model's parameters** at particular steps,
in the ``lightning_logs/my_first_model/version_0/checkpoints`` directory (change version number if you want to use another one).
There you will see a ``best-X.ckpt`` and a ``last.ckpt`` file. They contain the best performing parameters
and the last parameters, respectively.

Now, how do we use these models?

Until now, we have just used the ``fit`` subcommand. It's now time to introduce two new subcommands:

    - ``test``: This will test the model on the structures you provide and give you a report of the performance.
    - ``predict``: This will use the model to predict the matrix for a new structure.

But before we use them, it is wise to understand the concept of `lightning callbacks <https://lightning.ai/docs/pytorch/stable/extensions/callbacks.html>`_.
They contain functionality that is used at the beggining/end of epochs/batches.
They are meant to be used as **plugins for the training, testing and predicting processes**.
In ``graph2mat``, we provide the following callbacks:

    - ``MatrixWriter``: Writes the computed matrices to files.
    - ``SamplewiseMetricsLogger``: Creates a csv file with the metrics individually for each structure in the dataset.
    - ``PlotMatrixError``: Plots the error of the matrices. It can add the plots to the logger or show them in the browser.

We will use them throughout this section.

First, let's say we want to test how good our model does in a particular structure in the dataset.
We can use the ``test`` subcommand to do that. It will need:

    - The checkpoint file with the model that we want to test. Passed to ``--ckpt_path``.
    - The paths of the structures that we want to test. Passed to ``--data.test_runs``.
    - Optionally, some callbacks to get more details.

To test structure 25, we can do (change name of the checkpoint by your best performing one):

.. code-block:: bash

    graph2mat models mace main test \
       --ckpt_path lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt  \
       --data.test_runs ../dataset/MD_steps/25/RUN.fdf \
       --trainer.callbacks+ PlotMatrixError --trainer.callbacks.show True \
       --trainer.callbacks+ SamplewiseMetricsLogger

This will have three outcomes:

    - In the **terminal you will see a quick summary** of the testing process, as a table. Something like this:

.. code-block:: bash

    ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃        Test metric        ┃       DataLoader 0        ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
    │       test_edge_max       │   0.0035408437252044678   │
    │      test_edge_mean       │   0.0003855021495837718   │
    │       test_edge_std       │   0.0006314062047749758   │
    │         test_loss         │   0.0007163776317611337   │
    │       test_node_max       │   0.006912112236022949    │
    │      test_node_mean       │   0.0003308755112811923   │
    │       test_node_std       │   0.0007246236782521009   │
    └───────────────────────────┴───────────────────────────┘

..

    - Due to the ``PlotMatrixError`` callback, a **plot with the matrix error** should pop **in your browser**. There you will be able to see which matrix elements have the largest error.
    - Due to the ``SamplewiseMetricsLogger`` callback, a ``sample_metrics.csv`` **file with the metrics for each structure** will be created.

Finally, we can use the model to write predictions to files.
This is possible by using the ``predict`` subcommand and the ``MatrixWriter`` callback:

.. code-block:: bash

    graph2mat models mace main predict \
       --ckpt_path lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt  \
       --data.predict_structs ../dataset/MD_steps/*/RUN.fdf \
       --trainer.callbacks+ MatrixWriter --trainer.callbacks.output_file ML_prediction.DM

This will write the predictions to each directory of the ``MD_datataset`` folder.
With this, you can do whatever you want. For example, you could **run a SIESTA calculation
using the prediction as an initial DM**, do some **further analysis of the errors** or **use it
as the true density matrix** of the system.

There is one particular use case that requires some more advanced usage of the models: using the
predictions as initial DM for each step of a MD run. This is what we will discuss in the following section.

Molecular dynamics with predictions
-----------------------------------

In this section we are going to use our model to **produce predictions for the next 50 steps of the MD**.
This process involves two parts:

  - Launching a python server that will produce the predictions.
  - In SIESTA, running a lua script that will request predictions for each MD step.

First, we will create a directory where we will run the molecular dynamics. Let's call it ``MD_continuation``.

.. code-block:: bash

    cd .. # Go to the root (graph2mat_tutorial) directory
    mkdir MD_continuation
    cd MD_continuation

Now, let's set it up. We will use the ``graph2mat siesta md setup`` command:

.. code-block:: bash

    graph2mat siesta md setup --ml 0 --inplace

We just asked to set up the current directory (``--inplace``) to use a ML model with 0 history depth. We will see what
this 0 means later, right now all you have to know is that predictions of the model will be used directly as the first
guess for the DM.

The command will create an ``graph2mat.fdf`` file containing the logic for initializing the DM at each step. If you open it,
you will see that it contains:

    - Some fdf keys.
    - The inclusion of the ``graph2mat.lua`` script.

The ``graph2mat.lua`` script is really what gets the predictions. At each step, it requests predictions to a server that
is running the ML model. This means that there has to be a server running, we will cover this in a moment!

But first, let's set up the rest of the inputs for the MD run. We need:

    - The pseudopotential files ``O.psml`` and ``H.psml``.
    - The file containing the last step of the dataset, which is the ``siesta.XV`` file inside the dataset directory.
    - The fdf file for the MD run (`RUN.fdf`). It looks very similar to the one we used to build the dataset, with the
    difference that we will ask for the XV file to be used, we won't include the ``md_store`` lua script and
    we will include the ``graph2mat.fdf`` file:

.. code-block::

    # Include the file for DM initialization at each step
    %include graph2mat.fdf
    # Use the siesta.XV file as the initial coordinates for the MD
    MD.UseSaveXV t

    # The rest are just the options that we used to generate the dataset,
    # except that we removed the lua script line.

    # Run 50 steps of a verlet MD
    MD.TypeOfRun verlet
    MD.Steps 50

    # Use the default Double Zeta Polarized basis.
    PAO.BasisSize DZP

    # Save all matrices
    TS.HS.Save t
    TS.DE.Save t

    # ForceAuxCell is not really needed here, but you will need it if you are
    # computing a periodic system only at the Gamma point.
    ForceAuxCell t

    # And then the information about the structure

    # The lattice is just a box big enough so that periodic images don't interact.
    LatticeConstant 1.0 Ang
    %block LatticeVectors
    10.00000000 0.00000000 0.00000000
    0.00000000 10.00000000 0.00000000
    0.00000000 0.00000000 10.00000000
    %endblock LatticeVectors

    # Two species, Oxygen and Hydrogen
    NumberOfSpecies 2
    %block ChemicalSpeciesLabel
    1 8 O
    2 1 H
    %endblock ChemicalSpeciesLabel

    # The coordinates of the water molecule
    NumberOfAtoms 3
    AtomicCoordinatesFormat Ang
    %block AtomicCoordinatesAndAtomicSpecies
    5.00000000  5.00000000  0.11926200 1 # 1: O
    5.00000000  5.76323900 -0.47704700 2 # 2: H
    5.00000000  4.33683900 -0.47704700 2 # 3: H
    %endblock AtomicCoordinatesAndAtomicSpecies

We are now ready to run the MD. But first, we need to **start the server that will produce the predictions**!

Open a new terminal and type (from the ``graph2mat_tutorial`` directory):

.. code-block:: bash

    graph2mat serve training/lightning_logs/my_first_model/version_0/checkpoints/best-2040.ckpt

replacing ``best-2040.ckpt`` by the checkpoint file that you have. If everything was succesful,
you should see something like:

.. code-block:: bash

    INFO:     Started server process [121733]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://localhost:56000 (Press CTRL+C to quit)

Congratulations, the server is running! You can check that it is working fine by using the ``request`` command:

.. code-block:: bash

    graph2mat request avail-models

Which should return a list of the available model names, in this case ``["0"]``. You should also see in the server
output that it succesfully handled the request.

We are ready now to run the MD using the predictions, let's do it! We just need to run siesta as usual:

.. code-block:: bash

    siesta < RUN.fdf | tee RUN.out

When the run finishes, it is time to evaluate how it went. A quick way to have an impression of the
performance is to run:

.. code-block:: bash

    graph2mat siesta md analyze RUN.out

It will open a table in the browser summarizing the performance of the SCF cycles. You can also
save the results to a csv file with the ``--save`` option.

.. code-block:: bash

    # Ask only for the mean
    graph2mat siesta md analyze RUN.out --agg mean --save results.csv

But these results are not enough to understand if using the ML predictions was beneficial. For that,
we need to compare with how it performed previously. Step back one directory and pass both output
files to ``analyze``:

.. code-block:: bash

    cd .. # To the graph2mat_tutorial directory
    graph2mat siesta md analyze dataset/RUN.out MD_continuation/RUN.out

The table should now contain the metrics for both runs. Has the ML improved the performance?

Whatever the answer is, take into account that **this is a super simple ML model**. It is a very small
model, and it has been trained only on 50 structures for a very short time.

Benchmarking molecular dynamics
-----------------------------------

At the end of the last section, we have seen how to compare the performance of two MD runs. However,
we have compared two different runs. A more robust test would be to test on exactly the same run.
We can also test a more diverse set of DM initialization methods. For example:

    - **Atomic densities** (*siesta_0*). This is what we do when we have no information about the system.
    - **DM from the last step** (*siesta_1*). This is the simplest approach to use the information from the dynamics.
    - **Extrapolating from the last 7 steps** (*siesta_7*). SIESTA has a built-in simple extrapolation method that can use
        the last **N** steps to extrapolate a new DM based on the atomic coordinates.
    - **ML predictions** (*ml_0*). The most straightforward method to use the predictions from the model.
    - **ML predictions + last step error** (*ml_1*). This is a very simple correction to the ML predictions, which adds
       the error from the previous step to the prediction, so that the initial DM is "prediction + previous error".

Directories with the necessary inputs for these five methods can be created with the following command:

.. code-block:: bash

    graph2mat siesta md setup --ml 0,1 --siesta 0,1,7

Follow the same procedure that we followed in the previous section to run the MD for each of these directories.
Remember, you will need to start the server for the ML prediction runs (not for the ``siesta_*`` ones)!
