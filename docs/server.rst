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
