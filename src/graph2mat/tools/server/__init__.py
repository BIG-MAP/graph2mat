"""Utilites to serve and request matrix predictions.

Initializing a model takes time. For this reason, sometimes
it is useful to have a server running the model, waiting for
requests to compute predictions.

This module implements a very simple HTTP server to provide matrix
predictions, as well as a very simple HTML front end and a very
simple python client API so that you are not forced to use raw requests.

Launching the server is easiest from the ``e3mat`` CLI with ``e3mat serve``.
You can also use ``e3mat request``, which uses the client.
"""

from .api_client import ServerClient
from .server_app import create_server_app, create_server_app_from_filesystem

__all__ = ["ServerClient", "create_server_app", "create_server_app_from_filesystem"]
