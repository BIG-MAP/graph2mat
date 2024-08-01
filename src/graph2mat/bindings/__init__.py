"""Specific interfaces to other codes.

The core functionality of `graph2mat` is agnostic to the framework,
and it is based on pure python and `numpy`.

For running a specific ML workflow, we need interfaces of the core
functionality with ML frameworks, e.g. `torch`, `e3nn`. The implementation
of the interfaces are mostly thin wrappers around the core functionality,
as well as functions that only make sense on the specific framework. For
example, the `e3nn` bindings contains functions that use irreps.

Whatever framework that we interface with, it should not be a required import
for the core functionality. So basically, the criteria for creating a new
submodule in `bindings` is that we can't add the functionality to the core
without requiring the framework as a dependency.
"""
